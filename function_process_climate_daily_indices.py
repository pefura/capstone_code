# FUNCTION FOR DAILY SPATIAL AGGREGATION OF CLIMATE DATA ONLY

import xarray as xr
import numpy as np
import pandas as pd
import os
import traceback
import zarr # Ensure zarr is imported
import regionmask
from dask.diagnostics import ProgressBar # To show progress on compute()
import warnings
# Ensure pyarrow is installed for parquet support
try:
    import pyarrow
except ImportError:
    print("Warning: pyarrow not found. Parquet saving might fail or use a different engine.")


def process_climate_daily(
    # --- Input/Output Paths ---
    zarr_store_path: str,
    output_dir: str,
    output_filename_base: str, # Base name for the aggregated climate output file

    # --- Processing Config ---
    initial_time_dim_name: str = "time",
    dask_chunks_config: dict = {'time': 'auto', 'latitude': 'auto', 'longitude': 'auto'},
    # *** PARAMETER FOR TIMEDELTA CONVERSION ***
    timedelta_vars_to_numeric: list = None, # e.g., ['HeatwaveIndex']

    # --- Climate Aggregation Config ---
    # perform_spatial_aggregation is assumed True for this function's purpose
    spatial_aggregation_level: str = "states", # Currently only 'states' for regionmask
    spatial_aggregation_method: str = 'mean', # Spatial agg method (e.g., 'mean', 'max')
    key_region_abbrs: list = None,        # GENERIC: List of region abbreviations (e.g., ['KS', 'IA'] or None if no filtering)

    # --- Output Format Config ---
    save_parquet: bool = True,
    save_csv: bool = False,
    verbose: bool = True
) -> dict:
    """
    Loads daily gridded climate data (from Zarr), aggregates it spatially
    (e.g., to states), converts specified timedelta climate variables to
    numeric (total seconds), and saves the resulting daily aggregated
    climate data as a DataFrame.

    NOTE: Assumes the input climate Zarr store contains DAILY data.
    NOTE: This function specifically performs spatial aggregation.

    Args:
        zarr_store_path (str): Full path to the climate indices Zarr store (expected to be daily).
        output_dir (str): Directory to save the output file(s).
        output_filename_base (str): Base name for the output file(s).
        initial_time_dim_name (str, optional): Initial guess for the time dimension name in Zarr. Defaults to "time".
        dask_chunks_config (dict, optional): Chunking configuration for Dask.
        timedelta_vars_to_numeric (list, optional): List of climate variable root names
                                                    (e.g., ['HeatwaveIndex']) to convert to numeric.
        spatial_aggregation_level (str, optional): Spatial aggregation level (currently only 'states'). Defaults to "states".
        spatial_aggregation_method (str, optional): Method for spatial aggregation ('mean', 'max', etc.). Defaults to 'mean'.
        key_region_abbrs (list, optional): List of region abbreviations to select after aggregation. Defaults to None.
        save_parquet (bool, optional): Save output as Parquet. Defaults to True.
        save_csv (bool, optional): Save output as CSV. Defaults to False.
        verbose (bool, optional): Print detailed progress messages. Defaults to True.

    Returns:
        dict: Status dictionary including output file paths if successful.
    """
    if verbose:
        print(f"--- Function Start: process_climate_daily ---")
        print(f"Zarr Store Path: {zarr_store_path} (Expected Daily)")
        print(f"Output Directory: {output_dir}")
        print(f"Output Base Name: {output_filename_base}")
        print(f"Spatial Aggregation: Level={spatial_aggregation_level}, Method={spatial_aggregation_method}, Key Regions={key_region_abbrs or 'All'}")
        if timedelta_vars_to_numeric:
            print(f"Timedelta variables to convert to numeric: {timedelta_vars_to_numeric}")
        print("-" * 50)

    # --- Validation ---
    if spatial_aggregation_level != "states":
         print("Warning: Currently only spatial_aggregation_level='states' is fully implemented with regionmask.")
         # raise NotImplementedError("Only spatial_aggregation_level='states' is currently supported.")

    ds_indices_daily = None # Input daily dataset
    climate_features_daily_df = None # Output aggregated dataframe
    aggregated_computed = None      # Intermediate computed dataset
    final_parquet_path = None
    final_csv_path = None
    time_dim_name = initial_time_dim_name

    try:
        # --- Step 1: Load Climate Data --- (Equivalent to Step 2 previously)
        if not os.path.exists(zarr_store_path): raise FileNotFoundError(f"Zarr store not found: {zarr_store_path}")
        if verbose: print(f"\n--- Step 1: Loading Climate Data (Expected Daily) ---")
        try:
            # Load Zarr
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=FutureWarning) # Ignore xarray's cftimeindex warning
                 ds_indices_daily = xr.open_zarr(zarr_store_path, consolidated=True, chunks={})
            if verbose: print(f"  Climate Zarr store loaded.")

             # Verify/Update Time Dim Name
            possible_time_dims = [initial_time_dim_name] + [d for d in ds_indices_daily.dims if d != initial_time_dim_name and ('time' in d.lower() or 'day' in d.lower() or 'date' in d.lower())]
            found_time_dim = False
            for dim in possible_time_dims:
                if dim in ds_indices_daily.dims:
                    if dim != initial_time_dim_name:
                        print(f"  Using time dim name: '{dim}' (found)")
                    else:
                         if verbose: print(f"  Confirmed time dimension: '{dim}'")
                    time_dim_name = dim
                    found_time_dim = True
                    break
            if not found_time_dim:
                 raise ValueError(f"Could not find suitable time dimension (checked: {possible_time_dims}) in Zarr store dims: {list(ds_indices_daily.dims)}")


            # Verify/Rename Coordinates
            rename_dict = {}
            current_coords = list(ds_indices_daily.coords)
            if 'latitude' not in current_coords:
                potential_lat = next((c for c in current_coords if 'lat' in c.lower()), None)
                if potential_lat: rename_dict[potential_lat] = 'latitude'
            if 'longitude' not in current_coords:
                potential_lon = next((c for c in current_coords if 'lon' in c.lower() or 'long' in c.lower()), None)
                if potential_lon: rename_dict[potential_lon] = 'longitude'
            if rename_dict: ds_indices_daily = ds_indices_daily.rename(rename_dict); print(f"  Renamed coordinates: {rename_dict}")
            if not all(c in ds_indices_daily.coords for c in ['latitude', 'longitude']): raise ValueError("Missing lat/lon coords.")

            # Remove variables without the time dimension
            vars_to_keep = [v for v in ds_indices_daily.data_vars if time_dim_name in ds_indices_daily[v].dims]
            if len(vars_to_keep) < len(ds_indices_daily.data_vars):
                 dropped_vars = set(ds_indices_daily.data_vars) - set(vars_to_keep)
                 if verbose: print(f"  Dropping climate variables without time dimension '{time_dim_name}': {list(dropped_vars)}")
                 ds_indices_daily = ds_indices_daily[vars_to_keep]

        except (FileNotFoundError, ValueError) as e: raise
        except Exception as e: raise RuntimeError(f"Error loading Zarr data: {e}") from e

        # --- Step 2: Process Climate Data (Aggregation) --- (Equivalent to Step 3 previously)
        if verbose: print("\n--- Step 2: Processing Climate Data (Daily Aggregation) ---")
        try:
            # Apply Dask chunking
            chunks_to_apply = { k: v for k, v in dask_chunks_config.items() if k in ds_indices_daily.dims }
            if chunks_to_apply: ds_indices_daily = ds_indices_daily.chunk(chunks_to_apply)
            if verbose and chunks_to_apply: print("  Applied Dask chunks.")

            # Perform Spatial Aggregation
            if verbose: print(f"  Performing spatial aggregation (Level: {spatial_aggregation_level}, Method: {spatial_aggregation_method})...")
            if spatial_aggregation_level == "states":
                # --- Sub-try block specifically for compute and conversion ---
                try:
                    # Create Mask
                    if verbose: print("    Creating US States region mask...")
                    us_states_mask = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
                    lon_coord = ds_indices_daily['longitude']
                    # Check if data longitude needs wrapping
                    if lon_coord.min() >= 0 and lon_coord.max() > 180:
                        if verbose: print("      Attempting to wrap data longitude from 0-360 to -180-180...")
                        ds_indices_daily.coords['longitude'] = (lon_coord + 180) % 360 - 180
                        ds_indices_daily = ds_indices_daily.sortby('longitude')
                        if verbose: print("      Data longitude wrapped and sorted.")
                        lon_coord = ds_indices_daily['longitude'] # Update reference

                    mask_da = us_states_mask.mask(
                        ds_indices_daily['longitude'],
                        ds_indices_daily['latitude'],
                        wrap_lon=False
                    ).rename("region")

                    mask_chunks = {k: v for k, v in dask_chunks_config.items() if k in mask_da.dims and k != time_dim_name}
                    if mask_chunks: mask_da = mask_da.chunk(mask_chunks)
                    if verbose: print(f"    Region mask created.")

                    # Aggregate
                    if verbose: print(f"    Aggregating spatially using '{spatial_aggregation_method}'...")
                    agg_func = getattr(ds_indices_daily.groupby(mask_da), spatial_aggregation_method)
                    regional_indices_daily_ds = agg_func(skipna=True)
                    if 'region' not in regional_indices_daily_ds.dims: raise ValueError("Dim 'region' missing after groupby.")
                    number_to_abbr = dict(zip(us_states_mask.numbers, us_states_mask.abbrevs))
                    present_abbrs = [number_to_abbr.get(int(num), f"Unk_{int(num)}") for num in regional_indices_daily_ds['region'].values]
                    regional_indices_daily_ds = regional_indices_daily_ds.assign_coords(region_abbr=('region', present_abbrs))
                    if verbose: print("    Spatial aggregation & region coords assigned.")

                    # Select Key Regions (if specified)
                    final_regional_ds = regional_indices_daily_ds
                    if key_region_abbrs:
                        if verbose: print(f"    Selecting key regions: {key_region_abbrs}")
                        valid_key_regions = list(key_region_abbrs) if key_region_abbrs else []
                        if valid_key_regions:
                            if 'region_abbr' not in final_regional_ds.coords:
                                raise ValueError("Coordinate 'region_abbr' not found in aggregated dataset before selection.")
                            final_regional_ds = regional_indices_daily_ds.sel(
                                region=regional_indices_daily_ds.region_abbr.isin(valid_key_regions)
                            )
                            if final_regional_ds.dims.get('region', 0) == 0:
                                print(f"      Warning: No data found for specified key regions.")
                            elif verbose: print(f"      Key regions selected. Count: {final_regional_ds.dims['region']}")
                        else:
                            if verbose: print("      No key regions specified, keeping all aggregated regions.")
                    else:
                         if verbose: print("      No key regions specified, keeping all aggregated regions.")

                    # Compute Aggregated Data
                    if verbose: print("    Computing aggregated climate data...")
                    with ProgressBar(): aggregated_computed = final_regional_ds.compute()
                    if verbose: print("    Computation complete.")
                    if not isinstance(aggregated_computed, xr.Dataset):
                         raise TypeError(f"Expected xr.Dataset after compute, but got {type(aggregated_computed)}")

                    # Convert Computed Aggregated Data to DataFrame
                    if verbose: print("    Converting aggregated data to Pandas DataFrame...")
                    if aggregated_computed.dims.get('region', 0) == 0:
                         print("      Warning: Aggregated dataset has 0 regions. Creating empty DataFrame.")
                         time_vals = aggregated_computed[time_dim_name].values if time_dim_name in aggregated_computed else []
                         time_idx = pd.to_datetime(np.array(time_vals, dtype='datetime64[ns]'))
                         if len(time_idx) > 0: time_idx = time_idx.tz_localize('UTC')
                         else: time_idx = pd.Index([], dtype='datetime64[ns, UTC]')
                         climate_features_daily_df = pd.DataFrame(index=pd.Index(time_idx, name='Date'))
                    else:
                         df_multi = aggregated_computed.to_dataframe()
                         expected_levels = [time_dim_name, 'region']
                         if not isinstance(df_multi.index, pd.MultiIndex) or not all(level in df_multi.index.names for level in expected_levels):
                             if verbose: print(f"    Warning: Unexpected index structure after to_dataframe(): {df_multi.index.names}. Resetting and setting index.")
                             df_multi = df_multi.reset_index()
                             missing_cols = [col for col in expected_levels if col not in df_multi.columns]
                             if missing_cols:
                                  raise ValueError(f"Cannot find expected columns {missing_cols} after reset_index. Available: {df_multi.columns.tolist()}")
                             if time_dim_name in df_multi.columns:
                                 df_multi[time_dim_name] = pd.to_datetime(df_multi[time_dim_name])
                             else: raise ValueError(f"Time dimension column '{time_dim_name}' missing after reset_index.")
                             if 'region' not in df_multi.columns: raise ValueError(f"Region column 'region' missing after reset_index.")
                             df_multi = df_multi.set_index(expected_levels)
                         else:
                             time_level_name = df_multi.index.names[df_multi.index.names.index(time_dim_name)]
                             if not pd.api.types.is_datetime64_any_dtype(df_multi.index.get_level_values(time_level_name)):
                                  if verbose: print(f"    Warning: Time level '{time_level_name}' is not datetime. Attempting conversion.")
                                  current_levels = list(df_multi.index.levels)
                                  current_names = list(df_multi.index.names)
                                  time_idx_pos = current_names.index(time_level_name)
                                  current_levels[time_idx_pos] = pd.to_datetime(current_levels[time_idx_pos])
                                  df_multi.index = pd.MultiIndex.set_levels(df_multi.index, current_levels)

                         df_wide = df_multi.unstack(level='region')
                         if 'region_abbr' not in aggregated_computed.coords:
                              raise ValueError("Coordinate 'region_abbr' not found in computed dataset for column renaming.")
                         region_abbr_map = dict(zip(aggregated_computed.region.values, aggregated_computed.region_abbr.values))
                         new_cols = [f"{var}_{region_abbr_map.get(int(num), f'Unk{int(num)}')}" for var, num in df_wide.columns]
                         df_wide.columns = new_cols
                         climate_features_daily_df = df_wide
                         if verbose: print("      DataFrame conversion and structuring complete.")

                # --- Catch errors specifically from compute/conversion ---
                except Exception as agg_conv_err:
                     print(f"ERROR during climate aggregation/computation or DataFrame conversion: {agg_conv_err}")
                     traceback.print_exc()
                     raise RuntimeError(f"Climate processing failed during aggregation/computation.") from agg_conv_err

            else: # spatial_aggregation_level != 'states'
                raise NotImplementedError(f"Spatial aggregation level '{spatial_aggregation_level}' not implemented.")


            # --- Check if climate_features_daily_df was successfully created ---
            if climate_features_daily_df is None:
                 raise RuntimeError("Climate features DataFrame was not created after processing step (check aggregation/computation errors).")

            # --- Finalize Climate DataFrame Index ---
            if verbose: print("  Ensuring final climate DataFrame index is DatetimeIndex/UTC/Name='Date'...")
            if not isinstance(climate_features_daily_df.index, pd.DatetimeIndex):
                if verbose: print(f"    Index is not DatetimeIndex ({climate_features_daily_df.index.dtype}), converting...")
                climate_features_daily_df.index = pd.to_datetime(climate_features_daily_df.index)
            if climate_features_daily_df.index.tz is None:
                if verbose: print("    Localizing index to UTC...")
                climate_features_daily_df.index = climate_features_daily_df.index.tz_localize('UTC')
            elif str(climate_features_daily_df.index.tz) != 'UTC':
                if verbose: print(f"    Converting index from {climate_features_daily_df.index.tz} to UTC...")
                climate_features_daily_df.index = climate_features_daily_df.index.tz_convert('UTC')
            climate_features_daily_df.index.name = 'Date'
            if verbose: print(f"    Climate index type: {climate_features_daily_df.index.dtype}")

            # --- Convert specified timedelta columns to numeric (total seconds) ---
            if timedelta_vars_to_numeric and not climate_features_daily_df.empty:
                if verbose: print(f"  Converting specified timedelta variables to numeric (total seconds)...")
                converted_cols_count = 0
                cols_already_numeric = 0
                cols_not_found = 0
                for var_root in timedelta_vars_to_numeric:
                    pattern = f"{var_root}_"
                    cols_to_check = [col for col in climate_features_daily_df.columns if col.startswith(pattern)]
                    if not cols_to_check:
                        if verbose: print(f"    No columns found starting with '{pattern}'.")
                        cols_not_found +=1
                        continue
                    for col in cols_to_check:
                        if col in climate_features_daily_df.columns:
                            if pd.api.types.is_timedelta64_dtype(climate_features_daily_df[col]):
                                try:
                                    climate_features_daily_df[col] = climate_features_daily_df[col].dt.total_seconds()
                                    converted_cols_count += 1
                                except Exception as conv_e: print(f"    Warning: Error converting column '{col}' to numeric: {conv_e}")
                            elif pd.api.types.is_numeric_dtype(climate_features_daily_df[col]): cols_already_numeric += 1
                            else:
                                col_dtype_before = climate_features_daily_df[col].dtype
                                try:
                                    climate_features_daily_df[col] = pd.to_numeric(climate_features_daily_df[col], errors='coerce')
                                    if climate_features_daily_df[col].isnull().all():
                                         if verbose: print(f"    Warning: Column '{col}' for variable '{var_root}' became all NaNs after numeric conversion attempt (original dtype: {col_dtype_before}).")
                                    else:
                                         if pd.api.types.is_numeric_dtype(climate_features_daily_df[col]):
                                             cols_already_numeric += 1
                                             if verbose: print(f"    Info: Column '{col}' for variable '{var_root}' converted to numeric (original dtype: {col_dtype_before}).")
                                         else:
                                             if verbose: print(f"    Warning: Column '{col}' for variable '{var_root}' did not become numeric after conversion attempt (original dtype: {col_dtype_before}, current: {climate_features_daily_df[col].dtype}). Skipping.")
                                except Exception as type_err:
                                     if verbose: print(f"    Warning: Column '{col}' for variable '{var_root}' is not timedelta and failed numeric conversion (dtype: {col_dtype_before}). Error: {type_err}. Skipping.")

                if verbose: print(f"    Timedelta conversion summary: {converted_cols_count} converted, {cols_already_numeric} numeric/converted, {cols_not_found} variable roots not found.")


        # --- Outer except block for Step 2 ---
        except (ValueError, KeyError, NotImplementedError, RuntimeError) as e: # Catch specific errors raised within
             raise # Re-raise them to be caught by the main function handler
        except Exception as e: # Catch unexpected errors during Step 2
             raise RuntimeError(f"Unexpected error during climate processing (Step 2): {e}") from e
        finally:
            # Ensure the dataset is closed even if errors occur
            if ds_indices_daily is not None:
                 try: ds_indices_daily.close()
                 except Exception: pass # Ignore errors during close if dataset is already problematic
                 if verbose: print("  Climate indices dataset closed.")


        # --- Step 3: Save Aggregated Climate Data --- (Equivalent to Step 5 previously)
        if verbose: print("\n--- Step 3: Saving Aggregated Climate Data ---")
        os.makedirs(output_dir, exist_ok=True)

        if climate_features_daily_df is None or climate_features_daily_df.empty:
             print("  Skipping save: Aggregated climate DataFrame is None or empty.")
             # You might want to return an error status if the DF is empty, depending on requirements
             # raise ValueError("Aggregated climate DataFrame is empty, cannot save.")
        else:
            # Save Parquet
            if save_parquet:
                final_parquet_path = os.path.join(output_dir, f"{output_filename_base}.parquet")
                if verbose: print(f"  Saving aggregated data to Parquet: {final_parquet_path}")
                try:
                    climate_features_daily_df.to_parquet(final_parquet_path, index=True, engine='pyarrow')
                    if verbose: print("    Parquet saved successfully.")
                except Exception as e:
                     print(f"    Error saving Parquet: {e}")
                     final_parquet_path = None # Indicate failure
            # Save CSV
            if save_csv:
                final_csv_path = os.path.join(output_dir, f"{output_filename_base}.csv")
                if verbose: print(f"  Saving aggregated data to CSV: {final_csv_path}")
                try:
                    climate_features_daily_df.to_csv(final_csv_path, index=True)
                    if verbose: print("    CSV saved successfully.")
                except Exception as e:
                     print(f"    Error saving CSV: {e}")
                     final_csv_path = None # Indicate failure

        # --- Success ---
        if verbose: print(f"\n--- Function process_climate_daily Finished Successfully ({output_filename_base}) ---")
        return {
            'status': 'success',
            'message': 'Daily climate data spatially aggregated and saved.',
            'output_parquet_path': final_parquet_path,
            'output_csv_path': final_csv_path,
            'error_details': None
        }

    # --- Error Handling ---
    except (FileNotFoundError, KeyError, ValueError, NotImplementedError, RuntimeError) as e:
        err_msg = f"Error in process_climate_daily: {e}"
        print(err_msg) # Always print errors
        return {'status': 'error', 'message': err_msg, 'output_parquet_path': None, 'output_csv_path': None, 'error_details': traceback.format_exc()}
    except Exception as e:
        err_msg = f"An unexpected error occurred in process_climate_daily: {e}"
        print(err_msg)
        traceback.print_exc() # Print full traceback for unexpected errors
        return {'status': 'error', 'message': err_msg, 'output_parquet_path': None, 'output_csv_path': None, 'error_details': traceback.format_exc()}


# --- Example Usage ---
if __name__ == "__main__":
    print("*"*20 + " Running Example Usage (Climate Aggregation Only) " + "*"*20)

    # --- !!! IMPORTANT: ADJUST THESE PATHS AND FILENAMES !!! ---
    EXAMPLE_OUTPUT_DIR = '/kaggle/working/'                       # ADJUST AS NEEDED (Where output will be saved)
    EXAMPLE_ZARR_PATH = "/kaggle/working/climate_indices.zarr"    # ADJUST AS NEEDED (Your DAILY climate Zarr store)
    EXAMPLE_KEY_REGIONS = ['KS', 'ND', 'MT', 'OK', 'WA', 'SD', 'CO', 'NE', 'TX', 'MN', 'ID', 'IL'] # ADJUST AS NEEDED (Or None)
    EXAMPLE_OUTPUT_BASE = "daily_aggregated_climate_states_mean" # Sensible base name
    # --- Timedelta Conversion ---
    EXAMPLE_TIMEDELTA_VARS = ['HeatwaveIndex']  # ADJUST or set to None if no timedelta vars
    # --- Initial Time Dim Name Guess ---
    EXAMPLE_INITIAL_TIME_DIM = 'time'           # Adjust if your zarr store uses something else


    # Check if required input file exists
    zarr_exists = os.path.exists(EXAMPLE_ZARR_PATH)
    print(f"Checking Zarr store at: {EXAMPLE_ZARR_PATH}")

    if not zarr_exists:
        print(f"ERROR: Required Zarr store not found: {EXAMPLE_ZARR_PATH}")
    else:
        print("\nCalling DAILY climate aggregation function...")
        result = process_climate_daily( # Call the new climate-only function
            zarr_store_path=EXAMPLE_ZARR_PATH,
            output_dir=EXAMPLE_OUTPUT_DIR,
            output_filename_base=EXAMPLE_OUTPUT_BASE,
            initial_time_dim_name=EXAMPLE_INITIAL_TIME_DIM,
            key_region_abbrs=EXAMPLE_KEY_REGIONS,
            spatial_aggregation_level='states',
            spatial_aggregation_method='mean',
            timedelta_vars_to_numeric=EXAMPLE_TIMEDELTA_VARS,
            verbose=True,
            save_parquet=True,
            save_csv=True
        )

        # --- Print Results ---
        print("\n" + "*"*20 + " Function Call Result " + "*"*20)
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        if result.get('status') == 'success':
            output_file_to_preview = result.get('output_parquet_path') or result.get('output_csv_path')
            if output_file_to_preview and os.path.exists(output_file_to_preview):
                 print(f"\nAggregated Climate Data Head (from {os.path.basename(output_file_to_preview)}):")
                 try:
                     if output_file_to_preview.endswith('.parquet'): preview_df = pd.read_parquet(output_file_to_preview)
                     else: preview_df = pd.read_csv(output_file_to_preview, index_col='Date', parse_dates=True)
                     print(preview_df.head())
                     print(f"\nAggregated Data Shape: {preview_df.shape}")
                     # Verify the dtype of a converted column if specified
                     if EXAMPLE_TIMEDELTA_VARS:
                         first_var_root = EXAMPLE_TIMEDELTA_VARS[0]
                         converted_cols_in_preview = [c for c in preview_df.columns if c.startswith(f"{first_var_root}_")]
                         if converted_cols_in_preview:
                             print(f"\nData type check for '{converted_cols_in_preview[0]}' after load: {preview_df[converted_cols_in_preview[0]].dtype}")
                         else:
                             print(f"\nCould not find columns starting with '{first_var_root}_' in output for dtype check.")

                 except Exception as load_err: print(f"Error loading preview: {load_err}")
            else: print("\nOutput file not found or not saved, cannot preview.")
        else:
            print(f"Error details available in result dictionary if needed.")
            print(f"\n----- Error Details -----")
            print(result.get('error_details'))
            print(f"----- End Error Details -----")

    print("\n--- Example Usage End ---")