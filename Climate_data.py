# -*- coding: utf-8 -*-
import os
import xarray as xr
import numpy as np
from glob import glob
from dask.diagnostics import ProgressBar
import warnings
import shutil
import traceback
import numcodecs # Import numcodecs for compressor objects
import dask.array as da # Import dask array for chunking checks
import dask # For dask.persist if needed
import time # For adding timestamps to checkpoints
import pickle # For saving/loading intermediate state
import zarr # <--- MAKE SURE THIS IMPORT IS PRESENT

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice*")
# Keep invalid value warnings for now
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide*")
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in greater")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in less") # Usually benign with skipna=True
warnings.filterwarnings("ignore", category=FutureWarning, message=".*elementwise comparison failed*")
warnings.filterwarnings("ignore", category=xr.SerializationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Large object passed.+")
warnings.filterwarnings("ignore", category=UserWarning, message="The `compressor` argument is deprecated.*") # Ignore zarr v3 warning
warnings.filterwarnings("ignore", category=UserWarning, message="Variable names are not unique.*") # Ignore if variable names are reused temporarily

# --- Checkpoint Directory ---
CHECKPOINT_DIR = "pipeline_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------
def save_checkpoint(stage_name, data):
    """Saves data to a checkpoint file."""
    filepath = os.path.join(CHECKPOINT_DIR, f"checkpoint_{stage_name}.pkl")
    print(f"--- Saving checkpoint: {stage_name} ---")
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint '{stage_name}' saved successfully.")
    except Exception as e:
        print(f"Error saving checkpoint '{stage_name}': {e}")

def load_checkpoint(stage_name):
    """Loads data from a checkpoint file if it exists."""
    filepath = os.path.join(CHECKPOINT_DIR, f"checkpoint_{stage_name}.pkl")
    if os.path.exists(filepath):
        print(f"--- Loading checkpoint: {stage_name} ---")
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"Checkpoint '{stage_name}' loaded successfully.")
            return data
        except Exception as e:
            print(f"Error loading checkpoint '{stage_name}': {e}. Proceeding without checkpoint.")
            return None
    else:
        # print(f"Checkpoint '{stage_name}' not found.") # Be less verbose if not found
        return None

def clear_checkpoints():
    """Removes all checkpoint files."""
    print("--- Clearing checkpoints ---")
    if os.path.exists(CHECKPOINT_DIR):
        try:
            shutil.rmtree(CHECKPOINT_DIR)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Recreate directory after removing
            print("Checkpoints cleared.")
        except Exception as e:
            print(f"Error clearing checkpoints: {e}")

# --- Basic Utility Functions ---
def kelvin_to_celsius(x):
    """Convert temperatures from Kelvin to Celsius."""
    return x - 273.15

def get_spatial_info(ds):
    """Extracts spatial dimension names, sizes, and coordinates."""
    lat_dim = next((d for d in ds.dims if d.lower() in ['lat', 'latitude']), None)
    lon_dim = next((d for d in ds.dims if d.lower() in ['lon', 'longitude']), None)
    lat_coord = ds[lat_dim] if lat_dim and lat_dim in ds.coords else None
    lon_coord = ds[lon_dim] if lon_dim and lon_dim in ds.coords else None
    shape = (ds.sizes.get(lat_dim, 0), ds.sizes.get(lon_dim, 0))
    is_spatial = bool(lat_dim and lon_dim and lat_coord is not None and lon_coord is not None)
    return {'lat_dim': lat_dim, 'lon_dim': lon_dim, 'lat_coord': lat_coord, 'lon_coord': lon_coord, 'shape': shape, 'is_spatial': is_spatial}

def suggest_chunks(ds, time_chunk_size=365, spatial_chunk_size=100, spatial_chunk_threshold=100): # Default time chunk = 1 year
    """Suggest chunk sizes as a dictionary suitable for ds.chunk()."""
    suggested = {};
    if not isinstance(ds, xr.Dataset) and not isinstance(ds, xr.DataArray):
        print(f"Warning: suggest_chunks received unexpected type {type(ds)}. Returning empty dict."); return suggested
    for dim in ds.dims:
        size = ds.sizes.get(dim, 0); dim_lower = dim.lower()
        if "time" in dim_lower: suggested[dim] = min(size, time_chunk_size) if size > 0 else 1
        elif dim_lower in ["latitude", "lat", "longitude", "lon"]: suggested[dim] = min(size, spatial_chunk_size) if size > spatial_chunk_threshold else -1
        else: suggested[dim] = min(size, 100) if size > 1000 else -1 # Default chunking for other dims
    final_chunks = {}
    for dim, chunk in suggested.items():
         dim_size = ds.sizes.get(dim, 0);
         if dim_size <= 0: continue # Skip zero-size dimensions
         if chunk == -1: final_chunks[dim] = dim_size # Use full dimension size if requested
         else: valid_chunk = max(1, min(chunk, dim_size)); final_chunks[dim] = valid_chunk
    return final_chunks

# --- Data Loading and Preprocessing Functions ---
def load_nc_dir(path, time_dim='time'):
    """Load all .nc files in a folder as a combined xarray Dataset."""
    files = sorted(glob(os.path.join(path, "*.nc")))
    if not files: raise FileNotFoundError(f"No .nc files found in {path}")

    sample_ds = None
    actual_time_dim_found = None
    try:
        # Use decode_times=False initially to inspect dims/coords safely
        # --- Try opening sample with h5netcdf too ---
        sample_ds = xr.open_dataset(files[0], decode_times=False, engine='h5netcdf')
        # Try to find the actual time dimension name
        potential_time_dims = [d for d in sample_ds.dims if 'time' in d.lower()]
        if time_dim in sample_ds.dims:
            actual_time_dim_found = time_dim
        elif time_dim in sample_ds.coords and time_dim not in sample_ds.dims:
             coord_dims = sample_ds[time_dim].dims
             if len(coord_dims) == 1: actual_time_dim_found = coord_dims[0]
             elif potential_time_dims: actual_time_dim_found = potential_time_dims[0]; print(f"Warning: '{time_dim}' ambiguous coord, using detected dim '{actual_time_dim_found}' for {path}.")
             else: raise KeyError(f"'{time_dim}' is coord but cannot map to single dim in {files[0]}. Dims: {list(sample_ds.dims)}")
        elif potential_time_dims: actual_time_dim_found = potential_time_dims[0]; print(f"Warning: '{time_dim}' not found, using detected '{actual_time_dim_found}' for {path}.")
        else: raise KeyError(f"No time-like dimension found in {files[0]}. Dims: {list(sample_ds.dims)}")
        time_dim = actual_time_dim_found # Use the found dimension name

    except Exception as e:
        print(f"Error inspecting sample file {files[0]}: {e}"); raise
    finally:
        if sample_ds: sample_ds.close()

    if not actual_time_dim_found: raise ValueError("Could not determine the time dimension.")

    chunks = {time_dim: 365}

    try:
        # --- !! USE h5netcdf ENGINE !! ---
        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            parallel=True,
            chunks=chunks,
            data_vars='minimal',
            coords='minimal',
            compat='override',
            decode_times=True,
            engine='h5netcdf' # Explicitly specify the available engine
        )
        # --- !! END CHANGE !! ---

        # Post-load checks
        if time_dim not in ds.coords: print(f"WARNING: Time dimension '{time_dim}' is not a coordinate after loading {path}.")
        elif ds[time_dim].isnull().any().compute(): raise ValueError(f"NaT values found post-load for {path} in time coord '{time_dim}'")
        else:
             try: # Check and enforce time monotonicity
                 time_coord = ds[time_dim].persist()
                 if not time_coord.to_index().is_monotonic_increasing:
                      print(f"Warning: Time dim '{time_dim}' not monotonic for {path} post-load. Sorting...")
                      with ProgressBar(): ds = ds.sortby(time_coord).persist()
                      if not ds[time_dim].to_index().is_monotonic_increasing: raise ValueError(f"Time sorting failed post-load for {path}")
             except TypeError: print(f"Warning: Could not check time monotonicity post-load for {path} due to index type.")

        ds.encoding['source_file_path'] = path
        return ds
    except ValueError as e: print(f"ValueError loading {path}: {e}"); raise
    except Exception as e: print(f"Unhandled Error loading {path}: {e}"); traceback.print_exc(); raise

def align_regrid_datasets(datasets, expected_time_dim, interpolation_method='linear'):
    """Aligns datasets by time, identifies target grid, and interpolates."""
    if not datasets: return []

    print("\n--- Processing Datasets for Alignment & Regridding ---")
    target_spatial_info = None
    max_spatial_pixels = 0
    dataset_info = []
    all_spatial_dims = set()

    print("Step 1: Gathering dataset info and finding target grid...")
    for i, ds in enumerate(datasets):
        if ds is None:
            dataset_info.append({'index': i, 'ds': None})
            continue

        source_path = ds.encoding.get('source_file_path', f'Input index {i}')
        spatial_info = get_spatial_info(ds)
        current_time_dim = None
        potential_time_dims = [d for d in ds.dims if 'time' in d.lower()]

        # Determine current time dimension
        if expected_time_dim in ds.dims:
            current_time_dim = expected_time_dim
        elif expected_time_dim in ds.coords and expected_time_dim not in ds.dims:
             coord_dims = ds[expected_time_dim].dims
             if len(coord_dims) == 1:
                 current_time_dim = coord_dims[0]
                 print(f"  Info: Using dim '{current_time_dim}' for coord '{expected_time_dim}' in dataset {i} ({source_path}).")
             elif potential_time_dims:
                 current_time_dim = potential_time_dims[0]
                 print(f"  Warning: Ambiguous coord '{expected_time_dim}', using detected dim '{current_time_dim}' for dataset {i} ({source_path}).")
             else:
                 print(f"  ERROR: Cannot map coord '{expected_time_dim}' to single dim for dataset {i} ({source_path}). Skipping.")
                 dataset_info.append({'index': i, 'ds': None, 'source_path': source_path}); continue
        elif potential_time_dims:
            current_time_dim = potential_time_dims[0]
            print(f"  Warning: Expected time dim '{expected_time_dim}' not found, using detected '{current_time_dim}' for dataset {i} ({source_path}).")
        else:
            print(f"  ERROR: No time-like dimension found for dataset {i} ({source_path}). Skipping.")
            dataset_info.append({'index': i, 'ds': None, 'source_path': source_path}); continue

        # Verify the identified time dimension actually exists in dims
        if current_time_dim not in ds.dims:
            print(f"  ERROR: Identified time dim '{current_time_dim}' not in dims for dataset {i} ({source_path}). Dims: {list(ds.dims)}. Skipping.")
            dataset_info.append({'index': i, 'ds': None, 'source_path': source_path}); continue


        info = {'index': i, 'ds': ds, 'spatial': spatial_info, 'time_dim': current_time_dim, 'source_path': source_path}
        dataset_info.append(info)

        # Update target spatial grid info
        if spatial_info['is_spatial']:
            num_pixels = spatial_info['shape'][0] * spatial_info['shape'][1]
            if num_pixels > max_spatial_pixels:
                max_spatial_pixels = num_pixels
                target_spatial_info = spatial_info
            # Collect all spatial dimension names encountered
            if spatial_info['lat_dim']: all_spatial_dims.add(spatial_info['lat_dim'])
            if spatial_info['lon_dim']: all_spatial_dims.add(spatial_info['lon_dim'])
        else:
             print(f"  Warning: Dataset {i} ({source_path}) lacks standard spatial dims/coords.")

    if target_spatial_info is None:
        print("ERROR: Could not determine target spatial grid. Cannot regrid.")
        return [None] * len(datasets)
    print(f"Target spatial grid determined: {target_spatial_info['shape']} (Lat: '{target_spatial_info['lat_dim']}', Lon: '{target_spatial_info['lon_dim']}')")
    if target_spatial_info['lat_coord'] is None or target_spatial_info['lon_coord'] is None:
        print("ERROR: Target spatial grid missing coordinate variables. Cannot interpolate.")
        return [None] * len(datasets)

    print("\nStep 2: Preparing datasets for time alignment...")
    processed_for_time_align = []
    original_indices_for_time = []
    rename_failures = False
    for info in dataset_info:
        if info.get('ds') is None: continue
        ds = info['ds']
        current_time_dim = info['time_dim']
        source_path = info['source_path']

        # Rename time dimension if necessary
        if current_time_dim != expected_time_dim:
            try:
                ds = ds.rename({current_time_dim: expected_time_dim})
                info['ds'] = ds # Update dataset in info dict
                info['time_dim'] = expected_time_dim # Update time dim name in info dict
            except ValueError as e:
                 print(f"  ERROR: Failed rename time dim for dataset {info['index']} ({source_path}): {e}. Skipping.")
                 info['ds'] = None # Mark as failed
                 rename_failures = True
                 continue # Skip this dataset

        processed_for_time_align.append(ds)
        original_indices_for_time.append(info['index'])

    if not processed_for_time_align:
        print("ERROR: No datasets available for time alignment.")
        return [None] * len(datasets)
    if rename_failures:
        print("Warning: Some datasets were skipped due to time dim renaming errors.")


    print(f"\nStep 3: Attempting time alignment on {len(processed_for_time_align)} datasets using dimension '{expected_time_dim}'...")
    try:
        # Define coordinates/dimensions to exclude from alignment checks
        # Add common non-geographic coordinate names here
        misc_coords_to_exclude = {'number', 'step', 'surface', 'heightAboveGround', 'depthBelowLandLayer'}
        coords_to_exclude_names = list(misc_coords_to_exclude | all_spatial_dims) # Combine misc with found spatial dims
        # Check which ones actually exist in the datasets being aligned
        actual_coords_to_exclude = [c for c in coords_to_exclude_names if any((c in ds.coords or c in ds.dims) for ds in processed_for_time_align if ds is not None)]

        print(f"  Excluding dimensions/coordinates from alignment: {actual_coords_to_exclude}")

        # Perform alignment
        time_aligned_datasets = xr.align(*processed_for_time_align, join='inner', copy=False, exclude=actual_coords_to_exclude)

        # Check results of alignment
        if not time_aligned_datasets or not all(ds.sizes.get(expected_time_dim, 0) > 0 for ds in time_aligned_datasets):
            print("ERROR: Time alignment resulted in empty datasets or zero length time dimension.")
            return [None] * len(datasets)
        print(f"Time alignment successful. Common time dimension size: {time_aligned_datasets[0].sizes[expected_time_dim]}")

    except Exception as e:
        print(f"Error during time alignment (xr.align): {e}")
        traceback.print_exc()
        return [None] * len(datasets)


    print(f"\nStep 4: Regridding datasets to target grid {target_spatial_info['shape']} using method '{interpolation_method}'...")
    final_datasets_map = {} # Use a map keyed by original index
    target_coords_for_interp = {
        target_spatial_info['lat_dim']: target_spatial_info['lat_coord'],
        target_spatial_info['lon_dim']: target_spatial_info['lon_coord']
    }
    target_lat_name = target_spatial_info['lat_dim']
    target_lon_name = target_spatial_info['lon_dim']

    for i, ds_time_aligned in enumerate(time_aligned_datasets):
        original_idx = original_indices_for_time[i]
        # Find the original info dict for logging path etc.
        original_info = next((info for info in dataset_info if info.get('index') == original_idx), None)
        source_path_log = original_info['source_path'] if original_info else f"Original index {original_idx}"

        current_spatial_info = get_spatial_info(ds_time_aligned)

        # Check if spatial info is valid after alignment
        if not current_spatial_info['is_spatial']:
            print(f"Warning: Dataset {original_idx} ({source_path_log}) missing spatial dims/coords after time alignment. Cannot regrid. Skipping.")
            final_datasets_map[original_idx] = None
            continue

        processed_ds = None
        # Check if regridding is needed
        if (current_spatial_info['shape'] == target_spatial_info['shape'] and
            current_spatial_info['lat_dim'] == target_lat_name and
            current_spatial_info['lon_dim'] == target_lon_name):
             # Grids match, just ensure chunking
             print(f"Dataset {original_idx} ({source_path_log}): Grid matches target. Ensuring chunking.")
             processed_ds = ds_time_aligned
             current_chunks = suggest_chunks(processed_ds) # Suggest chunks based on current structure
             processed_ds = processed_ds.chunk(current_chunks)

        else:
            # Grids differ, perform interpolation
            print(f"Dataset {original_idx} ({source_path_log}): Grid {current_spatial_info['shape']} or dims differ from target. Interpolating...")

            # Rename spatial dims if they differ from target names before interp
            rename_interp = {}
            if current_spatial_info['lat_dim'] != target_lat_name:
                rename_interp[current_spatial_info['lat_dim']] = target_lat_name
            if current_spatial_info['lon_dim'] != target_lon_name:
                rename_interp[current_spatial_info['lon_dim']] = target_lon_name

            ds_to_interp = ds_time_aligned.rename(rename_interp) if rename_interp else ds_time_aligned

            try:
                 # Ensure coordinate data types match before interpolation (can cause issues)
                 target_lat_dtype = target_coords_for_interp[target_lat_name].dtype
                 target_lon_dtype = target_coords_for_interp[target_lon_name].dtype
                 ds_to_interp[target_lat_name] = ds_to_interp[target_lat_name].astype(target_lat_dtype)
                 ds_to_interp[target_lon_name] = ds_to_interp[target_lon_name].astype(target_lon_dtype)

                 # Perform interpolation
                 processed_ds = ds_to_interp.interp(
                     coords=target_coords_for_interp,
                     method=interpolation_method,
                     kwargs={"fill_value": np.nan} # Use NaN for fill value
                 )
                 # Re-chunk after interpolation as grid/size changed
                 final_chunks_after_interp = suggest_chunks(processed_ds)
                 print(f"    Re-chunking interpolated dataset {original_idx} ({source_path_log}) with suggested chunks: {final_chunks_after_interp}")
                 processed_ds = processed_ds.chunk(final_chunks_after_interp)
                 print(f"    Interpolation successful for dataset {original_idx}.")

            except Exception as e:
                 print(f"  ERROR interpolating dataset {original_idx} ({source_path_log}): {e}")
                 traceback.print_exc()
                 print(f"  Skipping dataset {original_idx} ({source_path_log}).")
                 processed_ds = None # Mark as failed

        final_datasets_map[original_idx] = processed_ds

    print("\nStep 5: Assembling final dataset list...")
    final_datasets_ordered = []
    original_input_indices = [info['index'] for info in dataset_info] # Get original indices in order

    for original_input_index in original_input_indices:
        processed_ds = final_datasets_map.get(original_input_index) # Get processed DS by original index
        if processed_ds is None:
             # Log if a dataset was skipped or failed
             original_info = next((info for info in dataset_info if info.get('index') == original_input_index), None)
             source_path_log = original_info.get('source_path', f'Original index {original_input_index}') if original_info else f'Original index {original_input_index}'
             print(f"Dataset originally at index {original_input_index} (from {source_path_log}) was skipped or failed.")
        final_datasets_ordered.append(processed_ds)

    num_successful = sum(1 for ds in final_datasets_ordered if ds is not None)
    print(f"\nReturning {num_successful} successfully processed datasets (originally {len(datasets)} input).")
    if num_successful != len(datasets):
        print(f"Warning: Number of output datasets ({num_successful}) differs from input ({len(datasets)}).")

    return final_datasets_ordered


# --- Climate Index Calculation Functions ---
def compute_gdd(tmax, tmin, base_temp=10):
    """Compute Growing Degree Days (GDD)."""
    tmax_c = kelvin_to_celsius(tmax)
    tmin_c = kelvin_to_celsius(tmin)
    avg_temp = (tmax_c + tmin_c) / 2.0 # Use float division
    # Calculate GDD, ensuring it's non-negative
    gdd = xr.where(avg_temp > base_temp, avg_temp - base_temp, 0.0)
    gdd = gdd.rename("GDD")
    gdd.attrs = {'long_name': 'Growing Degree Days', 'units': 'degree C day', 'base_temperature_C': base_temp}
    return gdd

def compute_spi(precip, window=30, min_std=1e-6):
    """Compute Standardized Precipitation Index (SPI) using groupby.apply."""
    time_dim = next((dim for dim in precip.dims if "time" in dim.lower()), None)
    if not time_dim: raise ValueError("No time-like dimension found in precipitation data!")

    print("  Preparing precipitation data for SPI...")
    # Fill NaNs with 0 and ensure non-negative values BEFORE rolling sum
    precip_filled = precip.fillna(0)
    print("    Filled NaNs in precipitation data with 0.")
    precip_non_neg = xr.where(precip_filled < 0, 0, precip_filled)
    if (precip_filled < 0).any().compute(): # Check if any negative values were actually replaced
         print("    Replaced negative precipitation values with 0.")

    # Calculate rolling sum (use float32 for precision in sums/means/stds)
    print(f"  Calculating {window}-day rolling sum for SPI...")
    rolling_sum = precip_non_neg.rolling({time_dim: window}, min_periods=window).sum(skipna=False).astype(np.float32)

    # Group by day of year to calculate climatology
    grouped_clim = rolling_sum.groupby(f"{time_dim}.dayofyear")

    print("  Calculating SPI climatology (mean/std)...")
    # Persist mean and std for efficiency as they are used repeatedly
    with ProgressBar(dt=1.0):
        mean_clim = grouped_clim.mean(skipna=True).persist()
        # Fill NaN std deviations (e.g., from all-zero periods) with 0 AFTER calculation
        std_clim = grouped_clim.std(skipna=True).fillna(0).persist()
    print("  SPI climatology calculation complete.")

    # Define the standardization function to apply per group
    def standardize(x, means, stds):
        # Get the day of year for the current chunk
        day = x[f"{time_dim}.dayofyear"]
        # Ensure we handle both array and scalar day values during apply
        if hasattr(day, 'values'):
            day_values = day.values
        else:
            day_values = np.array([day]) # Handle scalar case from groupby

        if day_values.size > 0:
            day_val = day_values.flat[0] # Get the single day value for this group
            # Ensure all values in the chunk belong to the same day-of-year group
            assert day_values.size <= 1 or np.all(day_values == day_val), f"Days not uniform within standardization chunk! Got {np.unique(day_values)}"
        else:
            return x # Return unmodified if no day value found (should not happen with groupby)

        # Select the pre-calculated mean and std for this day
        mean_for_day = means.sel(dayofyear=day_val, drop=True)
        std_for_day = stds.sel(dayofyear=day_val, drop=True)

        # Calculate SPI: (value - mean) / std, handle near-zero std
        return xr.where(std_for_day > min_std, (x - mean_for_day) / std_for_day, 0.0)

    print("  Applying SPI standardization...")
    # Apply the standardization function
    spi = rolling_sum.groupby(f"{time_dim}.dayofyear").apply(standardize, means=mean_clim, stds=std_clim)
    print("  SPI standardization apply step defined.")

    # Ensure output has same dimensions and coords as input rolling sum
    spi = spi.rename("SPI").transpose(*rolling_sum.dims)

    print("  Restoring SPI dimensions and coordinates...")
    # Explicitly copy coordinates that might be lost during groupby.apply
    for coord_name, coord_array in rolling_sum.coords.items():
        if coord_name not in spi.coords:
             try:
                 # Check if all dimensions required by the coordinate exist in spi
                 if all(dim in spi.dims for dim in coord_array.dims):
                     spi.coords[coord_name] = coord_array
                 else:
                     print(f"    Skipping coord '{coord_name}' for SPI (dim mismatch: expected {coord_array.dims}, got {spi.dims})")
             except Exception as e:
                 print(f"    Warning: Could not restore coord '{coord_name}' for SPI: {e}")

    spi.attrs = {'long_name': 'Standardized Precipitation Index', 'units': 'dimensionless', 'window_days': window}
    print("  SPI calculation graph fully defined.")
    return spi

def compute_sma(soil, min_std=1e-6):
    """Compute Soil Moisture Anomaly (SMA) using groupby.apply."""
    time_dim = next((dim for dim in soil.dims if "time" in dim.lower()), None)
    if not time_dim: raise ValueError("No time-like dimension found in soil moisture data!")

    print("  Preparing soil moisture data for SMA...")
    # Ensure float32 for calculations
    soil_float = soil.astype(np.float32)
    soil_input = soil_float # Use this variable going forward
    # Note: NaNs in input soil moisture are handled by skipna=True in mean/std

    print("  Proceeding with SMA calculation (NaNs handled by skipna=True in mean/std).")
    grouped_clim = soil_input.groupby(f"{time_dim}.dayofyear")

    print("  Calculating SMA climatology (mean/std)...")
    with ProgressBar(dt=1.0):
        mean_clim = grouped_clim.mean(skipna=True).persist()
        std_clim = grouped_clim.std(skipna=True).fillna(0).persist() # Fill NaN std dev with 0
    print("  SMA climatology calculation complete (NaN std filled with 0).")

    # Define standardization function for SMA
    def standardize_sma(x, means, stds):
        day = x[f"{time_dim}.dayofyear"]
        if hasattr(day, 'values'): day_values = day.values
        else: day_values = np.array([day])

        if day_values.size > 0:
            day_val = day_values.flat[0]
            assert day_values.size <= 1 or np.all(day_values == day_val), f"Days not uniform! Got {np.unique(day_values)}"
        else: return x

        mean_for_day = means.sel(dayofyear=day_val, drop=True)
        std_for_day = stds.sel(dayofyear=day_val, drop=True)

        # Calculate standardized anomaly
        result = xr.where(std_for_day > min_std, (x - mean_for_day) / std_for_day, 0.0)

        # Re-mask NaNs: if original input was NaN, output should be NaN
        # Check for isnan method based on xarray/numpy version
        try:
            final_result = xr.where(xr.ufuncs.isnan(x), np.nan, result)
        except AttributeError: # Older numpy/xarray might not have xr.ufuncs.isnan
            final_result = xr.where(np.isnan(x), np.nan, result)
        return final_result

    print("  Applying SMA standardization...")
    sma = soil_input.groupby(f"{time_dim}.dayofyear").apply(standardize_sma, means=mean_clim, stds=std_clim)
    print("  SMA standardization apply step defined.")
    sma = sma.rename("SoilMoistureAnomaly").transpose(*soil_input.dims)

    print("  Restoring SMA dimensions and coordinates...")
    for coord_name, coord_array in soil_input.coords.items():
        if coord_name not in sma.coords:
             try:
                 if all(dim in sma.dims for dim in coord_array.dims):
                      sma.coords[coord_name] = coord_array
                 else: print(f"    Skipping coord '{coord_name}' for SMA (dim mismatch)")
             except Exception as e: print(f"    Warning: Could not restore coord '{coord_name}' for SMA: {e}")

    sma.attrs = {'long_name': 'Soil Moisture Anomaly', 'units': 'dimensionless'}
    print("  SMA calculation graph fully defined.")
    return sma

def compute_et_ratio(evap, precip, min_precip=1e-4):
    """Compute the Evapotranspiration Ratio (E/P)."""
    print("  Calculating Evapotranspiration Ratio...")
    # Ensure precipitation is non-negative
    precip_non_neg = xr.where(precip < 0, 0, precip)
    # Calculate ratio where precipitation is above threshold, otherwise NaN
    # Use absolute value of evaporation (often negative in models like ERA5)
    ratio = xr.where(precip_non_neg >= min_precip, abs(evap) / precip_non_neg, np.nan)
    ratio = ratio.rename("EvapotranspirationRatio")
    ratio.attrs = {'long_name': 'Evapotranspiration Ratio (E/P)', 'units': 'dimensionless', 'min_precip_threshold': min_precip}
    print("  Evapotranspiration Ratio calculation defined.")
    return ratio

def compute_heatwave_index(tmax, threshold_c=35, window=7):
    """Compute the Heatwave Index (count of days above threshold in window)."""
    time_dim = next((dim for dim in tmax.dims if "time" in dim.lower()), None)
    assert time_dim, "No time-like dimension found in tmax data!"
    print(f"  Calculating Heatwave Index (threshold={threshold_c}C, window={window} days)...")
    tmax_c = kelvin_to_celsius(tmax)
    # Create a binary array: 1 if hot day, 0 otherwise (handle NaNs as 0)
    is_hot_day = (tmax_c > threshold_c).astype(np.int8).fillna(0)
    # Sum over a rolling window (use min_periods=1 to get counts even at start)
    hwi = is_hot_day.rolling({time_dim: window}, min_periods=1).sum(skipna=False)
    hwi = hwi.rename("HeatwaveIndex")
    # CHANGED: units to '1' for dimensionless count, and long_name slightly adjusted
    hwi.attrs = {
        'long_name': f'Heatwave Index (count of days in {window}-day window with Tmax >= {threshold_c} C)',
        'units': '1', # CF convention for dimensionless count
        'threshold_C': threshold_c,
        'window_days': window
    }
    print("  Heatwave Index calculation defined.")
    return hwi

def compute_wind_speed(u, v):
    """Compute Wind Speed magnitude from U and V components."""
    print("  Calculating Wind Speed...")
    wind_speed = np.sqrt(u**2 + v**2).rename("WindSpeed")
    wind_speed.attrs = {'long_name': 'Wind Speed Magnitude', 'units': 'm s-1'} # Adjusted units
    print("  Wind Speed calculation defined.")
    return wind_speed

# ----------------------------------------
# DIAGNOSTICS FUNCTION
# ----------------------------------------
def run_dataset_diagnostics(base_path="/kaggle/working", expected_time_dim='valid_time'):
    """Run diagnostics on input datasets. Returns True if critical problems found."""
    print("--- Running Dataset Diagnostics ---")
    print(f"Checking base path: {base_path}")
    print(f"Expecting time dimension name like: '{expected_time_dim}'")
    folders = [
        "era5_max_2m_temperature", "era5_min_2m_temperature",
        "total_precipitation", "evaporation",
        "volumetric_soil_water_layer_1",
        "10m_u_component_of_wind_mean", "10m_v_component_of_wind_mean"
    ]
    all_time_dims_found = []
    all_calendars = []
    critical_problems_found = False
    grid_mismatch_found = False
    folder_file_counts = {}
    spatial_grids = {}

    print("\nChecking folder existence and file counts:")
    for folder in folders:
        path = os.path.join(base_path, folder)
        if not os.path.exists(path):
            print(f"  ERROR: Folder {path} does not exist!")
            critical_problems_found = True
            folder_file_counts[folder] = 0
            continue
        nc_files = sorted(glob(os.path.join(path, "*.nc")))
        count = len(nc_files)
        folder_file_counts[folder] = count
        print(f"  {folder}: Found {count} .nc files.")
        if not nc_files:
            print(f"  WARNING: No .nc files found in {path}")

    if critical_problems_found:
        print("\nCritical errors found (missing folders). Diagnostics aborted.")
        return True # Abort early

    print("\nChecking individual dataset properties (using first file in each folder):")
    reference_spatial_info = None
    for folder, file_count in folder_file_counts.items():
        if file_count == 0: continue # Skip folders with no files

        path = os.path.join(base_path, folder)
        nc_files = sorted(glob(os.path.join(path, "*.nc")))
        print(f"\nDiagnostics for {folder}:")
        sample_ds = None
        try:
            # Open without decoding time first to check metadata safely
            sample_ds = xr.open_dataset(nc_files[0], decode_times=False, engine='h5netcdf') # Ensure engine for consistency
            print(f"  Variables: {list(sample_ds.data_vars)}")
            print(f"  Dimensions: {dict(sample_ds.sizes)}")

            # Check time dimension
            time_dim_actual = next((d for d in sample_ds.dims if 'time' in d.lower()), None)
            if not time_dim_actual:
                print(f"  ERROR: No time-like dimension found!")
                critical_problems_found = True
            elif time_dim_actual != expected_time_dim:
                 # This is now handled by align_regrid, so just a warning
                 print(f"  WARNING: Time dim is '{time_dim_actual}', pipeline expects '{expected_time_dim}' (will attempt rename).")
            if time_dim_actual and time_dim_actual in sample_ds.variables:
                 all_time_dims_found.append(time_dim_actual)
                 time_var = sample_ds.variables[time_dim_actual]
                 calendar = time_var.attrs.get('calendar', 'Not specified')
                 units = time_var.attrs.get('units', 'Not specified')
                 all_calendars.append(calendar)
                 print(f"    Time Dim: '{time_dim_actual}', Units: '{units}', Calendar: '{calendar}'")
                 if not units or 'since' not in units.lower():
                     print(f"    WARNING: Time units attribute '{units}' might be missing or non-CF compliant.")
            elif time_dim_actual:
                 print(f"    Time Dim: '{time_dim_actual}' (found in dims, not coords/vars).")
                 all_time_dims_found.append(time_dim_actual)
                 all_calendars.append('Not specified')


            # Check spatial dimensions
            current_spatial_info = get_spatial_info(sample_ds)
            spatial_grids[folder] = current_spatial_info['shape']
            if current_spatial_info['is_spatial']:
                print(f"    Spatial Dims: Lat='{current_spatial_info['lat_dim']}', Lon='{current_spatial_info['lon_dim']}', Shape={current_spatial_info['shape']}")
                if reference_spatial_info is None:
                    reference_spatial_info = current_spatial_info # Set reference from first spatial dataset
                elif (reference_spatial_info['shape'] != current_spatial_info['shape'] or
                      reference_spatial_info['lat_dim'] != current_spatial_info['lat_dim'] or
                      reference_spatial_info['lon_dim'] != current_spatial_info['lon_dim']):
                     print(f"    WARNING: Spatial grid shape/dims {current_spatial_info['shape']} ('{current_spatial_info['lat_dim']}', '{current_spatial_info['lon_dim']}') differs from reference {reference_spatial_info['shape']} ('{reference_spatial_info['lat_dim']}', '{reference_spatial_info['lon_dim']}'). Interpolation will be needed.")
                     grid_mismatch_found = True
            else:
                print(f"  WARNING: Standard latitude/longitude dimensions/coordinates not found.")

        except Exception as e:
            print(f"  ERROR processing sample file {nc_files[0]}: {e}")
            critical_problems_found = True
        finally:
            if sample_ds: sample_ds.close()

    if all_time_dims_found:
        print("\n--- Cross-Dataset Time Checks ---")
        unique_time_dims = set(all_time_dims_found)
        # We now handle differing time dim names, so this is less critical
        if len(unique_time_dims) > 1: print(f"INFO: Multiple time dimension names found: {unique_time_dims}. Alignment process will attempt renaming to '{expected_time_dim}'.")
        elif unique_time_dims: print(f"Consistent time dimension name found: '{list(unique_time_dims)[0]}'")
        else: print("No time dimensions found across datasets.") # Should not happen if checks above passed

        unique_calendars = set(all_calendars)
        filtered_calendars = {c for c in unique_calendars if c not in ['Not specified', 'Unknown', 'N/A', None, 'standard']} # 'standard' is often default
        if len(filtered_calendars) > 1: # Allow standard/proleptic_gregorian as consistent
             proleptic_present = 'proleptic_gregorian' in filtered_calendars
             other_calendars = {c for c in filtered_calendars if c != 'proleptic_gregorian'}
             if not (proleptic_present and not other_calendars): # If not just 'proleptic_gregorian' or empty set
                 print(f"ERROR: Potentially inconsistent calendars found: {filtered_calendars}. Check CF compliance. Forcing proleptic_gregorian.")
                 critical_problems_found = False # Downgrade to warning, pipeline will try to handle
                 print("  Pipeline will assume 'proleptic_gregorian' for time encoding.")
        elif len(filtered_calendars) == 1:
            print(f"Consistent calendar found: '{list(filtered_calendars)[0]}'")
        elif not filtered_calendars and any(c in ['Not specified', 'Unknown', 'N/A', None, 'standard'] for c in unique_calendars):
            print("WARNING: Calendar info missing or unspecified across datasets. Assuming 'proleptic_gregorian'.")
        elif not unique_calendars: pass # No calendars found
        else: print("Calendar check passed (or only one dataset).")

    print("\n--- Diagnostics Summary ---")
    if grid_mismatch_found:
        print("WARNING: Inconsistent spatial grids detected. Interpolation will be attempted.")
    if critical_problems_found:
        print("CRITICAL PROBLEMS detected during diagnostics. Review messages above. Pipeline may fail.")
    elif not grid_mismatch_found and not critical_problems_found:
        print("Basic diagnostics passed (no critical errors, grids match).")
    elif not critical_problems_found:
        print("Basic diagnostics passed (no critical errors, grid mismatch noted - interpolation will be attempted).")
    print("-----------------------------")
    return critical_problems_found


# ----------------------------------------
# MAIN PIPELINE
# ----------------------------------------
def compute_indices_lazy(base_path="/kaggle/input/climate-dataset-variables", save_path="climate_indices.zarr", time_dim_name='valid_time', force_recompute=False):
    """Compute climate indices, save result incrementally to Zarr store."""
    if not os.path.isdir(base_path): raise FileNotFoundError(f"Base path '{base_path}' does not exist.")
    if force_recompute: clear_checkpoints()

    # --- Checkpoint 0: Initial Loading ---
    datasets_raw = None; ordered_names = None
    loaded_datasets_checkpoint = load_checkpoint("0_loaded_datasets")
    if loaded_datasets_checkpoint is not None:
        print("Using loaded datasets from checkpoint.")
        datasets_raw = loaded_datasets_checkpoint
        ordered_names = list(datasets_raw.keys())
        print(f"  Dataset order from checkpoint: {ordered_names}")
    else:
        print("\nRunning pre-computation diagnostics...");
        critical_problems = run_dataset_diagnostics(base_path=base_path, expected_time_dim=time_dim_name);
        if critical_problems:
            print("Critical problems found during diagnostics. Aborting pipeline.")
            return None
        print("Diagnostics finished."); print("\n--- Starting Climate Indices Pipeline ---")
        paths = {
            "tmax": os.path.join(base_path, "era5_max_2m_temperature"),
            "tmin": os.path.join(base_path, "era5_min_2m_temperature"),
            "precip": os.path.join(base_path, "total_precipitation"),
            "evap": os.path.join(base_path, "evaporation"),
            "soil": os.path.join(base_path, "volumetric_soil_water_layer_1"),
            "u_wind": os.path.join(base_path, "10m_u_component_of_wind_mean"),
            "v_wind": os.path.join(base_path, "10m_v_component_of_wind_mean")
        }
        ordered_names = list(paths.keys())
        print("\nLoading datasets (lazily)..."); datasets_raw = {}; load_errors = False
        for name in ordered_names:
            path = paths[name]
            if not os.path.isdir(path):
                print(f"Warning: Input directory for '{name}' not found at {path}. Skipping.")
                datasets_raw[name] = None; continue
            if not glob(os.path.join(path, "*.nc")):
                print(f"Warning: No .nc files found for '{name}' at {path}. Skipping.")
                datasets_raw[name] = None; continue
            try:
                datasets_raw[name] = load_nc_dir(path, time_dim=time_dim_name)
                print(f"  Successfully initiated loading for '{name}'.")
            except Exception as e:
                print(f"\nError loading '{name}': {e}"); load_errors = True; datasets_raw[name] = None
        if all(ds is None for ds in datasets_raw.values()):
            raise RuntimeError("No datasets could be loaded.")
        if load_errors: print("Warning: Errors occurred during loading setup for some datasets.")
        save_checkpoint("0_loaded_datasets", datasets_raw)

    # --- Checkpoint 1: Align & Regrid ---
    processed_datasets_dict = None; interpolation_method = 'linear'
    processed_datasets_checkpoint = load_checkpoint("1_processed_datasets")
    if processed_datasets_checkpoint is not None:
         print("Using processed datasets from checkpoint.")
         processed_datasets_dict = processed_datasets_checkpoint
         interpolation_method = processed_datasets_dict.get('_metadata_', {}).get('interpolation_method', 'linear')
         loaded_names_from_chk1 = [name for name in processed_datasets_dict.keys() if name != '_metadata_']
         # Ensure ordered_names reflects what's actually in the checkpoint if it was loaded
         if ordered_names is None or set(ordered_names) != set(loaded_names_from_chk1):
             print("Warning: Order or keys differ from initial load. Using order from Checkpoint 1.")
             ordered_names = loaded_names_from_chk1 # Use the order from the checkpoint
         print(f"  Dataset order confirmed from checkpoint 1: {ordered_names}")
    else:
        if ordered_names is None or datasets_raw is None:
             raise RuntimeError("Internal error: ordered_names or datasets_raw not available for alignment.")
        print("\nAligning time and regridding spatial coordinates...");
        dataset_list_to_process = [datasets_raw.get(name) for name in ordered_names] # Ensure correct order
        try:
            processed_datasets_list = align_regrid_datasets(dataset_list_to_process, expected_time_dim=time_dim_name, interpolation_method=interpolation_method)
        except Exception as e:
            print(f"Coordinate alignment/regridding failed critically: {e}"); traceback.print_exc(); raise
        if len(processed_datasets_list) != len(ordered_names):
            print(f"Critical Warning: Mismatch in length after align/regrid. Expected {len(ordered_names)}, got {len(processed_datasets_list)}.")
            # This could happen if align_regrid returns None for some datasets
            # Rebuild dict carefully, preserving Nones for failed items
        processed_datasets_dict = {name: ds for name, ds in zip(ordered_names, processed_datasets_list)}
        processed_datasets_dict['_metadata_'] = {'interpolation_method': interpolation_method}
        save_checkpoint("1_processed_datasets", processed_datasets_dict)

    # --- Essential Dataset Checks ---
    essential_names = ["tmax", "tmin", "precip", "evap", "soil", "u_wind", "v_wind"]
    missing_essential = []
    if processed_datasets_dict is None:
        raise RuntimeError("Processed datasets dictionary is not available after alignment/regridding step.")
    for name in essential_names:
        ds = processed_datasets_dict.get(name)
        if ds is None:
            missing_essential.append(name)
        elif not ds.dims: # Check if dataset is empty (has no dimensions)
            missing_essential.append(f"{name} (empty)")
    if missing_essential:
        raise RuntimeError(f"Essential datasets missing or empty after alignment/regridding: {missing_essential}.")

    aligned_tmax_ds = processed_datasets_dict["tmax"] # Keep reference
    print("\nDimensions after alignment & regridding (Tmax dataset):"); print(dict(aligned_tmax_ds.sizes))

    # --- Checkpoint 2: Extracted Variables ---
    variables_to_chunk = None # Initialize to ensure it's defined
    extracted_vars_checkpoint = load_checkpoint("2_extracted_vars")
    if extracted_vars_checkpoint is not None:
         print("Using extracted variables from checkpoint.")
         # Unpack variables carefully, checking if they exist in the checkpoint
         tmax_var = extracted_vars_checkpoint.get('tmax')
         tmin_var = extracted_vars_checkpoint.get('tmin')
         precip_var = extracted_vars_checkpoint.get('precip')
         evap_var = extracted_vars_checkpoint.get('evap')
         soil_var = extracted_vars_checkpoint.get('soil')
         u_var = extracted_vars_checkpoint.get('u')
         v_var = extracted_vars_checkpoint.get('v')
         vars_to_check_chk2 = [tmax_var, tmin_var, precip_var, evap_var, soil_var, u_var, v_var];
         if any(var is None for var in vars_to_check_chk2):
             print("Warning: Some variables were None after loading checkpoint 2. Re-extracting all.")
             extracted_vars_checkpoint = None # Force re-extraction
         else:
             variables_to_chunk = {
                'tmax':tmax_var, 'tmin':tmin_var, 'precip':precip_var,
                'evap':evap_var, 'soil':soil_var, 'u':u_var, 'v':v_var
             }
    if extracted_vars_checkpoint is None: # Or if forced re-extraction
        print("\nExtracting primary variables from processed datasets...");
        def get_first_data_var(ds, ds_key_name): # Changed 'name' to 'ds_key_name'
            if ds is None: raise ValueError(f"Input dataset for '{ds_key_name}' is None.")
            if not isinstance(ds, xr.Dataset): raise ValueError(f"Input for '{ds_key_name}' is not an xarray.Dataset, but {type(ds)}.")
            if not ds.data_vars: raise ValueError(f"Dataset for '{ds_key_name}' contains no data variables.")
            # Map for common variable names within datasets
            common_name_map = {
                "tmax": ["mx2t", "t2mmax", "tasmax", "tmax"],
                "tmin": ["mn2t", "t2mmin", "tasmin", "tmin"],
                "precip": ["tp", "precip", "pr", "total_precipitation"],
                "evap": ["e", "evspsbl", "evspsblsoi", "evaporation"], # Added evspsbl
                "soil": ["swvl1", "volumetric_soil_water_layer_1", "soil_moisture"],
                "u_wind": ["u10", "uas", "10m_u_component_of_wind"],
                "v_wind": ["v10", "vas", "10m_v_component_of_wind"]
            }
            expected_vars = common_name_map.get(ds_key_name, [ds_key_name]) # Fallback to ds_key_name
            for var_name_in_ds in expected_vars:
                if var_name_in_ds in ds.data_vars:
                    print(f"  Found expected variable '{var_name_in_ds}' for dataset key '{ds_key_name}'.")
                    return ds[var_name_in_ds]
            # If specific names not found, try the first variable
            first_var_name = list(ds.data_vars)[0]
            print(f"  Warning: Expected var not found for dataset key '{ds_key_name}'. Extracting first var '{first_var_name}'.")
            return ds[first_var_name]
        try:
            tmax_var = get_first_data_var(processed_datasets_dict["tmax"], "tmax")
            tmin_var = get_first_data_var(processed_datasets_dict["tmin"], "tmin")
            precip_var = get_first_data_var(processed_datasets_dict["precip"], "precip")
            evap_var = get_first_data_var(processed_datasets_dict["evap"], "evap")
            soil_var = get_first_data_var(processed_datasets_dict["soil"], "soil")
            u_var = get_first_data_var(processed_datasets_dict["u_wind"], "u_wind")
            v_var = get_first_data_var(processed_datasets_dict["v_wind"], "v_wind")
            variables_to_chunk = {
                'tmax': tmax_var, 'tmin': tmin_var, 'precip': precip_var,
                'evap': evap_var, 'soil': soil_var, 'u': u_var, 'v': v_var
            }
            save_checkpoint("2_extracted_vars", variables_to_chunk)
        except ValueError as e:
            print(f"Error during variable extraction: {e}"); traceback.print_exc(); raise
        except KeyError as e:
            print(f"Error: Key not found during variable extraction, processed_datasets_dict keys: {list(processed_datasets_dict.keys())}, error: {e}"); traceback.print_exc(); raise

    if variables_to_chunk is None:
        raise RuntimeError("Failed to load or extract variables_to_chunk.")

    print("\nEnsuring consistent chunking across input variables...")
    target_chunks = suggest_chunks(aligned_tmax_ds); print(f"  Target chunks for data variables: {target_chunks}")
    chunked_vars = {}
    for name, var_da in variables_to_chunk.items(): # var_da is DataArray
         if var_da is None: raise ValueError(f"Variable '{name}' is None before chunking.")
         if not isinstance(var_da, xr.DataArray): raise TypeError(f"Variable '{name}' is not an xarray.DataArray, but {type(var_da)}.")

         var_dims = set(var_da.dims);
         applicable_chunks = {dim: chnk for dim, chnk in target_chunks.items() if dim in var_dims}
         if not applicable_chunks: # Handle scalar or 0-dim arrays
             print(f"  Variable '{name}' is scalar or has no dimensions matching target. Skipping chunking.")
             chunked_vars[name] = var_da
             continue

         current_chunks_match = False
         if var_da.chunks is not None:
              # var_da.chunks is a tuple of tuples, var_da.dims is a tuple of strings
              current_chunk_dict_from_da = {var_da.dims[i]: var_da.chunks[i] for i in range(len(var_da.dims))}
              # Compare only the first chunk size for each dimension
              if all(dim in current_chunk_dict_from_da and
                     current_chunk_dict_from_da[dim] and # Ensure chunk tuple is not empty
                     current_chunk_dict_from_da[dim][0] == applicable_chunks[dim]
                     for dim in applicable_chunks if dim in current_chunk_dict_from_da): # Only check dims present in both
                  current_chunks_match = True

         if not current_chunks_match:
             print(f"  Rechunking '{name}' from current {var_da.chunks} to target: {applicable_chunks}")
             try:
                 # Preserve attributes manually when chunking DataArray
                 original_attrs_var = var_da.attrs.copy()
                 chunked_vars[name] = var_da.chunk(applicable_chunks)
                 chunked_vars[name].attrs = original_attrs_var
             except ValueError as e:
                 print(f"    ERROR rechunking {name}: {e}. Its current chunks: {var_da.chunks}. Applicable: {applicable_chunks}"); raise
         else:
             print(f"  Chunks for '{name}' already match target.")
             chunked_vars[name] = var_da


    # --- Prepare Zarr Store (Coordinates) ---
    print(f"\nPreparing Zarr store: {save_path}")
    if os.path.exists(save_path):
        print(f"Warning: Output path {save_path} already exists. Overwriting.")
        shutil.rmtree(save_path)
    coord_encoding = {}
    # Use aligned_tmax_ds for reference coordinates. Chunk them if they have dimensions.
    coords_ds_for_chunk_suggestion = xr.Dataset({name: da for name, da in aligned_tmax_ds.coords.items() if da.dims})
    final_coord_chunks = suggest_chunks(coords_ds_for_chunk_suggestion) if coords_ds_for_chunk_suggestion.coords else {}

    print(f"  Coordinate target chunks: {final_coord_chunks}")
    fill_value_float = np.float32(np.nan) # Default fill for float coords
    ref_coords_dataset = aligned_tmax_ds.coords.to_dataset() # Convert Coords to Dataset for easier iteration
    coords_to_write = xr.Dataset()

    for coord_name in ref_coords_dataset: # Iterate over names in the Dataset
        coord_da = ref_coords_dataset[coord_name] # This is now a DataArray
        print(f"  Processing coordinate: {coord_name} (dtype: {coord_da.dtype})")
        enc = {}
        coord_da_to_write = coord_da # Start with original

        # Chunk dimensional coordinates
        if coord_da.dims:
            applicable_coord_chunks = {dim: chnk for dim, chnk in final_coord_chunks.items() if dim in coord_da.dims}
            if applicable_coord_chunks:
                needs_rechunk_coord = True
                if coord_da.chunks is not None:
                    current_coord_chunk_dict = {coord_da.dims[i]: coord_da.chunks[i] for i in range(len(coord_da.dims))}
                    if all(dim in current_coord_chunk_dict and current_coord_chunk_dict[dim] and current_coord_chunk_dict[dim][0] == applicable_coord_chunks[dim]
                           for dim in applicable_coord_chunks if dim in current_coord_chunk_dict):
                        needs_rechunk_coord = False
                if needs_rechunk_coord:
                    print(f"    Rechunking coordinate '{coord_name}' to {applicable_coord_chunks}")
                    try:
                        coord_da_to_write = coord_da.chunk(applicable_coord_chunks)
                    except Exception as e:
                        print(f"      Failed to chunk coordinate {coord_name}: {e}. Using original chunking or unchunked.")
                        coord_da_to_write = coord_da # Fallback

        coords_to_write[coord_name] = coord_da_to_write # Add potentially rechunked coord to dataset

        # Define encoding for coordinate based on the one to be written
        if hasattr(coord_da_to_write, 'dims') and coord_da_to_write.dims and coord_da_to_write.chunks is not None:
            try:
                # Ensure all dims are present in chunks mapping before creating tuple
                chunk_tuple_list = []
                for dim_name in coord_da_to_write.dims:
                    # coord_da_to_write.chunks is tuple of tuples e.g. ((10,), (20,)) for 2D
                    # coord_da_to_write.dims is tuple of strings e.g. ('lat', 'lon')
                    # We need to find the index of dim_name in coord_da_to_write.dims
                    dim_idx = coord_da_to_write.dims.index(dim_name)
                    chunk_tuple_list.append(coord_da_to_write.chunks[dim_idx][0]) # Get the first chunk size for that dim
                enc['chunks'] = tuple(chunk_tuple_list)
            except Exception as e:
                print(f"    Warning: Error getting chunks for coord '{coord_name}': {e}. Chunks: {coord_da_to_write.chunks}, Dims: {coord_da_to_write.dims}")

        if np.issubdtype(coord_da_to_write.dtype, np.floating):
            enc['_FillValue'] = fill_value_float
            enc['dtype'] = str(coord_da_to_write.dtype) # Ensure float32 or float64 is explicit
        elif 'datetime64' in str(coord_da_to_write.dtype):
            print(f"    Encoding for datetime coordinate '{coord_name}'.")
            enc['dtype'] = str(coord_da_to_write.dtype) # Keep original datetime64 dtype
            enc.setdefault('units', coord_da_to_write.attrs.get('units', 'nanoseconds since 1970-01-01T00:00:00Z'))
            enc.setdefault('calendar', coord_da_to_write.attrs.get('calendar', 'proleptic_gregorian'))
            print(f"      Using units: {enc['units']}, calendar: {enc['calendar']}")
        else: # Other types (int, bool, etc.)
            enc['dtype'] = str(coord_da_to_write.dtype)

        if enc: coord_encoding[coord_name] = enc
    try:
        print("  Writing coordinates to Zarr store...");
        coords_to_write.to_zarr(save_path, mode='w', encoding=coord_encoding, consolidated=True, compute=True)
        print("  Coordinates written successfully.")
    except Exception as e:
        print(f"    ERROR writing coordinates: {e}"); traceback.print_exc();
        raise RuntimeError("Failed to initialize Zarr store with coordinates.") from e

    # --- Compute and append each index ---
    print("\nComputing and saving indices incrementally...")
    indices_to_compute = [
        ("GDD", compute_gdd, [chunked_vars['tmax'], chunked_vars['tmin']]),
        ("SPI", compute_spi, [chunked_vars['precip']]),
        ("SoilMoistureAnomaly", compute_sma, [chunked_vars['soil']]),
        ("EvapotranspirationRatio", compute_et_ratio, [chunked_vars['evap'], chunked_vars['precip']]),
        ("HeatwaveIndex", compute_heatwave_index, [chunked_vars['tmax']]),
        ("WindSpeed", compute_wind_speed, [chunked_vars['u'], chunked_vars['v']])
    ]
    total_indices = len(indices_to_compute)
    for i, (index_name, compute_func, args) in enumerate(indices_to_compute):
        print(f"\n--- Processing Index {i+1}/{total_indices}: {index_name} ---")
        try:
            # 1. Compute the index (lazily first)
            index_da = compute_func(*args)
            if index_da.name != index_name: index_da = index_da.rename(index_name)

            # 2. Set appropriate dtype for the index
            # Default to float32, specify others as needed
            target_dtype_str = "float32"
            if index_name == "HeatwaveIndex":
                target_dtype_str = "int16" # For counts (e.g., 0-7 days)
                print(f"  Setting target dtype for {index_name} to {target_dtype_str} (count-based).")
            # Add other specific dtypes if needed for other indices
            # elif index_name == "SomeOtherIndex":
            # target_dtype_str = "some_other_type"

            actual_target_dtype = np.dtype(target_dtype_str)

            if index_da.dtype != actual_target_dtype:
                print(f"  Coercing {index_name} from {index_da.dtype} to {actual_target_dtype}.")
                # Preserve attributes during astype for xarray DataArrays
                original_attrs = index_da.attrs.copy()
                index_da = index_da.astype(actual_target_dtype)
                index_da.attrs = original_attrs # Restore attributes
            else:
                print(f"  {index_name} already has target dtype {actual_target_dtype}.")


            # 3. Rechunk index_da before saving to match target_chunks
            print(f"  Ensuring chunks for {index_name} align with target {target_chunks}...")
            index_dims_set = set(index_da.dims)
            # Use the same 'target_chunks' derived from aligned_tmax_ds (for data vars)
            applicable_chunks_for_index = {dim: chnk for dim, chnk in target_chunks.items() if dim in index_dims_set}

            if applicable_chunks_for_index and hasattr(index_da, 'dims') and index_da.dims:
                needs_rechunk_index = True
                if index_da.chunks is not None:
                    current_chunk_dict_index = {index_da.dims[k]: index_da.chunks[k] for k in range(len(index_da.dims))}
                    if all(dim in current_chunk_dict_index and current_chunk_dict_index[dim] and
                           current_chunk_dict_index[dim][0] == applicable_chunks_for_index[dim]
                           for dim in applicable_chunks_for_index if dim in current_chunk_dict_index):
                        needs_rechunk_index = False
                        print(f"    Chunks for {index_name} already match target: {applicable_chunks_for_index}.")
                if needs_rechunk_index:
                    print(f"    Rechunking {index_name} from {index_da.chunks} to: {applicable_chunks_for_index}")
                    try:
                        original_attrs_rechunk = index_da.attrs.copy()
                        index_da = index_da.chunk(applicable_chunks_for_index)
                        index_da.attrs = original_attrs_rechunk
                        print(f"    Rechunking for {index_name} successful. New chunks: {index_da.chunks}")
                    except Exception as e_rechunk:
                        print(f"    ERROR rechunking {index_name}: {e_rechunk}. Proceeding with current chunks: {index_da.chunks}")
            else:
                 print(f"    No applicable target chunks found or {index_name} is scalar/chunkless. Current chunks: {index_da.chunks}")


            # 4. Prepare encoding for Zarr storage
            var_encoding = {}
            # Explicitly set the dtype in encoding. This is crucial.
            var_encoding['dtype'] = str(index_da.dtype)

            if np.issubdtype(index_da.dtype, np.floating):
                # For float types, set a _FillValue (NaN of the correct float type)
                var_encoding['_FillValue'] = index_da.dtype.type(np.nan)
            elif np.issubdtype(index_da.dtype, np.integer):
                # For integer types (like HeatwaveIndex), a _FillValue is often not needed
                # if the calculation ensures no NaNs (e.g., HWI is a count, should be >= 0).
                # If you need one, it must be an integer of the same type, e.g., np.int16(-9999)
                pass # No _FillValue specified for integers by default here.

            # Let Zarr infer chunks from index_da.chunks directly; DO NOT set 'chunks' in var_encoding.
            encoding_dict_for_append = {index_name: var_encoding}

            # 5. Append to the Zarr store
            print(f"  Appending {index_name} (dtype: {index_da.dtype}, chunks: {index_da.chunks}) to {save_path}...")
            with ProgressBar(dt=2.0): # Increased dt for potentially long computations
                 index_da.to_zarr(save_path, mode='a', encoding=encoding_dict_for_append, compute=True, consolidated=False) # Consolidate at the end
            print(f"  {index_name} saved successfully.")

        except ValueError as ve:
             if "Final chunk of Zarr array must be the same size or smaller" in str(ve) or \
                "Zarr requires uniform chunk sizes" in str(ve) or \
                "chunksize is larger than dimension size" in str(ve): # Added more specific Zarr chunk errors
                 print(f"\nERROR saving index {index_name}: {ve}")
                 print(f"  Computed Chunks for index_da: {index_da.chunks if hasattr(index_da, 'chunks') else 'N/A'}")
                 print(f"  Target Chunks for data vars: {target_chunks}")
                 print(f"  This usually happens with complex operations like groupby.apply or if rechunking failed.")
                 print(f"  Skipping index {index_name} due to incompatible chunks.")
                 continue
             else:
                 print(f"\nERROR processing and saving index {index_name} (ValueError): {ve}")
                 traceback.print_exc(); print(f"Skipping index {index_name} due to error."); continue
        except Exception as e:
            print(f"\nERROR processing and saving index {index_name} (Other): {e}")
            traceback.print_exc(); print(f"Skipping index {index_name} due to error."); continue

    # Consolidate metadata at the very end
    try:
        print("\nConsolidating Zarr metadata for the entire store...")
        zarr.consolidate_metadata(save_path)
        print("Metadata consolidated.")
    except NameError:
        print("ERROR: zarr library not imported. Cannot consolidate metadata.") # Should not happen if imported globally
    except Exception as e:
        print(f"Warning: Failed to consolidate Zarr metadata: {e}")

    print(f"\n--- Incremental Pipeline Finished ---"); print(f"Climate indices incrementally saved to: {save_path}")
    clear_checkpoints(); return save_path # Return path on success


# ----------------------------------------
# EXECUTION BLOCK
# ----------------------------------------

if __name__ == "__main__":
    print("Running climate index pipeline...")
    # --- Imports --- (zarr is imported globally)

    # --- Configuration ---
    BASE_INPUT_PATH = "/kaggle/input/climate-dataset-variables"
    OUTPUT_ZARR_PATH = "/kaggle/working/climate_indices.zarr"
    EXPECTED_TIME_DIM = 'valid_time' # Make sure this matches your data's primary time dimension name
    FORCE_RECOMPUTE_ALL = False # Set to True to ignore checkpoints and recompute everything

    # Clean up previous output if it exists, to ensure a fresh run
    if os.path.exists(OUTPUT_ZARR_PATH):
        print(f"Removing existing output directory: {OUTPUT_ZARR_PATH}")
        try:
            shutil.rmtree(OUTPUT_ZARR_PATH)
        except Exception as e:
            print(f"Error removing existing output: {e}")

    if FORCE_RECOMPUTE_ALL:
        print("Force recompute enabled. Clearing all checkpoints.")
        clear_checkpoints()
    try:
        start_time = time.time()
        result_path = compute_indices_lazy(
            base_path=BASE_INPUT_PATH,
            save_path=OUTPUT_ZARR_PATH,
            time_dim_name=EXPECTED_TIME_DIM,
            force_recompute=FORCE_RECOMPUTE_ALL
        )
        end_time = time.time()

        if result_path and os.path.exists(result_path):
            print("\n--- Pipeline Summary ---")
            print(f"Output Zarr store generated at: {result_path}")
            print(f"Total execution time: {end_time - start_time:.2f} seconds")

            print("\nVerifying output Zarr store...")
            try:
                 # xarray options for cftimeindex are usually handled by default now
                 # with xr.set_options(enable_cftimeindex=True):
                 # Ensure metadata is consolidated before opening if needed (should be by pipeline)
                ds_check = xr.open_zarr(result_path, consolidated=True, decode_times=True) # Ensure decode_times
                print("Successfully opened Zarr store.")
                print("Variables:", list(ds_check.data_vars))
                print("Dimensions:", dict(ds_check.sizes))
                if EXPECTED_TIME_DIM in ds_check.coords:
                    print(f"Time coordinate ('{EXPECTED_TIME_DIM}') dtype after loading: {ds_check[EXPECTED_TIME_DIM].dtype}")
                    print(f"Sample time values: {ds_check[EXPECTED_TIME_DIM].isel({EXPECTED_TIME_DIM:slice(0,3)}).values}")


                expected_indices = ["GDD", "SPI", "SoilMoistureAnomaly", "EvapotranspirationRatio", "HeatwaveIndex", "WindSpeed"]
                present_indices = list(ds_check.data_vars)
                missing_indices_list = [idx for idx in expected_indices if idx not in present_indices]
                if missing_indices_list:
                    print(f"WARNING: Missing expected indices in output: {missing_indices_list}")

                print("\n--- Sample data and attributes for each variable ---")
                for var_name in present_indices:
                    print(f"\nVariable: {var_name}")
                    print(f"  Dtype: {ds_check[var_name].dtype}")
                    print(f"  Shape: {ds_check[var_name].shape}")
                    print(f"  Chunks: {ds_check[var_name].chunks}")
                    print(f"  Attributes: {ds_check[var_name].attrs}")
                    # Print a small sample of data
                    sample_slice = {dim: 0 for dim in ds_check[var_name].dims} # Take first element along each dim
                    try:
                        sample_data = ds_check[var_name].isel(**sample_slice).compute()
                        print(f"  Sample value (at {sample_slice}): {sample_data.item() if sample_data.size == 1 else sample_data.values}")

                        # Compute and print basic statistics
                        print(f"  Computing min/max for {var_name} (this might take a moment)...")
                        with ProgressBar():
                            var_min = ds_check[var_name].min().compute()
                            var_max = ds_check[var_name].max().compute()
                        print(f"    Min: {var_min.item() if hasattr(var_min, 'item') else var_min}")
                        print(f"    Max: {var_max.item() if hasattr(var_max, 'item') else var_max}")

                    except Exception as e_sample:
                        print(f"    Error getting sample/stats for {var_name}: {e_sample}")


                ds_check.close()
            except Exception as e_verify:
                print(f"Error verifying output Zarr store: {e_verify}")
                traceback.print_exc()
        else:
            print("\nPipeline did not complete successfully or output Zarr store not found.")

    except FileNotFoundError as e: print(f"\nPipeline Error: Input data not found. {e}")
    except RuntimeError as e: print(f"\nPipeline Error: {e}"); traceback.print_exc()
    except ValueError as e: print(f"\nPipeline Error (ValueError): {e}"); traceback.print_exc()
    except AttributeError as e: print(f"\nPipeline Error (AttributeError): {e}"); traceback.print_exc()
    except TypeError as e: print(f"\nPipeline Error (TypeError): {e}"); traceback.print_exc()
    except KeyError as e: print(f"\nPipeline Error (KeyError): {e}"); traceback.print_exc()
    except SyntaxError as e: print(f"\nPipeline Error (SyntaxError): {e}"); traceback.print_exc()
    except NameError as e: print(f"\nPipeline Error (NameError): {e}"); traceback.print_exc() # Catch NameError
    except Exception as e: print(f"\nAn unexpected error occurred in the pipeline:"); traceback.print_exc()
        
