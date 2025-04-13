# ERA5 downloader 
import os
import cdsapi
from typing import List, Optional, Union
from datetime import datetime

import xarray as xr
import numpy as np
from glob import glob
from dask.diagnostics import ProgressBar
import warnings
import shutil
import traceback
import numcodecs # Import numcodecs for compressor objects
import dask.array as da # Import dask array for chunking checks


# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning, message="Sending large graph.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*elementwise comparison failed*")
warnings.filterwarnings("ignore", category=xr.SerializationWarning)
# Suppress the specific Dask warning about object serialization (often seen with groupby.apply)
warnings.filterwarnings("ignore", category=UserWarning, message="Large object passed.+")



def download_era5_data(
    dataset: str = "derived-era5-single-levels-daily-statistics",
    variables: Union[str, List[str]] = "2m_temperature",
    start_year: int = 2010,
    end_year: int = 2025,
    months: Optional[List[str]] = None,
    days: Optional[List[str]] = None,
    time: Optional[List[str]] = None,
    bbox: Optional[List[float]] = None,
    output_dir: str = "era5_data",
    overwrite: bool = False,
    
    # Daily statistics parameters
    daily_statistic: Optional[str] = None,  # 'daily_mean', 'daily_max', etc.
    time_zone: Optional[str] = None,
    
    # Pressure level parameters
    pressure_level: Optional[Union[str, List[str]]] = None,
    
    # Hourly data parameters
    frequency: Optional[str] = None,
    
    # Common ERA5 parameters
    product_type: str = "reanalysis",
    format: str = "netcdf",
    
    # Derived datasets parameters
    derive_variable: Optional[bool] = None,
    statistical_measure: Optional[str] = None
) -> None:
    """
    Download ERA5 data with explicitly defined parameters.
    
    Args:
        dataset: CDS dataset name
        variables: Climate variable(s) to download
        start_year: First year to download
        end_year: Last year to download (inclusive)
        months: Months to download (01-12)
        days: Days to download (01-31)
        time: Specific times for hourly data (00:00, 06:00, etc.)
        bbox: Bounding box [north, west, south, east]
        output_dir: Output directory path
        overwrite: Overwrite existing files
        
        # Dataset-specific parameters
        daily_statistic: For daily datasets ('daily_mean', 'daily_max', etc.)
        time_zone: Timezone offset (e.g., 'utc-05:00')
        pressure_level: Pressure level(s) in hPa for pressure-level data
        frequency: Temporal frequency ('1h', '3h', '6h')
        
        # Additional parameters
        product_type: 'reanalysis' or 'ensemble_members'
        format: Output format ('netcdf' or 'grib')
        derive_variable: Whether to derive variables
        statistical_measure: Statistical measure for derived variables
    """
    
    # Convert single values to lists where needed
    if isinstance(variables, str):
        variables = [variables]
    if isinstance(pressure_level, str):
        pressure_level = [pressure_level]
    
    # Set defaults
    months = months or [f"{m:02d}" for m in range(1, 13)]
    days = days or [f"{d:02d}" for d in range(1, 32)]
    years = [str(y) for y in range(start_year, end_year + 1)]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CDS client
    client = cdsapi.Client()
    
    for year in years:
        # Create filename based on variables
        var_str = "_".join(v.lower().replace(" ", "_") for v in variables)
        if pressure_level:
            var_str += f"_plev_{'_'.join(pressure_level)}"
        output_path = os.path.join(output_dir, f"ERA5_{var_str}_{year}.nc")
        
        # Skip if file exists and not overwriting
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {year} (file exists)")
            continue
            
        print(f"Downloading {', '.join(variables)} for {year}...")
        
        try:
            # Base request parameters
            request = {
                "product_type": product_type,
                "variable": variables,
                "year": year,
                "month": months,
                "day": days,
                "format": format
            }
            
            # Add optional parameters
            if bbox:
                request["area"] = bbox
            if time:
                request["time"] = time
            if daily_statistic:
                request["daily_statistic"] = daily_statistic
            if time_zone:
                request["time_zone"] = time_zone
            if pressure_level:
                request["pressure_level"] = pressure_level
            if frequency:
                request["frequency"] = frequency
            if derive_variable is not None:
                request["derive_variable"] = derive_variable
            if statistical_measure:
                request["statistical_measure"] = statistical_measure
            
            client.retrieve(dataset, request, output_path)
            print(f"Successfully downloaded {output_path}")
            
        except Exception as e:
            print(f"Failed to download {year}: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)


# Climate indices computation
# ----------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------
def kelvin_to_celsius(x):
    """Convert temperatures from Kelvin to Celsius."""
    return x - 273.15

def load_nc_dir(path, time_dim='time'):
    """Load all .nc files in a folder as a combined xarray Dataset."""
    files = sorted(glob(os.path.join(path, "*.nc")))
    if not files: raise FileNotFoundError(f"No .nc files found in {path}")
    sample_ds = None
    try:
        sample_ds = xr.open_dataset(files[0], decode_times=False)
        original_time_dim = time_dim; actual_time_dim_found = None
        potential_time_dims = [d for d in sample_ds.dims if 'time' in d.lower()]
        if time_dim in sample_ds.dims: actual_time_dim_found = time_dim
        elif time_dim in sample_ds.coords and time_dim not in sample_ds.dims:
             coord_dims = sample_ds[time_dim].dims
             if len(coord_dims) == 1: actual_time_dim_found = coord_dims[0]; print(f"Warning: '{time_dim}' is coord, using its dim '{actual_time_dim_found}' for {path}.")
             elif potential_time_dims: actual_time_dim_found = potential_time_dims[0]; print(f"Warning: '{time_dim}' ambiguous coord, using detected dim '{actual_time_dim_found}' for {path}.")
             else: raise KeyError(f"'{time_dim}' is coord but cannot map to single dim in {files[0]}. Dims: {list(sample_ds.dims)}")
        elif potential_time_dims: actual_time_dim_found = potential_time_dims[0]; print(f"Warning: '{time_dim}' not found, using detected '{actual_time_dim_found}' for {path}.")
        else: raise KeyError(f"No time-like dimension found in {files[0]}. Dims: {list(sample_ds.dims)}")
        time_dim = actual_time_dim_found
    except Exception as e: print(f"Error inspecting sample file {files[0]}: {e}"); raise
    finally:
        if sample_ds: sample_ds.close()

    # Determine initial chunking strategy - focus on time
    chunks = {time_dim: 30} # Chunk primarily by time initially

    try:
        # Load with decode_times=True, minimal data/coords for efficiency
        ds = xr.open_mfdataset(
            files,
            combine='by_coords',
            parallel=True,
            chunks=chunks, # Apply initial chunking
            data_vars='minimal',
            coords='minimal',
            compat='override',
            decode_times=True # Decode time coordinates during load
        )

        if time_dim not in ds.coords: print(f"WARNING: Time dimension '{time_dim}' is not a coordinate after loading {path}.")
        # Use .persist() before isnull().any().compute() for potentially large time coords
        # Check for NaT requires computation
        elif ds[time_dim].persist().isnull().any().compute():
            raise ValueError(f"NaT values found post-load for {path} in time coord '{time_dim}'")
        else:
             try:
                 # Ensure time is monotonic (critical for alignment/interpolation)
                 if not ds[time_dim].to_index().is_monotonic_increasing:
                      print(f"Warning: Time dim '{time_dim}' not monotonic for {path} post-load. Sorting...")
                      ds = ds.sortby(time_dim)
                      # Verify after sorting - requires computation
                      if not ds[time_dim].persist().to_index().is_monotonic_increasing:
                          raise ValueError(f"Time sorting failed post-load for {path}")
             except TypeError: print(f"Warning: Could not check time monotonicity post-load for {path} due to index type.")

        ds.encoding['source_file_path'] = path
        return ds
    except ValueError as e: print(f"ValueError loading {path}: {e}"); raise
    except Exception as e: print(f"Unhandled Error loading {path}: {e}"); traceback.print_exc(); raise

def compute_gdd(tmax, tmin, base_temp=10):
    """Compute Growing Degree Days (GDD)."""
    tmax_c = kelvin_to_celsius(tmax); tmin_c = kelvin_to_celsius(tmin); avg_temp = (tmax_c + tmin_c) / 2
    gdd = xr.where(avg_temp > base_temp, avg_temp - base_temp, 0).rename("GDD")
    gdd.attrs = {'long_name': 'Growing Degree Days', 'units': 'degree C day', 'base_temperature_C': base_temp}
    return gdd

def compute_spi(precip, window=30, min_std=1e-6):
    """Compute Standardized Precipitation Index (SPI) using groupby.apply."""
    time_dim = next((dim for dim in precip.dims if "time" in dim.lower()), None)
    if not time_dim: raise ValueError("No time-like dimension found!")

    precip = xr.where(precip < 0, 0, precip)
    rolling_sum = precip.rolling({time_dim: window}, min_periods=window).sum(skipna=False).astype(np.float32)

    grouped_clim = rolling_sum.groupby(f"{time_dim}.dayofyear")

    # --- Compute climatology eagerly for groupby.apply ---
    # Persist helps avoid recomputation within the apply function
    print("  Calculating SPI climatology (mean/std)...")
    with ProgressBar():
        mean_clim = grouped_clim.mean(skipna=True).persist()
        std_clim = grouped_clim.std(skipna=True).persist()
    print("  SPI climatology calculation complete.")
    safe_std_clim = xr.where(std_clim < min_std, min_std, std_clim)

    def standardize(x, means, stds):
        # x is expected to be an xarray.DataArray chunk here when using .apply
        day = x[f"{time_dim}.dayofyear"]
        # Ensure day is scalar for .sel() - handles chunk behavior
        if hasattr(day, 'size') and day.size > 0: # Check if it's an array/DataArray with elements
             # Check if all days in the chunk are the same
             day_values = day.values
             first_day = day_values.flat[0] # Get first element regardless of shape
             if not np.all(day_values == first_day):
                 raise ValueError(f"Days within a {time_dim}.dayofyear group chunk are not uniform! Got {np.unique(day_values)}")
             day_val = first_day
        elif hasattr(day, 'item'): # Check if it's a 0-d array
             day_val = day.item()
        else: # Assume scalar or single value DataArray
             day_val = day

        # Extract the pre-computed mean/std for the specific day of year
        mean_for_day = means.sel(dayofyear=day_val, drop=True)
        std_for_day = stds.sel(dayofyear=day_val, drop=True)

        # Use xr.where for dask compatibility, check for std_for_day being close to zero
        return xr.where(abs(std_for_day) < min_std, 0.0, (x - mean_for_day) / std_for_day)

    # Apply using groupby.apply - this ensures 'x' passed to standardize has coords
    print("  Applying SPI standardization...")
    spi = rolling_sum.groupby(f"{time_dim}.dayofyear").apply(
        standardize, means=mean_clim, stds=safe_std_clim
    )
    print("  SPI standardization complete.")

    spi = spi.rename("SPI").transpose(*rolling_sum.dims) # Ensure original dimension order
    # Restore coordinates that might be lost by apply
    if time_dim in rolling_sum.coords: spi.coords[time_dim] = rolling_sum[time_dim]
    for coord_name, coord_array in rolling_sum.coords.items():
        if coord_name != time_dim and coord_name not in spi.coords:
             # Only add coords that span dimensions present in spi
             if all(dim in spi.dims for dim in coord_array.dims):
                 spi.coords[coord_name] = coord_array
    spi.attrs = {'long_name': 'Standardized Precipitation Index', 'units': 'dimensionless', 'window_days': window}
    return spi

def compute_sma(soil, min_std=1e-6):
    """Compute Soil Moisture Anomaly (SMA) using groupby.apply."""
    time_dim = next((dim for dim in soil.dims if "time" in dim.lower()), None)
    if not time_dim: raise ValueError("No time-like dimension found!")

    soil = soil.astype(np.float32)

    grouped_clim = soil.groupby(f"{time_dim}.dayofyear")

    # --- Compute climatology eagerly for groupby.apply ---
    print("  Calculating SMA climatology (mean/std)...")
    with ProgressBar():
        mean_clim = grouped_clim.mean(skipna=True).persist()
        std_clim = grouped_clim.std(skipna=True).persist()
    print("  SMA climatology calculation complete.")
    safe_std_clim = xr.where(std_clim < min_std, min_std, std_clim)


    def standardize(x, means, stds):
        # x is expected to be an xarray.DataArray chunk here when using .apply
        day = x[f"{time_dim}.dayofyear"]
        # Ensure day is scalar for .sel() - handles chunk behavior
        if hasattr(day, 'size') and day.size > 0:
             day_values = day.values
             first_day = day_values.flat[0]
             if not np.all(day_values == first_day):
                 raise ValueError(f"Days within a {time_dim}.dayofyear group chunk are not uniform! Got {np.unique(day_values)}")
             day_val = first_day
        elif hasattr(day, 'item'):
             day_val = day.item()
        else:
             day_val = day

        # Extract the pre-computed mean/std for the specific day of year
        mean_for_day = means.sel(dayofyear=day_val, drop=True)
        std_for_day = stds.sel(dayofyear=day_val, drop=True)

        # Use xr.where for dask compatibility, check for std_for_day being close to zero
        return xr.where(abs(std_for_day) < min_std, 0.0, (x - mean_for_day) / std_for_day)

    # Apply using groupby.apply
    print("  Applying SMA standardization...")
    sma = soil.groupby(f"{time_dim}.dayofyear").apply(
        standardize, means=mean_clim, stds=safe_std_clim
    )
    print("  SMA standardization complete.")

    sma = sma.rename("SoilMoistureAnomaly").transpose(*soil.dims) # Ensure original dimension order
    # Restore coordinates that might be lost by apply
    if time_dim in soil.coords: sma.coords[time_dim] = soil[time_dim]
    for coord_name, coord_array in soil.coords.items():
        if coord_name != time_dim and coord_name not in sma.coords:
             if all(dim in sma.dims for dim in coord_array.dims):
                 sma.coords[coord_name] = coord_array
    sma.attrs = {'long_name': 'Soil Moisture Anomaly', 'units': 'dimensionless'}
    return sma


def compute_et_ratio(evap, precip, min_precip=0.1):
    """Compute the Evapotranspiration Ratio."""
    precip = xr.where(precip < 0, 0, precip); ratio = xr.where(precip >= min_precip, abs(evap) / precip, np.nan).rename("EvapotranspirationRatio")
    ratio.attrs = {'long_name': 'Evapotranspiration Ratio', 'units': 'dimensionless', 'min_precip_threshold': min_precip}
    return ratio

def compute_heatwave_index(tmax, threshold_c=35, window=7):
    """Compute the Heatwave Index."""
    time_dim = next((dim for dim in tmax.dims if "time" in dim.lower()), None); assert time_dim, "No time-like dimension found!"
    tmax_c = kelvin_to_celsius(tmax); is_hot_day = (tmax_c > threshold_c).astype(np.int8)
    hwi = is_hot_day.rolling({time_dim: window}, min_periods=1).sum(skipna=False).rename("HeatwaveIndex")
    hwi.attrs = {'long_name': 'Heatwave Index (days in window meeting threshold)', 'units': 'days', 'threshold_C': threshold_c, 'window_days': window}
    # Re-chunking handled later
    return hwi

def compute_wind_speed(u, v):
    """Compute Wind Speed magnitude."""
    wind_speed = np.sqrt(u ** 2 + v ** 2).rename("WindSpeed"); wind_speed.attrs = {'long_name': 'Wind Speed Magnitude', 'units': 'm/s'}
    # Re-chunking handled later
    return wind_speed

def coerce_dataset_dtypes(ds, target_dtype="float32", verbose=True):
    """Convert variables to a target float dtype if applicable."""
    converted_ds = ds.copy(deep=False); target_np_dtype = np.dtype(target_dtype)
    for var_name in list(converted_ds.data_vars):
        var = converted_ds[var_name]; original_dtype = var.dtype
        # Convert if floating and not target, or if object (might contain numbers)
        if (np.issubdtype(original_dtype, np.floating) and original_dtype != target_np_dtype) or original_dtype == object:
            if verbose: print(f"Converting variable '{var_name}' from {original_dtype} to {target_dtype}")
            try:
                attrs = var.attrs.copy() # Preserve attributes
                # Handle potential fill value conversion
                fill_value = attrs.get('_FillValue')
                if fill_value is not None:
                   try:
                       attrs['_FillValue'] = target_np_dtype.type(fill_value)
                   except (ValueError, TypeError):
                       print(f"  Warning: Could not convert _FillValue for '{var_name}' to {target_dtype}. Removing it.")
                       del attrs['_FillValue']

                converted_ds[var_name] = var.astype(target_np_dtype)
                converted_ds[var_name].attrs = attrs
            except Exception as e:
                print(f"Warning: Could not convert variable '{var_name}' to {target_dtype}. Error: {e}")
    return converted_ds

def suggest_chunks(ds, time_chunk_size=30, spatial_chunk_size=100, spatial_chunk_threshold=100):
    """
    Suggest chunk sizes as a dictionary suitable for ds.chunk().
    Prefers time chunking and avoids chunking small dimensions.
    Uses -1 notation which ds.chunk() interprets as 'no chunking along this dim'.
    """
    suggested = {}
    for dim in ds.dims:
        size = ds.dims[dim]; dim_lower = dim.lower()
        if "time" in dim_lower:
            suggested[dim] = min(size, time_chunk_size) if size > 0 else 1 # Ensure chunk > 0
        elif dim_lower in ["latitude", "lat", "longitude", "lon"]:
            # Use -1 (no chunking) if dim size is below threshold
            suggested[dim] = min(size, spatial_chunk_size) if size > spatial_chunk_threshold else -1
        else:
            # Default for other dimensions (e.g., levels) - chunk if large
            suggested[dim] = min(size, 100) if size > 1000 else -1 # Use -1 for no chunking

    # Ensure chunk size is not larger than dimension size (only for positive chunks)
    # The -1 value is handled directly by ds.chunk()
    for dim, chunk in suggested.items():
        if chunk != -1 and chunk > ds.dims[dim]:
            suggested[dim] = ds.dims[dim]
        if chunk == 0 and ds.dims[dim] > 0: # Handle zero-size chunk suggestion for non-empty dim
             suggested[dim] = 1
        # print(f"  Suggesting chunk for {dim}: {suggested[dim]}") # Debug

    return suggested


def get_spatial_info(ds):
    """Extracts spatial dimension names, sizes, and coordinates."""
    lat_dim = next((d for d in ds.dims if d.lower() in ['lat', 'latitude']), None)
    lon_dim = next((d for d in ds.dims if d.lower() in ['lon', 'longitude']), None)
    lat_coord = ds[lat_dim] if lat_dim and lat_dim in ds.coords else None # Check coords specifically
    lon_coord = ds[lon_dim] if lon_dim and lon_dim in ds.coords else None # Check coords specifically
    shape = (ds.dims.get(lat_dim, 0), ds.dims.get(lon_dim, 0))
    # A dataset is spatial if it has lat/lon dims AND corresponding coordinates
    is_spatial = bool(lat_dim and lon_dim and lat_coord is not None and lon_coord is not None)
    return {'lat_dim': lat_dim, 'lon_dim': lon_dim, 'lat_coord': lat_coord, 'lon_coord': lon_coord, 'shape': shape, 'is_spatial': is_spatial}


def align_regrid_datasets(datasets, expected_time_dim, interpolation_method='linear'):
    """
    Aligns datasets by time (inner join), identifies target grid (highest res),
    and interpolates others to it. Returns a list of processed datasets in the
    same order as input (with None for failures).
    """
    if not datasets: return []
    print("\n--- Processing Datasets for Alignment & Regridding ---")

    target_spatial_info = None
    max_spatial_pixels = 0
    dataset_info = []
    all_spatial_dims = set() # Keep track of all spatial dim names encountered

    # 1. Gather info, find target grid, identify time dims
    print("Step 1: Gathering dataset info and finding target grid...")
    for i, ds in enumerate(datasets):
        if ds is None: # Handle potential None inputs
             print(f"  Skipping dataset at index {i} as it is None.")
             dataset_info.append({'index': i, 'ds': None, 'spatial': None, 'time_dim': None, 'source_path': f'Input index {i} was None'})
             continue

        # Store source path early for better logging
        source_path = ds.encoding.get('source_file_path', f'Input index {i}')

        spatial_info = get_spatial_info(ds)
        current_time_dim = None
        potential_time_dims = [d for d in ds.dims if 'time' in d.lower()]
        if expected_time_dim in ds.dims:
            current_time_dim = expected_time_dim
        elif expected_time_dim in ds.coords and expected_time_dim not in ds.dims:
             coord_dims = ds[expected_time_dim].dims
             if len(coord_dims) == 1: current_time_dim = coord_dims[0]; print(f"  Info: Using dim '{current_time_dim}' for coord '{expected_time_dim}' in dataset {i} ({source_path}).")
             elif potential_time_dims: current_time_dim = potential_time_dims[0]; print(f"  Warning: Ambiguous coord '{expected_time_dim}', using detected dim '{current_time_dim}' for dataset {i} ({source_path}).")
             else: print(f"  ERROR: Cannot map coord '{expected_time_dim}' to single dim for dataset {i} ({source_path}). Skipping."); dataset_info.append({'index': i, 'ds': None, 'source_path': source_path}); continue # Mark for skipping
        elif potential_time_dims:
             current_time_dim = potential_time_dims[0]; print(f"  Warning: Expected time dim '{expected_time_dim}' not found, using detected '{current_time_dim}' for dataset {i} ({source_path}).")
        else:
             print(f"  ERROR: No time-like dimension found for dataset {i} ({source_path}). Skipping."); dataset_info.append({'index': i, 'ds': None, 'source_path': source_path}); continue # Mark for skipping

        if current_time_dim not in ds.dims:
             print(f"  ERROR: Identified time dim '{current_time_dim}' not actually in dimensions for dataset {i} ({source_path}). Dims: {list(ds.dims)}. Skipping."); dataset_info.append({'index': i, 'ds': None, 'source_path': source_path}); continue

        info = {'index': i, 'ds': ds, 'spatial': spatial_info, 'time_dim': current_time_dim, 'source_path': source_path}
        dataset_info.append(info)

        if spatial_info['is_spatial']:
            num_pixels = spatial_info['shape'][0] * spatial_info['shape'][1]
            if num_pixels > max_spatial_pixels:
                max_spatial_pixels = num_pixels
                target_spatial_info = spatial_info
            # Add spatial dims to the set for later exclusion during time alignment
            if spatial_info['lat_dim']: all_spatial_dims.add(spatial_info['lat_dim'])
            if spatial_info['lon_dim']: all_spatial_dims.add(spatial_info['lon_dim'])
        else:
            print(f"  Warning: Dataset {i} ({source_path}) does not seem to have standard spatial dimensions/coordinates.")

    if target_spatial_info is None:
        print("ERROR: Could not determine a target spatial grid with lat/lon coords. Cannot proceed with regridding.")
        return [None] * len(datasets)
    print(f"Target spatial grid determined: {target_spatial_info['shape']} (Lat: '{target_spatial_info['lat_dim']}', Lon: '{target_spatial_info['lon_dim']}')")
    if target_spatial_info['lat_coord'] is None or target_spatial_info['lon_coord'] is None:
         print("ERROR: Target spatial grid is missing coordinate variables. Cannot interpolate.")
         return [None] * len(datasets)

    # 2. Prepare for time alignment (rename time dims if necessary)
    print("\nStep 2: Preparing datasets for time alignment...")
    processed_for_time_align = []; original_indices_for_time = []; rename_failures = False
    for info in dataset_info:
        if info['ds'] is None: continue # Skip datasets marked for failure earlier
        ds = info['ds']; current_time_dim = info['time_dim']; source_path = info['source_path']
        if current_time_dim != expected_time_dim:
            try:
                print(f"  Renaming time dim '{current_time_dim}' to '{expected_time_dim}' for dataset {info['index']} ({source_path})")
                ds = ds.rename({current_time_dim: expected_time_dim})
                info['ds'] = ds # Update ds in info dict
                info['time_dim'] = expected_time_dim # Update time_dim name in info dict
            except ValueError as e:
                print(f"  ERROR: Failed rename time dim for dataset {info['index']} ({source_path}): {e}. Skipping dataset.");
                info['ds'] = None # Mark as failed
                rename_failures = True; continue
        processed_for_time_align.append(ds)
        original_indices_for_time.append(info['index'])

    if not processed_for_time_align:
        print("ERROR: No datasets available for time alignment (all failed or skipped).")
        return [None] * len(datasets)
    if rename_failures:
        print("Warning: Some datasets were skipped due to time dimension renaming errors.")

    # 3. Align time coordinates (Inner Join) - crucial to exclude spatial dims
    print(f"\nStep 3: Attempting time alignment on {len(processed_for_time_align)} datasets using dimension '{expected_time_dim}'...")
    try:
        # Exclude non-essential coords AND all identified spatial dimensions to focus alignment on time
        misc_coords_to_exclude = {'number', 'step', 'surface', 'heightAboveGround', 'depthBelowLandLayer'}
        # Make sure we don't exclude the time dimension itself if it ended up in all_spatial_dims (unlikely)
        coords_to_exclude_names = list(misc_coords_to_exclude | all_spatial_dims - {expected_time_dim})

        # Filter to only exclude coords actually present in *any* dataset's dims or coords
        actual_coords_to_exclude = [
            c for c in coords_to_exclude_names
            if any((c in ds.coords or c in ds.dims) for ds in processed_for_time_align if ds is not None)
        ]
        print(f"  Excluding dimensions/coordinates from alignment: {actual_coords_to_exclude}")

        time_aligned_datasets = xr.align(
            *processed_for_time_align,
            join='inner',    # Keep only common time steps
            copy=False,      # Avoid unnecessary copies
            exclude=actual_coords_to_exclude # IMPORTANT: Prevent spatial alignment here
        )
        if not time_aligned_datasets or not all(ds.dims.get(expected_time_dim, 0) > 0 for ds in time_aligned_datasets):
            print("ERROR: Time alignment resulted in empty datasets or zero length time dimension.")
            # Construct list of Nones to return
            return [None] * len(datasets) # Match original input length
        print(f"Time alignment successful. Common time dimension size: {time_aligned_datasets[0].dims[expected_time_dim]}")
    except Exception as e:
        print(f"Error during time alignment (xr.align): {e}"); traceback.print_exc()
        return [None] * len(datasets) # Match original input length

    # 4. Regrid spatial coordinates
    print(f"\nStep 4: Regridding datasets to target grid {target_spatial_info['shape']} using method '{interpolation_method}'...")
    final_datasets_map = {} # Map original index to processed dataset
    target_coords_for_interp = {
         target_spatial_info['lat_dim']: target_spatial_info['lat_coord'],
         target_spatial_info['lon_dim']: target_spatial_info['lon_coord']
    }

    for i, ds_time_aligned in enumerate(time_aligned_datasets):
        original_idx = original_indices_for_time[i]
        # Find original source path again for logging
        original_info = next((info for info in dataset_info if info['index'] == original_idx), None)
        source_path_log = original_info['source_path'] if original_info else f"Original index {original_idx}"

        current_spatial_info = get_spatial_info(ds_time_aligned) # Re-check spatial info after align

        if not current_spatial_info['is_spatial']:
            print(f"Warning: Dataset {original_idx} ({source_path_log}) missing spatial dims/coords after time alignment. Cannot regrid. Skipping.")
            final_datasets_map[original_idx] = None # Mark as failed
            continue

        processed_ds = None
        # Check if grid shape AND dimension names match
        if current_spatial_info['shape'] == target_spatial_info['shape'] and \
           current_spatial_info['lat_dim'] == target_spatial_info['lat_dim'] and \
           current_spatial_info['lon_dim'] == target_spatial_info['lon_dim']:
            print(f"Dataset {original_idx} ({source_path_log}): Grid {current_spatial_info['shape']} and dims match target. No interpolation needed.")
            processed_ds = ds_time_aligned
             # --- Ensure chunking even if no interpolation happens ---
            print(f"    Ensuring chunking for dataset {original_idx} ({source_path_log})")
            current_chunks = suggest_chunks(processed_ds) # Use the dictionary output
            processed_ds = processed_ds.chunk(current_chunks)

        else:
            print(f"Dataset {original_idx} ({source_path_log}): Grid {current_spatial_info['shape']} or dims differ from target. Interpolating...")
            # Prepare for interpolation (rename spatial dims if needed)
            rename_interp = {}
            if current_spatial_info['lat_dim'] != target_spatial_info['lat_dim']:
                rename_interp[current_spatial_info['lat_dim']] = target_spatial_info['lat_dim']
            if current_spatial_info['lon_dim'] != target_spatial_info['lon_dim']:
                rename_interp[current_spatial_info['lon_dim']] = target_spatial_info['lon_dim']

            ds_to_interp = ds_time_aligned.rename(rename_interp) if rename_interp else ds_time_aligned

            try:
                 # Perform interpolation using the target coordinate arrays
                 processed_ds = ds_to_interp.interp(
                     coords=target_coords_for_interp,
                     method=interpolation_method,
                     kwargs={"fill_value": np.nan} # Use NaN for fill value
                 )

                 # --- Re-chunk *after* interpolation ---
                 # Suggest chunks based on the *new* interpolated shape
                 new_chunks = suggest_chunks(processed_ds) # Use the dictionary output
                 print(f"    Re-chunking interpolated dataset {original_idx} ({source_path_log}) with suggested chunks: {new_chunks}")
                 processed_ds = processed_ds.chunk(new_chunks)
                 print(f"    Interpolation successful for dataset {original_idx}.")

            except ValueError as e:
                 # Catch the specific "Chunks do not add up to shape" error if it occurs here
                 print(f"  ERROR interpolating dataset {original_idx} ({source_path_log}). Possible chunk issue before interpolation or during interp. Error: {e}")
                 traceback.print_exc()
                 print(f"  Skipping dataset {original_idx} ({source_path_log}) due to interpolation error.")
                 processed_ds = None # Mark as failed
            except Exception as e:
                 print(f"  UNEXPECTED ERROR interpolating dataset {original_idx} ({source_path_log}): {e}")
                 traceback.print_exc()
                 print(f"  Skipping dataset {original_idx} ({source_path_log}) due to interpolation error.")
                 processed_ds = None # Mark as failed

        final_datasets_map[original_idx] = processed_ds # Store processed ds or None

    # 5. Assemble final list in original order
    print("\nStep 5: Assembling final dataset list...")
    final_datasets_ordered = []
    # Use the original dataset_info list to determine the original order and indices
    original_input_indices = [info['index'] for info in dataset_info]

    for original_input_index in original_input_indices:
        processed_ds = final_datasets_map.get(original_input_index) # Get from map using original index
        if processed_ds is None:
             # Find original source path for logging more reliably
             original_info = next((info for info in dataset_info if info['index'] == original_input_index), None)
             source_path_log = original_info['source_path'] if original_info else f"Original index {original_input_index}"
             print(f"Dataset originally at index {original_input_index} (from {source_path_log}) was skipped or failed during processing.")
        final_datasets_ordered.append(processed_ds) # Append the result (or None)

    num_successful = sum(1 for ds in final_datasets_ordered if ds is not None)
    print(f"\nReturning {num_successful} successfully processed datasets (originally {len(datasets)} input).")
    if num_successful != len(datasets):
        print(f"Warning: Number of output datasets ({num_successful}) differs from input ({len(datasets)}). Check logs for skipped datasets.")

    return final_datasets_ordered


# ----------------------------------------
# DIAGNOSTICS FUNCTION
# ----------------------------------------
def run_dataset_diagnostics(base_path="/kaggle/working", expected_time_dim='valid_time'):
    """Run diagnostics on input datasets. Returns True if critical problems found."""
    print("--- Running Dataset Diagnostics ---"); print(f"Checking base path: {base_path}"); print(f"Expecting time dimension name like: '{expected_time_dim}'")
    folders = ["era5_max_2m_temperature", "era5_min_2m_temperature", "total_precipitation", "evaporation", "volumetric_soil_water_layer_1", "10m_u_component_of_wind_mean", "10m_v_component_of_wind_mean"]
    all_time_dims_found = []; all_calendars = []; critical_problems_found = False; grid_mismatch_found = False; folder_file_counts = {}; spatial_grids = {}
    print("\nChecking folder existence and file counts:")
    for folder in folders:
        path = os.path.join(base_path, folder)
        if not os.path.exists(path): print(f"  ERROR: Folder {path} does not exist!"); critical_problems_found = True; folder_file_counts[folder] = 0; continue
        nc_files = sorted(glob(os.path.join(path, "*.nc"))); count = len(nc_files); folder_file_counts[folder] = count
        if not nc_files: print(f"  WARNING: No .nc files found in {path}")
    if critical_problems_found: print("\nCritical errors found (missing folders). Diagnostics aborted."); return True
    print("\nChecking individual dataset properties (using first file in each folder):"); reference_spatial_info = None
    for folder, file_count in folder_file_counts.items():
        if file_count == 0: continue
        path = os.path.join(base_path, folder); nc_files = sorted(glob(os.path.join(path, "*.nc")))
        print(f"\nDiagnostics for {folder}:"); sample_ds = None
        try:
            sample_ds = xr.open_dataset(nc_files[0], decode_times=False)
            print(f"  Variables: {list(sample_ds.data_vars)}"); print(f"  Dimensions: {dict(sample_ds.dims)}")
            time_dim_actual = next((d for d in sample_ds.dims if 'time' in d.lower()), None)
            if not time_dim_actual: print(f"  ERROR: No time-like dimension found!"); critical_problems_found = True
            elif time_dim_actual != expected_time_dim: print(f"  WARNING: Time dim is '{time_dim_actual}', expected '{expected_time_dim}'.")
            if time_dim_actual and time_dim_actual in sample_ds.variables:
                all_time_dims_found.append(time_dim_actual); time_var = sample_ds.variables[time_dim_actual]
                calendar = time_var.attrs.get('calendar', 'Not specified'); units = time_var.attrs.get('units', 'Not specified'); all_calendars.append(calendar)
                print(f"    Time Dim: '{time_dim_actual}', Units: '{units}', Calendar: '{calendar}'") # Added detail
                if not units or 'since' not in units.lower(): print(f"    WARNING: Time units attribute '{units}' might be missing or non-CF compliant.")
            current_spatial_info = get_spatial_info(sample_ds); spatial_grids[folder] = current_spatial_info['shape']
            if current_spatial_info['is_spatial']: # Use the refined check
                print(f"    Spatial Dims: Lat='{current_spatial_info['lat_dim']}', Lon='{current_spatial_info['lon_dim']}', Shape={current_spatial_info['shape']}")
                if reference_spatial_info is None: reference_spatial_info = current_spatial_info
                elif reference_spatial_info['shape'] != current_spatial_info['shape'] or \
                     reference_spatial_info['lat_dim'] != current_spatial_info['lat_dim'] or \
                     reference_spatial_info['lon_dim'] != current_spatial_info['lon_dim']:
                     print(f"    WARNING: Spatial grid shape/dims {current_spatial_info['shape']} ('{current_spatial_info['lat_dim']}', '{current_spatial_info['lon_dim']}') "
                           f"differs from reference {reference_spatial_info['shape']} ('{reference_spatial_info['lat_dim']}', '{reference_spatial_info['lon_dim']}').")
                     grid_mismatch_found = True
            else: print(f"  WARNING: Standard latitude/longitude dimensions/coordinates not found.")
        except Exception as e: print(f"  ERROR processing sample file {nc_files[0]}: {e}"); critical_problems_found = True
        finally:
            if sample_ds: sample_ds.close()
    if all_time_dims_found:
        print("\n--- Cross-Dataset Time Checks ---")
        unique_time_dims = set(all_time_dims_found)
        if len(unique_time_dims) > 1: print(f"ERROR: Inconsistent time dimension names found: {unique_time_dims}"); critical_problems_found = True
        elif unique_time_dims: print(f"Consistent time dimension name found: '{list(unique_time_dims)[0]}'")
        else: print("No time dimensions found across datasets.")
        unique_calendars = set(all_calendars); filtered_calendars = {c for c in unique_calendars if c not in ['Not specified', 'Unknown', 'N/A', None]}
        if len(filtered_calendars) > 1: print(f"ERROR: Inconsistent calendars found: {filtered_calendars}."); critical_problems_found = True
        elif len(filtered_calendars) == 1: print(f"Consistent calendar found: '{list(filtered_calendars)[0]}'")
        elif not filtered_calendars and any(c in ['Not specified', 'Unknown', 'N/A', None] for c in unique_calendars): print("WARNING: Calendar info missing or unspecified across datasets.")
        elif not unique_calendars: pass # No calendars to check
        else: print("Calendar check passed (or only one dataset).")
    print("\n--- Diagnostics Summary ---")
    if grid_mismatch_found: print("WARNING: Inconsistent spatial grids detected. Interpolation will be attempted.")
    if critical_problems_found: print("CRITICAL PROBLEMS detected. Review messages above.")
    elif not grid_mismatch_found and not critical_problems_found: print("Basic diagnostics passed (no critical errors, grids match).")
    elif not critical_problems_found: print("Basic diagnostics passed (no critical errors, grid mismatch noted).")
    print("-----------------------------")
    return critical_problems_found


# ----------------------------------------
# MAIN PIPELINE
# ----------------------------------------
def compute_indices_lazy(base_path="/kaggle/working", save_path="climate_indices.zarr", time_dim_name='valid_time'):
    """Compute climate indices, save result to Zarr. Uses time alignment followed by spatial regridding via interpolation."""
    if not os.path.isdir(base_path): raise FileNotFoundError(f"Base path '{base_path}' does not exist.")

    print("\nRunning pre-computation diagnostics...");
    critical_problems = run_dataset_diagnostics(base_path=base_path, expected_time_dim=time_dim_name)
    if critical_problems:
        print("\nCritical diagnostics problems found. Aborting pipeline."); return None
    print("Diagnostics finished.")

    # Use dask progress bar for the whole pipeline execution
    with ProgressBar():
        print("\n--- Starting Climate Indices Pipeline ---")
        print(f"Using base path: {base_path}"); print(f"Expected time dimension name: {time_dim_name}"); print(f"Output will be saved to: {save_path}")
        interpolation_method = 'linear' # Or 'nearest', etc.

        # Define input paths
        paths = {
            "tmax": os.path.join(base_path, "era5_max_2m_temperature"),
            "tmin": os.path.join(base_path, "era5_min_2m_temperature"),
            "precip": os.path.join(base_path, "total_precipitation"),
            "evap": os.path.join(base_path, "evaporation"),
            "soil": os.path.join(base_path, "volumetric_soil_water_layer_1"),
            "u_wind": os.path.join(base_path, "10m_u_component_of_wind_mean"),
            "v_wind": os.path.join(base_path, "10m_v_component_of_wind_mean")
        }

        # Load datasets lazily
        print("\nLoading datasets (lazily)...");
        datasets_raw = {}; load_errors = False; ordered_names = list(paths.keys())
        for name in ordered_names:
            path = paths[name]
            if not os.path.isdir(path):
                print(f"Warning: Input directory for '{name}' not found at {path}. Skipping."); datasets_raw[name] = None; continue
            if not glob(os.path.join(path, "*.nc")):
                print(f"Warning: No .nc files found for '{name}' at {path}. Skipping."); datasets_raw[name] = None; continue
            try:
                datasets_raw[name] = load_nc_dir(path, time_dim=time_dim_name)
                print(f"  Successfully initiated loading for '{name}'.")
            except (KeyError, ValueError, FileNotFoundError) as e:
                print(f"\nError loading '{name}': {e}"); load_errors = True; datasets_raw[name] = None
            except Exception as e:
                print(f"\nUnexpected error loading '{name}': {e}"); traceback.print_exc(); load_errors = True; datasets_raw[name] = None

        dataset_list_to_process = [datasets_raw[name] for name in ordered_names] # Keep None placeholders

        if all(ds is None for ds in dataset_list_to_process):
             raise RuntimeError("No datasets were successfully loaded or found.")
        if load_errors:
             print("Warning: Errors occurred during loading setup for some datasets.")

        # Align time and regrid spatial coordinates
        print("\nAligning time and regridding spatial coordinates...");
        try:
            # Pass the list including potential None values
            processed_datasets_list = align_regrid_datasets(dataset_list_to_process, expected_time_dim=time_dim_name, interpolation_method=interpolation_method)
        except Exception as e:
            print(f"Coordinate alignment/regridding failed critically: {e}"); traceback.print_exc(); raise

        # Check if the number of returned datasets matches the input order
        if len(processed_datasets_list) != len(ordered_names):
            raise RuntimeError(f"Mismatch in length between input names ({len(ordered_names)}) and processed datasets ({len(processed_datasets_list)}).")

        # Map processed datasets back to names
        processed_datasets_dict = {name: ds for name, ds in zip(ordered_names, processed_datasets_list)}

        # Check for essential datasets after processing
        essential_names = ["tmax", "tmin", "precip", "evap", "soil", "u_wind", "v_wind"]
        missing_essential = []
        for name in essential_names:
             if processed_datasets_dict.get(name) is None:
                 missing_essential.append(name)
                 print(f"ERROR: Essential dataset '{name}' is missing or failed processing.")

        if missing_essential:
             raise RuntimeError(f"Essential datasets missing after processing: {missing_essential}. Cannot compute indices.")

        # --- Datasets successfully aligned and regridded ---
        aligned_tmax_ds = processed_datasets_dict["tmax"]
        aligned_tmin_ds = processed_datasets_dict["tmin"]
        aligned_precip_ds = processed_datasets_dict["precip"]
        aligned_evap_ds = processed_datasets_dict["evap"]
        aligned_soil_ds = processed_datasets_dict["soil"]
        aligned_u_ds = processed_datasets_dict["u_wind"]
        aligned_v_ds = processed_datasets_dict["v_wind"]

        print("\nDimensions after alignment & regridding (Tmax dataset):"); print(dict(aligned_tmax_ds.dims))
        final_spatial_info = get_spatial_info(aligned_tmax_ds)
        if not final_spatial_info['is_spatial'] or final_spatial_info['shape'][0] <= 1 or final_spatial_info['shape'][1] <= 1:
             raise ValueError(f"Invalid spatial dimensions after processing: {final_spatial_info}")

        # Extract primary variables (assuming first data var if specific name not found)
        print("\nExtracting primary variables...");
        def get_first_data_var(ds, name):
            if ds is None: raise ValueError(f"Input dataset for '{name}' is None.") # Should be caught earlier
            if not ds.data_vars: raise ValueError(f"Dataset for '{name}' contains no data variables.")

            common_name_map = {
                "tmax": ["mx2t", "tmax", "tasmax"], "tmin": ["mn2t", "tmin", "tasmin"],
                "precip": ["tp", "precip", "pr"], "evap": ["e", "evap"],
                "soil": ["swvl1", "soil_moisture"], "u_wind": ["u10", "uas"], "v_wind": ["v10", "vas"]
            }
            expected_vars = common_name_map.get(name, [])
            for var_name in expected_vars:
                if var_name in ds.data_vars:
                    print(f"  Found expected variable '{var_name}' for '{name}'.")
                    return ds[var_name]

            # Fallback to first variable if expected not found
            var_name = list(ds.data_vars)[0]
            print(f"  Warning: Expected var not found for '{name}'. Extracting first var '{var_name}'.")
            return ds[var_name]

        tmax = get_first_data_var(aligned_tmax_ds, "tmax")
        tmin = get_first_data_var(aligned_tmin_ds, "tmin")
        precip = get_first_data_var(aligned_precip_ds, "precip")
        evap = get_first_data_var(aligned_evap_ds, "evap")
        soil = get_first_data_var(aligned_soil_ds, "soil")
        u = get_first_data_var(aligned_u_ds, "u_wind")
        v = get_first_data_var(aligned_v_ds, "v_wind")

        # Ensure inputs to computation have appropriate chunks (applied during align_regrid)
        print("\nVerifying input chunks before computation (Tmax example):")
        if hasattr(tmax, 'chunks') and tmax.chunks:
             print(f"  tmax chunks: {tmax.chunksizes}") # Display chunksizes tuple
        else:
             print(f"  tmax is not chunked or chunks info unavailable.")


        print("\nComputing climate indices (lazily)...");
        try:
            gdd = compute_gdd(tmax, tmin)
            print("- GDD computation graph defined.")
            spi = compute_spi(precip)
            print("- SPI computation graph defined.")
            sma = compute_sma(soil)
            print("- SMA computation graph defined.")
            et_ratio = compute_et_ratio(evap, precip)
            print("- ET Ratio computation graph defined.")
            hwi = compute_heatwave_index(tmax)
            print("- HWI computation graph defined.")
            wind_speed = compute_wind_speed(u, v)
            print("- Wind Speed computation graph defined.")
        except Exception as e:
            print(f"Error during index computation graph definition: {e}"); traceback.print_exc(); raise

        # Create output dataset
        print("\nCreating output dataset...");
        ds_out = xr.Dataset({
            "GDD": gdd,
            "SPI": spi,
            "SoilMoistureAnomaly": sma,
            "EvapotranspirationRatio": et_ratio,
            "HeatwaveIndex": hwi,
            "WindSpeed": wind_speed
        })

        # Coerce data types
        print("\nCoercing data types to float32...");
        ds_out = coerce_dataset_dtypes(ds_out, target_dtype="float32", verbose=False)

        # Apply final chunking for output (using suggest_chunks dictionary)
        print("\nApplying final chunking for Zarr output...");
        final_chunks_out_dict = suggest_chunks(ds_out, time_chunk_size=30, spatial_chunk_size=100) # Use dictionary output
        print(f"Applying final output chunk sizes (using dict): {final_chunks_out_dict}")
        ds_out = ds_out.chunk(final_chunks_out_dict)

        # Verify final chunks (optional debug)
        # print("\nVerifying chunks in final output dataset (GDD example):")
        # if 'GDD' in ds_out and hasattr(ds_out['GDD'], 'chunks') and ds_out['GDD'].chunks:
        #     print(f"  GDD chunks: {ds_out['GDD'].chunksizes}")
        # else:
        #     print("  GDD variable not found or not chunked.")


        # Add metadata
        print("\nAdding metadata...");
        ds_out.attrs['description'] = 'Climate indices computed from ERA5 data'
        ds_out.attrs['source_data_path'] = base_path
        ds_out.attrs['processing_script_name'] = __file__ if '__file__' in globals() else 'interactive_notebook'
        ds_out.attrs['creation_date'] = np.datetime_as_string(np.datetime64('now', 's'), unit='s') + ' UTC'
        ds_out.attrs['Conventions'] = 'CF-1.8'
        ds_out.attrs['regridding_method'] = interpolation_method
        # Add source file info if available from inputs
        source_files_info = {name: ds.encoding.get('source_file_path', 'N/A')
                             for name, ds in processed_datasets_dict.items() if ds is not None}
        ds_out.attrs['source_datasets'] = str(source_files_info)

        # Prepare encoding for Zarr
        print("\nPreparing encoding for Zarr store...");
        encoding = {};
        compressor = numcodecs.zlib.Zlib(level=1) # Choose compression
        fill_value_float = np.float32(np.nan)    # Define fill value for float32

        # Get the desired chunking from the *dataset* after the .chunk() call
        # ds_out.chunks is a FrozenDict {dim_name: chunk_tuple}
        target_chunks_per_dim = ds_out.chunks

        for var_name in list(ds_out.data_vars) + list(ds_out.coords):
            var = ds_out[var_name]
            enc = {'compressor': compressor}

            # --- Revised Chunk Encoding ---
            # Use the target chunks derived from the Dataset after the final .chunk() call
            if hasattr(var, 'chunks') and var.chunks is not None and var.ndim > 0:
                 # Construct the chunk tuple based on the variable's dimensions
                 # and the target chunk sizes stored in target_chunks_per_dim
                 try:
                     var_chunk_tuple = tuple(target_chunks_per_dim[dim][0] for dim in var.dims)
                     # Basic validation: ensure tuple length matches ndim
                     if len(var_chunk_tuple) == var.ndim:
                          # Ensure chunks are positive and not larger than actual dim size
                          validated_chunks = []
                          for i, chunk_size in enumerate(var_chunk_tuple):
                               dim_name = var.dims[i]
                               dim_total_size = ds_out.dims[dim_name]
                               # Use min(chunk_size, dim_total_size) and max(1, ...)
                               valid_chunk = max(1, min(chunk_size, dim_total_size) if dim_total_size > 0 else 1)
                               validated_chunks.append(valid_chunk)
                          enc['chunks'] = tuple(validated_chunks)
                          # print(f"  Encoding chunks for {var_name}: {enc['chunks']}") # Debug
                     else:
                          print(f"  Warning: Mismatched dimension count for '{var_name}' when creating chunk tuple. Skipping chunk encoding.")
                 except KeyError as e:
                      print(f"  Warning: Dimension '{e}' not found in dataset chunks dict for variable '{var_name}'. Skipping chunk encoding.")
                 except Exception as e:
                      print(f"  Warning: Error processing chunks for variable '{var_name}': {e}. Skipping chunk encoding.")
            # elif var.ndim == 0:
            #      print(f"  Skipping chunk encoding for 0-dim variable '{var_name}'.") # No chunks for scalars
            # else:
            #      print(f"  Variable '{var_name}' is not chunked. Skipping chunk encoding.")


            # Handle FillValue: Apply only to float data variables
            if var_name in ds_out.data_vars and np.issubdtype(var.dtype, np.floating):
                enc['_FillValue'] = fill_value_float
                # print(f"  Encoding _FillValue for {var_name}: {enc['_FillValue']}") # Debug

            # Add dtype to encoding (optional but can be helpful)
            enc['dtype'] = str(var.dtype)

            if enc: # Only add if encoding dict is not empty
                encoding[var_name] = enc

        # --- DEBUG: Print final encoding dict ---
        print(f"Final encoding dictionary (first few keys): {dict(list(encoding.items())[:5])}")

        # Save to Zarr
        print(f"\nSaving computed indices to Zarr store: {save_path}...");
        if os.path.exists(save_path):
            print(f"Warning: Output path {save_path} already exists. Overwriting.");
            shutil.rmtree(save_path)

        # Create the write job (compute=False means lazy)
        write_job = ds_out.to_zarr(
            save_path,
            mode='w',
            consolidated=True, # Improves read performance
            encoding=encoding,
            compute=False      # Keep it lazy until the final compute() call
        )

        # Trigger computation and writing
        print("Submitting Dask graph for computation and writing...");
        # The ProgressBar context manager handles the compute() call
        results = write_job.compute()
        print(f"\n--- Pipeline Finished ---");
        print(f"Climate indices successfully saved to: {save_path}")

        return ds_out # Return the computed dataset

# Clean previous output
if os.path.exists("/kaggle/working/climate_indices.zarr"):
    shutil.rmtree("/kaggle/working/climate_indices.zarr")

# Run pipeline
indices = compute_indices_lazy(base_path="/kaggle/working/", save_path="climate_indices.zarr")
