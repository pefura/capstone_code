# ERA5 downloader 
import os
import cdsapi
from typing import List, Optional, Union
from datetime import datetime

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
