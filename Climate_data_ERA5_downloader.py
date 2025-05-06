# ERA5 downloader 
!pip install cdsapi h5netcdf zarr numcodecs
import os
import cdsapi
from typing import List, Optional, Union
from datetime import datetime
import os
import zipfile

# Connect to CDS database via API
# Replace with your actual CDS API key!
cdsapirc_content = """
url: https://cds.climate.copernicus.eu/api
key: CDS_API_KEY
"""

# Write to Kaggle's working directory (not /root/)
config_file_path = "/kaggle/working/.cdsapirc"  
with open(config_file_path, "w") as f:
    f.write(cdsapirc_content.strip())

# Set environment variable to force CDS API to use this file
os.environ["CDSAPI_RC"] = "/kaggle/working/.cdsapirc"

# Verify the file exists
!cat /kaggle/working/.cdsapirc


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


# Example usages:

# 1. Daily mean temperature
download_era5_data(
    dataset = "derived-era5-single-levels-daily-statistics",
    variables="2m_temperature",
    start_year=2010,
    end_year=2025,
    output_dir="era5_2m_temperature",
    daily_statistic="daily_mean",
    time_zone="utc-05:00",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]
)

# 1bis. Daily land temperature

download_era5_data(
    dataset = "derived-era5-land-daily-statistics",
    variables="2m_temperature",
    start_year=2010,
    end_year=2025,
    output_dir="era5_land_2m_temperature",
    daily_statistic="daily_mean",
    time_zone="utc-05:00",
    frequency="6_hourly",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]

)

# 2. Daily max temperature [lancer le 06 avril 2025]

download_era5_data(
    dataset = "derived-era5-single-levels-daily-statistics",
    variables="maximum_2m_temperature_since_previous_post_processing",
    start_year=2010,
    end_year=2025,
    output_dir= "era5_max_2m_temperature",
    daily_statistic="daily_mean",
    time_zone="utc-05:00",
    frequency="6_hourly",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]
)

#3. Daily min temperature
download_era5_data(
    dataset = "derived-era5-single-levels-daily-statistics",
    variables= "minimum_2m_temperature_since_previous_post_processing",
    start_year=2010,
    end_year=2025,
    output_dir= "era5_min_2m_temperature",
    daily_statistic="daily_mean",
    time_zone="utc-05:00",
    frequency="6_hourly",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]
)

# 4. mean total precipitation
download_era5_data(
    dataset = "derived-era5-single-levels-daily-statistics",
    variables = "total_precipitation",
    start_year=2010,
    end_year=2025,
    output_dir= "total_precipitation",
    daily_statistic="daily_mean",
    time_zone="utc-05:00",
    frequency="6_hourly",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]
)

#5. soil moisture 1
download_era5_data(
    dataset = "derived-era5-land-daily-statistics",
    variables= "volumetric_soil_water_layer_1",
    start_year=2010,
    end_year=2025,
    output_dir= "volumetric_soil_water_layer_1",
    daily_statistic="daily_mean",
    time_zone="utc-05:00",
    frequency="6_hourly",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]
)


# 6. Evaporation
download_era5_data(
    dataset = "derived-era5-single-levels-daily-statistics",
    variables= "evaporation",
    start_year=2010,
    end_year=2025,
    output_dir= "evaporation",
    daily_statistic="daily_mean",
    time_zone="utc-05:00",
    frequency="6_hourly",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]
)

#7. mean wind u
download_era5_data(
    dataset = "derived-era5-land-daily-statistics",
    variables= "10m_u_component_of_wind",
    start_year=2010,
    end_year=2025,
    output_dir= "10m_u_component_of_wind_mean",
    daily_statistic="daily_mean",
    time_zone="utc-05:00",
    frequency="6_hourly",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]
)

#8. Max wing u
download_era5_data(
    dataset = "derived-era5-land-daily-statistics",
    variables= "10m_u_component_of_wind",
    start_year=2010,
    end_year=2025,
    output_dir= "10m_u_component_of_wind_max",
    daily_statistic="daily_maximum",
    time_zone="utc-05:00",
    frequency="6_hourly",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]
)

#9. mean wind v
download_era5_data(
    dataset = "derived-era5-land-daily-statistics",
    variables= "10m_v_component_of_wind",
    start_year=2010,
    end_year=2025,
    output_dir= "10m_v_component_of_wind_mean",
    daily_statistic="daily_mean",
    time_zone="utc-05:00",
    frequency="6_hourly",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]
)

#10. max wind v
download_era5_data(
    dataset = "derived-era5-land-daily-statistics",
    variables= "10m_v_component_of_wind",
    start_year=2010,
    end_year=2025,
    output_dir= "10m_v_component_of_wind_max",
    daily_statistic="daily_maximum",
    time_zone="utc-05:00",
    frequency="6_hourly",
    bbox= [49.3845, -124.8489, 24.3963, -66.8854]
)

# Compressed all files in zip format
def zip_folder_structure(base_dir, zip_filename):
    # Delete the old zip if it exists
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        print(f"🗑️ Deleted old: {zip_filename}")

    # Create a ZipFile object in write mode
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory and add files to the zip file
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                # Create a relative path for each file based on the base directory
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, arcname)

# Define the base directory and the output zip filename
base_directory = '/kaggle/working/' # Replace with the actual path
zip_filename = 'climate_data.zip'

# Zip the folder structure
zip_folder_structure(base_directory, zip_filename)








