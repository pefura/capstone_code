# 1. ERA5 Climate Data Downloader 🌍 (Climate_data_ERA5_downloader.py)

This repository provides a Python script to download ERA5 climate reanalysis data from the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/) in an efficient and flexible way.

The script uses the `cdsapi` package to query and retrieve climate variables (like temperature, precipitation, soil moisture, wind speed, and more) as NetCDF files, directly from the ECMWF CDS servers.

---

## Features

- Download ERA5 data for any date range (`start_year` → `end_year`).
- Select one or more **climate variables** (e.g., `2m_temperature`, `total_precipitation`, etc).
- Filter by:
  - Month, day, time.
  - Specific geographic **bounding box**.
  - Pressure levels (for pressure-level datasets).
- Supports both **daily** and **hourly** datasets.
- Automatically names output files and skips already downloaded files (unless `overwrite=True`).
- Handles advanced ERA5 options like:
  - `daily_statistic` (mean, max, min),
  - `time_zone` adjustment,
  - `frequency` for hourly data,
  - `derive_variable` & `statistical_measure` for post-processing.

---

# 2. Climate_data.py
A complete pipeline for preprocessing, cleaning, aligning, regridding, and calculating climate indices from ERA5 datasets.

Highlights:
- Uses xarray, dask, and zarr for efficient big-data handling.

- Supports automated chunking and checkpointing.

- Built-in diagnostics for grid consistency, missing files, and metadata checks.

- Computes advanced climate indices:

   - GDD — Growing Degree Days.

   - SPI — Standardized Precipitation Index.

   - SMA — Soil Moisture Anomaly.

   - EvapotranspirationRatio — E/P ratio.

   - HeatwaveIndex — number of extreme hot days.

   - WindSpeed — derived from u and v components.

💾 Output: multi-variable, consolidated Zarr data store for easy integration with machine learning pipelines.

# 3. function_process_climate_daily_indices()
Purpose: Aggregates daily gridded climate data (Zarr format) spatially (e.g., to US states) and exports to Parquet/CSV.

Key Features:
  - Spatial Aggregation: Uses regionmask to compute mean/max/min by region (currently supports US states).

  - Timedelta Support: Converts time-based variables (e.g., HeatwaveIndex) to numeric seconds.

  - Output Formats: Saves results as Parquet (recommended) or CSV.

  - Dask Integration: Handles large datasets via chunked processing.

Inputs:
  - Zarr store with daily climate data (must include latitude/longitude and a time dimension).

  - Optional: Region filter (e.g., ['CA', 'TX']), timedelta variables to convert.

Outputs:
  - DataFrame with Date index and {variable}_{region} columns (e.g., Temperature_CA).

  - Status dictionary with file paths or error details.

Dependencies:
  - xarray, pandas, regionmask, dask, zarr, pyarrow.


# 4. Cultuvation wheat data in US
wheat_table05_monthly_approx_200910_to_202425.csv

# 5. Fertilizers prices and crude oil price
commodities_prices_world_bank_fertilizers_oil_2010-01-01_to_2025-03-31.csv

# 6. Farm prices for wheat in US
all_wheat_monthly_prices_200910_to_202425.csv

# 7. Supply and peoduction of wheat in US
wheat_table05_monthly_approx_200910_to_202425.csv"

# 8. Macroeconomic data
fred_data_2010-01-01_to_2025-03-31.csv

# 9. Geopolitical risk
gpr_dataset.csv

# 10. Financial data and interest rate-10y treasury bond yield (from yahoo finance)
yfinance_data_Adj_Close_2010-01-01_to_2025-03-12.csv


