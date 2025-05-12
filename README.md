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

# 7. Supply and production of wheat in US
wheat_table05_monthly_approx_200910_to_202425.csv"

# 8. Macroeconomic data
fred_data_2010-01-01_to_2025-03-31.csv

# 9. Geopolitical risk
gpr_dataset.csv

# 10. Financial data and interest rate-10y treasury bond yield (from yahoo finance)
yfinance_data_Adj_Close_2010-01-01_to_2025-03-12.csv

# 11 Merged dataset for futures wheat prices prediction
dataset : merged_wheat_predictors_W-MON_from_daily_climate_v4.csv

# Wheat Supply and Production Dataset Documentation

## Overview
This dataset combines multiple data sources to create a comprehensive weekly dataset for wheat supply and production analysis. The data is merged and aligned to a weekly frequency (W-MON) from various daily, monthly, and quarterly sources.

## Data Sources and Variables

### 1. Climate Data (Daily Aggregated)
- **GDD (Growing Degree Days)**: Measures heat accumulation for crop growth
- **SPI (Standardized Precipitation Index)**: Measures precipitation anomalies
- **SoilMoistureAnomaly**: Soil moisture deviation from normal conditions
- **EvapotranspirationRatio**: Ratio of actual to potential evapotranspiration
- **WindSpeed**: Average wind speed measurements
- **HeatwaveIndex**: Measures extreme heat conditions

### 2. Cultivation Data (Monthly)
- **US_Planted_Acreage_M_Acres**: Total planted acreage in millions of acres
- **US_Harvested_Acreage_M_Acres**: Total harvested acreage in millions of acres
- **US_Production_M_Bu**: Total production in millions of bushels
- **US_Yield_Bu_Acre**: Yield per acre in bushels
- **US_Imports_M_Bu**: Total imports in millions of bushels
- **US_Food_Use_M_Bu**: Food usage in millions of bushels
- **US_Seed_Use_M_Bu**: Seed usage in millions of bushels
- **US_Feed_Residual_Use_M_Bu**: Feed and residual usage in millions of bushels
- **US_Total_Domestic_Use_M_Bu**: Total domestic usage in millions of bushels
- **US_Exports_M_Bu**: Total exports in millions of bushels
- **US_Total_Disappearance_M_Bu**: Total disappearance in millions of bushels
- **US_Beginning_Stocks_M_Bu**: Beginning stocks in millions of bushels
- **US_Ending_Stocks_M_Bu**: Ending stocks in millions of bushels
- **US_Total_Supply_M_Bu**: Total supply in millions of bushels

### 3. Fertilizer and Oil Prices (Monthly)
- **DAP**: Diammonium Phosphate price
- **TSP**: Triple Superphosphate price
- **Urea**: Urea price
- **Crude_oil_WTI**: West Texas Intermediate crude oil price

### 4. Farm Price Data (Monthly)
- **Price_USD_per_Bushel**: Wheat price in USD per bushel
- **US_Farm_Price_USD_Bu**: US farm price in USD per bushel

### 5. FRED Economic Indicators (Quarterly)
- **CPIAUCSL**: Consumer Price Index for All Urban Consumers
- **PPIACO**: Producer Price Index for All Commodities
- **UNRATE**: Unemployment Rate
- **GDPC1**: Real Gross Domestic Product

### 6. GPR Data (Daily)
- **GPR**: Geopolitical Risk Index

### 7. YFinance Data (Daily)
- **ZW=F**: Wheat Futures price
- **^TNX**: 10-Year Treasury Yield
- **DX-Y.NYB**: US Dollar Index

## Data Processing Notes

1. **Frequency Alignment**: All data is aligned to weekly frequency (W-MON)
2. **Aggregation Methods**:
   - Climate variables: Mean aggregation for most metrics, sum for GDD
   - Price data: Last value of the period
   - Supply/Demand metrics: Mean aggregation
   - Economic indicators: Mean aggregation

3. **Missing Value Handling**:
   - Forward fill (ffill) for most variables
   - Backward fill (bfill) with limit of 10 periods for remaining gaps

4. **Spatial Aggregation**:
   - Climate data is aggregated across key wheat-producing states:
     - CO, ID, IL, KS, MN, MT, ND, NE, OK, SD, TX, WA
   - Aggregation methods include:
     - Mean (Belt_Mean)
     - Standard Deviation (Belt_Std)
     - Minimum (for SPI and SoilMoisture)
     - Maximum (for HeatwaveIndex and GDD)

## Target Variable
- **Target_Return**: Weekly percentage return of wheat futures price (ZW=F)

## Data Range
- Start Date: 2010-01-01
- End Date: 2025-03-31

## File Information
- Output File: `merged_wheat_predictors_W-MON_from_daily_climate_v4.csv`
- Frequency: Weekly (W-MON)
- Format: CSV with Date as index 



