# 1. ERA5 Climate Data Downloader 🌍

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
