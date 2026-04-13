<<<<<<< HEAD
- the files for labelled point must be stored in /cofigs/points_files/ as a .shpp with the name format location_points.shpp, where location matches the region name in the attribute

the ee project name should be specified in an .env file as EE_PROJECT

# Project Overview

This repository contains code for building a labelled point dataset from Sentinel-2 imagery in Google Earth Engine and preparing those samples for downstream machine learning analysis. The workflow is designed for research settings where labelled observations come from multiple study sites and may span different continents and image projections.

# Sampling Workflow

The core sampling logic is implemented in [`src/gee_utils.py`](/Users/bensutton/Library/CloudStorage/Dropbox/MASTERS/Dissertation/code/src/gee_utils.py). The workflow is:

1. Read the labelled point shapefile for each study site.
2. Validate the minimum required schema, including `lc`, `obs_date`, geometry, and CRS.
3. Standardize `obs_date` to `YYYY-MM-DD` so that local labels and Earth Engine image metadata match exactly.
4. Build one Sentinel-2 image per site-date pair from `COPERNICUS/S2_SR_HARMONIZED`.
5. Apply a cloud mask using the Sentinel-2 SCL band.
6. Sample the selected bands at each labelled point.
7. Export the resulting records as a tabular training dataset.

# CRS Handling

The labelled point files may originate in geographic coordinates, while Sentinel-2 images are stored in their native projected CRS, typically a local UTM grid. This project keeps the points in `EPSG:4326` when converting them to Earth Engine, then samples pixel values using the projection of Sentinel-2 band `B2`.

This design is deliberate:

- the point files remain easy to inspect and merge across regions
- sampling is performed on the native Sentinel-2 grid
- the final machine learning table does not depend on a shared projected CRS across continents

Original longitude and latitude are retained as plain attributes (`lon`, `lat`) so that sampled rows can be traced back to their source locations without keeping full geometries in the exported table.

# Expected Inputs

The code assumes the following project structure:

- site metadata in `configs/sites.json`
- one labelled point shapefile per site in `configs/points_files/`
- point filenames following the pattern `<location>_points.shp`

Each point shapefile should contain at least:

- `lc`: the class label used for model training
- `obs_date`: the observation date in `YYYY-MM-DD` format
- point geometry

The `location` value used during sampling is derived from the site key in `sites.json`, which helps keep image metadata and point metadata consistent even if shapefile attributes differ in case or spelling.

# Output Structure

The sampling stage is intended to produce a flat table suitable for machine learning. A typical output row contains:

- label information such as `lc`
- site metadata such as `location`
- temporal metadata such as `obs_date`
- original coordinates as `lon` and `lat`
- sampled spectral bands such as `B2`, `B3`, `B4`, `B8`, `B11`, and `B12`

Because the classifier is trained on attributes rather than geometries, geometries can be dropped after sampling unless they are needed for a separate spatial validation workflow.

# Research Notes

For a research-facing codebase, useful documentation should explain not just what the code does, but why specific decisions were made. In this project, the important decisions worth documenting are:

- why dates are standardized before any Earth Engine operations
- why sampling is tied to the image projection rather than reprojecting the raster stack
- why geometries are dropped from the final training table
- what assumptions are made about file naming and site configuration

That style of documentation makes the workflow easier to reproduce, review, and defend in a methods chapter or appendix.
=======
# water_hyacinth_code
Repository of code for MSc dissertation project 
>>>>>>> 083346ae0783b35bc82029477c22eb44495fe78f
