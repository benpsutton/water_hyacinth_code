import json
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import pandas as pd


### Dont think i'll need this anymore as will do this validation in data_utils for the labels.gpkg now

# def validate_points_columns(points_gdf: gpd.GeoDataFrame, points_file: str | Path):
#     """Validate and standardize a labelled points GeoDataFrame."""
#     required_columns = {"lc", "obs_date"}

#     missing_columns = required_columns - set(points_gdf.columns)

#     if missing_columns:
#         missing_str = ", ".join(sorted(missing_columns))
#         raise ValueError(f"{points_file} is missing required columns: {missing_str}")

#     if points_gdf.geometry is None:
#         raise ValueError(f"{points_file} has no geometry column")

#     if points_gdf.empty:
#         raise ValueError(f"{points_file} contains no points")

#     if points_gdf.crs is None:
#         raise ValueError(f"{points_file} has no CRS")

#     cleaned_gdf = points_gdf.copy()

#     try:
#         cleaned_gdf["obs_date"] = pd.to_datetime(
#             cleaned_gdf["obs_date"],
#             format="%Y-%m-%d",
#             errors="raise",
#         ).dt.strftime("%Y-%m-%d")
#     except (TypeError, ValueError) as exc:
#         bad_values = cleaned_gdf["obs_date"].dropna().astype(str).unique().tolist()
#         raise ValueError(
#             f"{points_file} has invalid obs_date values. Expected YYYY-MM-DD."
#             f"Examples: {bad_values[:5]}"
#         ) from exc

#     if cleaned_gdf["obs_date"].isna().any():
#         raise ValueError(f"{points_file} containes null obs_date valeus")

#     return cleaned_gdf


def create_image_collection(
    sites_file: str | Path,
    location_gdf: gpd.GeoDataFrame,
    location: str,
    clip: bool = False,
):
    """Build a Sentinel-2 image collection for one study location."""
    if not isinstance(location, str):
        raise TypeError(f"location must be a string, got {type(location).__name__}")

    with Path(sites_file).open("r", encoding="utf-8") as s:
        sites = json.load(s)

    current_site_point = sites.get("sites", {}).get(location, {}).get("geometry", {}).get("coordinates", [])
    point_geom = ee.Geometry.Point(current_site_point)

    current_site_bbox = sites.get("sites", {}).get(location, {}).get("bbox", {}).get("coordinates", [])
    bbox_geom = ee.Geometry.Polygon(current_site_bbox)

    if "obs_date" not in location_gdf.columns:
        raise ValueError(f"The points file for {location} has no obs_date column")

    date_list = location_gdf["obs_date"].astype(str).unique().tolist()

    def mask_clouds_scl(image):
        scl = image.select("SCL")
        mask = scl.neq(3).And(scl.neq(9)).And(scl.neq(8))
        return image.updateMask(mask)

    image_list = []
    for date in date_list:
        ee_date = ee.Date.parse("yyyy-MM-dd", date)

        daily_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point_geom)
            .filterDate(ee_date, ee_date.advance(1, "day"))
            .map(mask_clouds_scl)
            .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
        )

        if daily_collection.size().getInfo() == 0:
            raise ValueError(
                f"No Sentinel-2 image found for location '{location}' on {date}."
            )

        image = daily_collection.first().set({"obs_date": date, "location": location})

        if clip:
            image = image.clip(bbox_geom)

        image_list.append(image)

    image_collection = ee.ImageCollection(image_list)

    print(f"Created image collection for: {location}")
    return image_collection


def create_images_for_all_locations(
    sites_file: str | Path,
    cleaned_label_fp: str | Path,
    clip: bool = False,
):
    """Build a combined image collection for all study locations."""

    
    with Path(sites_file).open("r", encoding="utf-8") as f:
        sites = json.load(f)


    locations_list = list(sites.get("sites", {}).keys())

    cleaned_label_fp = Path(cleaned_label_fp)

    label_gdf = gpd.read_file(cleaned_label_fp)

    merged_ic = ee.ImageCollection([])

    # This could be a slightly roundabout way of doing this, but in earlier version had separate files for the labelled points at each location

    for location in locations_list:

        location_gdf = label_gdf.loc[label_gdf["location"] == location]

        if location_gdf.empty:
            raise ValueError(f"There are no labelled points for {location}")

        ic = create_image_collection(
            sites_file=sites_file,
            location_gdf = location_gdf,
            location=location,
            clip=clip,
        )
        merged_ic = merged_ic.merge(ic)

    return merged_ic

####----------------------------------------------------------------------------------------------------------
#### -------- SAMPLING -------------------------------------------------------------------------------------

def sample_points_from_image(point_fc_for_image, image):
    """Sample pixel values from one image at a set of labelled point locations."""
    proj = image.select("B2").projection()

    samples = image.sampleRegions(
        collection=point_fc_for_image,
        properties=["lc", "binary", "obs_date", "location", "lon", "lat"],
        scale=10,
        projection=proj,
        geometries=False,
    )

    return samples


def sample_patches_from_image(point_fc_for_image, image):
    """Create image patch arrays and sample them at labelled points."""
    kernel_radius = 7
    bands = ["B2", "B3", "B4", "B5", "B8", "B11", "B12"]

    image = image.select(bands)
    proj = image.select("B2").projection()

    array_image = image.neighborhoodToArray(kernel=ee.Kernel.square(kernel_radius))

    sampled_patches = array_image.sampleRegions(
        collection=point_fc_for_image,
        properties=["lc", "binary", "obs_date", "location", "lon", "lat"],
        scale=10,
        projection=proj,
        geometries=False,
    )
    return sampled_patches


def sample_by_date(date, location_fc, merged_ic, location, points_or_patches):
    """Sample all labelled points for one date within one study location."""
    point_fc_for_image = location_fc.filter(ee.Filter.eq("obs_date", date))

    image = (
        merged_ic.filter(ee.Filter.eq("location", location))
        .filter(ee.Filter.eq("obs_date", date))
        .first()
    )

    if points_or_patches == "points":
        samples = sample_points_from_image(point_fc_for_image, image)
    elif points_or_patches == "patches":
        samples = sample_patches_from_image(point_fc_for_image, image)
    else:
        raise ValueError("points_or_patches argument must be either 'points' or 'patches'")

    return samples


def sample_by_location(location, merged_fc, merged_ic, points_or_patches):
    """Sample all labelled dates for a single study location."""
    location_fc = merged_fc.filter(ee.Filter.eq("location", location))
    date_list = ee.List(location_fc.aggregate_array("obs_date")).distinct()

    def map_over_date(date):
        return sample_by_date(date, location_fc, merged_ic, location, points_or_patches)

    samples_for_location = ee.FeatureCollection(date_list.map(map_over_date)).flatten()

    return samples_for_location


def subset_merged_fc_by_location(locations_ee_list, merged_ic, merged_fc, points_or_patches):
    """Sample all configured locations and merge the results."""
    valid_points_or_patch_values = {"points", "patches"}

    if points_or_patches not in valid_points_or_patch_values:
        raise ValueError("points_or_patches argument must be either 'points' or 'patches'")

    def map_over_location(location):
        return sample_by_location(location, merged_fc, merged_ic, points_or_patches)

    samples_all_locations = ee.FeatureCollection(locations_ee_list.map(map_over_location)).flatten()

    return samples_all_locations

# Can get rif of this once ive moved some of its functionality to the cleaning functions in data_utils

# def _load_merged_points(sites_file: str | Path, project_root: str | Path) -> gpd.GeoDataFrame:
#     project_root = Path(project_root)

#     with Path(sites_file).open("r", encoding="utf-8") as f:
#         sites = json.load(f)

#     locations_list = list(sites.get("sites", {}).keys())
#     gdf_list = []
    
#     for location in locations_list:
#         points_file = project_root / "configs" / "point_files" / f"{location}_points.shp"

#         if not points_file.exists():
#             raise FileNotFoundError(f"Points file not found for {location}: {points_file}")

#         points_gdf = gpd.read_file(points_file)
#         points_gdf = validate_points_columns(points_gdf, points_file)
#         points_gdf = points_gdf.to_crs("EPSG:4326")

#         def multipoint_to_point(geom):
#             if geom.geom_type == "Point":
#                 return geom
#             if geom.geom_type == "MultiPoint" and len(geom.geoms) == 1:
#                 return geom.geoms[0]
#             raise ValueError(f"Expected Point or singleton MultiPoint, got {geom.geom_type}")

#         points_gdf["geometry"] = points_gdf.geometry.apply(multipoint_to_point)
#         points_gdf["location"] = location
#         points_gdf["lon"] = points_gdf.geometry.x
#         points_gdf["lat"] = points_gdf.geometry.y

#         gdf_list.append(points_gdf)

#     return pd.concat(gdf_list, ignore_index=True)


# Change this so it is passed the cleaned_label_gpkg instead

def get_samples(merged_ic, cleaned_label_fp: str | Path) -> pd.DataFrame:
    """Prepare labelled points for sampling and extract all samples."""

    cleaned_label_gdf = gpd.read_file(cleaned_label_fp)

    merged_fc = geemap.gdf_to_ee(cleaned_label_gdf) 
    locations_ee_list = merged_fc.aggregate_array("location").distinct()

    all_samples = subset_merged_fc_by_location(
        locations_ee_list,
        merged_ic,
        merged_fc,
        points_or_patches="points",
    )

    all_samples_df = geemap.ee_to_df(all_samples)

    return all_samples_df


def export_patches(merged_ic, cleaned_label_fp: str | Path):

    cleaned_label_gdf = gpd.read_file(cleaned_label_fp)
    merged_fc = geemap.gdf_to_ee(cleaned_label_gdf)
    locations_ee_list = merged_fc.aggregate_array("location").distinct()

    all_samples = subset_merged_fc_by_location(
        locations_ee_list,
        merged_ic,
        merged_fc,
        points_or_patches="patches",
    )

    task = ee.batch.Export.table.toDrive(
        collection=all_samples,
        description="sampled_patches_as_arrays",
        folder="Dissertation",
        fileFormat="GeoJSON",
    )

    task.start()
    print("Exporting sampled patches as GeoJSON to drive/Dissertation")
