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

def export_awei_p95_asset_for_location(
    sites_file: str | Path,                                                                                                                                    
    location: str,                                                                                                                                           
    ee_project: str,                                                                                                                                           
    start_date: str = "2019-01-01",                                                                                                                          
    end_date: str = "2025-12-31",                                                                                                                              
    ):
    """Compute AWEIp95 for one location and export to an EE asset."""                                                                       
                                                                                                                                                                 
    with Path(sites_file).open("r", encoding="utf-8") as s:                                                                                                    
          sites = json.load(s)                                                                                                                                   
                                                                                                                                                                 
    site = sites["sites"][location]                                                                                                                          
    point_geom = ee.Geometry.Point(site["geometry"]["coordinates"])
    bbox_geom = ee.Geometry.Polygon(site["bbox"]["coordinates"])                                                                                               
   
    def mask_clouds_scl(image):
        scl = image.select("SCL")
        mask = scl.neq(3).And(scl.neq(9)).And(scl.neq(8))
        return image.updateMask(mask)
    
    sent2_ic = (                                                                                                                                               
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")                                                                                                    
        .filterBounds(point_geom)                                                                                                                              
        .filterDate(start_date, end_date)                                                                                                                      
        .map(mask_clouds_scl)
    )                                                                                                                                                          

    # calculate AWEIsh                                                                                       
    def make_awei(image):                                                                                                                                      
        expr = "b('B2')+2.5*b('B3')-1.5*(b('B8')+b('B11'))-0.25*b('B12')"                                                                                    
        return image.expression(expr).rename("AWEI")                                                                                                           
   
    awei_p95 = (                                                                                                                                               
        sent2_ic.map(make_awei)                                                                                                                              
        .reduce(ee.Reducer.percentile([95]))                                                                                                                   
        .rename("AWEIp95")                                                                                                                                   
        .round().toInt32()
    )                                                                                                                                                          
   
    asset_id = f"projects/{ee_project}/assets/awei_p95_{location}"                                                                                             
                                                                                                                                                               
    task = ee.batch.Export.image.toAsset(
        image=awei_p95,
        description=f"awei_p95_{location}",                                                                                                                    
        assetId=asset_id,
        region=bbox_geom,                                                                                                                                      
        scale=10,                                                                                                                                            
        maxPixels=1e10,                                                                                                                                        
    )

    try:
        ee.data.deleteAsset(asset_id)
        print(f"Deleted existing asset: {asset_id}")                                                                                                               
    except ee.EEException:
        pass   

    task.start()                                               

    print(f"Exporting AWEIp95 asset for {location} → {asset_id}")          

    return task, asset_id                                                 

def export_awei_p95_all_locations(
        sites_file: str | Path,
        ee_project: str,                                                      
        start_date: str = "2019-01-01",                                                                                                                          
        end_date: str = "2025-12-31",                                                                                                                              
        ):
    """ Exports an AWEIp95 image as an asset for each location in site file
    Returns an ee task_list that can be checked for export status"""

    with Path(sites_file).open("r", encoding="utf-8") as s:                                                                                                    
        sites = json.load(s)                                                                                                                                   
                                                                                                                                                                 
    location_list = list(sites.get("sites",{}).keys())

    task_list = []

    for location in location_list:
        t, _ = export_awei_p95_asset_for_location(sites_file, location, ee_project=ee_project, start_date= start_date, end_date= end_date)
        task_list.append(t)

    print("Ensure that the exports are complete before running create_image_collection_for_all_locations()") 
    return task_list


def create_image_collection(
    sites_file: str | Path,
    location_gdf: gpd.GeoDataFrame,
    location: str,
    ee_project: str,
    clip: bool = False,
    images_file: str | Path | None = None,
):
    """Build a Sentinel-2 image collection for one location."""

    if not isinstance(location, str):
        raise TypeError(f"location must be a string, got {type(location).__name__}")

    with Path(sites_file).open("r", encoding="utf-8") as s:
        sites = json.load(s)

    # Use the exact granule each date was labelled on (recorded by the labelling
    # app), not filterDate(...).first(). Inle straddles two MGRS tiles, so .first()
    # could return the wrong granule and drop points that fell in its no-data margin.
    if images_file is None:
        images_file = Path(sites_file).parent / "images.json"
    with Path(images_file).open("r", encoding="utf-8") as im:
        images_config = json.load(im)

    target_image_ids = {
        entry["obs_date"]: entry["s2_target_image_id"]
        for loc_key, entries in images_config.items()
        if loc_key.lower() == location.lower()
        for entry in entries
    }

    if "obs_date" not in location_gdf.columns:
        raise ValueError(f"The points file for {location} has no obs_date column")

    date_list = location_gdf["obs_date"].astype(str).unique().tolist()

    def mask_clouds_scl(image):
        scl = image.select("SCL")
        mask = scl.neq(3).And(scl.neq(9)).And(scl.neq(8))
        return image.updateMask(mask)

    AWEI_asset_id = f"projects/{ee_project}/assets/awei_p95_{location}"
    AWEI_p95 = ee.Image(AWEI_asset_id).round().toInt32() # value range is ~ -11000 to 12000 so rounding to int fine.

    image_list = []
    for date in date_list:
        target_image_id = target_image_ids.get(date)

        if target_image_id is None:
            raise ValueError(
                f"No labelled Sentinel-2 image id found for location '{location}' "
                f"on {date} in {images_file}."
            )

        image = (
            mask_clouds_scl(ee.Image(target_image_id))
            .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
            .set({"obs_date": date, "location": location})
        )

        if clip:
            current_site_bbox = sites.get("sites", {}).get(location, {}).get("bbox", {}).get("coordinates", [])
            bbox_geom = ee.Geometry.Polygon(current_site_bbox)
            image = image.clip(bbox_geom)

        image_plus_AWEIp95 = image.addBands(AWEI_p95)

        image_list.append(image_plus_AWEIp95)

    image_collection = ee.ImageCollection(image_list)

    print(f"Created image collection for: {location}")
    return image_collection


def create_images_for_all_locations(
    sites_file: str | Path,
    cleaned_label_fp: str | Path,
    ee_project: str,
    clip: bool = False,
    images_file: str | Path | None = None,
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
            ee_project= ee_project,
            clip=clip,
            images_file=images_file,
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
        properties=["lc", "class_label", "class_int", "obs_date", "location", "longitude", "latitude", "label_id"],
        scale=10,
        projection=proj,
        geometries=False,
    )

    return samples


def sample_patches_from_image(point_fc_for_image, image, kernel_size):
    """Create image patch arrays and sample them at labelled points."""

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    
    kernel_radius = (kernel_size -1)//2

    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8a","B11", "B12", "AWEIp95"]

    image = image.select(bands)
    proj = image.select("B2").projection()

    array_image = image.neighborhoodToArray(kernel=ee.Kernel.square(kernel_radius))

    sampled_patches = array_image.sampleRegions(
        collection=point_fc_for_image,
        properties=["lc", "class_label", "class_int", "obs_date", "location", "longitude", "latitude", "label_id"],
        scale=10,
        projection=proj,
        geometries=False,
    )
    return sampled_patches


def sample_by_date(date, location_fc, location_ic, points_or_patches, kernel_size):
    """Sample all labelled points for one date within one study location."""
    point_fc_for_image = location_fc.filter(ee.Filter.eq("obs_date", date))

    image = (
        location_ic
        .filter(ee.Filter.eq("obs_date", date))
        .first()
    )

    if points_or_patches == "points":
        samples = sample_points_from_image(point_fc_for_image, image)
    elif points_or_patches == "patches":
        samples = sample_patches_from_image(point_fc_for_image, image, kernel_size)
    else:
        raise ValueError("points_or_patches argument must be either 'points' or 'patches'")

    return samples


def sample_by_location(location, location_fc, location_ic, points_or_patches, kernel_size):
    """Sample all labelled dates for a single study location."""

    valid_points_or_patch_values = {"points", "patches"}

    if points_or_patches not in valid_points_or_patch_values:
        raise ValueError("points_or_patches argument must be either 'points' or 'patches'")
    
    date_list = ee.List(location_fc.aggregate_array("obs_date")).distinct()

    def map_over_date(date):
        return sample_by_date(date, location_fc, location_ic, points_or_patches, kernel_size)

    samples_for_location = ee.FeatureCollection(date_list.map(map_over_date)).flatten()

    return samples_for_location


def subset_merged_fc_by_location(locations_ee_list, merged_ic, merged_fc, points_or_patches, kernel_size: int | None = None):
    """Sample all configured locations and merge the results."""
    valid_points_or_patch_values = {"points", "patches"}

    if points_or_patches not in valid_points_or_patch_values:
        raise ValueError("points_or_patches argument must be either 'points' or 'patches'")

    def map_over_location(location):
        return sample_by_location(location, merged_fc, merged_ic, points_or_patches, kernel_size)

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

    # cleaned_label_gdf = gpd.read_file(cleaned_label_fp)

    # merged_fc = geemap.gdf_to_ee(cleaned_label_gdf) 
    # locations_ee_list = merged_fc.aggregate_array("location").distinct()

    # all_samples = subset_merged_fc_by_location(
    #     locations_ee_list,
    #     merged_ic,
    #     merged_fc,
    #     points_or_patches="points",
    # )

    # all_samples_df = geemap.ee_to_df(all_samples)

    cleaned_label_gdf = gpd.read_file(cleaned_label_fp)

    # Where should i get location from? Sites or cleaned_labels. 
    
    location_list = cleaned_label_gdf["location"].unique()

    for location in location_list:
        location_gdf = cleaned_label_gdf.loc[cleaned_label_gdf["location"]== location]
        location_fc = geemap.gdf_to_ee(location_gdf)

        location_ic= merged_ic.filter(ee.Filter.eq("location", location))

        location_samples = sample_by_location(
            location, location_fc,
            location_ic,
            points_or_patches= "points", 
            kernel_size = None)
        
        task = ee.batch.Export.table.toDrive(
            collection= location_samples,
            description= f"sampled_points_{location}",
            folder= "Dissertation",
            fileFormat = "CSV"
        )
        task.start()

        print(f"Exporting sampled points for {location} to drive/Dissertation")

    

def export_patches(merged_ic, cleaned_label_fp: str | Path, kernel_size: int):

    if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
        raise TypeError(f"kernel_size must be an int, got {type(kernel_size).__name__}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    

    cleaned_label_gdf = gpd.read_file(cleaned_label_fp)
    
    locations_list = list(cleaned_label_gdf["location"].unique())

    for location in locations_list:
        location_gdf = cleaned_label_gdf.loc[cleaned_label_gdf["location"] == location]
        location_fc = geemap.gdf_to_ee(location_gdf)
        location_ic = merged_ic.filter(ee.Filter.eq("location", location))

        location_samples = sample_by_location(location= location,
                            location_fc= location_fc,
                            location_ic= location_ic,
                            points_or_patches= 'patches',
                            kernel_size= kernel_size
                            )

        task = ee.batch.Export.table.toDrive(
            collection=location_samples,
            description=f"sampled_{kernel_size}_pixel_patches_for_{location}",
            folder="Dissertation",
            fileFormat="GeoJSON",
        )

        task.start()
        
        print(f"Exporting sampled patches for {location} as GeoJSON to drive/Dissertation")
