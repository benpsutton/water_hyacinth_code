import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import geemap
import ee

def validate_points_columns(points_gdf: gpd.GeoDataFrame, points_file: str| Path):
    """Validate and standardize a labelled points GeoDataFrame.

    This helper is used before both image retrieval and sampling so that the
    pipeline works from a single, predictable schema. It checks that the input
    file contains the minimum required columns, has valid geometry and CRS
    information, and stores observation dates in a consistent ``YYYY-MM-DD``
    string format.

    Args:
        points_gdf: Labelled point data loaded from a shapefile.
        points_file: Path to the source file, used in error messages.

    Returns:
        A copy of ``points_gdf`` with ``obs_date`` normalized to ``YYYY-MM-DD``
        strings.

    Raises:
        ValueError: If required columns, geometry, CRS, or valid ``obs_date``
            values are missing.
    """

    required_columns = {"lc", "obs_date"}

    missing_columns = required_columns - set(points_gdf.columns)

    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"{points_file} is missing required columns: {missing_str}")

    if points_gdf.geometry is None:
        raise ValueError(f"{points_file} has no geometry column")

    if points_gdf.empty:
        raise ValueError(f"{points_file} contains no points")

    if points_gdf.crs is None:
        raise ValueError(f"{points_file} has no CRS")
    
    cleaned_gdf = points_gdf.copy()

    try:
        cleaned_gdf["obs_date"] = pd.to_datetime(
        cleaned_gdf["obs_date"],
        format = "%Y-%m-%d",
        errors= "raise",
        ).dt.strftime("%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        bad_values = cleaned_gdf["obs_date"].dropna().astype(str).unique().tolist()
        raise ValueError(
            f"{points_file} has invalid obs_date values. Expected YYYY-MM-DD."
            f"Examples: {bad_values[:5]}"
        ) from exc
    
    if cleaned_gdf["obs_date"].isna().any():
        raise ValueError(f"{points_file} containes null obs_date valeus")
    
    return cleaned_gdf


def create_image_collection(sites_file : str | Path,
                            points_file: str | Path,
                            location: str,
                            ):
    """Build a Sentinel-2 image collection for one study location.

    The function reads the labelled points for a single site, validates their
    schema and observation dates, and then retrieves one Sentinel-2 SR image
    for each labelled observation date. Each image is filtered to the site,
    cloud-masked using the SCL band, restricted to the selected spectral bands,
    and clipped to the site's bounding box.

    Args:
        sites_file: JSON file containing per-site metadata, including point
            geometry and bounding box coordinates.
        points_file: Shapefile containing labelled points for the target site.
            The file must contain at least ``lc`` and ``obs_date`` columns.
        location: Site name matching a key in ``sites_file``.

    Returns:
        An ``ee.ImageCollection`` containing one image per observation date for
        the requested site. Each image is tagged with ``obs_date`` and
        ``location`` properties for later joins during sampling.

    Raises:
        TypeError: If ``location`` is not a string.
        ValueError: If the points file fails validation.
    """
    
    if not isinstance(location, str):
        raise TypeError(f"location must be a string, got {type(location).__name__}")
    # location = location.lower() I previously had the location keys as lower case in the sites file

    with Path(sites_file).open("r", encoding = "utf-8") as s:
        sites = json.load(s)
    
    # Create gdf from labelled points
    points_file = Path(points_file)
    points_gdf = gpd.read_file(points_file)

    points_gdf = validate_points_columns(points_gdf, points_file)
    
    # Get coordinates of current location 
    current_site_point = sites.get("sites", {}).get(location,{}).get("geometry", {}).get("coordinates", [])
    # convert to a ee point. 
    point_geom = ee.Geometry.Point(current_site_point)

    # Get bbox to clip
    current_site_bbox = sites.get("sites", {}).get(location,{}).get("bbox", {}).get("coordinates", [])
    bbox_geom = ee.Geometry.Polygon(current_site_bbox)

    # Get date list from labelled points using the obs_date column in points_file:
    if not "obs_date" in points_gdf.columns:
        raise ValueError(f"The points file for {location} has no obs_date column")
    else: 
        date_list = points_gdf["obs_date"].astype(str).unique().tolist()

    # ------------------------------
    def mask_clouds_SCL(image):
            scl = image.select('SCL')
            mask = scl.neq(3).And(scl.neq(9)).And(scl.neq(8))                           
            return image.updateMask(mask)
    
    # Create an image collection from the date list

    image_list = []
    for date in date_list:
        ee_date = ee.Date.parse('yyyy-MM-dd', date)

        daily_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point_geom)
            .filterDate(ee_date, ee_date.advance(1,"day"))
            .map(mask_clouds_SCL)
            .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
        )

        # Skip observation dates that have no Sentinel-2 image so downstream
        # sampling fails with a clear Python-side error instead of a null EE
        # object deep in the call stack.
        if daily_collection.size().getInfo() == 0:
            raise ValueError(
                f"No Sentinel-2 image found for location '{location}' on {date}."
            )

        image = (daily_collection
            .first()
            .set({
                 "obs_date": date,
                 "location":location
            })
            #.clip(bbox_geom) revisit if need this when not bringing the image out of GEE
           )

        image_list.append(image)
    # ---------------------------------------------
    # create an image ffrom python python list of individual images
    image_collection = ee.ImageCollection(image_list)

    print(f"Created image collection for: {location}")
    return image_collection



def create_images_for_all_locations(sites_file: str | Path, project_root: str | Path):
    """Build a combined image collection for all configured study locations.

    This is a thin wrapper around :func:`create_image_collection`. It reads the
    site names from the project configuration, locates the corresponding point
    shapefile for each site, and merges the resulting per-site image
    collections into one ``ee.ImageCollection``.

    Args:
        sites_file: JSON file describing all study sites.
        project_root: Root directory of the code project. Point files are
            expected at ``configs/points_files/<location>_points.shp``.

    Returns:
        An ``ee.ImageCollection`` containing all site/date images used for
        point sampling.

    Raises:
        FileNotFoundError: If a required point shapefile is missing.
    """
    project_root = Path(project_root)
    with Path(sites_file).open("r", encoding="utf-8") as f:
        sites = json.load(f)

    locations_list = list(sites.get("sites", {}).keys())

    # then need to load the points files to pass to create_image_collection()
    # could do this in the bewow loop?

    merged_ic = ee.ImageCollection([])

    for location in locations_list:
            points_file = project_root / "configs" / "point_files" / f"{location}_points.shp"

            if not points_file.exists():
                raise FileNotFoundError(f"Points file not found for {location}: {points_file}")
    
            ic = create_image_collection(sites_file = sites_file, points_file = points_file, location = location)
            merged_ic = merged_ic.merge(ic)
            # Should i flatten this into one image collection and then select by obs_date and loaction in a later function?
    return merged_ic

def sample_points_from_image(point_fc_for_image,
                             image,
                             ):
    """Sample pixel values from one image at a set of labelled point locations.

    Sampling is locked to the native projection of Sentinel-2 band ``B2`` so
    that point extraction happens on the image grid rather than in geographic
    coordinates. The input points do not need to be manually reprojected in
    Earth Engine for this operation. Geometries are dropped from the output to
    keep the result compact for tabular machine learning workflows.

    Args:
        point_fc_for_image: ``ee.FeatureCollection`` of points associated with
            a single image date and location.
        image: ``ee.Image`` from which spectral bands will be sampled.

    Returns:
        An ``ee.FeatureCollection`` containing the copied point attributes and
        sampled band values.
    """
    proj = image.select("B2").projection()

    # dont need to reproject the feature collection as EE handles that, as long as we arent keeping the geometries.

    samples = image.sampleRegions(
        collection=point_fc_for_image,
        properties=["lc", "binary", "obs_date", "location", "lon", "lat"],
        scale=10,
        projection=proj,    # <-- locks the CRS and avoids pyramid ambiguity
        geometries=False
        )
     
    return samples

def sample_patches_from_image(point_fc_for_image,
                            image,
                           ):
    """
    To create an image where each pixel is an array, then sample the arrays for the labelled pixels
    """
    kernel_radius = 7 # How many pixels either side of the labelled pixel e.g. radius of 7 gives 15x15
    bands = ["B2", "B3", "B4", "B5", "B8", "B11", "B12"]

    image = image.select(bands)
    proj = image.select("B2").projection()

    array_image = image.neighborhoodToArray(kernel = ee.Kernel.square(kernel_radius), )

    sampled_patches = array_image.sampleRegions(collection = point_fc_for_image,
                                                properties = ["lc", "binary", "obs_date", "location", "lon", "lat"],
                                                scale = 10,
                                                projection = proj,
                                                geometries = False
                                                )
    return sampled_patches

def sample_by_date(date,
                   location_fc,
                   merged_ic,
                   location,
                   points_or_patches):
    """Sample all labelled points for one date within one study location.

    Args:
        date: Observation date used to match both the label points and the
            image metadata.
        location_fc: ``ee.FeatureCollection`` containing all labelled points
            for the current location.
        merged_ic: Combined ``ee.ImageCollection`` for all locations.
        location: Site identifier used to filter the image collection.

    Returns:
        An ``ee.FeatureCollection`` of sampled points for the requested
        location-date pair.
    """
    point_fc_for_image = location_fc.filter(ee.Filter.eq("obs_date", date))

    image = (merged_ic
        .filter(ee.Filter.eq("location", location))
        .filter(ee.Filter.eq("obs_date", date))
        .first()
    )
    
    if points_or_patches == "points":
        samples = sample_points_from_image(point_fc_for_image, image)
    elif points_or_patches == "patches":
        samples = sample_patches_from_image(point_fc_for_image, image)

    return samples
     

def sample_by_location(location,
                       merged_fc,
                       merged_ic,
                       points_or_patches):
    """Sample all labelled dates for a single study location.

    Args:
        location: Site identifier to extract from the merged point collection.
        merged_fc: ``ee.FeatureCollection`` containing labelled points from all
            sites.
        merged_ic: ``ee.ImageCollection`` containing images for all
            site-date pairs.

    Returns:
        An ``ee.FeatureCollection`` of sampled points for one location across
        all available observation dates.
    """
    location_fc = merged_fc.filter(ee.Filter.eq("location", location))
    
    date_list = ee.List(location_fc.aggregate_array("obs_date")).distinct()

    def map_over_date(date):
        return sample_by_date(date, location_fc, merged_ic, location, points_or_patches)

    samples_for_location = ee.FeatureCollection(date_list.map(map_over_date)
    ).flatten()

    return samples_for_location

def subset_merged_fc_by_location(locations_ee_list,
                                 merged_ic,
                                 merged_fc,
                                 points_or_patches):
    """Sample all configured locations and merge the results.

    Args:
        locations_ee_list: Earth Engine list of unique location identifiers.
        merged_ic: ``ee.ImageCollection`` containing all site-date images.
        merged_fc: ``ee.FeatureCollection`` containing all labelled points.

    Returns:
        An ``ee.FeatureCollection`` containing sampled points from every
        location in ``locations_ee_list``.
    """

    valid_points_or_patch_values = {"points", "patches"}

    if points_or_patches not in valid_points_or_patch_values:
        raise ValueError("points_or_patches argument must be either 'points' or 'patches'")
                         
    def map_over_location(location):
        return sample_by_location(location, merged_fc, merged_ic, points_or_patches)
    
    samples_all_locations = ee.FeatureCollection(locations_ee_list.map(map_over_location)).flatten()

    return samples_all_locations

def get_samples(merged_ic,
                sites_file: str | Path,
                project_root: str | Path
                )-> pd.DataFrame:
    """Prepare labelled points for sampling and extract all samples.

    This function loads every site-specific point shapefile, validates the
    schema, standardizes dates, reprojects geometries to ``EPSG:4326`` for
    Earth Engine upload, and stores original longitude and latitude as regular
    attributes. The merged point set is then converted to an Earth Engine
    feature collection and sampled against the pre-built image collection.

    Args:
        merged_ic: ``ee.ImageCollection`` containing one image for each
            location-date pair.
        sites_file: JSON file listing all study sites.
        project_root: Root directory of the code project.

    Returns:
        An ``ee.FeatureCollection`` containing sampled spectral values together
        with label and metadata fields needed for downstream tabular analysis.

    Raises:
        FileNotFoundError: If a required point shapefile is missing.
        ValueError: If a point file fails schema or date validation.
    """
    project_root = Path(project_root)

    with Path(sites_file).open("r", encoding="utf-8") as f:
        sites = json.load(f)
    
    locations_list = list(sites.get("sites", {}).keys())

    gdf_list = []

    for location in locations_list:
            points_file = project_root / "configs" / "point_files" / f"{location}_points.shp"

            if not points_file.exists():
                raise FileNotFoundError(f"Points file not found for {location}: {points_file}")
            
            points_gdf = gpd.read_file(points_file)

            points_gdf = validate_points_columns(points_gdf, points_file)

            points_gdf = points_gdf.to_crs("EPSG:4326")

            # Had issue with multipoint geometreis so small helper function
            def multipoint_to_point(geom):
                if geom.geom_type == "Point":
                    return geom
                if geom.geom_type == "MultiPoint" and len(geom.geoms) == 1:
                    return geom.geoms[0]
                raise ValueError(f"Expected Point or singleton MultiPoint, got {geom.geom_type}")

            points_gdf["geometry"] = points_gdf.geometry.apply(multipoint_to_point)

            # Force a consistent join key between labels and the image collection.
            points_gdf["location"] = location
            points_gdf["lon"] = points_gdf.geometry.x
            points_gdf["lat"] = points_gdf.geometry.y

            gdf_list.append(points_gdf)
    
    merged_gdf = pd.concat(gdf_list, ignore_index=True)

    # Convert merged_gdf to ee.FeatureCollection

    merged_fc = geemap.gdf_to_ee(merged_gdf)

    locations_ee_list = merged_fc.aggregate_array("location").distinct()

    all_samples =  subset_merged_fc_by_location(
         locations_ee_list,
         merged_ic,
         merged_fc,
         points_or_patches= "points"
    )

    all_samples_df = geemap.ee_to_df(all_samples)
    
    return all_samples_df
    
    
def export_patches(merged_ic, sites_file: str | Path, project_root: str | Path):

    project_root = Path(project_root)

    with Path(sites_file).open("r", encoding="utf-8") as f:
        sites = json.load(f)

    locations_list = list(sites.get("sites", {}).keys())

    gdf_list = []

    for location in locations_list:
            points_file = project_root / "configs" / "point_files" / f"{location}_points.shp"

            if not points_file.exists():
                raise FileNotFoundError(f"Points file not found for {location}: {points_file}")
            
            points_gdf = gpd.read_file(points_file)

            points_gdf = validate_points_columns(points_gdf, points_file)

            points_gdf = points_gdf.to_crs("EPSG:4326")

            # Had issue with multipoint geometreis so small helper function
            def multipoint_to_point(geom):
                if geom.geom_type == "Point":
                    return geom
                if geom.geom_type == "MultiPoint" and len(geom.geoms) == 1:
                    return geom.geoms[0]
                raise ValueError(f"Expected Point or singleton MultiPoint, got {geom.geom_type}")

            points_gdf["geometry"] = points_gdf.geometry.apply(multipoint_to_point)

            # Force a consistent join key between labels and the image collection.
            points_gdf["location"] = location
            points_gdf["lon"] = points_gdf.geometry.x
            points_gdf["lat"] = points_gdf.geometry.y

            gdf_list.append(points_gdf)
    
    merged_gdf = pd.concat(gdf_list, ignore_index=True)

    # Convert merged_gdf to ee.FeatureCollection

    merged_fc = geemap.gdf_to_ee(merged_gdf)

    locations_ee_list = merged_fc.aggregate_array("location").distinct()

    all_samples =  subset_merged_fc_by_location(
         locations_ee_list,
         merged_ic,
         merged_fc, 
         points_or_patches= "patches"
    )

    task = ee.batch.Export.table.toDrive(
                                        collection= all_samples,
                                        description= 'sampled_patches_as_arrays',
                                        folder= 'Dissertation',
                                        fileFormat= 'GeoJSON'
                                        )
    
    task.start() # check this

    return print("Exporting sampled patches to drive/Dissertation")
