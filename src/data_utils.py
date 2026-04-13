import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import geemap
import ee

INDICES_PATH = Path(__file__).resolve().parent.parent / "configs" / "indices.json"

# Create a helper function that creates new indices as columns in a pd.DataFrame

def add_indices(df: pd.DataFrame, 
                indices_file : str | Path = INDICES_PATH
                ) -> pd.DataFrame:
    
    df = df.copy()

    with Path(indices_file).open("r", encoding="utf-8") as f:
        indices = json.load(f)

    for index_name, bands in indices.get("ND", {}).items():
        col_a, col_b = bands
        df[index_name] = (df[col_a] - df[col_b]) / (df[col_a] + df[col_b])

    for index_name, spec in indices.get("FORMULA", {}).items():
        expression = spec["expression"]
        namespace = {}

        for symbol, band_col in spec.get("bands", {}).items():
            namespace[symbol] = df[band_col]

        for const_name, const_value in spec.get("constants", {}).items():
            namespace[const_name] = const_value

         # this will give us namespace as:
         #  { "BLUE": df["B2"],
        #  "L": 0.5, etc }  

        df[index_name] = pd.eval(expression, {"__builtins__": {}}, namespace)

    return df

def correlation_matrix(df: pd.DataFrame,
                       font_size: int = 20,
                       columns: list[str]| None = None) -> pd.DataFrame:
    if columns:
        df = df[columns]


    corr_matrix = df.corr(numeric_only=True)

    
    fig = plt.figure(figsize=(12,10))
    hm = sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'Pearsons R-value',})
    cbar = hm.collections[0].colorbar
    cbar.set_label('Pearsons R-value',labelpad = 15, fontsize=font_size -2)

    plt.tick_params(axis='x', labelsize = font_size -8)
    plt.tick_params(axis='y',labelsize = font_size -8)
    plt.title('Correlation matrix of Sentinel-2 MSI bands and derived indices', fontweight = 'bold', fontsize = 20, y = 1.02, x= 0.55)
    plt.show()

    # could add code to save this at same time

    save_path = Path(__file__).resolve().parent.parent / "outputs" / "corr_matrix.png"

    fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return corr_matrix


def create_image_collection(sites_file : str | Path,
                            points_file: str | Path,
                            location: str,
                            ):
    
    """
    Creates an image collection for a study site and creates a collection of sentinel-2 images from the study dates

    Arguments: sites_file is a json with the a point geometry and bbox for each site location. Points_file is an\
     .shp with labelled point geometries and the acquisition date (in yyyy-MM-dd) for each label.\
     Location is thename of the site, corresponding to a key in sites_file

     Returns: an image collection of images for a single location for every date that coresponds to labelled points.
    """

    if not isinstance(location, str):
        raise TypeError(f"location must be a string, got {type(location).__name__}")
    location = location.lower()

    with Path(sites_file).open("r", encoding = "utf-8") as s:
        sites = json.load(s)
    
    # Create gdf from labelled points
    points_file = Path(points_file)
    points_gdf = gpd.read_file(points_file)
    
    # Get coordinates of current location 
    current_site_point = sites.get("sites", {}).get(location,{}).get("geometry", {}).get("coordinates", [])
    # convert to a ee point. 
    point_geom = ee.Geometry.Point(current_site_point)

    # Get bbox to clip
    current_site_bbox = sites.get("sites", {}).get(location,{}).get("bbox", {}).get("coordinates", [])
    bbox_geom = ee.Geometry.Polygon(current_site_bbox)

    # Get date list from labelled points:
    if not "obs_date" in points_gdf.columns:
        raise ValueError(f"The points file for {location} has no obs_date column")
    else: 
        date_list = points_gdf["obs_date"].astype(str).unique().tolist()
    
    # Convert list of dates to an EE list
    date_list = ee.List(date_list)

    # ------------------------------
    # Define a function to create an image collection from the date list
    # This could be defined separately above, but this is the only place it will be used
    def create_image_for_date(date, 
                           point_geom=point_geom,
                           bbox_geom = bbox_geom,
                           ):
        """
        Create an ee.Image for a specified location on a specified date and clipped to a specified bbox

        Returns: an ee image
        """

        ee_date = ee.Date.parse('yyyy-MM-dd', date)

        def mask_clouds_SCL(image):
            scl = image.select('SCL')
            mask = scl.neq(3).And(scl.neq(9)).And(scl.neq(8))                           
            return image.updateMask(mask)

        image = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point_geom)
            .filterDate(ee_date, ee_date.advance(1,"day"))
            .map(mask_clouds_SCL)
            .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
            .first()
            .clip(bbox_geom)
           )

        return image
    # ---------------------------------------------
    # create an image for each date in the list

    image_list = date_list.map(create_image_for_date)
    image_collection = ee.ImageCollection(image_list)

    print(f"Created image collection for: {location}")
    return image_collection
           


def create_images_for_all_locations(sites_file: str | Path, project_root: str | Path):

    """
    Accesses all the site locations in sites_file and calls create_image_collection() for each site. 
    project_root is used to access the files for lebelled points. They must have a path /configs/points_file/location_point.shp
    Returns: a list of ee ImageCollections, one for each site
    """
    project_root = Path(project_root)
    with Path(sites_file).open("r", encoding="utf-8") as f:
        sites = json.load(f)

    locations_list = list(sites.get("sites", {}).keys())

    # then need to load the points files to pass to create_image_collection()
    

    ic_list = []
    for location in locations_list:
            points_file = project_root / "configs" / "points_files" / f"{location}_points.shp"

            if not points_file.exists():
                raise FileNotFoundError(f"Points file not found for {location}: {points_file}")
    
            ic = create_image_collection(sites_file = sites_file, points_file = points_file, location = location)
            ic_list.append(ic)

    return ic_list


#  date = ee.Date.parse('yyyy-MM-dd', feature.get('obs_date'))


#     points_feature = geemap.gdf_to_ee(points)

# def export_image_to_drive(date,
#                           location,


   
#     tasks = []

#     for i in range(image_list.size().getInfo()):
#         img = ee.Image(image_list.get(i))
#         year = img.get("year").getInfo()
#         nodata_val = -9999
        
#         unmasked_image = img.unmask(value=nodata_val, sameFootprint=False)
#         # Use the "noData" key in the "formatOptions" parameter to set the nodata value
#         # (GeoTIFF format only).

#         task = ee.batch.Export.image.toDrive(
#             image=unmasked_image,
#             description=f'gebe_composite_{year}',
#             folder='EORSS/CW',
#             region=gebe,  # full image bounds
#             scale=10,  # large scale for minimal demo
#             crs=crs,
#             fileFormat='GeoTIFF',
#             formatOptions={
#                 'noData': nodata_val
#             }
#         )
#         task.start()
#         tasks.append(task)
        