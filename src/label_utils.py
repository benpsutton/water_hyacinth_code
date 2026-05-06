###### Helpers for creating ee Image ids for images to label. 
## Will be used to populate configs/images.json 

import ee
import json
from pathlib import Path

# step one:
# need to generate feature collection with the s2 image dates that have less then x% cloud for a goven location

def create_s2_list_for_location(location: str,
                                site_fp: str | Path,
                                cloud_perc: int,
                                ):
    
    # we dont need to filter by date but i will restrict it to the years of interest
    

    with open(Path(site_fp)) as f:
        sites = json.load(f)


    # Get coordinates of current location 
    current_site_point = sites.get("sites", {}).get(location,{}).get("geometry", {}).get("coordinates", [])
    # convert to a ee point. 
    point_geom = ee.Geometry.Point(current_site_point)

    image_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterDate(2019,2026)
                        .filterBounds(point_geom)
                        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_perc))
    )

    s2_list = image_collection.map(lambda img: img.set('date', img.date().format('YYYY-MM-dd'))).aggregate_array('date')

    return s2_list

def create_s1_list_for_location(location: str,
                                site_fp: str | Path,
                                ):
    
    with open(Path(site_fp)) as f:
        sites = json.load(f)

    # Get coordinates of current location 
    current_site_point = sites.get("sites", {}).get(location,{}).get("geometry", {}).get("coordinates", [])
    # convert to a ee point. 

    point_geom = ee.Geometry.Point(current_site_point)

    image_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterDate(2019,2026)
                        .filterBounds(point_geom)
    )

    s1_list = image_collection.map(lambda img: img.set('date', img.date().format('YYYY-MM-dd'))).aggregate_array('date')

    return s1_list

def get_s2_s1_matching_dates(location: str,
                             site_fp: str | Path,
                             cloud_perc: Int = 10):
    
    s1_list = create_s1_list_for_location(location = location, site_fp= site_fp)

    s2_list = create_s2_list_for_location(location= location, site_fp=site_fp, cloud_perc= cloud_perc)

    matching_dates = s1_list.filter(ee.Filter.inList("item", s2_list))
                                    
    return matching_dates

