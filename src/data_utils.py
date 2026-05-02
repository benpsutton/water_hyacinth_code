import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import geemap
import ee

INDICES_PATH = Path(__file__).resolve().parent.parent / "configs" / "indices.json"


def create_bounding_box(location: str,
                        project_root: str | Path):
    location_fp = project_root / "configs"/ "point_files" / f"{location}_points.shp"
    
    location_gdf = gpd.read_file(location_fp)
    location_gdf = location_gdf.to_crs("EPSG:4326")

    min_X, min_Y, max_X, max_Y = location_gdf.total_bounds

    return min_X, min_Y, max_X, max_Y


def make_padded_bbox_all_location(sites_file: str | Path,
                                  project_root: str | Path):
    
    sites_fp = Path(sites_file)
    project_root = Path(project_root)

    with open(sites_fp) as f:
        sites = json.load(f)

    location_list = list(sites.get("sites", {}).keys())

    for location in location_list:

        min_X, min_Y, max_X, max_Y = create_bounding_box(location = location, project_root= project_root)

        pad = 0.015

        min_X -= pad
        max_X += pad
        min_Y -= pad
        max_Y += pad

        data = {
                "type": "Polygon",
                "coordinates": [[
                    [min_X, min_Y],
                    [min_X, max_Y],
                    [max_X, max_Y],
                    [max_X, min_Y],
                    [min_X, min_Y]
                ]]
                }
            
        sites["sites"][location]["bbox"] = data
        
    with open(sites_fp, "w") as f:
        json.dump(sites, f, indent=4)

    location_string = ", ".join(location_list)

    return print(f"Created padded bbox from the points files for {location_string}")





     
    

