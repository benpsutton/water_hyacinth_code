import json
from pathlib import Path

import geopandas as gpd


def create_bounding_box(location: str, project_root: str | Path):
    project_root = Path(project_root)
    location_fp = project_root / "configs" / "point_files" / f"{location}_points.shp"

    location_gdf = gpd.read_file(location_fp)
    location_gdf = location_gdf.to_crs("EPSG:4326")

    min_x, min_y, max_x, max_y = location_gdf.total_bounds

    return min_x, min_y, max_x, max_y


def make_padded_bbox_all_location(sites_file: str | Path, project_root: str | Path):
    sites_fp = Path(sites_file)
    project_root = Path(project_root)

    with sites_fp.open(encoding="utf-8") as f:
        sites = json.load(f)

    location_list = list(sites.get("sites", {}).keys())

    for location in location_list:
        min_x, min_y, max_x, max_y = create_bounding_box(
            location=location,
            project_root=project_root,
        )

        pad = 0.015

        min_x -= pad
        max_x += pad
        min_y -= pad
        max_y += pad

        data = {
            "type": "Polygon",
            "coordinates": [[
                [min_x, min_y],
                [min_x, max_y],
                [max_x, max_y],
                [max_x, min_y],
                [min_x, min_y],
            ]],
        }

        sites["sites"][location]["bbox"] = data

    with sites_fp.open("w", encoding="utf-8") as f:
        json.dump(sites, f, indent=4)

    location_string = ", ".join(location_list)

    print(f"Created padded bbox from the points files for {location_string}")
