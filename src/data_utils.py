import json
from pathlib import Path

import geopandas as gpd
######---- I dont think i want to do this approach now i have labels in a single file
# def create_shp_for_each_location(label_fp: str | Path, sites_fp: str | Path, project_root: str | Path):
#     sites_fp = Path(sites_fp)
#     project_root = Path(project_root)

#     with sites_fp.open(encoding="utf-8") as f:
#         sites = json.load(f)

#     label_gdf = gpd.read_file(label_fp)

#     if 'location' not in label_gdf.columns:
#         raise ValueError("Missing 'location' column")

#     location_list = list(sites.get("sites", {}).keys())

#     for location in location_list:
#         if location not in set(label_gdf['location']):
#             raise ValueError(f"There are no labels for {location}")
        
#         location_fp = project_root / "configs" / "point_files" / f"{location}_points.shp"

#         location_gdf = label_gdf.loc[label_gdf["location"] == location]
#         location_gdf.to_file(location_fp)

#         print(f"Created {location}_points.shp")

def validate_label_file_columns(label_gdf: gpd.GeoDataFrame, label_fp: str | Path):

    required_cols = ["location", "obs_date", "class_label", "label_id"]

    missing_cols = required_cols - set(label_gdf.columns)

    if missing_cols:
        missing_string = ",".join(missing_cols)
        raise ValueError(f"{label_fp} is missing columns: {missing_string}")
    
    if label_gdf.geometry is None:
        raise ValueError(f"{label_fp} has no geometry")

    if label_gdf.empty:
        raise ValueError(f"{label_fp} is has no points")

    # Drop old target binary column as had mistakes, will replace it
    if "target_binary" in label_gdf.columns:
        label_gdf = label_gdf.drop("target_binary", axis = 1)


    cleaned_gdf = label_gdf.copy()

    try:
        cleaned_gdf["obs_date"] = pd.to_datetime(
            cleaned_gdf["obs_date"],
            format="%Y-%m-%d",
            errors="raise",
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

def convert_class_labels(label_gdf: gdp.GeoDataFrame, label_fp: str | path):

    """ Create two new columns with class_label as a number and as binary. Other functions expect the 
    points to have a class column 'lc', therefore the binary column will be called this"""

    label_gdf= label_gdf.copy()

    if "class_label" not in label_gdf.columns:
        raise ValueError(f"{label_fp} is mssing a 'class_label' column")
    
    class_map = {
        "LEV": 1,
        "open_water": 2,
        "floating_plants" : 3,
        "surface_algae": 4
    }

    label_gdf["class_int"] = label_gdf["class_label"].map(class_map)
    
    if label_gdf["class_int"].isna().any():
        raise ValueError("Unmapped class labels found")

    binary_map = {
        "LEV": 0,
        "open_water": 0,
        "floating_plants": 1,
        "surface_algae": 0
    }

    label_gdf["lc"] = label_gdf["class_label"].map(binary_map)

    if label_gdf["lc"].isna().any():
        raise ValueError("Unmapped class labels found")
    
    return label_gdf


def clean_labelled_data(label_fp: str | Path, cleaned_label_fp: str | Path):
    label_fp = Path(label_fp)
    cleaned_label_fp = Path(cleaned_label_fp)

    label_gdf = gpd.read_file(label_fp)

    cleaned_label_gdf = validate_label_file_columns(label_gdf= label_gdf, label_fp= label_fp)
    cleaned_label_gdf = convert_class_labels(label_gdf= cleaned_label_gdf)
    
    cleaned_label_gdf.to_file(cleaned_label_fp)

    print(f"Saved a cleaned version of {label_fp} as {cleaned_label_fp}")


def create_bounding_box(location: str, project_root: str | Path):
    """ Creates a bounding box around the labelled points in labels.gpkg for a given location"""

    project_root = Path(project_root)

    label_fp = project_root / "configs" / "labels.gpkg"

    label_gdf = gpd.read_file(label_fp)
    location_gdf = label_gdf.loc[label_gdf["location"] == location]

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

        pad = 0.02

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
