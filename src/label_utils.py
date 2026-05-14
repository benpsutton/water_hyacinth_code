import json
from pathlib import Path
import plotly.express as px
import ee
import pandas as pd
import geemap


def prepare_matching_dates_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise matching-date data for plotting and comparison selection.

    The exported CSV can contain multiple rows for the same location/date when
    more than one Sentinel-2 tile intersects the study area. This helper keeps
    the original rows, but makes sure key columns are in the right dtype and
    derives the extra metadata used downstream.
    """
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if "S1_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["S1_time"]):
        df["S1_time"] = pd.to_datetime(df["S1_time"], unit="ms", utc=True)

    if "S2_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["S2_time"]):
        df["S2_time"] = pd.to_datetime(df["S2_time"], unit="ms", utc=True)

    if "time_diff" not in df.columns and {"S1_time", "S2_time"}.issubset(df.columns):
        # Both timestamps come from GEE system:time_start.
        df["time_diff"] = (df["S1_time"] - df["S2_time"]).abs()

    if "season" not in df.columns and "date" in df.columns:
        def get_season(dt):
            month = dt.month
            if month in (12, 1, 2):
                return "winter"
            if month in (3, 4, 5):
                return "spring"
            if month in (6, 7, 8):
                return "summer"
            return "autumn"

        df["season"] = df["date"].apply(get_season)

    return df


def _best_row_per_location_date(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate tiles for the same location/date to one preferred row."""
    sort_cols = []

    if "cloud_perc" in df.columns:
        sort_cols.append("cloud_perc")
    if "time_diff" in df.columns:
        sort_cols.append("time_diff")
    if "S2_time" in df.columns:
        sort_cols.append("S2_time")

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=True)

    deduped = df.drop_duplicates(subset=["location", "date"], keep="first")
    return deduped.sort_values(["location", "date"]).reset_index(drop=True)


def select_comparison_dates(
    matching_dates: pd.DataFrame | str | Path,
    months_back: int = 3,
    n_comparisons: int = 2,
    max_cloud_perc: float | None = None,
    min_gap_days: int = 14,
) -> pd.DataFrame:
    """Pick earlier comparison images automatically for each target date.

    Strategy:
    1. Standardise timestamps and derive helper columns.
    2. Reduce duplicate location/date rows to the best available scene.
    3. For each target date, search the preceding `months_back` months.
    4. Prefer one recent and one older comparison date when possible, then
       fill any remaining slots greedily with low-cloud dates that are not too
       close to already selected comparisons.

    Returns one row per target image date with flattened comparison columns.
    """
    if isinstance(matching_dates, (str, Path)):
        df = pd.read_csv(matching_dates)
    else:
        df = matching_dates.copy()

    df = prepare_matching_dates_df(df)
    base_df = _best_row_per_location_date(df)

    results: list[dict] = []

    for _, target in base_df.iterrows():
        location = target["location"]
        target_date = target["date"]
        window_start = target_date - pd.DateOffset(months=months_back)

        candidates = base_df[
            (base_df["location"] == location)
            & (base_df["date"] < target_date)
            & (base_df["date"] >= window_start)
        ].copy()

        if max_cloud_perc is not None and "cloud_perc" in candidates.columns:
            candidates = candidates[candidates["cloud_perc"] <= max_cloud_perc]

        if candidates.empty:
            row = target.to_dict()
            row["comparison_count"] = 0
            results.append(row)
            continue

        candidates["age_days"] = (target_date - candidates["date"]).dt.days
        candidates["window_bucket"] = pd.cut(
            candidates["age_days"],
            bins=[0, 45, 10_000],
            labels=["recent", "older"],
            include_lowest=True,
        )

        ranking_cols = []
        if "cloud_perc" in candidates.columns:
            ranking_cols.append("cloud_perc")
        if "time_diff" in candidates.columns:
            ranking_cols.append("time_diff")
        ranking_cols.append("age_days")

        candidates = candidates.sort_values(ranking_cols, ascending=True)

        selected_rows = []
        selected_dates = []

        for bucket in ["recent", "older"]:
            bucket_rows = candidates[candidates["window_bucket"] == bucket]
            for _, candidate in bucket_rows.iterrows():
                if all(abs((candidate["date"] - d).days) >= min_gap_days for d in selected_dates):
                    selected_rows.append(candidate)
                    selected_dates.append(candidate["date"])
                    break
            if len(selected_rows) >= n_comparisons:
                break

        if len(selected_rows) < n_comparisons:
            for _, candidate in candidates.iterrows():
                if any(candidate["date"] == existing["date"] for existing in selected_rows):
                    continue
                if all(abs((candidate["date"] - d).days) >= min_gap_days for d in selected_dates):
                    selected_rows.append(candidate)
                    selected_dates.append(candidate["date"])
                if len(selected_rows) >= n_comparisons:
                    break

        row = target.to_dict()
        row["comparison_count"] = len(selected_rows)

        for idx, candidate in enumerate(selected_rows, start=1):
            row[f"comparison_{idx}_date"] = candidate["date"]
            row[f"comparison_{idx}_s2_img_id"] = candidate.get("S2_img_id")
            row[f"comparison_{idx}_s1_img_id"] = candidate.get("S1_img_id")
            row[f"comparison_{idx}_cloud_perc"] = candidate.get("cloud_perc")
            row[f"comparison_{idx}_age_days"] = int(candidate["age_days"])

        results.append(row)

    result_df = pd.DataFrame(results).sort_values(["location", "date"]).reset_index(drop=True)

    for idx in range(1, n_comparisons + 1):
        date_col = f"comparison_{idx}_date"
        if date_col in result_df.columns:
            result_df[date_col] = pd.to_datetime(result_df[date_col])

    return result_df

def create_s2_list_for_location(location: str, site_fp: str | Path, cloud_perc: int):
    with Path(site_fp).open(encoding="utf-8") as f:
        sites = json.load(f)

    current_site_point = sites.get("sites", {}).get(location, {}).get("geometry", {}).get("coordinates", [])
    point_geom = ee.Geometry.Point(current_site_point)

    image_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate("2019", "2026")
        .filterBounds(point_geom)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_perc))
    )

    s2_fc = ee.FeatureCollection(image_collection.map(
        lambda img: ee.Feature(None, {
                                "location": location,
                               "date": img.date().format("yyyy-MM-dd"),
                               "S2_img_id": img.get("system:id"),
                                "S2_time": img.get("system:time_start"),
                                "cloud_perc": img.get("CLOUDY_PIXEL_PERCENTAGE")
        }
        )
    ))
    

    return s2_fc


def create_s1_list_for_location(location: str, site_fp: str | Path):
    with Path(site_fp).open(encoding="utf-8") as f:
        sites = json.load(f)

    current_site_point = sites.get("sites", {}).get(location, {}).get("geometry", {}).get("coordinates", [])
    point_geom = ee.Geometry.Point(current_site_point)

    image_collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate("2019", "2026")
        .filterBounds(point_geom)
    )

    s1_fc = ee.FeatureCollection(image_collection.map(
        lambda img: ee.Feature(None, {
            "date": img.date().format("yyyy-MM-dd"),
            "S1_img_id": img.get('system:id'),
            "S1_time": img.get('system:time_start')
            }))
    )

    return s1_fc


def get_s2_s1_matching_dates(location: str, site_fp: str | Path, project_root: str |Path, cloud_perc: int = 10):

    s1_fc = create_s1_list_for_location(location=location, site_fp=site_fp)
    s2_fc = create_s2_list_for_location(
        location=location,
        site_fp=site_fp,
        cloud_perc=cloud_perc,
    )

    filter = ee.Filter.equals(
        leftField= 'date',
        rightField= 'date'
    )

    matching_dates_fc = ee.Join.saveFirst(matchKey = 's1_images').apply(
        primary = s2_fc,
        secondary = s1_fc,
        condition = filter
    )



    matching_dates_fc = ee.FeatureCollection(matching_dates_fc).map(
        lambda feature: ee.Feature(
            None,
            ee.Dictionary({
                'location': ee.Feature(feature).get('location'),
                'date': ee.Feature(feature).get('date'),
                'S2_img_id': ee.Feature(feature).get('S2_img_id'),
                'S2_time': ee.Feature(feature).get('S2_time'),
                'cloud_perc': ee.Feature(feature).get('cloud_perc'),
                'S1_img_id': ee.Feature(feature.get('s1_images')).get('S1_img_id'),
                'S1_time': ee.Feature(feature.get('s1_images')).get('S1_time'),
            }),
            ),
        )
    
    matching_dates_df = geemap.ee_to_df(matching_dates_fc)

    PROJECT_ROOT = Path(project_root)

    dates_output_fp = PROJECT_ROOT / "outputs" / f"{location}_matching_dates.csv"

    matching_dates_df.to_csv(dates_output_fp)

    return matching_dates_df

# Plotting the matching dates along with cloud percentage and season

def plot_dates(location: str,
               project_root:str | Path,
               start_year = 2019,
               end_year = 2025):
    
    PROJECT_ROOT = Path(project_root)

    dates_output_fp = PROJECT_ROOT / "outputs" / f"{location}_matching_dates.csv"


    df = pd.read_csv(dates_output_fp)
    df = prepare_matching_dates_df(df)
    df = df.set_index('date', drop = False)

    # make certain date is a datetime column

    df["date"] = pd.to_datetime(df["date"])

    location = df["location"].iloc[0]

    # Specify which years to plot
    start_year = str(start_year)
    end_year = str(end_year)

    df = df.loc[start_year:end_year]

    # using plotly so i can hover over points to see the date attributes
    fig = px.scatter(
        df,
        x="date",
        y="cloud_perc",
        color="season",
        title=f"Dates with matching S1 and S2 images for {location}",
        hover_data={
            "date": "|%Y-%m-%d",
            "cloud_perc": True,
            "season": True
        }
    )

    fig.update_layout(
        width=900,
        height=560,
        xaxis_title="Date",
        yaxis_title="Cloud percentage",
        template = "plotly"
    )

    return fig

## Function for writing image and comparison images detials to json for the labelling app. 
   
def image_details_to_json(comparison_df: pd.DataFrame,
                           images_fp: str | Path):
    
    images_fp = Path(images_fp)

    with open(images_fp, "r") as f:
        Image_dict = json.load(f)

    # Each comparison_df will be for a single location
    location = comparison_df["location"].iloc[0]

    # Each location key has a list of images as its value
    image_list_for_location = []

    for _, row in comparison_df.iterrows():

        location = row["location"]
        alt_date = row['date'].strftime(format = "%Y%m%d")
        entry_id = location + "_" + alt_date
        obs_date = row['date'].strftime(format = "%Y-%m-%d")
        display_label = location + " - " + row["date"].strftime(format = "%d %B %Y")
        s2_target_image_id = row["S2_img_id"]
        s1_target_image_id = row["S1_img_id"]

        compar_list = []
        count = int(row["comparison_count"])

        for i in range(1, count+1):
            img_col = f"comparison_{i}_s2_img_id"
            date_col = f"comparison_{i}_date"

            if img_col in comparison_df.columns and date_col in comparison_df.columns:
                img = row[img_col]
                date = row[date_col].strftime(format = "%Y-%m-%d")

                if pd.notna(img) and pd.notna(date):
                    comp_dict = {
                        "date" : date, "image_id": img
                        }
                    
                    compar_list.append(comp_dict)

        Dict = {"entry_id": entry_id,
                "location": location,
                "obs_date": obs_date,
                "display_label": display_label,
                "s2_target_image_id": s2_target_image_id,
                "s1_image_id": s1_target_image_id,
                "s2_comparison_images": compar_list  
                }
        
        image_list_for_location.append(Dict)
    
    # Assign the list of image-dates to the location key in Image_dict

    Image_dict[location]= image_list_for_location

    with open(images_fp, "w") as f:
        json.dump(Image_dict, f, indent = 4)

    return Image_dict