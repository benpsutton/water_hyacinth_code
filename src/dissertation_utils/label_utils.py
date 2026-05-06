import json
from pathlib import Path

import ee


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


def get_s2_s1_matching_dates(location: str, site_fp: str | Path, cloud_perc: int = 10):
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
    


    return matching_dates_fc
