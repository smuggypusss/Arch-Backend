import math
from typing import List
from ..models.project import Region, RegionPoint


def polygon_area(points: List[RegionPoint]) -> float:
    """Calculate area of a polygon using the shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y
    return abs(area) / 2.0


def pixels_to_sqft(pixel_area: float, reference_height_ft: float, reference_pixels: float) -> float:
    scale = reference_height_ft / reference_pixels
    return pixel_area * scale * scale


def estimate_region_area(
    region: Region,
    reference_height_ft: float = 7.0,
    reference_pixels: float = 100.0,
) -> float:
    pixel_area = polygon_area(region.polygon)
    area_sqft = pixels_to_sqft(pixel_area, reference_height_ft, reference_pixels)

    if region.type == "window":
        area_sqft *= 0.85
    elif region.type == "railing":
        area_sqft *= 0.4
    elif region.type == "pillar":
        area_sqft *= 1.3

    return round(area_sqft, 2)


def estimate_all_regions(
    regions: List[Region],
    reference_height_ft: float = 7.0,
    reference_pixels: float = 100.0,
) -> dict:
    results = {}
    for region in regions:
        area = estimate_region_area(region, reference_height_ft, reference_pixels)
        results[region.id] = {
            "area_sqft": area,
            "region_type": region.type,
        }
    return results