#!/usr/bin/env python3
"""
Create agricultural regions shapefile for SEM notebook
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


# Define region geometries (4x5 grid)
def create_region_polygon(row, col, size=0.1):
    """Create a square polygon for a region."""
    x0 = -71.8 + col * size
    y0 = 42.0 + row * size
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size), (x0, y0)])


# Create grid of regions
regions = []
region_id = 1
for row in range(4):
    for col in range(5):
        poly = create_region_polygon(row, col)
        regions.append({"region_id": region_id, "geometry": poly})
        region_id += 1

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(regions, crs="EPSG:4326")

# Save shapefile
output_path = Path(__file__).parent / "agricultural_regions.shp"
gdf.to_file(output_path)

print(f"Created shapefile: {output_path}")
print(f"Number of regions: {len(gdf)}")
