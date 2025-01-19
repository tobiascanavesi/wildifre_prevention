import geopandas as gpd
import shapely.geometry as geom
from shapely.ops import unary_union
import numpy as np
import os

def create_fishnet(bounds, cell_size_km=10.0):
    """
    Create a regular grid (fishnet) of squares (cell_size_km) 
    covering the provided bounding box (minx, miny, maxx, maxy), 
    in degrees if using EPSG:4326 (lat/lon).
    
    For better accuracy, you might want to project to a meter-based
    projection (e.g., EPSG:3310 for California), then convert 10km 
    to 10,000 meters. This example uses lat/lon for simplicity.
    """
    minx, miny, maxx, maxy = bounds
    # cell_size in degrees: ~0.1 deg ~ 11 km at mid-latitude. 
    # This is approximate. For truly 10 km cells, 
    # reproject to a meter-based coordinate system first.
    cell_size_deg = 0.1  # approximate for 10km

    # Ranges
    x_coords = np.arange(minx, maxx, cell_size_deg)
    y_coords = np.arange(miny, maxy, cell_size_deg)

    polygons = []
    cell_id = 0
    for x in x_coords:
        for y in y_coords:
            # bottom-left corner (x, y)
            # top-right corner (x+cell_size_deg, y+cell_size_deg)
            poly = geom.Polygon([
                (x, y),
                (x + cell_size_deg, y),
                (x + cell_size_deg, y + cell_size_deg),
                (x, y + cell_size_deg)
            ])
            polygons.append((cell_id, poly))
            cell_id += 1

    gdf = gpd.GeoDataFrame(polygons, columns=["cell_id", "geometry"], crs="EPSG:4326")
    return gdf

def main():
    # 1. Create the grid
    bounds_ca = (-124.48, 32.53, -114.13, 42.01)
    grid_gdf = create_fishnet(bounds_ca, cell_size_km=10.0)
    print(f"Created grid with {len(grid_gdf)} cells.")

    # 2. Ensure the output directory exists
    out_dir = "../data/raw/"
    os.makedirs(out_dir, exist_ok=True)

    # 3. Save to shapefile (or any other supported format)
    out_path = os.path.join(out_dir, "ca_grid_10km.shp")
    grid_gdf.to_file(out_path)
    print(f"Saved 10 km grid shapefile to {out_path}")

if __name__ == "__main__":
    main()