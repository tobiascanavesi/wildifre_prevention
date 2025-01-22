# src/create_grid.py

import geopandas as gpd
import shapely.geometry as geom
import numpy as np
import os

def create_fishnet(bounds, cell_size_km=10.0, crs_original="EPSG:4326", crs_projected="EPSG:3310"):
    """
    Create a regular grid (fishnet) of squares covering the provided bounding box.

    Parameters:
        bounds (tuple): (minx, miny, maxx, maxy) in degrees (EPSG:4326).
        cell_size_km (float): Desired cell size in kilometers.
        crs_original (str): Original CRS of the bounds.
        crs_projected (str): Projected CRS for accurate distance calculations.

    Returns:
        GeoDataFrame: GeoDataFrame containing the grid cells with 'cell_id' and 'geometry'.
    """
    minx, miny, maxx, maxy = bounds

    # Create a GeoDataFrame with the bounding box
    bbox = geom.box(minx, miny, maxx, maxy)
    grid_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs=crs_original)

    # Reproject to projected CRS
    grid_projected = grid_gdf.to_crs(crs_projected)

    # Calculate the number of cells in x and y directions
    width = grid_projected.total_bounds[2] - grid_projected.total_bounds[0]
    height = grid_projected.total_bounds[3] - grid_projected.total_bounds[1]

    num_cols = int(np.ceil(width / (cell_size_km * 1000)))  # meters
    num_rows = int(np.ceil(height / (cell_size_km * 1000)))  # meters

    # Generate grid cells
    grids = []
    cell_id = 0
    for i in range(num_cols):
        for j in range(num_rows):
            x_min = grid_projected.bounds.iloc[0].minx + i * cell_size_km * 1000
            y_min = grid_projected.bounds.iloc[0].miny + j * cell_size_km * 1000
            x_max = x_min + cell_size_km * 1000
            y_max = y_min + cell_size_km * 1000
            poly = geom.box(x_min, y_min, x_max, y_max)
            grids.append((cell_id, poly))
            cell_id += 1

    # Create GeoDataFrame with 'cell_id' as a column
    grid_cells = gpd.GeoDataFrame(grids, columns=["cell_id", "geometry"], crs=crs_projected)

    # Clip grid to original bounding box using gpd.clip to preserve 'cell_id'
    grid_clipped = gpd.clip(grid_cells, grid_projected)

    # Remove empty geometries
    grid_clipped = grid_clipped[~grid_clipped.is_empty]

    return grid_clipped

def extract_centroids(shapefile_path, output_csv_path, crs_original="EPSG:4326", crs_projected="EPSG:3310"):
    """
    Extract centroid coordinates from the shapefile and save to CSV.

    Parameters:
        shapefile_path (str): Path to the shapefile.
        output_csv_path (str): Path where the centroid CSV will be saved.
        crs_original (str): Original CRS of the shapefile.
        crs_projected (str): Projected CRS used for accurate centroid calculation.
    """
    try:
        gdf = gpd.read_file(shapefile_path)
        print("Shapefile columns:", gdf.columns.tolist())  # Debugging line

        # Reproject to projected CRS for accurate centroids
        gdf_projected = gdf.to_crs(crs_projected)

        # Calculate centroids in projected CRS
        gdf_projected['centroid'] = gdf_projected['geometry'].centroid

        # Reproject centroids back to original CRS (EPSG:4326) to get lat/lon
        gdf_projected['centroid'] = gdf_projected['centroid'].to_crs(crs_original)

        # Extract latitude and longitude
        gdf_projected['centroid_lat'] = gdf_projected['centroid'].y
        gdf_projected['centroid_lon'] = gdf_projected['centroid'].x

        # Check if 'cell_id' is present
        if 'cell_id' not in gdf_projected.columns:
            print("Error: 'cell_id' column not found in shapefile.")
            return

        # Select required columns
        centroids_df = gdf_projected[['cell_id', 'centroid_lat', 'centroid_lon']]

        # Save to CSV
        centroids_df.to_csv(output_csv_path, index=False)
        print(f"Saved cell centroids to {output_csv_path}")
    except Exception as e:
        print(f"Error extracting centroids: {e}")

def main():
    # 1. Define the bounding box for California
    bounds_ca = (-124.48, 32.53, -114.13, 42.01)  # (minx, miny, maxx, maxy)

    # 2. Create the grid
    grid_gdf = create_fishnet(bounds_ca, cell_size_km=10.0)
    print(f"Created grid with {len(grid_gdf)} cells.")

    # 3. Define output directory and paths
    out_dir = "data/raw/"
    shapefile_path = os.path.join(out_dir, "ca_grid_10km.shp")
    centroids_csv_path = os.path.join(out_dir, "cell_centroids.csv")

    # 4. Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # 5. Save the grid to a shapefile
    grid_gdf.to_file(shapefile_path)
    print(f"Saved 10 km grid shapefile to {shapefile_path}")

    # 6. Extract and save centroids
    extract_centroids(shapefile_path, centroids_csv_path)

if __name__ == "__main__":
    main()
