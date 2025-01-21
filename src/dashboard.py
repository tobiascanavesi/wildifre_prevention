# src/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from shapely.geometry import Point, Polygon

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Set the page configuration
st.set_page_config(
    page_title="Wildfire Prevention Dashboard for California",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the Dashboard
st.title("🌲 Wildfire Prevention Dashboard")

# Sidebar for Filters
st.sidebar.header("Filters")

# Function to load data
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    return df

# Function to load grid shapefile
@st.cache_data
def load_grid(shapefile_path):
    grid_gdf = gpd.read_file(shapefile_path)
    return grid_gdf

# Function to load grid centroids
@st.cache_data
def load_grid_centroids(_grid_gdf):
    """
    Compute centroids in projected CRS and reproject them to geographic CRS.

    Parameters:
        _grid_gdf (GeoDataFrame): Projected GeoDataFrame of grid cells.

    Returns:
        GeoDataFrame: DataFrame with 'cell_id', 'centroid_lon', 'centroid_lat'.
    """
    # Compute centroids in projected CRS (EPSG:3310)
    centroids_projected = _grid_gdf.geometry.centroid

    # Create a GeoDataFrame with centroids
    gdf_centroids = gpd.GeoDataFrame(
        _grid_gdf[['cell_id']].copy(),
        geometry=centroids_projected,
        crs=_grid_gdf.crs
    )

    # Reproject centroids to geographic CRS (EPSG:4326)
    gdf_centroids = gdf_centroids.to_crs(epsg=4326)

    # Extract longitude and latitude
    gdf_centroids['centroid_lon'] = gdf_centroids.geometry.x
    gdf_centroids['centroid_lat'] = gdf_centroids.geometry.y

    # Return only necessary columns
    return gdf_centroids[['cell_id', 'centroid_lon', 'centroid_lat']]

# Load the merged dataset
DATA_CSV_PATH = "/Users/tobiascanavesi/Documents/wildifre_prevention/data/processed/merged_weather_fire_ndvi.csv"
df = load_data(DATA_CSV_PATH)

# Load grid shapefile
GRID_SHP_PATH = "/Users/tobiascanavesi/Documents/wildifre_prevention/data/raw/ca_grid_10km.shp"
grid_gdf = load_grid(GRID_SHP_PATH)

# Load centroids
grid_centroids = load_grid_centroids(grid_gdf)

# Display dataset information
st.markdown("### 📊 Dataset Overview")
st.write(f"**Total Records:** {df.shape[0]}")
st.write(f"**Features:** {', '.join(df.columns)}")
st.dataframe(df.head())

# Sidebar Filters

# Year Selection with "Select All" Option
select_all_years = st.sidebar.checkbox("Select All Years", value=True)

if select_all_years:
    selected_years = sorted(df['date'].dt.year.unique())
    st.sidebar.write(f"**All {len(selected_years)} Years Selected**")
else:
    selected_years = st.sidebar.multiselect(
        "Select Year(s)",
        options=sorted(df['date'].dt.year.unique()),
        default=[sorted(df['date'].dt.year.unique())[-1]]  # Default to the latest year
    )

# Month Selection with "Select All" Option
select_all_months = st.sidebar.checkbox("Select All Months", value=True)

months_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

months_available = df['date'].dt.month_name().unique()
months_sorted = [month for month in months_order if month in months_available]

if select_all_months:
    selected_months = months_sorted
    st.sidebar.write(f"**All {len(selected_months)} Months Selected**")
else:
    selected_months = st.sidebar.multiselect(
        "Select Month(s)",
        options=months_sorted,
        default=[months_sorted[0]]  # Default to the first month
    )

# Toggle for Grid Cell Selection Method
selection_method = st.sidebar.radio(
    "Select Grid Cells Method",
    ("Manual Selection", "Map Selection")
)

selected_cells = []

if selection_method == "Manual Selection":
    # "Select All" Checkbox for Grid Cells
    select_all_cells = st.sidebar.checkbox("Select All Grid Cells", value=True)

    if select_all_cells:
        # Automatically select all grid cell IDs
        selected_cells = df['cell_id'].unique().tolist()
        st.sidebar.write(f"**All {len(selected_cells)} Grid Cells Selected**")
    else:
        # Allow users to select specific grid cells
        selected_cells = st.sidebar.multiselect(
            "Select Grid Cell IDs",
            options=df['cell_id'].unique(),
            default=None
        )
elif selection_method == "Map Selection":
    # Map Selection
    st.sidebar.write("### 🗺️ Draw a Region to Select Grid Cells")

    # Initialize Folium map centered on California
    m = folium.Map(location=[36.7783, -119.4179], zoom_start=6)

    # Add Draw plugin to the map
    draw = Draw(
        draw_options={
            'polyline': False,
            'rectangle': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'polygon': {
                'allowIntersection': False,
                'showArea': True,
                'drawError': {
                    'color': '#e1e100',
                    'message': "<strong>Error:<strong> Shape cannot intersect!"
                },
                'shapeOptions': {
                    'color': '#97009c'
                }
            },
        },
        edit_options={
            'edit': False
        }
    )
    draw.add_to(m)

    # Display Folium map and capture drawn shapes
    output = st_folium(m, width=700, height=500)

    # Process drawn shapes
    if output and 'all_drawings' in output and output['all_drawings']:
        drawings = output['all_drawings']

        # Extract all polygons drawn
        polygons = []
        for drawing in drawings:
            if drawing['geometry']['type'] == 'Polygon':
                coords = drawing['geometry']['coordinates'][0]  # List of [lon, lat] pairs
                polygon = Polygon(coords)
                polygons.append(polygon)

        if polygons:
            # Create a GeoDataFrame from the polygons
            gdf_user_polygons = gpd.GeoDataFrame(
                geometry=polygons,
                crs="EPSG:4326"
            )

            # Reproject polygons to match grid CRS
            gdf_user_polygons_projected = gdf_user_polygons.to_crs(grid_gdf.crs)

            # Reproject grid to match CRS (if not already)
            grid_gdf_projected = grid_gdf.to_crs(grid_gdf.crs)

            # Spatial join to find grid cells within the polygons
            selected_cells_gdf = gpd.sjoin(grid_gdf_projected, gdf_user_polygons_projected, how='inner', predicate='within')

            # Extract selected cell_ids
            selected_cells = selected_cells_gdf['cell_id'].unique().tolist()

            st.sidebar.write(f"**{len(selected_cells)} Grid Cells Selected via Map**")
    else:
        if selection_method == "Map Selection":
            st.sidebar.write("**Selected Grid Cells via Map:** None")

# Filter Data Based on Selection
def filter_data(df, years, months, cell_ids):
    """
    Filter the DataFrame based on selected years, months, and grid cell IDs.

    Parameters:
        df (DataFrame): The original DataFrame.
        years (list): List of selected years.
        months (list): List of selected months.
        cell_ids (list): List of selected grid cell IDs.

    Returns:
        DataFrame: The filtered DataFrame.
    """
    df_filtered = df.copy()
    if years:
        df_filtered = df_filtered[df_filtered['date'].dt.year.isin(years)]
    if months:
        df_filtered = df_filtered[df_filtered['date'].dt.month_name().isin(months)]
    if cell_ids:
        df_filtered = df_filtered[df_filtered['cell_id'].isin(cell_ids)]
    return df_filtered

df_filtered = filter_data(df, selected_years, selected_months, selected_cells)

# Debugging Information
st.write("### 🔍 Debugging Information")
st.write(f"**Selected Years:** {selected_years}")
st.write(f"**Selected Months:** {selected_months}")
st.write(f"**Selected Grid Cells:** {selected_cells}")
st.write(f"**Number of Selected Grid Cells:** {len(selected_cells)}")

# Display Filtered Data Info
st.markdown("### 📈 Filtered Data Overview")
st.write(f"**Total Records After Filtering:** {df_filtered.shape[0]}")
st.write(f"**Selected Years:** {selected_years}")
st.write(f"**Selected Months:** {selected_months}")
if selection_method == "Map Selection":
    if selected_cells:
        st.write(f"**Selected Grid Cells via Map:** {selected_cells}")
    else:
        st.write("**Selected Grid Cells via Map:** None")
else:
    if selection_method == "Manual Selection":
        if select_all_cells:
            st.write("**Selected Grid Cells:** All")
        elif selected_cells:
            st.write(f"**Selected Grid Cells:** {selected_cells}")
        else:
            st.write("**Selected Grid Cells:** None")

# 🔥 Fire Occurrences Map

st.markdown("### 🔥 Fire Occurrences Map")

# Merge centroids with filtered data
df_plot = pd.merge(df_filtered, grid_centroids, on='cell_id', how='left')

# Debugging: Check if 'fire_occurred' exists and has any 1's
st.write("### Fire Occurrences in Filtered Data")
st.write(df_filtered[df_filtered['fire_occurred'] == 1].shape)

# Aggregate fires by date and location
df_fires = df_plot[df_plot['fire_occurred'] == 1]
df_fires_agg = df_fires.groupby(['date', 'centroid_lat', 'centroid_lon']).size().reset_index(name='fires')

# Check if df_fires_agg is not empty
if df_fires_agg.empty:
    st.warning("No fire occurrences found for the selected filters.")
else:
    # Extract month for animation
    df_fires_agg['month'] = df_fires_agg['date'].dt.to_period('M').dt.to_timestamp()

    # Create the interactive scatter map
    fig_map = px.scatter_mapbox(
        df_fires_agg,
        lat="centroid_lat",
        lon="centroid_lon",
        color="fires",
        size="fires",
        animation_frame='month',
        mapbox_style="carto-positron",
        title="Monthly Fire Occurrences Across Grid Cells",
        color_continuous_scale="YlOrRd",
        size_max=15,
        range_color=(0, df_fires_agg['fires'].max())
    )

    fig_map.update_layout(
        autosize=True,
        hovermode='closest',
        title_x=0.5,
        margin={"r":0,"t":30,"l":0,"b":0},
        mapbox=dict(
            center=dict(lat=df_fires_agg['centroid_lat'].mean(), lon=df_fires_agg['centroid_lon'].mean()),
            zoom=6  # Adjust zoom level as needed
        )
    )

    st.plotly_chart(fig_map, use_container_width=True)

# 🌿 NDVI Distribution by Month and Fire Occurrence

st.markdown("### 🌿 NDVI Distribution by Month and Fire Occurrence")

# Prepare data for boxplot
df_box = df_filtered.copy()

# Extract month from date
df_box['month'] = df_box['date'].dt.month_name()

# Convert 'month' to a categorical type with a specific order
df_box['month'] = pd.Categorical(df_box['month'], categories=months_order, ordered=True)

# Plot using Seaborn
plt.figure(figsize=(16, 8))
sns.boxplot(
    data=df_box, 
    x='month', 
    y='ndvi', 
    hue='fire_occurred', 
    palette={0: 'blue', 1: 'red'}
)
plt.title('NDVI Distribution by Month and Fire Occurrence')
plt.xlabel('Month')
plt.ylabel('NDVI')
plt.legend(title='Fire Occurred', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)

# 📈 Monthly Fire Counts

st.markdown("### 📈 Monthly Fire Counts")

# Aggregate fires per month
fires_monthly = df_fires_agg.groupby('month')['fires'].sum().reset_index()

# Plot using Plotly Express
fig_ts = px.line(
    fires_monthly, 
    x='month', 
    y='fires',
    markers=True,
    title='Total Fire Occurrences per Month',
    labels={'month': 'Month', 'fires': 'Number of Fires'}
)

fig_ts.update_layout(
    xaxis_title='Month',
    yaxis_title='Number of Fires',
    title_x=0.5,
    hovermode='x unified'
)

st.plotly_chart(fig_ts, use_container_width=True)

# 🧮 Cumulative Fire Counts Over Time

st.markdown("### 🧮 Cumulative Fire Counts Over Time")

# Calculate cumulative fires
fires_monthly_sorted = fires_monthly.sort_values('month')
fires_monthly_sorted['cumulative_fires'] = fires_monthly_sorted['fires'].cumsum()

# Plot using Plotly Express
fig_cumulative = px.line(
    fires_monthly_sorted, 
    x='month', 
    y='cumulative_fires',
    markers=True,
    title='Cumulative Fire Occurrences Over Time',
    labels={'month': 'Month', 'cumulative_fires': 'Cumulative Number of Fires'}
)

fig_cumulative.update_layout(
    xaxis_title='Month',
    yaxis_title='Cumulative Number of Fires',
    title_x=0.5,
    hovermode='x unified'
)

st.plotly_chart(fig_cumulative, use_container_width=True)

# 🔍 Interactive Filters

st.markdown("### 🔍 Interactive Filters")

# Temperature Range Slider
tmax_min, tmax_max = int(df['tmax_F'].min()), int(df['tmax_F'].max())
tmax_selected = st.slider("Select Maximum Temperature (°F)", tmax_min, tmax_max, (tmax_min, tmax_max))

# NDVI Range Slider
ndvi_min, ndvi_max = float(df['ndvi'].min()), float(df['ndvi'].max())
ndvi_selected = st.slider("Select NDVI Range", ndvi_min, ndvi_max, (ndvi_min, ndvi_max))

# Apply additional filters
def apply_additional_filters(df, tmax_range, ndvi_range):
    df_filtered = df.copy()
    df_filtered = df_filtered[(df_filtered['tmax_F'] >= tmax_range[0]) & (df_filtered['tmax_F'] <= tmax_range[1])]
    df_filtered = df_filtered[(df_filtered['ndvi'] >= ndvi_range[0]) & (df_filtered['ndvi'] <= ndvi_range[1])]
    return df_filtered

df_filtered = apply_additional_filters(df_filtered, tmax_selected, ndvi_selected)

# 📊 Key Statistics

st.markdown("### 📊 Key Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    total_fires = df_fires_agg['fires'].sum()
    st.metric("Total Fires", total_fires)

with col2:
    avg_ndvi = df_filtered['ndvi'].mean()
    st.metric("Average NDVI", f"{avg_ndvi:.4f}")

with col3:
    avg_precip = df_filtered['precip_in'].mean()
    st.metric("Average Precipitation (in)", f"{avg_precip:.2f}")

# 📥 Download Filtered Data

st.markdown("### 📥 Download Filtered Data")

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(df_filtered)

st.download_button(
    label="Download Data as CSV",
    data=csv_data,
    file_name=f"filtered_fire_data_{'_'.join(map(str, selected_years))}_{'_'.join(selected_months)}.csv",
    mime='text/csv',
)

# 🔗 Correlation Heatmap

st.markdown("### 🔗 Correlation Heatmap")

# Select relevant features
features = ['precip_in', 'tmax_F', 'tmin_F', 'ndvi', 'fire_occurred']

# Check if there are enough data points to compute correlation
if df_filtered[features].shape[0] > 1:
    corr = df_filtered[features].corr()
    
    # Plot heatmap
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    plt.title('Correlation Matrix of Features')
    st.pyplot(fig_corr)
else:
    st.write("Not enough data points to display correlation heatmap.")
