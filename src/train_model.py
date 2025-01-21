# src/model_training.py

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load data
def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    return df

def load_grid(shapefile_path):
    grid_gdf = gpd.read_file(shapefile_path)
    return grid_gdf

def load_grid_centroids(grid_gdf):
    centroids_projected = grid_gdf.geometry.centroid
    gdf_centroids = gpd.GeoDataFrame(
        grid_gdf[['cell_id']].copy(),
        geometry=centroids_projected,
        crs=grid_gdf.crs
    )
    gdf_centroids = gdf_centroids.to_crs(epsg=4326)
    gdf_centroids['centroid_lon'] = gdf_centroids.geometry.x
    gdf_centroids['centroid_lat'] = gdf_centroids.geometry.y
    return gdf_centroids[['cell_id', 'centroid_lon', 'centroid_lat']]

# Feature Engineering
def feature_engineering(df):
    # Time-based features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    
    def get_season(month):
        if month in ['December', 'January', 'February']:
            return 'Winter'
        elif month in ['March', 'April', 'May']:
            return 'Spring'
        elif month in ['June', 'July', 'August']:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    # Moving averages
    window_sizes = [7, 15, 30, 60, 90, 180, 360]
    
    for window in window_sizes:
        df[f'precip_ma_{window}d'] = df.groupby('cell_id')['precip_in'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'tmax_ma_{window}d'] = df.groupby('cell_id')['tmax_F'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'tmin_ma_{window}d'] = df.groupby('cell_id')['tmin_F'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'ndvi_ma_{window}d'] = df.groupby('cell_id')['ndvi'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
    # NDVI lag feature
    df['ndvi_lag_1d'] = df.groupby('cell_id')['ndvi'].shift(1)
    
    # Previous fires
    df['fires_last_7d'] = df.groupby('cell_id')['fire_occurred'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    
    return df

# Encoding and Scaling
def encode_and_scale(df):
    categorical_cols = ['month', 'day_of_week', 'season']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Scaling
    numerical_cols = [col for col in df_encoded.columns if df_encoded[col].dtype in ['int64', 'float64'] and col not in ['fire_occurred', 'cell_id', 'date']]
    
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    return df_encoded, scaler

# Automated Clustering
def perform_clustering(df_encoded, clustering_features, k_min=2, k_max=10):
    clustering_data = df_encoded.groupby('cell_id')[clustering_features].mean().reset_index()
    clustering_data = clustering_data.dropna()
    
    best_k = 2
    best_score = -1
    for k in range(k_min, k_max+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(clustering_data[clustering_features])
        score = silhouette_score(clustering_data[clustering_features], labels)
        print(f'For k={k}, the Silhouette Score is {score:.4f}')
        if score > best_score:
            best_k = k
            best_score = score
    
    print(f'Optimal number of clusters determined: {best_k} with Silhouette Score: {best_score:.4f}')
    
    # Final clustering with optimal k
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    clustering_data['cluster'] = final_kmeans.fit_predict(clustering_data[clustering_features])
    
    return clustering_data[['cell_id', 'cluster']]

# Model Training
def train_models(df_encoded, feature_cols, target='fire_occurred'):
    models = {}
    clusters = df_encoded['cluster'].unique()
    
    for cluster in clusters:
        print(f'Training models for Cluster {cluster}')
        cluster_data = df_encoded[df_encoded['cluster'] == cluster]
        X = cluster_data[feature_cols]
        y = cluster_data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Initialize LightGBM classifier
        lgbm = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42
        )
        
        # Train model
        lgbm.fit(X_train_res, y_train_res)
        
        # Store model
        models[cluster] = lgbm
        
        # Evaluation
        y_pred = lgbm.predict(X_test)
        y_proba = lgbm.predict_proba(X_test)[:,1]
        
        from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
        
        print(f'--- Evaluation for Cluster {cluster} ---')
        print(classification_report(y_test, y_pred))
        print(f'ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}')
        cm = confusion_matrix(y_test, y_pred)
        print('Confusion Matrix:')
        print(cm)
    
    return models

# Save Models
def save_models(models, save_path='models/'):
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for cluster, model in models.items():
        joblib.dump(model, f'{save_path}/LightGBM_cluster_{cluster}.joblib')
        print(f'Model for Cluster {cluster} saved.')

def main():
    # Paths
    DATA_CSV_PATH = "/Users/tobiascanavesi/Documents/wildifre_prevention/data/processed/merged_weather_fire_ndvi.csv"
    GRID_SHP_PATH = "/Users/tobiascanavesi/Documents/wildifre_prevention/data/raw/ca_grid_10km.shp"
    
    # Load data
    df = load_data(DATA_CSV_PATH)
    grid_gdf = load_grid(GRID_SHP_PATH)
    grid_centroids = load_grid_centroids(grid_gdf)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Encode and scale
    df_encoded, scaler = encode_and_scale(df)
    
    # Define clustering features
    clustering_features = ['precip_ma_180d', 'tmax_ma_180d', 'tmin_ma_180d', 'ndvi_ma_180d', 'fires_last_7d']
    
    # Perform clustering
    clustering_labels = perform_clustering(df_encoded, clustering_features, k_min=2, k_max=10)
    
    # Assign clusters to the main DataFrame
    df_encoded = pd.merge(df_encoded, clustering_labels, on='cell_id', how='left')
    
    # Define feature columns for modeling
    feature_cols = [col for col in df_encoded.columns if col not in ['fire_occurred', 'cell_id', 'date']]
    
    # Train models
    models = train_models(df_encoded, feature_cols, target='fire_occurred')
    
    # Save models
    save_models(models, save_path='models/')
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    print('Scaler saved.')

if __name__ == "__main__":
    main()
