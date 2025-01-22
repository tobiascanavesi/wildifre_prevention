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
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import logging
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    filename='model_training.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load data
def load_data(csv_path):
    logging.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    # only keep year 2020
    df = df[df['date'].dt.year == 2020]
    logging.info(f"Data loaded successfully with shape {df.shape}")
    return df

def load_grid(shapefile_path):
    logging.info(f"Loading grid shapefile from {shapefile_path}")
    grid_gdf = gpd.read_file(shapefile_path)
    logging.info(f"Grid shapefile loaded successfully with {len(grid_gdf)} records")
    return grid_gdf

def load_grid_centroids(grid_gdf):
    logging.info("Calculating centroids for grid cells")
    centroids_projected = grid_gdf.geometry.centroid
    gdf_centroids = gpd.GeoDataFrame(
        grid_gdf[['cell_id']].copy(),
        geometry=centroids_projected,
        crs=grid_gdf.crs
    )
    gdf_centroids = gdf_centroids.to_crs(epsg=4326)
    gdf_centroids['centroid_lon'] = gdf_centroids.geometry.x
    gdf_centroids['centroid_lat'] = gdf_centroids.geometry.y
    logging.info("Centroids calculated successfully")
    return gdf_centroids[['cell_id', 'centroid_lon', 'centroid_lat']]

# Feature Engineering
def feature_engineering(df):
    logging.info("Starting feature engineering")
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
        logging.info(f"Calculating moving averages with window size: {window} days")
        df[f'precip_ma_{window}d'] = df.groupby('cell_id')['precip_in'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'tmax_ma_{window}d'] = df.groupby('cell_id')['tmax_F'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'tmin_ma_{window}d'] = df.groupby('cell_id')['tmin_F'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'ndvi_ma_{window}d'] = df.groupby('cell_id')['ndvi'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
    # NDVI lag feature
    logging.info("Creating NDVI lag feature")
    df['ndvi_lag_1d'] = df.groupby('cell_id')['ndvi'].shift(1)
    
    # Previous fires
    logging.info("Creating previous fires feature")
    df['fires_last_7d'] = df.groupby('cell_id')['fire_occurred'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    
    logging.info("Feature engineering completed")
    return df

# Optimized Missing Value Handling
def handle_missing_values(df):
    logging.info("Handling missing values after feature engineering")
    
    # Ensure the DataFrame is sorted by 'cell_id' and 'date' for correct forward-fill
    if 'date' in df.columns:
        df = df.sort_values(['cell_id', 'date'])
        logging.info("DataFrame sorted by 'cell_id' and 'date'")
    else:
        df = df.sort_values('cell_id')
        logging.info("DataFrame sorted by 'cell_id'")
    
    # Forward-fill within each 'cell_id' group
    df[['precip_in', 'tmax_F', 'tmin_F', 'ndvi', 'fire_occurred']] = df.groupby('cell_id')[['precip_in', 'tmax_F', 'tmin_F', 'ndvi', 'fire_occurred']].ffill()
    logging.info("Forward-fill applied within each 'cell_id' group")
    
    # Backward-fill to handle any remaining NaNs at the start of each group
    df[['precip_in', 'tmax_F', 'tmin_F', 'ndvi', 'fire_occurred']] = df.groupby('cell_id')[['precip_in', 'tmax_F', 'tmin_F', 'ndvi', 'fire_occurred']].bfill()
    logging.info("Backward-fill applied within each 'cell_id' group")
    
    # Drop any remaining NaNs
    df = df.dropna()
    logging.info("NaN values dropped")
    
    return df

# Encoding and Scaling
def encode_and_scale(df):
    logging.info("Starting encoding and scaling")
    categorical_cols = ['month', 'day_of_week', 'season']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    logging.info("Performing One-Hot Encoding")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Identify numerical columns for scaling
    numerical_cols = [col for col in df_encoded.columns if df_encoded[col].dtype in ['int64', 'float64'] and col not in ['fire_occurred', 'cell_id', 'date']]
    
    logging.info(f"Imputing missing numerical values in columns: {numerical_cols}")
    numerical_features_path = os.path.join('models', 'numerical_features.txt')
    with open(numerical_features_path, 'w') as f:
        for feature in numerical_cols:
            f.write(f"{feature}\n")
    logging.info(f"Numerical features saved to {numerical_features_path}")
    
    # Impute numerical missing values with mean (if any remain)
    imputer = SimpleImputer(strategy='mean')
    df_encoded[numerical_cols] = imputer.fit_transform(df_encoded[numerical_cols])
    
    logging.info("Scaling numerical features")
    # Scaling
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    logging.info("Encoding and scaling completed")

    return df_encoded, scaler

# Automated Clustering
def perform_clustering(df_encoded, clustering_features, k_min=2, k_max=10):
    logging.info("Starting automated clustering")
    clustering_data = df_encoded.groupby('cell_id')[clustering_features].mean().reset_index()
    clustering_data = clustering_data.dropna()
    
    best_k = 2
    best_score = -1
    logging.info("Determining the optimal number of clusters using Silhouette Score")
    for k in tqdm(range(k_min, k_max+1), desc="Clustering Progress"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(clustering_data[clustering_features])
        score = silhouette_score(clustering_data[clustering_features], labels)
        logging.info(f'For k={k}, the Silhouette Score is {score:.4f}')
        if score > best_score:
            best_k = k
            best_score = score
    
    logging.info(f'Optimal number of clusters determined: {best_k} with Silhouette Score: {best_score:.4f}')
    
    # Final clustering with optimal k
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    clustering_data['cluster'] = final_kmeans.fit_predict(clustering_data[clustering_features])
    
    logging.info("Clustering completed")
    return clustering_data[['cell_id', 'cluster']]

# Model Training
def train_models(df_encoded, feature_cols, target='fire_occurred'):
    logging.info("Starting model training")
    models = {}
    clusters = df_encoded['cluster'].unique()
    
    for cluster in tqdm(clusters, desc="Model Training Progress"):
        logging.info(f'Training models for Cluster {cluster}')
        cluster_data = df_encoded[df_encoded['cluster'] == cluster]
        X = cluster_data[feature_cols]
        y = cluster_data[target]
        
        # Check for missing values
        if X.isnull().sum().sum() > 0:
            logging.warning(f"Missing values detected in Cluster {cluster}. Imputing missing values.")
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Split data
        logging.info("Splitting data into training and testing sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        logging.info("Applying SMOTE to handle class imbalance")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logging.info(f"After SMOTE, training data shape: {X_train_res.shape}")
        
        # Initialize LightGBM classifier
        lgbm = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42
        )
        
        # Train model
        logging.info("Training LightGBM model")
        lgbm.fit(X_train_res, y_train_res)
        logging.info("Model training completed")
        
        # Store model
        models[cluster] = lgbm
        logging.info(f"Model for Cluster {cluster} stored")
        
        # Evaluation
        logging.info("Evaluating model performance")
        y_pred = lgbm.predict(X_test)
        y_proba = lgbm.predict_proba(X_test)[:,1]
        
        from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
        
        report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        logging.info(f'--- Evaluation for Cluster {cluster} ---')
        logging.info(f'Classification Report:\n{report}')
        logging.info(f'ROC AUC Score: {roc_auc:.2f}')
        logging.info(f'Confusion Matrix:\n{cm}')
    
    logging.info("Model training completed for all clusters")
    return models

# Save Models
def save_models(models, save_path='models/'):
    logging.info(f"Saving models to {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logging.info(f"Created directory {save_path}")
    for cluster, model in models.items():
        model_filename = f'LightGBM_cluster_{cluster}.joblib'
        joblib.dump(model, os.path.join(save_path, model_filename))
        logging.info(f'Model for Cluster {cluster} saved as {model_filename}')
    logging.info("All models saved successfully")

# Save Scaler
def save_scaler(scaler, save_path='models/scaler.joblib'):
    logging.info(f"Saving scaler to {save_path}")
    joblib.dump(scaler, save_path)
    logging.info("Scaler saved successfully")

# Save Clustering Labels
def save_clustering_labels(clustering_labels, save_path='data/processed/clustering_labels.csv'):
    logging.info(f"Saving clustering labels to {save_path}")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        logging.info(f"Created directory {os.path.dirname(save_path)}")
    clustering_labels.to_csv(save_path, index=False)
    logging.info("Clustering labels saved successfully")

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
    
    # Handle missing values after feature engineering
    df = handle_missing_values(df)
    
    # Encode and scale
    df_encoded, scaler = encode_and_scale(df)
    
    # Define clustering features
    clustering_features = ['precip_ma_180d', 'tmax_ma_180d', 'tmin_ma_180d', 'ndvi_ma_180d', 'fires_last_7d']
    
    # Perform clustering
    clustering_labels = perform_clustering(df_encoded, clustering_features, k_min=2, k_max=10)
    
    # Save clustering labels
    save_clustering_labels(clustering_labels, save_path='data/processed/clustering_labels.csv')
    
    # Assign clusters to the main DataFrame
    logging.info("Assigning cluster labels to the main DataFrame")
    df_encoded = pd.merge(df_encoded, clustering_labels, on='cell_id', how='left')
    
    # Define feature columns for modeling
    feature_cols = [col for col in df_encoded.columns if col not in ['fire_occurred', 'cell_id', 'date', 'cluster']]
    logging.info(f"Feature columns for modeling: {feature_cols}")
    
    # Train models
    models = train_models(df_encoded, feature_cols, target='fire_occurred')

    feature_names_path = os.path.join('models', 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    logging.info(f"Feature names saved to {feature_names_path}")
    # Save models
    save_models(models, save_path='models/')
    
    # Save scaler
    save_scaler(scaler, save_path='models/scaler.joblib')
    logging.info('Scaler saved.')

if __name__ == "__main__":
    main()
