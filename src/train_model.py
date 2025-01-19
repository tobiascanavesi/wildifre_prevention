import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

PROCESSED_DIR = os.path.join("..", "data", "processed")
MODELS_DIR = os.path.join("..", "models")

def load_local_data():
    # For simplicity, just load the most recent processed file
    # Or load multiple CSV files and concatenate
    files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")])
    if not files:
        raise ValueError("No processed data found! Run ingest_data.py first.")
    
    latest_file = os.path.join(PROCESSED_DIR, files[-1])
    df = pd.read_csv(latest_file)
    
    # Example: We'll pretend we have a 'fire_occurred' column for supervised learning
    # In reality, you'd merge with historical fire data or engineer that label
    # For demonstration, let's create a dummy label
    df["fire_occurred"] = (df["temp_C"] > 30) & (df["humidity"] < 30)
    df["fire_occurred"] = df["fire_occurred"].astype(int)
    
    return df

def main():
    df = load_local_data()
    
    # Prepare features and target
    X = df.drop(columns=["fire_occurred", "time"])
    y = df["fire_occurred"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'boosting_type': 'gbdt'
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, valid_data],
        valid_names=['train','valid'],
        early_stopping_rounds=10
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    print("Validation AUC:", auc)
    
    # Save the model locally
    model_path = os.path.join(MODELS_DIR, "lgb_model.txt")
    model.save_model(model_path)
    print(f"[Local] Model saved to {model_path}")

if __name__ == "__main__":
    main()
