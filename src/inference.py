import os
import json
import lightgbm as lgb
import pandas as pd

MODELS_DIR = os.path.join("..", "models")

# Arbitrary threshold for "high risk"
RISK_THRESHOLD = 0.7

def load_model():
    model_path = os.path.join(MODELS_DIR, "lgb_model.txt")
    booster = lgb.Booster(model_file=model_path)
    return booster

def score_instance(data):
    # data: dict of features, e.g. {"temp_C": 35, "humidity": 20, ...}
    booster = load_model()
    df = pd.DataFrame([data])
    predictions = booster.predict(df)
    return predictions[0]

def main():
    sample_data = {
        "temp_C": 35,
        "humidity": 20,
        "wind_speed": 10
    }
    
    risk_score = score_instance(sample_data)
    print(f"Risk Score: {risk_score}")
    
    if risk_score > RISK_THRESHOLD:
        alert_msg = f"ALERT: High wildfire risk detected (score={risk_score:.2f})"
        print(alert_msg)
        # Here you could integrate with local email / Slack / etc.
    else:
        print("Risk is below threshold.")

if __name__ == "__main__":
    main()
