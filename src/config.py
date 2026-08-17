import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA = os.path.join(BASE_DIR, 'data', 'raw', 'online_retail_II.csv')
INTERIM_DATA = os.path.join(BASE_DIR, 'data', 'interim', 'retail_interim.csv')
PROCESSED_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'reatil_processed.csv')
CUSTOMER_FEATURES = os.path.join(BASE_DIR, "data", "processed", "customer_features.csv")

MODEL_COMPARISON = os.path.join(BASE_DIR, "artifacts", "model_comparison.csv")
CHURN_MODEL = os.path.join(BASE_DIR, "models", "churn_model.pkl")

ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

CHURN_RECENCY = 90

RANDOM_STATE = 42