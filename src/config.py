import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA = os.path.join(BASE_DIR, 'data', 'raw', 'online_retail_II.csv')
INTERIM_DATA = os.path.join(BASE_DIR, 'data', 'interim', 'retail_interim.csv')
PROCESSED_DATA = os.path.join(BASE_DIR, 'data', 'processed', 'reatil_processed.csv')

ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

CHURN_RECENCY = 90

RANDOM_STATE = 42