import sys
import os
import pandas as pd

from src.logger import get_logger
from src.exception import CustomException
from src.config import PROCESSED_DATA, CUSTOMER_FEATURES, CHURN_RECENCY

logger = get_logger(__name__)


class FeatureEngineering:
    def __init__(self):
        self.processed_data_path = PROCESSED_DATA
        self.customer_features_path = CUSTOMER_FEATURES
        self.churn_threshold = CHURN_RECENCY

    def initiate_feature_engineering(self):
        logger.info("Starting feature engineering")
        try:
            df = pd.read_csv(self.processed_data_path, encoding="ISO-8859-1")
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            logger.info(f"Loaded processed data with shape: {df.shape}")

            reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
            non_cancelled = df[df["is_cancelled"] == False]

            rfm = non_cancelled.groupby("CustomerID").agg(
                last_purchase_date=("InvoiceDate", "max"),
                frequency=("Invoice", "nunique"),
                monetary=("Revenue", "sum")
            ).reset_index()

            rfm["recency_days"] = (reference_date - rfm["last_purchase_date"]).dt.days
            rfm = rfm[["CustomerID", "recency_days", "frequency", "monetary"]]

            # Churn label
            rfm["is_churned"] = (rfm["recency_days"] > self.churn_threshold).astype(int)
            logger.info(f"Churn rate: {rfm['is_churned'].mean() * 100:.2f}%")

            # Average order value
            rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]

            # RFM scores and segments
            rfm["R_score"] = pd.qcut(rfm["recency_days"], 4, labels=[4, 3, 2, 1]).astype(int)
            rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
            rfm["M_score"] = pd.qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
            rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

            def segment_customer(score):
                if score >= 10:
                    return "Champion"
                elif score >= 8:
                    return "Loyal"
                elif score >= 6:
                    return "At Risk"
                else:
                    return "Lost"

            rfm["segment"] = rfm["RFM_score"].apply(segment_customer)

            # Country (most common per customer)
            customer_country = df.groupby("CustomerID")["Country"].agg(lambda x: x.mode()[0]).reset_index()
            rfm = rfm.merge(customer_country, on="CustomerID", how="left")

            rfm.to_csv(self.customer_features_path, index=False)
            logger.info(f"Feature-engineered data saved to {self.customer_features_path}, shape: {rfm.shape}")

            return self.customer_features_path

        except Exception as e:
            logger.error("Error occurred during feature engineering")
            raise CustomException(e, sys)


if __name__ == "__main__":
    fe = FeatureEngineering()
    output_path = fe.initiate_feature_engineering()
    print(f"Feature engineering completed. Output saved at: {output_path}")