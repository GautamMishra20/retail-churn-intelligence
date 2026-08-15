import sys
import pandas as pd

from src.logger import get_logger
from src.exception import CustomException
from src.config import INTERIM_DATA, PROCESSED_DATA

logger = get_logger(__name__)


class DataTransformation:
    def __init__(self):
        self.interim_data_path = INTERIM_DATA
        self.processed_data_path = PROCESSED_DATA

    def initiate_data_transformation(self):
        logger.info("Starting data transformation")
        try:
            df = pd.read_csv(self.interim_data_path, encoding="ISO-8859-1")
            initial_shape = df.shape
            logger.info(f"Loaded interim data with shape: {initial_shape}")

            # Flag cancellations before any filtering
            df["is_cancelled"] = df["Invoice"].astype(str).str.startswith("C")

            # Drop rows with missing Customer ID (can't do churn/RFM without it)
            before = len(df)
            df = df.dropna(subset=["Customer ID"])
            logger.info(f"Dropped {before - len(df)} rows with missing Customer ID")

            # Drop rows with missing Description
            before = len(df)
            df = df.dropna(subset=["Description"])
            logger.info(f"Dropped {before - len(df)} rows with missing Description")

            # Drop exact duplicate rows
            before = len(df)
            df = df.drop_duplicates()
            logger.info(f"Dropped {before - len(df)} duplicate rows")

            # Remove invalid price rows (Price <= 0), except legitimate cancellations
            before = len(df)
            df = df[~((df["Price"] <= 0) & (df["is_cancelled"] == False))]
            logger.info(f"Dropped {before - len(df)} rows with invalid non-cancellation price <= 0")

            # Parse dates
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

            # Compute revenue
            df["Revenue"] = df["Quantity"] * df["Price"]

            # Rename Customer ID for consistency downstream
            df = df.rename(columns={"Customer ID": "CustomerID"})

            df["CustomerID"] = df["CustomerID"].astype(int)

            final_shape = df.shape
            logger.info(f"Transformation complete. Shape: {initial_shape} -> {final_shape}")

            df.to_csv(self.processed_data_path, index=False)
            logger.info(f"Processed data saved to {self.processed_data_path}")

            return self.processed_data_path

        except Exception as e:
            logger.error("Error occurred during data transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformer = DataTransformation()
    output_path = transformer.initiate_data_transformation()
    print(f"Data transformation completed. Output saved at: {output_path}")