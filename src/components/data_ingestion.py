import sys
import pandas as pd

from src.logger import get_logger
from src.exception import CustomException
from src.config import RAW_DATA, INTERIM_DATA

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.raw_data = RAW_DATA
        self.interim_data = INTERIM_DATA
        
    def initiate_data_ingestion(self):
        logger.info("Data Ingestion Starts...")
        try:
            df = pd.read_csv(self.raw_data, encoding="ISO-8859-1")
            logger.info(f"Raw data loaded successfully with shape: {df.shape}")
            
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Missing values per column:\n{df.isnull().sum()}")
            logger.info(f"Data types:\n{df.dtypes}")
            
            df.to_csv(self.interim_data, index=False)
            logger.info(f"Interim data saved to {self.interim_data}")

            return self.interim_data
        
        except Exception as e:
            logger.error("Error occured during data ingestion")
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    ingestion = DataIngestion()
    output_path = ingestion.initiate_data_ingestion()
    print(f"Data ingestion completed. Output saved at: {output_path}")