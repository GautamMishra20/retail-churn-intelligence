import sys
import pandas as pd
import joblib

from src.logger import get_logger
from src.exception import CustomException
from src.config import CHURN_MODEL

logger = get_logger(__name__)


class CustomerData:
    """
    Represents a single customer's input data for churn prediction.
    Mirrors the exact feature set the model was trained on.
    """
    def __init__(self, frequency: int, monetary: float, avg_order_value: float, country: str):
        self.frequency = frequency
        self.monetary = monetary
        self.avg_order_value = avg_order_value
        self.country = country

    def to_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "frequency": [self.frequency],
                "monetary": [self.monetary],
                "avg_order_value": [self.avg_order_value],
                "Country": [self.country],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)


class PredictionPipeline:
    def __init__(self):
        self.model_path = CHURN_MODEL
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading model from {self.model_path}")
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, input_df: pd.DataFrame):
        logger.info(f"Running prediction on input shape: {input_df.shape}")
        try:
            model = self._load_model()

            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)[:, 1]

            logger.info(f"Prediction: {prediction}, Probability: {prediction_proba}")

            return prediction, prediction_proba

        except Exception as e:
            logger.error("Error occurred during prediction")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example: predict churn for a sample customer
    # sample_customer = CustomerData(
    #     frequency=3,
    #     monetary=1200.50,
    #     avg_order_value=400.17,
    #     country="United Kingdom"
    # )
    
    # sample_customer = CustomerData(
    #     frequency=1,
    #     monetary=50.00,
    #     avg_order_value=50.00,
    #     country="United Kingdom"
    # )

    sample_customer = CustomerData(
        frequency=15,
        monetary=8500.00,
        avg_order_value=566.67,
        country="United Kingdom"
    )

    input_df = sample_customer.to_dataframe()

    pipeline = PredictionPipeline()
    prediction, probability = pipeline.predict(input_df)

    result = "Churned" if prediction[0] == 1 else "Active"
    print(f"Prediction: {result}")
    print(f"Churn probability: {probability[0]:.2%}")