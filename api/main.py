import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.prediction_pipeline import CustomerData, PredictionPipeline
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

app = FastAPI(
    title="Retail Customer Churn Prediction API",
    description="Predicts whether a customer is likely to churn based on RFM behavior.",
    version="1.0.0"
)


class CustomerInput(BaseModel):
    frequency: int = Field(..., ge=0, description="Number of distinct orders placed")
    monetary: float = Field(..., ge=0, description="Total amount spent by the customer")
    avg_order_value: float = Field(..., ge=0, description="Average value per order")
    country: str = Field(..., description="Customer's country")

    class Config:
        json_schema_extra = {
            "example": {
                "frequency": 5,
                "monetary": 1200.50,
                "avg_order_value": 240.10,
                "country": "United Kingdom"
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float


@app.get("/")
def root():
    return {
        "message": "Retail Customer Churn Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    try:
        logger.info(f"Received prediction request: {customer.dict()}")

        customer_data = CustomerData(
            frequency=customer.frequency,
            monetary=customer.monetary,
            avg_order_value=customer.avg_order_value,
            country=customer.country
        )

        input_df = customer_data.to_dataframe()

        pipeline = PredictionPipeline()
        prediction, probability = pipeline.predict(input_df)

        result = "Churned" if prediction[0] == 1 else "Active"

        response = PredictionResponse(
            prediction=result,
            churn_probability=round(float(probability[0]), 4)
        )

        logger.info(f"Prediction result: {response}")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")