import sys
import pandas as pd

from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)

from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)


class ModelEvaluation:
    def __init__(self):
        pass

    def evaluate(self, pipeline, X_test, y_test):
        logger.info("Starting model evaluation")
        try:
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
            }

            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")

            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            logger.info(f"Confusion Matrix:\n{cm}")
            logger.info(f"Classification Report:\n{report}")

            return metrics, cm, report

        except Exception as e:
            logger.error("Error occurred during model evaluation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    import joblib
    from src.config import CHURN_MODEL, CUSTOMER_FEATURES
    from sklearn.model_selection import train_test_split
    from src.config import RANDOM_STATE

    pipeline = joblib.load(CHURN_MODEL)
    df = pd.read_csv(CUSTOMER_FEATURES)

    features = ["frequency", "monetary", "avg_order_value", "Country"]
    target = "is_churned"
    X = df[features]
    y = df[target]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    evaluator = ModelEvaluation()
    metrics, cm, report = evaluator.evaluate(pipeline, X_test, y_test)

    print("Metrics:", metrics)
    print("\nClassification Report:\n", report)