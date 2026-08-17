import sys

from src.logger import get_logger
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.feature_engineering import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

logger = get_logger(__name__)


class TrainingPipeline:
    def run_pipeline(self):
        logger.info("========== TRAINING PIPELINE STARTED ==========")
        try:
            # Step 1: Data Ingestion
            logger.info("Step 1/5: Data Ingestion")
            ingestion = DataIngestion()
            ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            logger.info("Step 2/5: Data Transformation")
            transformation = DataTransformation()
            transformation.initiate_data_transformation()

            # Step 3: Feature Engineering
            logger.info("Step 3/5: Feature Engineering")
            feature_engineering = FeatureEngineering()
            feature_engineering.initiate_feature_engineering()

            # Step 4: Model Training
            logger.info("Step 4/5: Model Training")
            trainer = ModelTrainer()
            pipeline, X_test, y_test, best_model_name, best_params = trainer.initiate_model_training()

            # Step 5: Model Evaluation
            logger.info("Step 5/5: Model Evaluation")
            evaluator = ModelEvaluation()
            metrics, cm, report = evaluator.evaluate(pipeline, X_test, y_test)

            logger.info("========== TRAINING PIPELINE COMPLETED ==========")
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best params: {best_params}")
            logger.info(f"Final metrics: {metrics}")

            return {
                "best_model_name": best_model_name,
                "best_params": best_params,
                "metrics": metrics,
            }

        except Exception as e:
            logger.error("Training pipeline failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    result = pipeline.run_pipeline()
    print("\n=== TRAINING PIPELINE SUMMARY ===")
    print(f"Best model: {result['best_model_name']}")
    print(f"Best params: {result['best_params']}")
    print(f"Metrics: {result['metrics']}")