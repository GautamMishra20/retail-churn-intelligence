import sys
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.logger import get_logger
from src.exception import CustomException
from src.config import (
    CUSTOMER_FEATURES,
    RANDOM_STATE,
    MODEL_COMPARISON,
    CHURN_MODEL,
)

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self):
        self.customer_features_path = CUSTOMER_FEATURES
        self.model_comparison_path = MODEL_COMPARISON
        self.churn_model_path = CHURN_MODEL
        self.random_state = RANDOM_STATE

        self.numeric_features = ["frequency", "monetary", "avg_order_value"]
        self.categorical_features = ["Country"]
        self.target = "is_churned"

    def _build_preprocessor(self):
        return ColumnTransformer(transformers=[
            ("num", StandardScaler(), self.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features)
        ])

    def _get_candidate_models(self):
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_state),
            "RandomForest": RandomForestClassifier(random_state=self.random_state),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=self.random_state),
        }

    def _get_param_grid(self, model_name):
        grids = {
            "LogisticRegression": {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs"],
            },
            "DecisionTree": {
                "classifier__max_depth": [3, 5, 7, 10],
                "classifier__min_samples_split": [2, 5, 10],
            },
            "RandomForest": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10, None],
            },
            "XGBoost": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
            },
        }
        return grids[model_name]

    def initiate_model_training(self):
        logger.info("Starting model training")
        try:
            df = pd.read_csv(self.customer_features_path)
            logger.info(f"Loaded customer features with shape: {df.shape}")

            features = self.numeric_features + self.categorical_features
            X = df[features]
            y = df[self.target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            preprocessor = self._build_preprocessor()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

            # --- Multi-model comparison ---
            results = []
            models = self._get_candidate_models()

            for name, model in models.items():
                pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
                f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
                roc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

                results.append({
                    "model": name,
                    "f1_mean": f1_scores.mean(),
                    "f1_std": f1_scores.std(),
                    "roc_auc_mean": roc_scores.mean(),
                    "roc_auc_std": roc_scores.std(),
                })
                logger.info(f"{name}: F1={f1_scores.mean():.4f}, ROC-AUC={roc_scores.mean():.4f}")

            results_df = pd.DataFrame(results).sort_values("f1_mean", ascending=False)
            os.makedirs(os.path.dirname(self.model_comparison_path), exist_ok=True)
            results_df.to_csv(self.model_comparison_path, index=False)
            logger.info(f"Model comparison saved to {self.model_comparison_path}")

            # --- Tune the winner ---
            best_model_name = results_df.iloc[0]["model"]
            logger.info(f"Best model from comparison: {best_model_name}")

            best_model = models[best_model_name]
            param_grid = self._get_param_grid(best_model_name)

            tuning_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", best_model)])
            grid_search = GridSearchCV(tuning_pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
            grid_search.fit(X_train, y_train)

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

            best_pipeline = grid_search.best_estimator_

            # --- Save model ---
            os.makedirs(os.path.dirname(self.churn_model_path), exist_ok=True)
            joblib.dump(best_pipeline, self.churn_model_path)
            logger.info(f"Model saved to {self.churn_model_path}")

            return best_pipeline, X_test, y_test, best_model_name, grid_search.best_params_

        except Exception as e:
            logger.error("Error occurred during model training")
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    pipeline, X_test, y_test, best_model_name, best_params = trainer.initiate_model_training()
    print(f"Training completed. Best model: {best_model_name}")
    print(f"Best params: {best_params}")