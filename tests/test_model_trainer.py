import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def sample_customer_features():
    return pd.DataFrame({
        "frequency": [12, 1, 5, 2, 8],
        "monetary": [5000.0, 45.0, 1200.0, 300.0, 3000.0],
        "avg_order_value": [416.67, 45.0, 240.0, 150.0, 375.0],
        "Country": ["United Kingdom", "United Kingdom", "Germany", "France", "United Kingdom"],
        "is_churned": [0, 1, 0, 1, 0],
    })


def test_preprocessor_transforms_correctly(sample_customer_features):
    numeric_features = ["frequency", "monetary", "avg_order_value"]
    categorical_features = ["Country"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    X = sample_customer_features[numeric_features + categorical_features]
    transformed = preprocessor.fit_transform(X)

    # 3 numeric + N one-hot country columns, should have same number of rows as input
    assert transformed.shape[0] == len(X)


def test_pipeline_fits_and_predicts(sample_customer_features):
    numeric_features = ["frequency", "monetary", "avg_order_value"]
    categorical_features = ["Country"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    X = sample_customer_features[numeric_features + categorical_features]
    y = sample_customer_features["is_churned"]

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_unseen_country_handled_gracefully(sample_customer_features):
    """Ensures OneHotEncoder doesn't crash on a country not seen during training."""
    numeric_features = ["frequency", "monetary", "avg_order_value"]
    categorical_features = ["Country"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    X_train = sample_customer_features[numeric_features + categorical_features]
    preprocessor.fit(X_train)

    new_data = pd.DataFrame({
        "frequency": [3],
        "monetary": [500.0],
        "avg_order_value": [166.67],
        "Country": ["Japan"]  # not in training data
    })

    # Should not raise an error due to handle_unknown="ignore"
    transformed = preprocessor.transform(new_data)
    assert transformed.shape[0] == 1