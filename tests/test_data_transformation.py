import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "Invoice": ["536365", "536365", "C536366", "536367", "536367"],
        "StockCode": ["85123A", "71053", "84029E", "22752", "22752"],
        "Description": ["WHITE HANGING HEART", "WHITE METAL LANTERN", None, "SET 7 BABUSHKA", "SET 7 BABUSHKA"],
        "Quantity": [6, 6, -1, 2, 2],
        "InvoiceDate": ["2010-12-01 08:26:00"] * 5,
        "Price": [2.55, 3.39, 1.65, 7.65, 7.65],
        "Customer ID": [17850, 17850, 17850, None, 13047],
        "Country": ["United Kingdom"] * 5,
    })


def test_drops_missing_customer_id(sample_raw_df):
    cleaned = sample_raw_df.dropna(subset=["Customer ID"])
    assert cleaned["Customer ID"].isnull().sum() == 0
    assert len(cleaned) == 4


def test_drops_missing_description(sample_raw_df):
    cleaned = sample_raw_df.dropna(subset=["Description"])
    assert cleaned["Description"].isnull().sum() == 0
    assert len(cleaned) == 4


def test_cancellation_flag_detection(sample_raw_df):
    sample_raw_df["is_cancelled"] = sample_raw_df["Invoice"].astype(str).str.startswith("C")
    assert sample_raw_df["is_cancelled"].sum() == 1
    assert sample_raw_df.loc[2, "is_cancelled"] == True


def test_revenue_calculation(sample_raw_df):
    sample_raw_df["Revenue"] = sample_raw_df["Quantity"] * sample_raw_df["Price"]
    expected_first_row_revenue = 6 * 2.55
    assert sample_raw_df.loc[0, "Revenue"] == pytest.approx(expected_first_row_revenue)


def test_duplicate_removal():
    df = pd.DataFrame({
        "Invoice": ["536367", "536367"],
        "StockCode": ["22752", "22752"],
        "Quantity": [2, 2],
        "Price": [7.65, 7.65],
    })
    deduped = df.drop_duplicates()
    assert len(deduped) == 1