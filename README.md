# Retail Sales & Customer Churn Intelligence Platform

![CI](https://github.com/GautamMishra20/retail-churn-intelligence/actions/workflows/ci.yml/badge.svg)

An end-to-end data science and ML engineering project built on real UK e-commerce
transaction data. Covers the full lifecycle: exploratory data analysis, statistical
hypothesis testing, RFM-based feature engineering, multi-algorithm model comparison,
and deployment via a containerized FastAPI service.

## Problem Statement

Retail businesses lose revenue when they can't identify disengaging customers before
it's too late. This project builds a **customer churn prediction model** that flags
at-risk customers based on their purchasing behavior — Recency, Frequency, and
Monetary value (RFM) — so retention efforts can be targeted before a customer is lost.

## Dataset

[Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) (Kaggle/UCI) —
real UK-based e-commerce transactional data, December 2009 to December 2011.
~1.07 million rows covering invoices, products, quantities, prices, customer IDs, and countries.

## Tech Stack

| Layer               | Tools                           |
| ------------------- | ------------------------------- |
| Language            | Python                          |
| Data handling       | pandas, numpy                   |
| Visualization (EDA) | matplotlib, seaborn             |
| Statistics          | scipy                           |
| Machine Learning    | scikit-learn, XGBoost, LightGBM |
| Explainability      | SHAP                            |
| Notebooks           | Jupyter                         |
| API                 | FastAPI, Uvicorn, Pydantic      |
| Containerization    | Docker                          |
| Testing             | pytest                          |
| Version Control     | Git / GitHub                    |

## Project Structure

retail-churn-intelligence/
├── data/
│ ├── raw/ # original dataset (gitignored)
│ ├── interim/ # post-ingestion, pre-cleaning (gitignored)
│ └── processed/ # cleaned + feature-engineered data (gitignored)
├── notebooks/
│ ├── 01_eda.ipynb # univariate, bivariate, multivariate analysis
│ ├── 02_feature_engineering.ipynb # RFM, churn label, segmentation
│ ├── 03_statistical_tests.ipynb # hypothesis testing on churn drivers
│ └── 04_modeling.ipynb # multi-algorithm comparison, tuning, SHAP
├── src/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ ├── feature_engineering.py
│ │ ├── model_trainer.py
│ │ └── model_evaluation.py
│ ├── pipeline/
│ │ ├── training_pipeline.py # end-to-end orchestration
│ │ └── prediction_pipeline.py # inference
│ ├── utils/
│ ├── config.py
│ ├── logger.py
│ └── exception.py
├── api/
│ ├── main.py # FastAPI app
│ └── Dockerfile
├── powerbi/ # dashboard (planned)
├── models/ # trained model artifacts (gitignored)
├── artifacts/ # model comparison results
├── tests/
├── requirements.txt
└── setup.py

## Pipeline Overview

1. **Data Ingestion** — load raw CSV with pandas, validate shape/dtypes/missing values
2. **Data Cleaning** — drop missing Customer IDs (~22.8%) and descriptions, remove duplicates (~3.2%), filter invalid prices, compute revenue
3. **EDA** — distributions, outlier detection (IQR), cancellations, seasonality, country and product-level analysis
4. **Feature Engineering** — RFM computation per customer, 90-day recency churn label, quantile-based RFM scoring, customer segmentation (Champion/Loyal/At Risk/Lost)
5. **Statistical Testing** — t-tests, ANOVA, chi-square, and correlation analysis to validate churn drivers
6. **Modeling** — Logistic Regression, Decision Tree, Random Forest, and XGBoost compared via 5-fold stratified cross-validation; best model tuned with GridSearchCV
7. **Evaluation** — F1, ROC-AUC, precision/recall, confusion matrix, SHAP feature importance
8. **Deployment** — FastAPI `/predict` endpoint, containerized with Docker

## Key Results

**Best model: Logistic Regression**
| Metric | CV Score | Test Score |
|---|---|---|
| F1 | 0.735 | 0.743 |
| ROC-AUC | 0.760 | 0.779 |

Logistic Regression outperformed all three tree-based ensembles, suggesting the
relationship between RFM features and churn is largely linear/monotonic in this dataset.

**Model comparison (5-fold CV, F1 score):**
| Model | F1 Mean | ROC-AUC Mean |
|---|---|---|
| Logistic Regression | 0.7265 | 0.7605 |
| XGBoost | 0.6932 | 0.7285 |
| Random Forest | 0.6329 | 0.6883 |
| Decision Tree | 0.6249 | 0.6132 |

**Key business insights:**

- Churn rate at a 90-day recency threshold: **~50.85%**
- **Frequency** is the strongest churn signal — active customers order ~3x more often than churned ones (9.64 vs 3.05 orders on average)
- **Monetary value** is also a strong signal — active customers spend over 4x more on average (£4,840 vs £1,135)
- Churn shows a statistically significant association with country (chi-square p = 0.0003), though the effect is modest outside the UK given smaller sample sizes
- SHAP analysis confirms frequency and monetary value as the top drivers of individual predictions, consistent with the statistical testing results
- ~75% of customers are repeat buyers, indicating a loyalty-driven customer base well suited to churn modeling

## Setup

### 1. Clone and set up environment

```bash
git clone https://github.com/GautamMishra20/retail-churn-intelligence.git
cd retail-churn-intelligence
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

### 2. Add the dataset

Download [Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
and place it at `data/raw/online_retail_II.csv`.

### 3. Run the full training pipeline

```bash
python -m src.pipeline.training_pipeline
```

This runs ingestion → cleaning → feature engineering → model training → evaluation,
and saves the trained model to `models/churn_model.pkl`.

### 4. Run the API

```bash
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### 5. Run with Docker (alternative to step 4)

```bash
docker build -t retail-churn-api -f api/Dockerfile .
docker run -d -p 8000:8000 --name churn-api retail-churn-api
```

### 6. Run tests

```bash
pytest tests/ -v
```

## Example API Request

```bash
POST /predict
{
  "frequency": 3,
  "monetary": 1200.50,
  "avg_order_value": 400.17,
  "country": "United Kingdom"
}
```

Response:

```json
{
  "prediction": "Active",
  "churn_probability": 0.2317
}
```

## Roadmap

- [ ] Sales forecasting model (regression) for revenue prediction
- [ ] Power BI dashboard connected to processed data (sales overview + churn risk views)
- [ ] Model monitoring / drift detection on API predictions

## Notes on Design Decisions

- A PostgreSQL/Docker layer was initially built for RFM and cohort analysis via SQL,
  but was later removed in favor of a fully pandas-based pipeline, after working through
  a Windows-native PostgreSQL port conflict. The RFM and cohort logic now lives in
  Python, which is equally standard practice in real-world data science workflows.
- `recency_days` is deliberately excluded from the model's feature set despite being
  part of RFM — since it directly defines the churn label (`is_churned = recency_days > 90`),
  including it would cause data leakage. This was caught after an initial 100% F1 score
  during model comparison flagged the issue.

## Author

Gautam Mishra
_(LinkedIn / portfolio link)_
"@ | Set-Content -Path README.md -Encoding utf8

````

## Commit

```powershell
git add README.md
git commit -m "docs: finalize README with setup instructions, results, and design notes"
git push
```

---

That's the last step — your project is now complete, deployed, tested, and documented, with an honest roadmap for what's next (forecasting + Power BI). A few final housekeeping suggestions before you call it done:

1. **Double check `data/raw/online_retail_II.csv` and `models/*.pkl` are actually gitignored** — run `git status` one more time to be sure nothing large accidentally got committed
2. **Look at your commit history on GitHub** (`git log --oneline`) — it should read as a clean, logical progression, which is genuinely a strong signal on its own
3. When you come back to add forecasting and Power BI, just pick up at "Step 19" again — the roadmap section in the README will remind you (and anyone reviewing it) exactly what's left

Nice work getting this all the way through.
````
