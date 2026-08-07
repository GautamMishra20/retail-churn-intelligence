# Retail Sales & Customer Churn Intelligence Platform

An end-to-end data science and ML engineering project simulating a real-world
retail analytics platform. Covers the full pipeline: SQL data modeling, EDA,
statistical analysis, feature engineering, multi-model ML comparison,
deployment via API, and business dashboards in Power BI.

## Problem Statement

Retail businesses need to understand **why customers churn** and **forecast
future sales** to make informed decisions. This project builds:

1. A **churn prediction model** (classification) to flag at-risk customers
2. A **sales forecasting model** (regression) to predict future revenue

## Tech Stack

- **Database**: PostgreSQL (Dockerized)
- **Languages**: Python, SQL
- **Libraries**: pandas, scikit-learn, XGBoost, LightGBM, SQLAlchemy, psycopg2
- **Deployment**: FastAPI, Docker
- **Visualization**: Power BI
- **Version Control**: Git

## Project Structure

retail-churn-intelligence/
├── data/ # raw, interim, processed data (gitignored)
├── sql/ # schema and analytical queries
├── notebooks/ # EDA, feature engineering, stats, modeling
├── src/
│ ├── components/ # ingestion, transformation, training, evaluation
│ ├── pipeline/ # training and prediction orchestration
│ ├── utils/ # shared helper functions
├── api/ # FastAPI serving layer
├── powerbi/ # Power BI dashboard file
├── models/ # trained model artifacts (gitignored)
├── artifacts/ # intermediate outputs (encoders, splits, comparisons)
├── tests/ # unit tests

## Pipeline Overview

1. **Data ingestion** — load raw transactional data into PostgreSQL
2. **SQL layer** — RFM scoring, cohort analysis, revenue rollups via window functions
3. **EDA** — distributions, missingness, correlations, seasonality
4. **Feature engineering** — RFM features, lag/rolling features, encoding
5. **Statistical testing** — hypothesis tests on churn drivers and campaign impact
6. **Modeling** — multi-algorithm comparison (Logistic Regression → Random Forest → XGBoost/LightGBM)
7. **Evaluation** — F1/ROC-AUC (churn), RMSE/MAPE (forecasting)
8. **Deployment** — FastAPI endpoint, Dockerized
9. **Dashboarding** — Power BI reports connected to PostgreSQL

## Setup

_(to be added once the pipeline is functional)_

## Results

_(to be added after modeling phase — will include model comparison table and key insights)_

## Author

_(your name / LinkedIn / portfolio link)_
