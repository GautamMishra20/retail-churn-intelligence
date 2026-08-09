# Retail Sales & Customer Churn Intelligence Platform

An end-to-end data science and ML engineering project simulating a real-world
retail analytics platform. Covers the full pipeline: EDA, statistical analysis,
feature engineering (including RFM analysis), multi-model ML comparison,
deployment via API, and business dashboards in Power BI.

## Problem Statement

Retail businesses need to understand **why customers churn** and **forecast
future sales** to make informed decisions. This project builds:

1. A **churn prediction model** (classification) to flag at-risk customers
2. A **sales forecasting model** (regression) to predict future revenue

## Tech Stack

- **Languages**: Python
- **Libraries**: pandas, numpy, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn
- **Deployment**: FastAPI, Docker
- **Visualization**: Power BI
- **Version Control**: Git

## Project Structure

retail-churn-intelligence/
├── data/ # raw, interim, processed data (gitignored)
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

1. **Data ingestion** — load raw transactional data (CSV) with pandas
2. **Data cleaning** — handle missing customer IDs, filter cancelled orders, remove invalid records
3. **EDA** — distributions, missingness, correlations, seasonality
4. **Feature engineering** — RFM (Recency, Frequency, Monetary) analysis, lag/rolling features, churn labeling, encoding
5. **Statistical testing** — hypothesis tests on churn drivers and campaign impact
6. **Modeling** — multi-algorithm comparison (Logistic Regression → Decision Tree → Random Forest → XGBoost/LightGBM)
7. **Evaluation** — F1/ROC-AUC (churn), RMSE/MAPE (forecasting)
8. **Deployment** — FastAPI endpoint, Dockerized
9. **Dashboarding** — Power BI reports connected to processed data

## Setup

_(to be added once the pipeline is functional)_

## Results

_(to be added after modeling phase — will include model comparison table and key insights)_

## Author

_(your name / LinkedIn / portfolio link)_
