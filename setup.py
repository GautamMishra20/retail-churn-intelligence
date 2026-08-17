from setuptools import setup, find_packages

setup(
    name="retail-churn-intelligence",
    version="0.1.0",
    author="Gautam Mishra",
    description="Retail Sales & Customer Churn Intelligence Platform - end-to-end data science project",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "shap",
        "fastapi",
        "uvicorn",
        "pydantic",
        "joblib",
        "python-dotenv",
    ],
    python_requires=">=3.9",
)
