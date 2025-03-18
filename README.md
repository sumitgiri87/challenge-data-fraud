# Fraud Detection Challenge

## Introduction
This project is part of the **Challenge Data** platform hosted by **École Normale Supérieure (ENS)**. The challenge, sponsored by **BNP Paribas Personal Finance (BNPP PF)**, aims to improve fraud detection using basket purchase data. The original challenge details can be found at [Challenge Data ENS](https://challengedata.ens.fr/participants/challenges/104/).

## Challenge Context
BNP Paribas Personal Finance is a leader in consumer credit across Europe. As part of its risk management strategy, the company leverages advanced statistical models to mitigate credit risks. Fraud detection is a crucial aspect of this process since fraudsters continuously attempt to manipulate financial systems to their advantage.

In this challenge, the focus is on analyzing transaction-level basket data from one of BNPP PF's retail partners. The goal is to develop machine learning models that can accurately detect fraudulent transactions and reduce financial risks.

## Objective
The challenge is to identify fraudulent transactions based on basket data, which includes details about purchased items, their categories, prices, manufacturers, models, and unique retailer codes. The task is to:
- **Develop a fraud detection model** that learns patterns from transaction data.
- **Predict fraudulent cases** based on a given test dataset.
- **Maximize model performance** using the PR-AUC (Precision-Recall Area Under the Curve) metric.

## Data Description
The dataset consists of **115,988 observations** with **147 columns**, including:
- **Item details:** Category, manufacturer, model, price, and quantity.
- **Fraud flag:** Whether a transaction is fraudulent (`1`) or legitimate (`0`).
- **Highly imbalanced data:** Only **1.4%** of the transactions are fraudulent.

## Expected Outcome
The goal is to build an effective fraud detection model that assigns a probability score to each transaction. A higher score indicates a higher likelihood of fraud. The evaluation metric is **PR-AUC**, which is suited for imbalanced classification problems.

## Next Steps
- **Exploratory Data Analysis (EDA):** Understand the dataset and identify key patterns.
- **Feature Engineering:** Transform raw transaction data into meaningful features.
- **Model Development:** Experiment with different machine learning models.
- **Performance Tuning:** Optimize hyperparameters and improve fraud detection accuracy.

This README will be updated as we progress with the project. Stay tuned for more details!

---

📌 **Contact & Credits**  
This challenge was organized by **BNP Paribas Personal Finance** in collaboration with **École Normale Supérieure**. The challenge details and data were provided by the Challenge Data platform.


