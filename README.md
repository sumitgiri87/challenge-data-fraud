# 🕵️‍♂️ Fraud Detection Challenge

## 📘 Overview

This project was completed as part of the **Challenge Data Platform** hosted by [École Normale Supérieure (ENS)](https://challengedata.ens.fr/), in collaboration with **BNP Paribas Personal Finance (BNPP PF)**.  
The goal is to build a **machine learning model** that detects **fraudulent transactions** based on basket-level purchase data.

---

## 🎯 Objective

The challenge involves predicting whether a transaction is **fraudulent (1)** or **legitimate (0)** using item-level features such as:

- Category  
- Manufacturer  
- Model  
- Price  
- Quantity  
- Retailer codes  

Because only **1.4%** of the transactions are fraudulent, the task is a **highly imbalanced classification problem**, evaluated using the **Precision-Recall AUC (PR-AUC)** metric.

---

## 📊 Dataset Summary

| Property | Description |
|-----------|--------------|
| Observations | 115,988 |
| Features | 147 |
| Fraud Rate | 1.4% |
| Evaluation Metric | PR-AUC |

---

## 🔍 Methodology

The project followed a typical ML pipeline:

1. **Exploratory Data Analysis (EDA)**  
   - Distribution of features, missing values, correlations  
   - Identification of useful categorical variables  

2. **Feature Engineering**  
   - Frequency encoding for categorical variables  
   - Price normalization and log-scaling  
   - Chi-squared test for feature relevance  

3. **Model Development**  
   - Random Forest (baseline)  
   - XGBoost (experiment)  
   - Evaluated using PR-AUC and recall metrics  

4. **Evaluation & Submission**  
   - Model inference on test data  
   - Output stored as probability scores in `final_result.csv`

---

## 🧠 Results

| Model | PR-AUC | Comments |
|--------|--------|----------|
| Random Forest | ~0.15 | Stable baseline |
| XGBoost | ~0.24 | Better recall on minority class |

*(Scores approximate based on internal evaluation; not final leaderboard results.)*

---

## 🧩 Repository Structure

```
fraud-detection-challenge/
│
├── data/
│   ├── raw/                ← original challenge datasets
│   └── processed/          ← intermediate feature files
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── legacy_experiments/
│       └── train_model.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── models/
│   └── trained_rf_classifier.pkl
│
├── submissions/
│   └── final_result.csv
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

Install the dependencies before running the notebooks:

```bash
pip install -r requirements.txt
```

Key libraries used:
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- joblib / pickle

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/sumitgiri87/challenge-data-fraud
   cd challenge-data-fraud
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (or venv\Scripts\activate on Windows)
   pip install -r requirements.txt
   ```

3. **Run EDA and Training**
   ```bash
   jupyter notebook src/model_training.ipynb
   ```

4. **Generate Predictions**
   - The model saves outputs to `submissions/final_result.csv`

---

## 🏁 Next Steps

- Add automated feature selection and class-weight tuning.  
- Implement explainability (e.g., SHAP values).  
- Deploy model as an API or streamlit dashboard.

---

## 📜 Credits

- **Challenge Organizer:** BNP Paribas Personal Finance  
- **Platform:** [Challenge Data (ENS)](https://challengedata.ens.fr/)  
- **Author:** Sumit Giri  
- **Contact:** [LinkedIn](https://www.linkedin.com/in/sumit-giri-0111/)  

---

## 💬 Note

This repository is a cleaned-up presentation of my experimentation and results for the BNPP PF Fraud Detection Challenge.  
It aims to demonstrate practical ML workflow understanding — from data exploration to model evaluation — in a realistic fraud detection context.
