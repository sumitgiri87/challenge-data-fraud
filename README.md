# Fraud Detection Challenge - BNP Paribas PF

## Project Description
This repository contains the code, data, and resources for the BNP Paribas Personal Finance fraud detection challenge. The goal of this project is to develop machine learning models to identify fraudulent transactions based on basket data from retail purchases.

## Requirements
To participate in this challenge, you need to build a model that can accurately detect fraudulent transactions. The dataset contains 115,988 transactions with 147 features, including information about the purchased items, their prices, manufacturers, and other attributes.

### Data Breakdown
- **Input Variables (X)**: 144 columns representing item attributes (e.g., `item1` to `item24`, `cash_price1` to `cash_price24`, etc.).
- **Target Variable (Y)**: `fraud_flag` (1 = fraud, 0 = non-fraud).
- **Dataset Split**:
  - **Training Set**: 92,790 observations (1,319 fraud, 91,471 non-fraud)
  - **Test Set**: 23,198 observations (362 fraud, 22,836 non-fraud)

### Evaluation Metric
The performance of fraud detection models is measured using **Precision-Recall AUC (PR-AUC)**, which focuses on the minority class (fraud cases). The higher the PR-AUC score, the better the model is at detecting fraudulent transactions.

## How to Use This Repository
### 1. Clone the Repository
```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/challenge-data-fraud.git
cd challenge-data-fraud
```

### 2. Install Dependencies
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Explore the Data
- **Data files are stored in the `data/` folder (not included in the repo for security reasons).**
- **Use Jupyter notebooks in the `notebooks/` directory for data exploration and preprocessing.**
```sh
jupyter notebook
```

### 4. Train the Model
```sh
python src/train_model.py
```

### 5. Generate and Submit Predictions
```sh
python src/generate_submission.py
```

## Repository Structure
```
├── data/                # Raw and processed dataset files (not included)
├── notebooks/           # Jupyter notebooks for exploration & model development
├── src/                 # Source code for preprocessing, training, and evaluation
├── models/              # Trained models and artifacts
├── submissions/         # Output predictions for submission
├── README.md            # Project documentation
└── requirements.txt     # List of dependencies
```

## Additional Resources
- **Official Challenge Page:** [BNP Paribas PF Fraud Challenge](#) *(Insert actual link if available)*
- **Scikit-learn Documentation:** [https://scikit-learn.org](https://scikit-learn.org)
- **Precision-Recall AUC Explanation:** [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

## Contributing
If you would like to contribute to this project, feel free to fork the repository and submit a pull request.

## License
This project is for educational purposes only and is not affiliated with BNP Paribas PF.