
# Import necessary libraries 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import gensim.downloader as api
import logging
import xgboost as xgb
import os
import joblib

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Word2Vec model
Word2V = api.load("word2vec-google-news-300")

# Set data directory
ROOT_DIR = os.getcwd()  # Current working directory
DATA_DIR = os.path.join(ROOT_DIR, "data")
print("Data directory:", DATA_DIR)

# Helper function to check if file exists
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    else:
        print(f"Found file: {file_path}")

# Function to get Word2Vec vector
def get_w2v_vector(sentence, model):
    vec = np.zeros(300)
    try:
        for word in sentence.split():
            if word in model.key_to_index:
                vec += model[word]
    except:
        return vec
    return vec / len(sentence.split())

# Preprocessing function
def preprocess(X_filename, Y_filename,
               cash_price_mean=None,
               cash_price_std=None,
               nbr_of_prod_purchas_mean=None,
               nbr_of_prod_purchas_std=None):
    
    # Build full paths
    X_path = os.path.join(DATA_DIR, X_filename)
    Y_path = os.path.join(DATA_DIR, Y_filename)
    
    # Check if files exist
    check_file_exists(X_path)
    check_file_exists(Y_path)
    
    n = 25
    X = pd.read_csv(X_path)
    y = pd.read_csv(Y_path)
    y = y.drop(['index', 'ID'], axis=1)

    start = 5
    columns_to_drop = ['item' + str(i) for i in range(start, n)] + \
                      ['make' + str(i) for i in range(start, n)] + \
                      ['model' + str(i) for i in range(start, n)] + \
                      ['goods_code' + str(i) for i in range(start, n)] + \
                      ['Nbr_of_prod_purchas' + str(i) for i in range(start, n)] + \
                      ['cash_price' + str(i) for i in range(start, n)]

    X = X.drop(columns_to_drop, axis=1)

    columns = ['item' + str(i) for i in range(1, n)] + \
              ['make' + str(i) for i in range(1, n)] + \
              ['model' + str(i) for i in range(1, n)] + \
              ['goods_code' + str(i) for i in range(1, n)]
    
    n = start
    df = X
    categorical_columns = ['item', 'make', 'model']
    categorical_columns = [col + str(i) for col in categorical_columns for i in range(1, n)]

    for col in df.columns:
        if col in categorical_columns:
            w2v_df = df[col].apply(lambda x: pd.Series(get_w2v_vector(str(x), Word2V), dtype=np.float32))
            w2v_df.columns = [f'{col}_w2v_{i}' for i in range(300)]
            df = pd.concat([df, w2v_df], axis=1)
            df = df.drop(col, axis=1)
        elif col in columns:
            df = df.drop(col, axis=1)

    # Numerical columns
    cash_price_cols = [f'cash_price{i}' for i in range(1, n)]
    nbr_of_prod_purchas_cols = [f'Nbr_of_prod_purchas{i}' for i in range(1, n)]

    # Replace NaNs with 0
    for col in cash_price_cols + nbr_of_prod_purchas_cols:
        df[col].fillna(0, inplace=True)

    if cash_price_mean is None:
        cash_price_mean = df[cash_price_cols].values.flatten().mean()
        cash_price_std = df[cash_price_cols].values.flatten().std()

        nbr_of_prod_purchas_mean = df[nbr_of_prod_purchas_cols].values.flatten().mean()
        nbr_of_prod_purchas_std = df[nbr_of_prod_purchas_cols].values.flatten().std()

    # Normalize
    for col in cash_price_cols:
        df[col] = (df[col] - cash_price_mean) / cash_price_std

    for col in nbr_of_prod_purchas_cols:
        df[col] = (df[col] - nbr_of_prod_purchas_mean) / nbr_of_prod_purchas_std

    return df, y, cash_price_mean, cash_price_std, nbr_of_prod_purchas_mean, nbr_of_prod_purchas_std

### Load training data
X_train_df, y_train_df, cash_price_mean, cash_price_std, nbr_of_prod_purchas_mean, nbr_of_prod_purchas_std = preprocess(
    X_filename="X_train_G3tdtEn.csv",
    Y_filename="Y_train_2_XPXJDyy.csv"
)

# Define model
xgb_classifier = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    min_child_weight=0,
    eta=0.01,
    gamma=0.1,
    subsample=0.97,
    colsample_bytree=0.98,
    objective='binary:logistic',
    reg_alpha=0,
    reg_lambda=0,
    scale_pos_weight=23.5,
    random_state=42
)

classifier = xgb_classifier

# Create pipeline
pipeline = make_pipeline(
    classifier
)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate on training set
print("Train score")
y_pred_train = pipeline.predict_proba(X_train)
average_precision_train = average_precision_score(y_train, y_pred_train[:, 1]) * 100
logger.info(f"Average precision score (Train): {average_precision_train:.2f}")

# Evaluate on validation set
print("Validation score")
y_pred_val = pipeline.predict_proba(X_val)
average_precision_val = average_precision_score(y_val, y_pred_val[:, 1]) * 100
logger.info(f"Average precision score (Validation): {average_precision_val:.2f}")

# Save trained model
model_filename = "trained_rf_classifier.pkl"
joblib.dump(pipeline, model_filename)
logger.info(f"Trained model saved as {model_filename}")

#####################
##### Prediction ####
#####################

# Preprocess test set
X_test_df, _, _, _, _, _ = preprocess(
    X_filename="X_test_8skS2ey.csv",
    Y_filename="Y_test_random_2.csv",
    cash_price_mean=cash_price_mean,
    cash_price_std=cash_price_std,
    nbr_of_prod_purchas_mean=nbr_of_prod_purchas_mean,
    nbr_of_prod_purchas_std=nbr_of_prod_purchas_std
)

# Predict on test
y_pred_test = pipeline.predict_proba(X_test_df)[:, 1]

# Create submission DataFrame
submission_df = pd.DataFrame({
    'ID': X_test_df['ID'],
    'fraud_flag': y_pred_test
})

# Save submission
submission_filename = os.path.join(DATA_DIR, 'y_pred.csv')
submission_df.to_csv(submission_filename, index=False)
logger.info(f"Prediction saved to {submission_filename}")
