import os
import re
import logging
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes special JSON characters from column names for LightGBM compatibility.
    """
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    return df

def train_and_evaluate():
    logging.info("Loading modeling-ready datasets...")
    X_train = pd.read_csv('../data/processed/X_train_balanced.csv')
    y_train = pd.read_csv('../data/processed/y_train_balanced.csv').squeeze()
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv').squeeze()

    # Sanitize column names to prevent JSON character error
    X_train = clean_feature_names(X_train)
    X_test = clean_feature_names(X_test)

    logging.info("Training LightGBM classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    logging.info("Evaluating model on the hold-out test set...")
    # Predict probabilities for ROC-AUC, and binary classes for Precision/Recall
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logging.info(f"Test Set ROC-AUC Score: {roc_auc:.4f}")

    # Compare thresholds to find the best Recall/Precision balance
    print("\n--- Threshold Comparison ---")
    for thresh in [0.50, 0.35, 0.25, 0.20]:
        preds = (y_pred_proba >= thresh).astype(int)
        cm = confusion_matrix(y_test, preds)
        tp, fn, fp = cm[1, 1], cm[1, 0], cm[0, 1]
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"Threshold {thresh:.2f} | Recall: {recall:.2f} | Precision: {precision:.2f} | Caught: {tp} | False Alarms: {fp}")

    # 2. Apply your chosen threshold for the detailed report
    chosen_threshold = 0.25
    y_pred = (y_pred_proba >= chosen_threshold).astype(int)

    print(f"\n--- Classification Report (Threshold = {chosen_threshold}) ---")
    print(classification_report(y_test, y_pred, target_names=['Standard Risk (0)', 'High Risk <30 (1)']))

    print(f"--- Confusion Matrix (Threshold = {chosen_threshold}) ---")
    print(confusion_matrix(y_test, y_pred))

    # Save the trained model
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/lgbm_readmission_model.pkl'
    joblib.dump(model, model_path)
    logging.info(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()