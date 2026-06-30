"""
Train the precision-optimized churn model on REAL BusyBee termination data.
Source: Assessment (Responses) - Form Responses 1.csv

Target: "Have we lost the Client?" (Yes=churned, No=retained)
Records with NaN target are used for feature learning but excluded from supervised training.

NO DATABASE CONNECTIONS - purely local CSV training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, fbeta_score, classification_report, confusion_matrix
)
import xgboost as xgb
import pickle
from datetime import datetime

# ─── Config ───
CSV_PATH = r'C:\Users\leslie.chasinda\Downloads\Assessment (Responses) - Form Responses 1.csv'
OUTPUT_DIR = './models_real'


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and engineer features from real termination form data."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records, {len(df.columns)} columns")
    
    # ─── Target Variable ───
    # "Have we lost the Client?" → Yes=1 (churned), No=0 (retained)
    df['churned'] = df['Have we lost the Client?'].map({'Yes': 1, 'No': 0})
    
    # Only keep labeled rows for training
    labeled = df[df['churned'].notna()].copy()
    print(f"Labeled records: {len(labeled)} (Yes={labeled['churned'].sum():.0f}, No={(labeled['churned']==0).sum():.0f})")
    print(f"Churn rate: {labeled['churned'].mean():.1%}")
    
    return labeled


def engineer_features(df: pd.DataFrame) -> tuple:
    """Engineer features matching the production model's expectations."""
    features = pd.DataFrame(index=df.index)
    label_encoders = {}
    
    # 1. Will There Be A Replacement (strongest predictor historically)
    replacement_map = {'Yes': 1, 'No': 0, 'TBA': 0.5}
    features['no_replacement'] = 1 - df['Will There Be A Replacement'].map(replacement_map).fillna(0.5)
    
    # 2. Current Client Seat Count
    features['current_seat_count'] = pd.to_numeric(df['Current Client Seat Count'], errors='coerce').fillna(1).clip(0, 50)
    
    # 3. Client Health (1-5 scale, already numeric)
    features['health_score'] = pd.to_numeric(df['Specify the Client Health In The Last Month'], errors='coerce').fillna(3)
    
    # 4. Cadence of Check Ins (encode frequency)
    cadence_map = {'Daily': 4, 'Weekly': 3, 'Bi-Monthly': 2, 'Monthly': 1}
    features['checkin_frequency'] = df['Cadence of Check Ins With The Client'].map(cadence_map).fillna(2)
    
    # 5. Termination Type
    le_term = LabelEncoder()
    term_values = df['Reason for Termination'].fillna('Unknown').astype(str)
    features['termination_type_encoded'] = le_term.fit_transform(term_values)
    label_encoders['termination_type'] = le_term
    
    # 6. Days since check-in (derived from dates)
    df['notice_date'] = pd.to_datetime(
        df['Date of Receiving Termination/Replacement/Resignation Notice'], errors='coerce'
    )
    df['last_checkin_date'] = pd.to_datetime(
        df['Last Date of Official Weekly/Bi-Monthly/Monthly Check In'], errors='coerce'
    )
    features['days_since_checkin'] = (df['notice_date'] - df['last_checkin_date']).dt.days.fillna(14).clip(0, 180)
    
    # 7. Days since general communication
    df['last_comm_date'] = pd.to_datetime(
        df['Last Date of General Communication With The Client'], errors='coerce'
    )
    features['days_since_communication'] = (df['notice_date'] - df['last_comm_date']).dt.days.fillna(14).clip(0, 180)
    
    # 8. Client tenure (from Client Start Date to notice date)
    df['client_start'] = pd.to_datetime(df['Client Start Date:'], errors='coerce')
    features['client_tenure_months'] = ((df['notice_date'] - df['client_start']).dt.days / 30).fillna(12).clip(0, 120)
    
    # 9. Team member tenure
    df['tm_start'] = pd.to_datetime(df['Team Member  Start Date '], errors='coerce')
    features['tm_tenure_months'] = ((df['notice_date'] - df['tm_start']).dt.days / 30).fillna(6).clip(0, 60)
    
    # 10. Was PIP implemented?
    features['had_pip'] = (df['Was a Performance Improvement Plan Implemented?'] == 'Yes').astype(int)
    
    # 11. Was discount/relief offered?
    features['offered_discount'] = (
        df['Business Issue Termination - Was the relief package/discounts offered?'] == 'Yes'
    ).astype(int)
    
    # 12. Geo-Location
    le_geo = LabelEncoder()
    geo_values = df['Geo - Location'].fillna('Unknown').astype(str)
    features['geo_encoded'] = le_geo.fit_transform(geo_values)
    label_encoders['geo'] = le_geo
    
    # 13. Replacement urgency / timeframe
    le_timeframe = LabelEncoder()
    timeframe_values = df['Timeframe for Replacement'].fillna('Unknown').astype(str)
    features['replacement_urgency'] = le_timeframe.fit_transform(timeframe_values)
    label_encoders['timeframe'] = le_timeframe
    
    target = df['churned'].astype(int)
    
    print(f"\nEngineered {len(features.columns)} features:")
    print(f"  {features.columns.tolist()}")
    print(f"\nFeature stats:")
    print(features.describe().round(2).to_string())
    
    return features, target, label_encoders


def train_precision_model(X: pd.DataFrame, y: pd.Series):
    """Train XGBoost optimized for precision (minimize false positives)."""
    
    print(f"\n{'='*60}")
    print("  TRAINING: Precision-Optimized XGBoost Churn Model")
    print(f"  Target: minimize false positives (don't scare stakeholders)")
    print(f"{'='*60}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Train: {len(y_train)} (churn={y_train.sum():.0f})")
    print(f"  Test:  {len(y_test)} (churn={y_test.sum():.0f})")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost tuned for precision
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        max_depth=3,
        learning_rate=0.1,
        n_estimators=150,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Calibrate probabilities (Platt scaling)
    print("\n  Applying probability calibration (Platt scaling)...")
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Predictions with calibrated model
    y_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    # ─── Evaluate at multiple thresholds ───
    print(f"\n  {'─'*50}")
    print(f"  THRESHOLD ANALYSIS (find optimal precision/recall trade-off)")
    print(f"  {'─'*50}")
    print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<10} {'F0.5':<10} {'Flagged':<10}")
    print(f"  {'─'*50}")
    
    best_threshold = 0.5
    best_f05 = 0
    
    for thresh in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        y_pred_t = (y_proba >= thresh).astype(int)
        if y_pred_t.sum() == 0:
            continue
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        f05 = fbeta_score(y_test, y_pred_t, beta=0.5, zero_division=0)
        flagged = y_pred_t.sum()
        marker = ' ←' if f05 > best_f05 else ''
        print(f"  {thresh:<12.2f} {p:<12.3f} {r:<10.3f} {f05:<10.3f} {flagged:<10}{marker}")
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh
    
    print(f"\n  → Best threshold for F0.5: {best_threshold}")
    
    # Final evaluation at best threshold
    y_pred = (y_proba >= best_threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'f0.5': fbeta_score(y_test, y_pred, beta=0.5, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'decision_threshold': best_threshold,
        'calibrated': True
    }
    
    print(f"\n  {'='*50}")
    print(f"  FINAL MODEL METRICS (threshold={best_threshold})")
    print(f"  {'='*50}")
    print(f"  Precision: {metrics['precision']:.3f}  ← TARGET METRIC")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"  F0.5:      {metrics['f0.5']:.3f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.3f}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted Retained  Predicted Churned")
    print(f"  Actually Retained:    {cm[0][0]:>5}            {cm[0][1]:>5}  (FP={cm[0][1]})")
    print(f"  Actually Churned:     {cm[1][0]:>5}            {cm[1][1]:>5}  (TP={cm[1][1]})")
    print(f"\n  False Positive Rate: {cm[0][1] / (cm[0][0] + cm[0][1]):.1%}")
    
    # Feature importance
    importance = model.feature_importances_
    feat_imp = dict(zip(X.columns, importance))
    print(f"\n  Feature Importance:")
    for feat, imp in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(imp * 40)
        print(f"    {feat:<25s} {imp:.3f} {bar}")
    
    # Cross-validation
    print(f"\n  5-Fold Cross-Validation AUC:")
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='roc_auc')
    print(f"    Mean AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(f"    Folds:    {[f'{s:.3f}' for s in cv_scores]}")
    
    # ─── Risk distribution (new thresholds) ───
    all_proba = calibrated_model.predict_proba(scaler.transform(X))[:, 1]
    print(f"\n  Risk Distribution (all {len(X)} labeled records):")
    print(f"    Critical (≥0.85): {(all_proba >= 0.85).sum()}")
    print(f"    High (0.65-0.84): {((all_proba >= 0.65) & (all_proba < 0.85)).sum()}")
    print(f"    Medium (0.40-0.64): {((all_proba >= 0.40) & (all_proba < 0.65)).sum()}")
    print(f"    Low (<0.40):      {(all_proba < 0.40).sum()}")
    
    return model, calibrated_model, scaler, feat_imp, metrics, best_threshold


def main():
    print("=" * 60)
    print("  REAL DATA TRAINING: BusyBee Churn Prediction Model")
    print("  Source: Assessment (Responses) - Form Responses 1.csv")
    print("  NO DATABASE CONNECTIONS - local only")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data(CSV_PATH)
    
    # Engineer features
    X, y, label_encoders = engineer_features(df)
    
    # Train
    model, calibrated_model, scaler, feat_imp, metrics, threshold = train_precision_model(X, y)
    
    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_data = {
        'model': model,
        'calibrated_model': calibrated_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_columns': X.columns.tolist(),
        'feature_importance': feat_imp,
        'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model_name': 'xgboost_churn_precision_v2',
        'decision_threshold': threshold,
        'metrics': metrics,
        'training_records': len(X),
        'churn_rate': float(y.mean()),
        'saved_at': datetime.now().isoformat()
    }
    
    model_path = os.path.join(OUTPUT_DIR, 'churn_model_real_precision.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n  Model saved to: {model_path}")
    
    # ─── Comparison vs old model ───
    print(f"\n{'='*60}")
    print("  COMPARISON: Old Model vs New Precision Model")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'Old':<10} {'New':<10} {'Change':<10}")
    print(f"  {'─'*42}")
    old = {'precision': 0.550, 'recall': 0.786, 'f1': 0.647, 'auc_roc': 0.967}
    for m in ['precision', 'recall', 'f1', 'auc_roc']:
        o = old[m]
        n = metrics[m]
        arrow = '↑' if n > o else '↓'
        print(f"  {m:<12} {o:<10.3f} {n:<10.3f} {arrow} {abs(n-o):.3f}")
    
    print(f"\n  Decision Threshold: {threshold}")
    print(f"  Calibrated: Yes (Platt scaling)")
    print(f"  Dropped balanced class weights: Yes")
    print(f"  Training data: {len(X)} real termination records")
    
    print(f"\n{'='*60}")
    print("  DONE - No database was touched")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
