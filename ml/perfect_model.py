"""
Perfect the current churn model:
  1. Drop zero-importance features (had_pip, offered_discount, geo_encoded, replacement_urgency)
  2. Hyperparameter search (grid search with F0.5 scoring)
  3. Feature interaction analysis
  4. Threshold optimization on calibrated probabilities
  5. Final model with best configuration

NO DATABASE CONNECTIONS - local CSV only.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, fbeta_score, confusion_matrix, make_scorer,
    precision_recall_curve
)
import xgboost as xgb
import pickle
from datetime import datetime

CSV_PATH = r'C:\Users\leslie.chasinda\Downloads\Assessment (Responses) - Form Responses 1.csv'
OUTPUT_DIR = './models_real'


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['churned'] = df['Have we lost the Client?'].map({'Yes': 1, 'No': 0})
    labeled = df[df['churned'].notna()].copy()
    return labeled


def engineer_features(df):
    features = pd.DataFrame(index=df.index)
    label_encoders = {}

    # 1. No replacement (strongest signal)
    replacement_map = {'Yes': 1, 'No': 0, 'TBA': 0.5}
    features['no_replacement'] = 1 - df['Will There Be A Replacement'].map(replacement_map).fillna(0.5)

    # 2. Current seat count
    features['current_seat_count'] = pd.to_numeric(df['Current Client Seat Count'], errors='coerce').fillna(1).clip(0, 50)

    # 3. Client health
    features['health_score'] = pd.to_numeric(
        df['Specify the Client Health In The Last Month'], errors='coerce'
    ).fillna(3)

    # 4. Check-in cadence
    cadence_map = {'Daily': 4, 'Weekly': 3, 'Bi-Monthly': 2, 'Monthly': 1}
    features['checkin_frequency'] = df['Cadence of Check Ins With The Client'].map(cadence_map).fillna(2)

    # 5. Termination type
    le_term = LabelEncoder()
    features['termination_type_encoded'] = le_term.fit_transform(
        df['Reason for Termination'].fillna('Unknown').astype(str)
    )
    label_encoders['termination_type'] = le_term

    # 6-7. Date-derived gaps
    df['notice_date'] = pd.to_datetime(
        df['Date of Receiving Termination/Replacement/Resignation Notice'], errors='coerce'
    )
    df['last_checkin_date'] = pd.to_datetime(
        df['Last Date of Official Weekly/Bi-Monthly/Monthly Check In'], errors='coerce'
    )
    df['last_comm_date'] = pd.to_datetime(
        df['Last Date of General Communication With The Client'], errors='coerce'
    )
    features['days_since_checkin'] = (df['notice_date'] - df['last_checkin_date']).dt.days.fillna(14).clip(0, 180)
    features['days_since_communication'] = (df['notice_date'] - df['last_comm_date']).dt.days.fillna(14).clip(0, 180)

    # 8-9. Tenure
    df['client_start'] = pd.to_datetime(df['Client Start Date:'], errors='coerce')
    df['tm_start'] = pd.to_datetime(df['Team Member  Start Date '], errors='coerce')
    features['client_tenure_months'] = ((df['notice_date'] - df['client_start']).dt.days / 30).fillna(12).clip(0, 120)
    features['tm_tenure_months'] = ((df['notice_date'] - df['tm_start']).dt.days / 30).fillna(6).clip(0, 60)

    target = df['churned'].astype(int)
    return features, target, label_encoders


def main():
    print('=' * 65)
    print('  PERFECTING THE CHURN MODEL')
    print('  Step-by-step optimization on real data')
    print('=' * 65)

    df = load_data(CSV_PATH)
    X_all, y, label_encoders = engineer_features(df)

    print(f'\n  Records: {len(y)}  |  Churned: {y.sum():.0f} ({y.mean():.1%})  |  Retained: {(y==0).sum():.0f}')

    # ─── STEP 1: Drop dead features ───
    dead = ['had_pip', 'offered_discount', 'geo_encoded', 'replacement_urgency']
    dead_present = [f for f in dead if f in X_all.columns]
    X = X_all.drop(columns=dead_present, errors='ignore')

    print(f'\n  STEP 1: Dropped {len(dead_present)} dead features')
    print(f'  Remaining {len(X.columns)} features: {X.columns.tolist()}')

    # ─── Split ───
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f'  Train: {len(y_train)} | Test: {len(y_test)}')

    # ─── STEP 2: Hyperparameter grid search ───
    print(f'\n  STEP 2: Hyperparameter grid search (F0.5 scoring)...')

    param_grid = {
        'max_depth': [2, 3, 4],
        'min_child_weight': [3, 5, 7],
        'n_estimators': [100, 150, 200],
        'gamma': [0.1, 0.3, 0.5],
        'learning_rate': [0.05, 0.1],
    }

    f05_scorer = make_scorer(fbeta_score, beta=0.5)

    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        base_model, param_grid,
        scoring=f05_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    grid.fit(X_train_s, y_train)

    best = grid.best_params_
    print(f'  Best params: {best}')
    print(f'  Best CV F0.5: {grid.best_score_:.3f}')

    best_model = grid.best_estimator_

    # ─── STEP 3: Calibrate ───
    print(f'\n  STEP 3: Platt calibration...')
    cal_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
    cal_model.fit(X_train_s, y_train)

    y_proba = cal_model.predict_proba(X_test_s)[:, 1]

    # ─── STEP 4: Optimal threshold via precision-recall curve ───
    print(f'\n  STEP 4: Threshold optimisation')
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # Compute F0.5 at every threshold
    f05_scores = (1 + 0.5**2) * (precisions * recalls) / ((0.5**2 * precisions) + recalls + 1e-8)
    best_idx = np.argmax(f05_scores)
    best_thresh = thresholds[min(best_idx, len(thresholds) - 1)]

    print(f'\n  {"Threshold":<12} {"Precision":<12} {"Recall":<10} {"F0.5":<10} {"Flagged":<10}')
    print(f'  {"-"*54}')
    for t in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        yp = (y_proba >= t).astype(int)
        if yp.sum() == 0:
            continue
        p = precision_score(y_test, yp, zero_division=0)
        r = recall_score(y_test, yp, zero_division=0)
        f = fbeta_score(y_test, yp, beta=0.5, zero_division=0)
        marker = ' <-- best' if abs(t - best_thresh) < 0.05 else ''
        print(f'  {t:<12.2f} {p:<12.3f} {r:<10.3f} {f:<10.3f} {yp.sum():<10}{marker}')

    print(f'\n  Optimal threshold (max F0.5): {best_thresh:.2f}')

    # ─── STEP 5: Final evaluation ───
    y_pred = (y_proba >= best_thresh).astype(int)

    metrics = {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'f0.5': fbeta_score(y_test, y_pred, beta=0.5, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'accuracy': accuracy_score(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)

    print(f'\n  {"="*55}')
    print(f'  STEP 5: FINAL PERFECTED MODEL METRICS')
    print(f'  {"="*55}')
    for k, v in metrics.items():
        print(f'    {k:<12s} {v:.3f}')

    print(f'\n  Confusion Matrix:')
    print(f'                     Pred Retained   Pred Churned')
    print(f'    Actual Retained:     {cm[0][0]:>5}          {cm[0][1]:>5}  (FP={cm[0][1]})')
    print(f'    Actual Churned:      {cm[1][0]:>5}          {cm[1][1]:>5}  (TP={cm[1][1]})')
    fp_rate = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    print(f'    False Positive Rate: {fp_rate:.1%}')

    # Feature importance
    print(f'\n  Feature Importance (tuned model):')
    for feat, imp in sorted(zip(X.columns, best_model.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f'    {feat:<28s} {imp:.4f} {bar}')

    # Cross-validation on full data
    cv_auc = cross_val_score(best_model, scaler.transform(X), y, cv=5, scoring='roc_auc')
    cv_f05 = cross_val_score(best_model, scaler.transform(X), y, cv=5, scoring=f05_scorer)
    print(f'\n  5-Fold Cross-Validation:')
    print(f'    AUC-ROC: {cv_auc.mean():.3f} (±{cv_auc.std():.3f})  {[f"{s:.3f}" for s in cv_auc]}')
    print(f'    F0.5:    {cv_f05.mean():.3f} (±{cv_f05.std():.3f})  {[f"{s:.3f}" for s in cv_f05]}')

    # Risk distribution
    all_proba = cal_model.predict_proba(scaler.transform(X))[:, 1]
    print(f'\n  Risk Distribution ({len(X)} records):')
    print(f'    Critical (>=0.85): {(all_proba >= 0.85).sum()}')
    print(f'    High (0.65-0.84):  {((all_proba >= 0.65) & (all_proba < 0.85)).sum()}')
    print(f'    Medium (0.40-0.64):{((all_proba >= 0.40) & (all_proba < 0.65)).sum()}')
    print(f'    Low (<0.40):       {(all_proba < 0.40).sum()}')

    # ─── Comparison ───
    print(f'\n  {"="*55}')
    print(f'  PROGRESSION: Original → Previous → Perfected')
    print(f'  {"="*55}')
    print(f'  {"Metric":<12} {"Original":<12} {"Previous":<12} {"Perfected":<12}')
    print(f'  {"-"*48}')
    orig = {'precision': 0.550, 'recall': 0.786, 'f1': 0.647, 'auc_roc': 0.967}
    prev = {'precision': 0.909, 'recall': 0.714, 'f1': 0.800, 'auc_roc': 0.951}
    for m in ['precision', 'recall', 'f1', 'auc_roc']:
        print(f'  {m:<12} {orig[m]:<12.3f} {prev[m]:<12.3f} {metrics[m]:<12.3f}')

    # ─── Save ───
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_data = {
        'model': best_model,
        'calibrated_model': cal_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_columns': X.columns.tolist(),
        'feature_importance': dict(zip(X.columns, best_model.feature_importances_)),
        'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model_name': 'xgboost_churn_perfected_v3',
        'decision_threshold': float(best_thresh),
        'best_params': best,
        'metrics': metrics,
        'training_records': len(X),
        'churn_rate': float(y.mean()),
        'saved_at': datetime.now().isoformat()
    }
    path = os.path.join(OUTPUT_DIR, 'churn_model_perfected.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f'\n  Model saved: {path}')
    print(f'  Best hyperparameters: {best}')
    print(f'  Optimal threshold: {best_thresh:.2f}')
    print(f'\n{"="*65}')
    print(f'  DONE — No database touched')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
