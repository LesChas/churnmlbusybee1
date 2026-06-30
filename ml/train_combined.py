"""
Train churn model on COMBINED real + synthetic data (~623 labeled records).
Same pipeline as perfect_model.py:
  - Feature engineering
  - Grid search (F0.5 scoring)
  - Platt calibration
  - Threshold optimization

NO DATABASE CONNECTIONS - local CSVs only.
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

REAL_CSV = r'C:\Users\leslie.chasinda\Downloads\Assessment (Responses) - Form Responses 1.csv'
SYNTHETIC_CSV = r'C:\Users\leslie.chasinda\Downloads\synthetic_assessment_responses.csv'
OUTPUT_DIR = './models_combined'


def load_combined_data():
    """Load both CSVs and combine labeled records."""
    df_real = pd.read_csv(REAL_CSV)
    df_synth = pd.read_csv(SYNTHETIC_CSV)

    # Standardize to common columns
    common_cols = list(set(df_real.columns) & set(df_synth.columns))
    df_real = df_real[common_cols].copy()
    df_synth = df_synth[common_cols].copy()

    # Tag source for diagnostics
    df_real['_source'] = 'real'
    df_synth['_source'] = 'synthetic'

    # Combine
    df = pd.concat([df_real, df_synth], ignore_index=True)

    # Label: churned
    df['churned'] = df['Have we lost the Client?'].map({'Yes': 1, 'No': 0})
    labeled = df[df['churned'].notna()].copy()

    return labeled


def engineer_features(df):
    """Same feature engineering as perfect_model.py."""
    features = pd.DataFrame(index=df.index)
    label_encoders = {}

    # 1. No replacement (strongest signal)
    replacement_map = {'Yes': 1, 'No': 0, 'TBA': 0.5}
    features['no_replacement'] = 1 - df['Will There Be A Replacement'].map(replacement_map).fillna(0.5)

    # 2. Current seat count (clip outliers at 50)
    features['current_seat_count'] = pd.to_numeric(
        df['Current Client Seat Count'], errors='coerce'
    ).fillna(1).clip(0, 50)

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
    print('  COMBINED MODEL TRAINING (Real + Synthetic)')
    print('  Grid search + Calibration + Threshold optimization')
    print('=' * 65)

    df = load_combined_data()

    # Show source breakdown
    src_counts = df['_source'].value_counts()
    print(f'\n  Data sources:')
    for src, cnt in src_counts.items():
        churn_in_src = df[df['_source'] == src]['churned'].sum()
        print(f'    {src:<12s}: {cnt:>4} records ({churn_in_src:.0f} churned)')

    X_all, y, label_encoders = engineer_features(df)

    print(f'\n  Combined: {len(y)} labeled records')
    print(f'  Churned: {y.sum():.0f} ({y.mean():.1%})  |  Retained: {(y==0).sum():.0f} ({(y==0).mean():.1%})')
    print(f'  Features: {X_all.columns.tolist()}')

    X = X_all.copy()

    # ─── Split ───
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f'  Train: {len(y_train)} | Test: {len(y_test)}')

    # ─── Grid Search ───
    print(f'\n  STEP 1: Hyperparameter grid search (F0.5 scoring)...')

    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'min_child_weight': [3, 5, 7],
        'n_estimators': [100, 150, 200, 300],
        'gamma': [0.1, 0.3, 0.5],
        'learning_rate': [0.03, 0.05, 0.1],
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

    # ─── Calibrate ───
    print(f'\n  STEP 2: Platt calibration...')
    cal_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
    cal_model.fit(X_train_s, y_train)

    y_proba = cal_model.predict_proba(X_test_s)[:, 1]

    # ─── Threshold optimization ───
    print(f'\n  STEP 3: Threshold optimization')
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    f05_scores = (1 + 0.5**2) * (precisions * recalls) / ((0.5**2 * precisions) + recalls + 1e-8)
    best_idx = np.argmax(f05_scores)
    best_thresh = thresholds[min(best_idx, len(thresholds) - 1)]

    print(f'\n  {"Threshold":<12} {"Precision":<12} {"Recall":<10} {"F0.5":<10} {"Flagged":<10}')
    print(f'  {"-"*54}')
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        yp = (y_proba >= t).astype(int)
        if yp.sum() == 0:
            continue
        p = precision_score(y_test, yp, zero_division=0)
        r = recall_score(y_test, yp, zero_division=0)
        f = fbeta_score(y_test, yp, beta=0.5, zero_division=0)
        marker = ' <-- best' if abs(t - best_thresh) < 0.025 else ''
        print(f'  {t:<12.2f} {p:<12.3f} {r:<10.3f} {f:<10.3f} {yp.sum():<10}{marker}')

    print(f'\n  Optimal threshold (max F0.5): {best_thresh:.2f}')

    # ─── Final evaluation ───
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
    print(f'  FINAL COMBINED MODEL METRICS')
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
    print(f'\n  Feature Importance (combined model):')
    for feat, imp in sorted(zip(X.columns, best_model.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f'    {feat:<28s} {imp:.4f} {bar}')

    # Cross-validation on full data
    cv_auc = cross_val_score(best_model, scaler.transform(X), y, cv=5, scoring='roc_auc')
    cv_f05 = cross_val_score(best_model, scaler.transform(X), y, cv=5, scoring=f05_scorer)
    print(f'\n  5-Fold Cross-Validation (full combined data):')
    print(f'    AUC-ROC: {cv_auc.mean():.3f} (±{cv_auc.std():.3f})  {[f"{s:.3f}" for s in cv_auc]}')
    print(f'    F0.5:    {cv_f05.mean():.3f} (±{cv_f05.std():.3f})  {[f"{s:.3f}" for s in cv_f05]}')

    # Risk distribution
    all_proba = cal_model.predict_proba(scaler.transform(X))[:, 1]
    print(f'\n  Risk Distribution ({len(X)} combined records):')
    print(f'    Critical (>=0.85): {(all_proba >= 0.85).sum()}')
    print(f'    High (0.65-0.84):  {((all_proba >= 0.65) & (all_proba < 0.85)).sum()}')
    print(f'    Medium (0.40-0.64):{((all_proba >= 0.40) & (all_proba < 0.65)).sum()}')
    print(f'    Low (<0.40):       {(all_proba < 0.40).sum()}')

    # ─── Comparison vs previous ───
    print(f'\n  {"="*55}')
    print(f'  PROGRESSION: v1 → v2 → v3(perfected) → v4(combined)')
    print(f'  {"="*55}')
    print(f'  {"Metric":<12} {"v1(orig)":<12} {"v2(real)":<12} {"v3(perf)":<12} {"v4(comb)":<12}')
    print(f'  {"-"*60}')
    v1 = {'precision': 0.550, 'recall': 0.786, 'f1': 0.647, 'auc_roc': 0.967}
    v2 = {'precision': 0.909, 'recall': 0.714, 'f1': 0.800, 'auc_roc': 0.951}
    v3 = {'precision': 0.875, 'recall': 1.000, 'f1': 0.933, 'auc_roc': 0.983}
    for m in ['precision', 'recall', 'f1', 'auc_roc']:
        print(f'  {m:<12} {v1[m]:<12.3f} {v2[m]:<12.3f} {v3[m]:<12.3f} {metrics[m]:<12.3f}')

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
        'model_name': 'xgboost_churn_combined_v4',
        'decision_threshold': float(best_thresh),
        'best_params': best,
        'metrics': metrics,
        'training_records': len(X),
        'data_sources': {'real': int(src_counts.get('real', 0)), 'synthetic': int(src_counts.get('synthetic', 0))},
        'churn_rate': float(y.mean()),
        'saved_at': datetime.now().isoformat()
    }
    path = os.path.join(OUTPUT_DIR, 'churn_model_combined_v4.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f'\n  Model saved: {path}')
    print(f'  Best hyperparameters: {best}')
    print(f'  Optimal threshold: {best_thresh:.2f}')
    print(f'  Training data: {len(X)} records (real={src_counts.get("real", 0)}, synthetic={src_counts.get("synthetic", 0)})')
    print(f'\n{"="*65}')
    print(f'  DONE — No database touched')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
