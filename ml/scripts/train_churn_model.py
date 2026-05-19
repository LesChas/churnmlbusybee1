#!/usr/bin/env python3
"""
BusyBee ML - Train Client Churn Model
======================================
Trains the XGBoost churn prediction model using local SQLite data.
Does NOT touch Supabase - only reads from local ml_training.db

Usage:
    python train_churn_model.py
    
    # With custom parameters:
    python train_churn_model.py --test-size 0.3 --n-estimators 200
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import pickle

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import xgboost as xgb

from database.ml_db import MLDatabase


# Feature columns used for training
FEATURE_COLUMNS = [
    'days_since_communication',
    'days_since_checkin', 
    'health_score',
    'checkin_frequency_encoded',
    'no_replacement',
    'replacement_urgency',
    'had_pip',
    'offered_discount',
    'current_seat_count',
    'client_tenure_at_termination',
    'termination_type_encoded',
    'geo_encoded'
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for training."""
    df = df.copy()
    
    # Convert numeric columns to proper types (handle mixed types from SQLite)
    numeric_columns = [
        'days_since_communication', 'days_since_checkin', 'health_score',
        'checkin_frequency_encoded', 'no_replacement', 'replacement_urgency',
        'had_pip', 'offered_discount', 'current_seat_count', 
        'client_tenure_at_termination', 'client_lost'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Fill missing values
    df['days_since_communication'] = df['days_since_communication'].fillna(999).clip(0, 999)
    df['days_since_checkin'] = df['days_since_checkin'].fillna(999).clip(0, 999)
    df['current_seat_count'] = df['current_seat_count'].fillna(1)
    df['client_tenure_at_termination'] = df['client_tenure_at_termination'].fillna(0)
    
    # Encode categorical variables
    if 'termination_type' in df.columns:
        le_term = LabelEncoder()
        df['termination_type_encoded'] = le_term.fit_transform(
            df['termination_type'].fillna('Unknown').astype(str)
        )
    else:
        df['termination_type_encoded'] = 0
    
    if 'geo_location' in df.columns:
        le_geo = LabelEncoder()
        df['geo_encoded'] = le_geo.fit_transform(
            df['geo_location'].fillna('Unknown').astype(str)
        )
    else:
        df['geo_encoded'] = 0
    
    return df


def train_model(df: pd.DataFrame, test_size: float = 0.2, 
                n_estimators: int = 100, random_state: int = 42) -> dict:
    """
    Train XGBoost churn prediction model.
    
    Args:
        df: Training data with features and client_lost column
        test_size: Fraction of data to use for testing
        n_estimators: Number of trees
        random_state: Random seed
        
    Returns:
        Dictionary with model, metrics, and feature importance
    """
    # Prepare features
    df = prepare_features(df)
    
    # Get feature columns that exist
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    X = df[available_features]
    y = df['client_lost']
    
    print(f"\nTraining with {len(available_features)} features:")
    for f in available_features:
        print(f"  - {f}")
    
    # Split data BEFORE resampling (prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nData split (before resampling):")
    print(f"  Training: {len(X_train)} ({int(y_train.sum())} churned)")
    print(f"  Testing:  {len(X_test)} ({int(y_test.sum())} churned)")
    
    # Handle class imbalance with SMOTE (if available) or class weights
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state, k_neighbors=min(5, int(y_train.sum()) - 1))
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"\n✓ SMOTE applied:")
        print(f"  Training after SMOTE: {len(X_train_resampled)} ({int(y_train_resampled.sum())} churned)")
        X_train, y_train = X_train_resampled, y_train_resampled
        scale_pos_weight = 1  # Balanced now
    except ImportError:
        print("\n⚠️  SMOTE not available (pip install imbalanced-learn)")
        print("    Using class weights instead...")
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'training_records': len(X_train),
        'test_records': len(X_test),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    metrics['cv_auc_mean'] = cv_scores.mean()
    metrics['cv_auc_std'] = cv_scores.std()
    
    # Feature importance
    feature_importance = dict(zip(available_features, model.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Print results
    print(f"\n{'='*60}")
    print("Model Performance:")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.3f}")
    print(f"  CV AUC:    {metrics['cv_auc_mean']:.3f} (+/- {metrics['cv_auc_std']:.3f})")
    
    print(f"\nFeature Importance:")
    for feature, importance in feature_importance.items():
        bar = "█" * int(importance * 50)
        print(f"  {feature:35} {importance:.3f} {bar}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'features': available_features
    }


def save_model(result: dict, model_dir: Path):
    """Save trained model and metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)
    
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = model_dir / f"churn_model_{version}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)
    
    # Save latest as well
    latest_path = model_dir / "churn_model_latest.pkl"
    with open(latest_path, 'wb') as f:
        pickle.dump(result['model'], f)
    
    # Convert numpy types to Python native for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Save metadata
    metadata = {
        'version': version,
        'trained_at': datetime.now().isoformat(),
        'metrics': convert_to_native(result['metrics']),
        'feature_importance': convert_to_native(result['feature_importance']),
        'features': result['features']
    }
    
    metadata_path = model_dir / f"churn_model_{version}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    latest_metadata_path = model_dir / "churn_model_latest_metadata.json"
    with open(latest_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return version


def main():
    parser = argparse.ArgumentParser(description='Train BusyBee churn prediction model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction (default: 0.2)')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("BusyBee ML - Client Churn Model Training")
    print(f"{'='*60}")
    print(f"  Using LOCAL SQLite database (not Supabase)")
    print(f"{'='*60}\n")
    
    # Load data from SQLite
    db = MLDatabase()
    stats = db.get_stats()
    
    if stats['total_records'] == 0:
        print("Error: No training data found!")
        print("Run 'python import_terminations.py <csv_file>' first")
        sys.exit(1)
    
    print(f"Database: {db.db_path}")
    print(f"Records: {stats['total_records']}")
    print(f"Churn rate: {stats['churn_rate']:.1%}")
    
    if stats['churned_clients'] < 5:
        print("\n⚠️  Warning: Very few churned clients in training data")
        print("   Model may not learn churn patterns well")
    
    # Get training data
    df = db.get_training_data()
    
    # Train model
    result = train_model(
        df, 
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        random_state=args.seed
    )
    
    # Save model
    model_dir = Path(__file__).parent.parent / "models"
    version = save_model(result, model_dir)
    
    # Log to database
    db.log_model_run(
        model_name='client_churn_xgboost',
        version=version,
        metrics=result['metrics'],
        feature_importance=result['feature_importance'],
        hyperparameters={
            'n_estimators': args.n_estimators,
            'test_size': args.test_size,
            'random_seed': args.seed
        }
    )
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Review model metrics above")
    print(f"  2. Run predictions: python predict_churn.py")
    print(f"  3. Push predictions to Supabase: python sync_predictions.py")


if __name__ == "__main__":
    main()
