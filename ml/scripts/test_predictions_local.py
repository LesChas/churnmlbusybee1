#!/usr/bin/env python3
"""
BusyBee ML - Test Predictions Locally (SQLite)
==============================================
Run predictions and save to LOCAL SQLite database for testing.
Does NOT touch Supabase production.

Usage:
    cd ml
    python scripts/test_predictions_local.py
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
import json
from datetime import datetime

from database.ml_db import MLDatabase


def load_model():
    """Load trained model and metadata."""
    model_dir = Path(__file__).parent.parent / "models"
    
    model_path = model_dir / "churn_model_latest.pkl"
    metadata_path = model_dir / "churn_model_latest_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run 'python train_churn_model.py' first"
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    
    return model, metadata


def create_test_clients() -> pd.DataFrame:
    """Create sample test clients for prediction."""
    test_clients = [
        {
            'client_id': 'test-001',
            'client_name': 'Acme Healthcare',
            'health': 'poor',
            'days_since_communication': 45,
            'days_since_checkin': 30,
            'current_seat_count': 2,  # Reduced seats
            'no_replacement': 1,  # Not replacing
            'client_tenure_days': 180,
        },
        {
            'client_id': 'test-002',
            'client_name': 'Sunrise Dental',
            'health': 'fair',
            'days_since_communication': 14,
            'days_since_checkin': 7,
            'current_seat_count': 5,
            'no_replacement': 0,
            'client_tenure_days': 365,
        },
        {
            'client_id': 'test-003',
            'client_name': 'Metro Medical Group',
            'health': 'good',
            'days_since_communication': 7,
            'days_since_checkin': 7,
            'current_seat_count': 8,
            'no_replacement': 0,
            'client_tenure_days': 730,
        },
        {
            'client_id': 'test-004',
            'client_name': 'Valley Orthodontics',
            'health': 'excellent',
            'days_since_communication': 3,
            'days_since_checkin': 3,
            'current_seat_count': 12,
            'no_replacement': 0,
            'client_tenure_days': 1095,
        },
        {
            'client_id': 'test-005',
            'client_name': 'Coastal Clinic',
            'health': 'poor',
            'days_since_communication': 60,
            'days_since_checkin': 45,
            'current_seat_count': 1,  # Very low
            'no_replacement': 1,  # Red flag
            'client_tenure_days': 90,  # New client
        },
    ]
    
    return pd.DataFrame(test_clients)


def prepare_features(clients_df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """Prepare features for model prediction."""
    # Health score encoding
    health_map = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
    
    features_df = clients_df.copy()
    
    # Add encoded features
    features_df['health_score'] = features_df['health'].map(health_map).fillna(2)
    features_df['checkin_frequency_encoded'] = 2  # Default bi-monthly
    features_df['replacement_urgency'] = 0
    features_df['had_pip'] = 0
    features_df['offered_discount'] = 0
    features_df['client_tenure_at_termination'] = features_df.get('client_tenure_days', 365)
    features_df['termination_type_encoded'] = 0
    features_df['geo_encoded'] = 0
    
    # Ensure no_replacement is numeric
    features_df['no_replacement'] = pd.to_numeric(features_df['no_replacement'], errors='coerce').fillna(0)
    
    # Ensure all model features exist
    for col in model_features:
        if col not in features_df.columns:
            features_df[col] = 0
    
    return features_df


def classify_risk(probability: float) -> tuple:
    """Convert probability to risk level and score."""
    score = int(probability * 100)
    
    if probability >= 0.75:
        level = 'critical'
    elif probability >= 0.50:
        level = 'high'
    elif probability >= 0.25:
        level = 'medium'
    else:
        level = 'low'
    
    return level, score


def get_top_risk_factors(features: dict, feature_importance: dict) -> list:
    """Identify top contributing risk factors."""
    risk_factors = []
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]  # Top 5
    
    for feature, importance in sorted_features:
        value = features.get(feature, 0)
        
        # Determine if this feature indicates risk
        is_risky = False
        description = ""
        
        if feature == 'current_seat_count' and value < 3:
            is_risky = True
            description = f"Low seat count ({value}) indicates potential downsizing"
        elif feature == 'no_replacement' and value == 1:
            is_risky = True
            description = "Client not requesting TM replacement"
        elif feature == 'days_since_communication' and value > 30:
            is_risky = True
            description = f"No communication in {value} days"
        elif feature == 'days_since_checkin' and value > 21:
            is_risky = True
            description = f"No check-in in {value} days"
        elif feature == 'health_score' and value <= 2:
            is_risky = True
            description = f"Low health score ({value}/4)"
        elif feature == 'client_tenure_at_termination' and value < 180:
            is_risky = True
            description = f"New client ({value} days tenure)"
        
        if is_risky:
            risk_factors.append({
                'factor': feature,
                'importance': round(importance * 100, 1),
                'value': value,
                'description': description
            })
    
    return risk_factors


def main():
    print("=" * 60)
    print("BusyBee ML - Local Prediction Test")
    print("=" * 60)
    print("\n🔒 Running in LOCAL mode - using SQLite database")
    print("   No Supabase connection required\n")
    
    # Initialize local database
    db = MLDatabase()
    
    # Load model
    print("📦 Loading trained model...")
    try:
        model, metadata = load_model()
        model_version = metadata.get('version', 'unknown')
        model_features = metadata.get('features', [])
        feature_importance = metadata.get('feature_importance', {})
        print(f"   Model version: {model_version}")
        print(f"   Features: {len(model_features)}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Create test clients
    print("\n📊 Creating test clients...")
    clients_df = create_test_clients()
    print(f"   Created {len(clients_df)} test clients")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    features_df = prepare_features(clients_df, model_features)
    X = features_df[model_features]
    
    # Run predictions
    print("\n🧠 Running predictions...")
    probabilities = model.predict_proba(X)[:, 1]
    
    # Build results
    results = []
    for i, (_, row) in enumerate(clients_df.iterrows()):
        prob = probabilities[i]
        risk_level, risk_score = classify_risk(prob)
        
        # Get feature values for this client
        client_features = {f: float(features_df.iloc[i][f]) for f in model_features}
        
        # Get top risk factors
        risk_factors = get_top_risk_factors(client_features, feature_importance)
        
        results.append({
            'client_id': row['client_id'],
            'client_name': row['client_name'],
            'churn_probability': float(prob),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'features': client_features,
            'top_risk_factors': risk_factors
        })
    
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    for _, row in results_df.iterrows():
        emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }[row['risk_level']]
        
        print(f"\n{emoji} {row['client_name']}")
        print(f"   Risk: {row['risk_level'].upper()} ({row['risk_score']}%)")
        print(f"   Churn Probability: {row['churn_probability']:.1%}")
        
        if row['top_risk_factors']:
            print("   Risk Factors:")
            for factor in row['top_risk_factors'][:3]:
                print(f"      - {factor['description']}")
    
    # Save to local database
    print("\n" + "=" * 60)
    print("SAVING TO LOCAL DATABASE")
    print("=" * 60)
    
    saved = db.save_predictions(results_df, model_version)
    print(f"\n✓ Saved {saved} predictions to: {db.db_path}")
    
    # Show statistics
    print("\n📊 Database Statistics:")
    stats = db.get_prediction_stats()
    print(f"   Total predictions: {stats['total_predictions']}")
    print(f"   Unique clients scored: {stats['unique_clients_scored']}")
    print(f"   Risk distribution: {stats['risk_distribution']}")
    print(f"   Total prediction runs: {stats['total_prediction_runs']}")
    
    # Show how to view results
    print("\n" + "=" * 60)
    print("HOW TO VIEW RESULTS")
    print("=" * 60)
    print("""
You can query the local SQLite database:

    from ml.database.ml_db import MLDatabase
    
    db = MLDatabase()
    
    # Get latest predictions
    predictions = db.get_latest_predictions()
    print(predictions)
    
    # Get high-risk clients (>50%)
    high_risk = db.get_high_risk_clients(threshold=0.5)
    print(high_risk)
    
    # Get prediction run history
    runs = db.get_prediction_runs()
    print(runs)

Or open the SQLite file directly:
    """)
    print(f"    sqlite3 {db.db_path}")
    print("""
    .tables
    SELECT * FROM client_churn_predictions ORDER BY risk_score DESC;
    SELECT * FROM prediction_runs;
    """)
    
    print("\n✅ Local test complete! When ready, run sync_predictions.py to push to Supabase.")


if __name__ == "__main__":
    main()
