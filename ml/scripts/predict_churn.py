#!/usr/bin/env python3
"""
BusyBee ML - Run Churn Predictions
===================================
Uses the trained model to predict churn for all active clients.
Saves predictions to CSV (local) - use sync_predictions.py to push to Supabase.

Usage:
    python predict_churn.py
    
    # Predict for specific clients:
    python predict_churn.py --clients "Acme Corp,Tech Inc"
"""

import sys
import os
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from database.ml_db import MLDatabase

# Load LOCAL environment first (disables external connections)
# Then fall back to parent .env if not in local mode
local_env = Path(__file__).parent.parent / '.env.local'
if local_env.exists():
    from dotenv import load_dotenv
    load_dotenv(local_env)
    print("🔒 Running in LOCAL mode - no external connections")

# Check if we should skip external connections
SKIP_EXTERNAL = os.environ.get('ML_SKIP_EXTERNAL_CONNECTIONS', 'true').lower() == 'true'


def get_supabase_client():
    """Get Supabase client (read-only for fetching active clients)."""
    # Skip if in local mode
    if SKIP_EXTERNAL:
        return None
    
    try:
        from supabase import create_client
        url = os.environ.get('VITE_SUPABASE_URL') or os.environ.get('SUPABASE_URL')
        key = os.environ.get('VITE_SUPABASE_ANON_KEY') or os.environ.get('SUPABASE_ANON_KEY')
        
        if not url or not key:
            return None
        
        return create_client(url, key)
    except ImportError:
        return None


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


def fetch_active_clients(supabase) -> pd.DataFrame:
    """Fetch active clients from Supabase."""
    if not supabase:
        print("⚠️  Supabase not configured. Using sample data.")
        return get_sample_clients()
    
    try:
        result = supabase.table('clients').select(
            'id, name, health, industry_type, client_location, start_date'
        ).eq('is_active', True).execute()
        
        if result.data:
            return pd.DataFrame(result.data)
    except Exception as e:
        print(f"⚠️  Could not fetch from Supabase: {e}")
    
    return get_sample_clients()


def get_sample_clients() -> pd.DataFrame:
    """Return sample client data for testing."""
    return pd.DataFrame([
        {'id': '1', 'name': 'Sample Client A', 'health': 'good'},
        {'id': '2', 'name': 'Sample Client B', 'health': 'fair'},
        {'id': '3', 'name': 'Sample Client C', 'health': 'poor'},
    ])


def prepare_client_features(clients_df: pd.DataFrame, db: MLDatabase) -> pd.DataFrame:
    """
    Prepare features for prediction from client data.
    Uses historical patterns from training data.
    """
    # Get average feature values from training data as defaults
    training_data = db.get_training_data()
    
    features = []
    for _, client in clients_df.iterrows():
        # Get client-specific history from training data if available
        client_history = training_data[
            training_data['client_name'].str.lower() == str(client.get('name', '')).lower()
        ]
        
        # Health score encoding
        health_map = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        health = str(client.get('health', '')).lower()
        health_score = health_map.get(health, 2)  # Default to fair
        
        # Use historical averages or sensible defaults
        if len(client_history) > 0:
            last_record = client_history.iloc[-1]
            feature_row = {
                'client_id': client.get('id'),
                'client_name': client.get('name'),
                'days_since_communication': last_record.get('days_since_communication', 14),
                'days_since_checkin': last_record.get('days_since_checkin', 14),
                'health_score': health_score,
                'checkin_frequency_encoded': last_record.get('checkin_frequency_encoded', 2),
                'no_replacement': 0,  # Not applicable for active clients
                'replacement_urgency': 0,
                'had_pip': 0,
                'offered_discount': 0,
                'current_seat_count': int(last_record.get('current_seat_count', 1) or 1),
                'client_tenure_at_termination': 365,  # Placeholder
                'termination_type_encoded': 0,
                'geo_encoded': 0
            }
        else:
            # No history - use defaults based on current health
            feature_row = {
                'client_id': client.get('id'),
                'client_name': client.get('name'),
                'days_since_communication': 14 if health_score >= 3 else 30,
                'days_since_checkin': 7 if health_score >= 3 else 21,
                'health_score': health_score,
                'checkin_frequency_encoded': 2,  # Bi-monthly
                'no_replacement': 0,
                'replacement_urgency': 0,
                'had_pip': 0,
                'offered_discount': 0,
                'current_seat_count': 1,
                'client_tenure_at_termination': 365,
                'termination_type_encoded': 0,
                'geo_encoded': 0
            }
        
        features.append(feature_row)
    
    return pd.DataFrame(features)


def predict_churn(model, features_df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """Run churn predictions."""
    # Ensure all required features exist
    for col in model_features:
        if col not in features_df.columns:
            features_df[col] = 0
    
    X = features_df[model_features]
    
    # Predict
    probabilities = model.predict_proba(X)[:, 1]
    
    # Add predictions to dataframe
    features_df['churn_probability'] = probabilities
    features_df['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['low', 'medium', 'high', 'critical']
    )
    features_df['risk_score'] = (probabilities * 100).astype(int)
    
    return features_df


def main():
    parser = argparse.ArgumentParser(description='Run churn predictions')
    parser.add_argument('--clients', type=str, help='Comma-separated client names to predict')
    parser.add_argument('--output', type=str, help='Output CSV path')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("BusyBee ML - Client Churn Predictions")
    print(f"{'='*60}\n")
    
    # Load model
    try:
        model, metadata = load_model()
        print(f"✓ Loaded model version: {metadata.get('version', 'unknown')}")
        print(f"  AUC-ROC: {metadata.get('metrics', {}).get('auc_roc', 'unknown')}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Get model features
    model_features = metadata.get('features', [
        'days_since_communication', 'days_since_checkin', 'health_score',
        'checkin_frequency_encoded', 'no_replacement', 'replacement_urgency',
        'had_pip', 'offered_discount', 'current_seat_count',
        'client_tenure_at_termination', 'termination_type_encoded', 'geo_encoded'
    ])
    
    # Connect to data sources
    db = MLDatabase()
    supabase = get_supabase_client()
    
    # Fetch clients
    print("\nFetching active clients...")
    clients_df = fetch_active_clients(supabase)
    
    if args.clients:
        client_names = [c.strip() for c in args.clients.split(',')]
        clients_df = clients_df[clients_df['name'].isin(client_names)]
    
    print(f"  Found {len(clients_df)} clients")
    
    if len(clients_df) == 0:
        print("No clients to predict. Exiting.")
        sys.exit(0)
    
    # Prepare features
    print("\nPreparing features...")
    features_df = prepare_client_features(clients_df, db)
    
    # Run predictions
    print("Running predictions...")
    results_df = predict_churn(model, features_df, model_features)
    
    # Sort by risk
    results_df = results_df.sort_values('churn_probability', ascending=False)
    
    # Display results
    print(f"\n{'='*60}")
    print("Prediction Results")
    print(f"{'='*60}\n")
    
    for _, row in results_df.iterrows():
        risk_indicator = "🔴" if row['risk_level'] == 'critical' else \
                        "🟠" if row['risk_level'] == 'high' else \
                        "🟡" if row['risk_level'] == 'medium' else "🟢"
        print(f"{risk_indicator} {row['client_name']}: {row['churn_probability']*100:.1f}% ({row['risk_level']})")
    
    print(f"\nRisk Distribution:")
    print(results_df['risk_level'].value_counts())
    
    # Save results
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else data_dir / f"predictions_{timestamp}.csv"
    
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    print(f"\nNext step: Run 'python sync_predictions.py' to push to Supabase dashboard")


if __name__ == "__main__":
    main()
