#!/usr/bin/env python3
"""
BusyBee ML - Sync Predictions to Supabase
==========================================
After training and running predictions locally, this script
pushes the results to Supabase for the dashboard to display.

This is the ONLY script that writes to Supabase.
All other ML operations use the local SQLite database.

⚠️  DISABLED BY DEFAULT in local mode (.env.local)
    To enable, set ML_SKIP_EXTERNAL_CONNECTIONS=false

Usage:
    python sync_predictions.py
    
    # Or sync from a specific predictions file:
    python sync_predictions.py --file predictions_20260409.csv
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# Load LOCAL environment first (disables external connections)
local_env = Path(__file__).parent.parent / '.env.local'
if local_env.exists():
    from dotenv import load_dotenv
    load_dotenv(local_env)

# Check if we should skip external connections
SKIP_EXTERNAL = os.environ.get('ML_SKIP_EXTERNAL_CONNECTIONS', 'true').lower() == 'true'


def get_supabase_client():
    """Get Supabase client from environment variables."""
    # Block if in local mode
    if SKIP_EXTERNAL:
        raise ValueError(
            "\n🔒 BLOCKED: Running in LOCAL mode - Supabase sync is disabled.\n"
            "   This prevents accidental writes to production.\n\n"
            "   To enable sync, edit ml/.env.local and set:\n"
            "   ML_SKIP_EXTERNAL_CONNECTIONS=false\n"
        )
    
    try:
        from supabase import create_client
    except ImportError:
        raise ImportError("supabase package not installed. Run: pip install supabase")
    
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / '.env')
    
    url = os.environ.get('VITE_SUPABASE_URL') or os.environ.get('SUPABASE_URL')
    key = os.environ.get('VITE_SUPABASE_ANON_KEY') or os.environ.get('SUPABASE_SERVICE_KEY')
    
    if not url or not key:
        raise ValueError(
            "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_SERVICE_KEY "
            "in your .env file"
        )
    
    return create_client(url, key)


def load_predictions(predictions_file: Path = None) -> pd.DataFrame:
    """Load predictions from file or generate from model."""
    if predictions_file and predictions_file.exists():
        return pd.read_csv(predictions_file)
    
    # Try to load latest predictions
    data_dir = Path(__file__).parent.parent / "data"
    prediction_files = list(data_dir.glob("predictions_*.csv"))
    
    if prediction_files:
        latest = max(prediction_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading predictions from: {latest}")
        return pd.read_csv(latest)
    
    raise FileNotFoundError(
        "No predictions file found. Run 'python predict_churn.py' first"
    )


def sync_to_supabase(df: pd.DataFrame, supabase):
    """Sync predictions to Supabase."""
    
    # Get model metadata
    metadata_path = Path(__file__).parent.parent / "models" / "churn_model_latest_metadata.json"
    model_version = "unknown"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            model_version = metadata.get('version', 'unknown')
    
    # Prepare records for Supabase
    records = []
    for _, row in df.iterrows():
        # Determine risk level
        prob = row['churn_probability']
        if prob >= 0.75:
            risk_level = 'critical'
        elif prob >= 0.5:
            risk_level = 'high'
        elif prob >= 0.25:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        record = {
            'client_id': row.get('client_id'),
            'client_name': row['client_name'],
            'churn_probability': float(prob),
            'risk_level': risk_level,
            'risk_score': int(prob * 100),
            'prediction_date': datetime.now().isoformat(),
            'model_version': model_version,
            'model_name': 'client_churn_xgboost',
            'features': json.dumps({
                'days_since_communication': row.get('days_since_communication'),
                'days_since_checkin': row.get('days_since_checkin'),
                'health_score': row.get('health_score'),
                'seat_count': row.get('current_seat_count')
            }),
            'top_risk_factors': json.dumps(get_risk_factors(row))
        }
        records.append(record)
    
    print(f"\nSyncing {len(records)} predictions to Supabase...")
    
    # Insert in batches
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            result = supabase.table('client_churn_predictions').upsert(
                batch, 
                on_conflict='client_id,prediction_date'
            ).execute()
            print(f"  Synced batch {i//batch_size + 1}: {len(batch)} records")
        except Exception as e:
            print(f"  Error syncing batch: {e}")
            # Try inserting without upsert
            try:
                result = supabase.table('client_churn_predictions').insert(batch).execute()
                print(f"  Inserted batch {i//batch_size + 1}: {len(batch)} records")
            except Exception as e2:
                print(f"  Failed to insert batch: {e2}")
    
    print(f"\n✓ Sync complete!")


def get_risk_factors(row) -> list:
    """Generate risk factors based on feature values."""
    factors = []
    
    # Communication gap
    days_comm = row.get('days_since_communication', 0)
    if days_comm > 30:
        factors.append({
            'factor': 'Communication Gap',
            'impact': 'high' if days_comm > 60 else 'medium',
            'description': f'No communication in {int(days_comm)} days'
        })
    
    # Missed check-ins
    days_checkin = row.get('days_since_checkin', 0)
    if days_checkin > 14:
        factors.append({
            'factor': 'Missed Check-ins',
            'impact': 'high' if days_checkin > 30 else 'medium',
            'description': f'Last official check-in was {int(days_checkin)} days ago'
        })
    
    # Poor health
    health = row.get('health_score', 0)
    if health <= 2:
        factors.append({
            'factor': 'Client Health Declining',
            'impact': 'high' if health == 1 else 'medium',
            'description': 'Recent health status was Fair or Poor'
        })
    
    # No replacement
    if row.get('no_replacement', 0) == 1:
        factors.append({
            'factor': 'No Replacement Requested',
            'impact': 'high',
            'description': 'Client did not request team member replacement'
        })
    
    # Had PIP
    if row.get('had_pip', 0) == 1:
        factors.append({
            'factor': 'Performance Issues',
            'impact': 'medium',
            'description': 'Performance Improvement Plan was implemented'
        })
    
    return factors[:5]  # Return top 5 factors


def main():
    parser = argparse.ArgumentParser(description='Sync ML predictions to Supabase')
    parser.add_argument('--file', type=str, help='Predictions CSV file path')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be synced without syncing')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("BusyBee ML - Sync Predictions to Supabase")
    print(f"{'='*60}\n")
    
    # Load predictions
    predictions_file = Path(args.file) if args.file else None
    try:
        df = load_predictions(predictions_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} predictions")
    print(f"\nRisk distribution:")
    
    # Show distribution
    df['risk_level'] = df['churn_probability'].apply(
        lambda p: 'critical' if p >= 0.75 else 'high' if p >= 0.5 else 'medium' if p >= 0.25 else 'low'
    )
    print(df['risk_level'].value_counts())
    
    if args.dry_run:
        print("\n[DRY RUN] Would sync above predictions to Supabase")
        return
    
    # Connect to Supabase
    try:
        supabase = get_supabase_client()
        print("\n✓ Connected to Supabase")
    except Exception as e:
        print(f"\nError connecting to Supabase: {e}")
        print("Make sure your .env file has SUPABASE_URL and SUPABASE_SERVICE_KEY")
        sys.exit(1)
    
    # Sync predictions
    sync_to_supabase(df, supabase)
    
    print(f"\n{'='*60}")
    print("Predictions are now visible in the ML Dashboard!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
