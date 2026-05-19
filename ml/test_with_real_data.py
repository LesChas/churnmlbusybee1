"""
BusyBee ML Predictions - Read-Only Production Test
===================================================
Safely tests ML models with REAL production data.
- READ-ONLY: Never writes to the database
- Displays predictions in console only
- Saves results to local JSON file

Usage:
    python test_with_real_data.py --supabase-url "YOUR_URL" --supabase-key "YOUR_ANON_KEY"
    
    Or set environment variables:
    $env:SUPABASE_URL = "your-url"
    $env:SUPABASE_ANON_KEY = "your-anon-key"  # Use ANON key, not service key!
    python test_with_real_data.py
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Check dependencies
try:
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install pandas numpy xgboost scikit-learn")
    sys.exit(1)

try:
    from supabase import create_client, Client
except ImportError:
    print("❌ Supabase client not installed. Run: pip install supabase")
    sys.exit(1)


class ReadOnlySupabaseClient:
    """Read-only wrapper for Supabase - prevents accidental writes."""
    
    def __init__(self, url: str, key: str):
        self._client = create_client(url, key)
        print("🔒 Connected to Supabase in READ-ONLY mode")
    
    def fetch_clients(self) -> pd.DataFrame:
        """Fetch clients (READ-ONLY)."""
        response = self._client.table('clients').select(
            'id, name, industry_type, client_location, start_date, '
            'status, is_active, health, attrition_date, created_at, updated_at'
        ).execute()
        
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    
    def fetch_team_members(self) -> pd.DataFrame:
        """Fetch team members (READ-ONLY)."""
        response = self._client.table('team_members').select(
            'id, first_name, last_name, email, role, status, '
            'start_date, client_id, created_at, updated_at'
        ).execute()
        
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    
    # Block all write operations
    def insert(self, *args, **kwargs):
        raise PermissionError("❌ BLOCKED: Write operations are disabled in read-only mode")
    
    def update(self, *args, **kwargs):
        raise PermissionError("❌ BLOCKED: Write operations are disabled in read-only mode")
    
    def delete(self, *args, **kwargs):
        raise PermissionError("❌ BLOCKED: Write operations are disabled in read-only mode")
    
    def upsert(self, *args, **kwargs):
        raise PermissionError("❌ BLOCKED: Write operations are disabled in read-only mode")


class ChurnModel:
    """Client churn prediction model."""
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        reference_date = pd.to_datetime('today')
        
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        
        start_dates = df['start_date'].fillna(df['created_at'])
        
        features['tenure_months'] = ((reference_date - start_dates).dt.days / 30).fillna(0).astype(int)
        
        health_mapping = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        features['health_score'] = df['health'].str.lower().map(health_mapping).fillna(2)
        
        features['days_since_activity'] = ((reference_date - df['updated_at']).dt.days).fillna(30).clip(0, 365)
        features['industry_encoded'] = pd.factorize(df['industry_type'].fillna('OTHER'))[0]
        features['location_encoded'] = pd.factorize(df['client_location'].fillna('Unknown'))[0]
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def train(self, df: pd.DataFrame) -> Dict:
        X = self.prepare_features(df.copy())
        y = (df['attrition_date'].notna() | (df['status'] == 'inactive') | (df['is_active'] == False)).astype(int)
        
        if len(y.unique()) < 2:
            print("⚠️  Warning: Not enough variance in target (all same class)")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc_roc': 0.5}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.1, n_estimators=50,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        self.model.fit(X_train, y_train, verbose=False)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        active_df = df[df['is_active'] == True].copy()
        if active_df.empty:
            return pd.DataFrame()
        
        X = self.prepare_features(active_df)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        def risk_level(p):
            if p >= 0.75: return 'critical'
            if p >= 0.50: return 'high'
            if p >= 0.25: return 'medium'
            return 'low'
        
        return pd.DataFrame({
            'client_id': active_df['id'].values,
            'client_name': active_df['name'].values,
            'industry': active_df['industry_type'].values,
            'health': active_df['health'].values,
            'churn_probability': probabilities,
            'risk_score': (probabilities * 100).astype(int),
            'risk_level': [risk_level(p) for p in probabilities]
        }).sort_values('churn_probability', ascending=False)


class AttritionModel:
    """Team member attrition prediction model."""
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        reference_date = pd.to_datetime('today')
        
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        
        start_dates = df['start_date'].fillna(df['created_at'])
        
        features['tenure_months'] = ((reference_date - start_dates).dt.days / 30).fillna(0).astype(int)
        features['role_encoded'] = pd.factorize(df['role'].fillna('team_member'))[0]
        features['days_since_activity'] = ((reference_date - df['updated_at']).dt.days).fillna(7).clip(0, 365)
        features['has_client'] = df['client_id'].notna().astype(int)
        features['is_new_employee'] = (features['tenure_months'] <= 3).astype(int)
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def train(self, df: pd.DataFrame) -> Dict:
        X = self.prepare_features(df.copy())
        y = (df['status'] != 'active').astype(int)
        
        if len(y.unique()) < 2:
            print("⚠️  Warning: Not enough variance in target")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc_roc': 0.5}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model = xgb.XGBClassifier(
            max_depth=3, learning_rate=0.1, n_estimators=50,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        self.model.fit(X_train, y_train, verbose=False)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        active_df = df[df['status'] == 'active'].copy()
        if active_df.empty:
            return pd.DataFrame()
        
        X = self.prepare_features(active_df)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        def risk_level(p):
            if p >= 0.75: return 'critical'
            if p >= 0.50: return 'high'
            if p >= 0.25: return 'medium'
            return 'low'
        
        return pd.DataFrame({
            'team_member_id': active_df['id'].values,
            'name': (active_df['first_name'].fillna('') + ' ' + active_df['last_name'].fillna('')).values,
            'role': active_df['role'].values,
            'attrition_probability': probabilities,
            'risk_score': (probabilities * 100).astype(int),
            'risk_level': [risk_level(p) for p in probabilities]
        }).sort_values('attrition_probability', ascending=False)


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_table(df: pd.DataFrame, title: str, columns: List[str], top_n: int = 15):
    """Print DataFrame as formatted table."""
    print(f"\n{title}")
    print("-" * 70)
    
    risk_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
    
    for idx, row in df.head(top_n).iterrows():
        emoji = risk_emoji.get(row.get('risk_level', 'low'), '⚪')
        
        name_col = 'client_name' if 'client_name' in row else 'name'
        prob_col = 'churn_probability' if 'churn_probability' in row else 'attrition_probability'
        
        name = str(row.get(name_col, 'Unknown'))[:25]
        prob = row.get(prob_col, 0) * 100
        risk = row.get('risk_level', 'low').upper()
        
        print(f"  {emoji} {name:<25} | {prob:>5.1f}% | {risk}")


def main():
    parser = argparse.ArgumentParser(description='Test ML predictions with real data (READ-ONLY)')
    parser.add_argument('--supabase-url', type=str, default=os.environ.get('SUPABASE_URL'))
    parser.add_argument('--supabase-key', type=str, default=os.environ.get('SUPABASE_ANON_KEY'))
    args = parser.parse_args()
    
    print_header("BusyBee ML Predictions - Production Test (READ-ONLY)")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("🔒 Mode: READ-ONLY (No database writes)")
    
    # Check credentials
    if not args.supabase_url or not args.supabase_key:
        print("\n❌ Supabase credentials required!")
        print("\nOption 1 - Command line:")
        print('  python test_with_real_data.py --supabase-url "YOUR_URL" --supabase-key "YOUR_ANON_KEY"')
        print("\nOption 2 - Environment variables:")
        print('  $env:SUPABASE_URL = "your-url"')
        print('  $env:SUPABASE_ANON_KEY = "your-anon-key"')
        print('  python test_with_real_data.py')
        return
    
    # Connect (read-only)
    print_header("Connecting to Supabase (READ-ONLY)")
    db = ReadOnlySupabaseClient(args.supabase_url, args.supabase_key)
    
    # Fetch data
    print_header("Fetching Data")
    clients_df = db.fetch_clients()
    team_members_df = db.fetch_team_members()
    
    print(f"✅ Fetched {len(clients_df)} clients")
    if not clients_df.empty:
        active_clients = clients_df['is_active'].sum() if 'is_active' in clients_df.columns else 0
        print(f"   - Active: {active_clients}")
        print(f"   - Churned: {len(clients_df) - active_clients}")
    
    print(f"\n✅ Fetched {len(team_members_df)} team members")
    if not team_members_df.empty:
        active_members = (team_members_df['status'] == 'active').sum()
        print(f"   - Active: {active_members}")
        print(f"   - Inactive: {len(team_members_df) - active_members}")
    
    results = {
        'test_date': datetime.now().isoformat(),
        'mode': 'read_only_production',
        'data': {
            'clients_total': len(clients_df),
            'team_members_total': len(team_members_df)
        }
    }
    
    # Train and predict - Churn
    if len(clients_df) >= 10:
        print_header("Client Churn Predictions")
        churn_model = ChurnModel()
        churn_metrics = churn_model.train(clients_df)
        
        print(f"\n📊 Model Performance:")
        print(f"   Accuracy: {churn_metrics['accuracy']:.1%}")
        print(f"   AUC-ROC:  {churn_metrics['auc_roc']:.3f}")
        
        churn_predictions = churn_model.predict(clients_df)
        
        if not churn_predictions.empty:
            high_risk = (churn_predictions['risk_level'].isin(['high', 'critical'])).sum()
            print(f"\n📈 Risk Distribution:")
            print(f"   High/Critical: {high_risk}")
            print(f"   Medium: {(churn_predictions['risk_level'] == 'medium').sum()}")
            print(f"   Low: {(churn_predictions['risk_level'] == 'low').sum()}")
            
            print_table(churn_predictions, "🎯 Clients at Risk of Churning:", ['client_name', 'risk_score', 'risk_level'])
            
            results['churn_predictions'] = {
                'total': len(churn_predictions),
                'high_risk': int(high_risk),
                'top_10': churn_predictions.head(10).to_dict(orient='records')
            }
    else:
        print("\n⚠️  Not enough clients to train churn model (need at least 10)")
    
    # Train and predict - Attrition
    if len(team_members_df) >= 10:
        print_header("Team Member Attrition Predictions")
        attrition_model = AttritionModel()
        attrition_metrics = attrition_model.train(team_members_df)
        
        print(f"\n📊 Model Performance:")
        print(f"   Accuracy: {attrition_metrics['accuracy']:.1%}")
        print(f"   AUC-ROC:  {attrition_metrics['auc_roc']:.3f}")
        
        attrition_predictions = attrition_model.predict(team_members_df)
        
        if not attrition_predictions.empty:
            high_risk = (attrition_predictions['risk_level'].isin(['high', 'critical'])).sum()
            print(f"\n📈 Risk Distribution:")
            print(f"   High/Critical: {high_risk}")
            print(f"   Medium: {(attrition_predictions['risk_level'] == 'medium').sum()}")
            print(f"   Low: {(attrition_predictions['risk_level'] == 'low').sum()}")
            
            print_table(attrition_predictions, "🎯 Team Members at Risk of Leaving:", ['name', 'risk_score', 'risk_level'])
            
            results['attrition_predictions'] = {
                'total': len(attrition_predictions),
                'high_risk': int(high_risk),
                'top_10': attrition_predictions.head(10).to_dict(orient='records')
            }
    else:
        print("\n⚠️  Not enough team members to train attrition model (need at least 10)")
    
    # Save results
    results_path = os.path.join(os.path.dirname(__file__), 'production_test_results.json')
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2, default=str)
    
    print_header("Test Complete!")
    print(f"""
✅ Predictions generated successfully (READ-ONLY mode)
📁 Results saved to: {results_path}

⚠️  IMPORTANT: These predictions were NOT written to the database.
    They are saved locally for review only.

📋 To deploy predictions to production:
    1. Review the results in the JSON file
    2. Set up the prediction tables in Supabase (staging first!)
    3. Deploy Lambda function to write predictions
""")


if __name__ == '__main__':
    main()
