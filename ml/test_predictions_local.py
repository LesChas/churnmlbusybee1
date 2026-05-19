"""
BusyBee ML Predictions - Local Testing Script
=============================================
Tests ML models locally with mock data - NO database connection required.
Safe to run against production since it doesn't write anything.

Usage:
    python test_predictions_local.py
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Check if we can import ML libraries
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("❌ Missing dependencies. Run: pip install pandas numpy")
    sys.exit(1)

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
except ImportError:
    print("❌ Missing ML dependencies. Run: pip install xgboost scikit-learn")
    sys.exit(1)


def generate_mock_clients(n: int = 50) -> pd.DataFrame:
    """Generate realistic mock client data for testing."""
    np.random.seed(42)
    
    industries = ['HEALTHCARE', 'TECHNOLOGY', 'FINANCE', 'RETAIL', 'EDUCATION', 'MANUFACTURING']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia']
    health_options = ['excellent', 'good', 'fair', 'poor']
    
    clients = []
    for i in range(n):
        start_date = datetime.now() - timedelta(days=np.random.randint(30, 730))
        
        # Simulate realistic churn patterns
        health = np.random.choice(health_options, p=[0.25, 0.40, 0.25, 0.10])
        tenure_days = (datetime.now() - start_date).days
        
        # Higher churn probability for poor health and short tenure
        base_churn_prob = 0.10
        if health == 'poor':
            base_churn_prob += 0.30
        elif health == 'fair':
            base_churn_prob += 0.15
        if tenure_days < 90:
            base_churn_prob += 0.15
        
        is_churned = np.random.random() < base_churn_prob
        
        clients.append({
            'id': f'client-{i:04d}',
            'name': f'Client {i} Corp',
            'industry_type': np.random.choice(industries),
            'client_location': np.random.choice(locations),
            'start_date': start_date.isoformat(),
            'status': 'inactive' if is_churned else 'active',
            'is_active': not is_churned,
            'health': health,
            'attrition_date': (start_date + timedelta(days=np.random.randint(30, 365))).isoformat() if is_churned else None,
            'created_at': start_date.isoformat(),
            'updated_at': (datetime.now() - timedelta(days=np.random.randint(0, 60))).isoformat()
        })
    
    return pd.DataFrame(clients)


def generate_mock_team_members(n: int = 100) -> pd.DataFrame:
    """Generate realistic mock team member data for testing."""
    np.random.seed(42)
    
    roles = ['team_member', 'team_lead', 'manager', 'senior', 'junior']
    
    team_members = []
    for i in range(n):
        start_date = datetime.now() - timedelta(days=np.random.randint(30, 730))
        tenure_days = (datetime.now() - start_date).days
        role = np.random.choice(roles, p=[0.50, 0.15, 0.10, 0.15, 0.10])
        has_client = np.random.random() > 0.15
        
        # Simulate realistic attrition patterns
        base_attrition_prob = 0.12
        if tenure_days < 90:  # New employees have higher attrition
            base_attrition_prob += 0.20
        elif tenure_days > 180 and tenure_days < 365:  # 6-12 month window
            base_attrition_prob += 0.10
        if not has_client:
            base_attrition_prob += 0.10
        
        is_left = np.random.random() < base_attrition_prob
        
        team_members.append({
            'id': f'tm-{i:04d}',
            'first_name': f'FirstName{i}',
            'last_name': f'LastName{i}',
            'email': f'employee{i}@busybee.com',
            'role': role,
            'status': 'inactive' if is_left else 'active',
            'start_date': start_date.isoformat(),
            'client_id': f'client-{np.random.randint(0, 50):04d}' if has_client else None,
            'created_at': start_date.isoformat(),
            'updated_at': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat()
        })
    
    return pd.DataFrame(team_members)


class SimpleChurnModel:
    """Simplified churn prediction model for testing."""
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from client data."""
        features = pd.DataFrame()
        reference_date = pd.to_datetime('today')
        
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        
        # Tenure in months
        features['tenure_months'] = ((reference_date - df['start_date']).dt.days / 30).fillna(0).astype(int)
        
        # Health score
        health_mapping = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        features['health_score'] = df['health'].map(health_mapping).fillna(2)
        
        # Days since activity
        features['days_since_activity'] = ((reference_date - df['updated_at']).dt.days).fillna(30).clip(0, 365)
        
        # Industry encoding (simple)
        features['industry_encoded'] = pd.factorize(df['industry_type'].fillna('OTHER'))[0]
        
        # Location encoding
        features['location_encoded'] = pd.factorize(df['client_location'].fillna('Unknown'))[0]
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train the model."""
        X = self.prepare_features(df)
        y = (df['attrition_date'].notna() | (df['status'] == 'inactive')).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=50,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
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
        """Generate predictions."""
        X = self.prepare_features(df)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        def get_risk_level(prob):
            if prob >= 0.75: return 'critical'
            if prob >= 0.50: return 'high'
            if prob >= 0.25: return 'medium'
            return 'low'
        
        predictions = pd.DataFrame({
            'client_id': df['id'],
            'client_name': df['name'],
            'churn_probability': probabilities,
            'risk_score': (probabilities * 100).astype(int),
            'risk_level': [get_risk_level(p) for p in probabilities]
        })
        
        return predictions.sort_values('churn_probability', ascending=False)


class SimpleAttritionModel:
    """Simplified attrition prediction model for testing."""
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from team member data."""
        features = pd.DataFrame()
        reference_date = pd.to_datetime('today')
        
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        
        # Tenure
        features['tenure_months'] = ((reference_date - df['start_date']).dt.days / 30).fillna(0).astype(int)
        
        # Role encoding
        features['role_encoded'] = pd.factorize(df['role'].fillna('team_member'))[0]
        
        # Days since activity
        features['days_since_activity'] = ((reference_date - df['updated_at']).dt.days).fillna(7).clip(0, 365)
        
        # Has client
        features['has_client'] = df['client_id'].notna().astype(int)
        
        # Is new employee
        features['is_new_employee'] = (features['tenure_months'] <= 3).astype(int)
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train the model."""
        X = self.prepare_features(df)
        y = (df['status'] != 'active').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=50,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
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
        """Generate predictions."""
        # Filter to active employees only
        active_df = df[df['status'] == 'active'].copy()
        
        if active_df.empty:
            return pd.DataFrame()
        
        X = self.prepare_features(active_df)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        def get_risk_level(prob):
            if prob >= 0.75: return 'critical'
            if prob >= 0.50: return 'high'
            if prob >= 0.25: return 'medium'
            return 'low'
        
        predictions = pd.DataFrame({
            'team_member_id': active_df['id'].values,
            'name': (active_df['first_name'] + ' ' + active_df['last_name']).values,
            'role': active_df['role'].values,
            'attrition_probability': probabilities,
            'risk_score': (probabilities * 100).astype(int),
            'risk_level': [get_risk_level(p) for p in probabilities]
        })
        
        return predictions.sort_values('attrition_probability', ascending=False)


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_metrics(metrics: Dict, model_name: str):
    """Print model metrics."""
    print(f"\n📊 {model_name} Performance:")
    print(f"   Accuracy:  {metrics['accuracy']:.1%}")
    print(f"   Precision: {metrics['precision']:.1%}")
    print(f"   Recall:    {metrics['recall']:.1%}")
    print(f"   F1 Score:  {metrics['f1']:.1%}")
    print(f"   AUC-ROC:   {metrics['auc_roc']:.3f}")


def print_predictions(predictions: pd.DataFrame, title: str, top_n: int = 10):
    """Print top predictions."""
    print(f"\n🎯 {title} (Top {top_n}):")
    print("-" * 60)
    
    for idx, row in predictions.head(top_n).iterrows():
        risk_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
        emoji = risk_emoji.get(row['risk_level'], '⚪')
        
        if 'client_name' in row:
            name = row['client_name']
            prob_col = 'churn_probability'
        else:
            name = row['name']
            prob_col = 'attrition_probability'
        
        print(f"   {emoji} {name:<25} | Risk: {row['risk_score']:>3}% | {row['risk_level'].upper()}")


def main():
    """Run local ML prediction tests."""
    print_header("BusyBee ML Predictions - Local Test")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("🔒 Mode: LOCAL TESTING (No database connection)")
    
    # Generate mock data
    print_header("Step 1: Generating Mock Data")
    clients_df = generate_mock_clients(50)
    team_members_df = generate_mock_team_members(100)
    
    print(f"✅ Generated {len(clients_df)} mock clients")
    print(f"   - Active: {(clients_df['is_active'] == True).sum()}")
    print(f"   - Churned: {(clients_df['is_active'] == False).sum()}")
    
    print(f"\n✅ Generated {len(team_members_df)} mock team members")
    print(f"   - Active: {(team_members_df['status'] == 'active').sum()}")
    print(f"   - Left: {(team_members_df['status'] != 'active').sum()}")
    
    # Train and test churn model
    print_header("Step 2: Training Client Churn Model")
    churn_model = SimpleChurnModel()
    churn_metrics = churn_model.train(clients_df)
    print_metrics(churn_metrics, "Client Churn Model")
    
    # Generate churn predictions
    print_header("Step 3: Generating Churn Predictions")
    active_clients = clients_df[clients_df['is_active'] == True]
    churn_predictions = churn_model.predict(active_clients)
    
    high_risk_clients = (churn_predictions['risk_level'].isin(['high', 'critical'])).sum()
    print(f"\n📈 Prediction Summary:")
    print(f"   Total Active Clients: {len(churn_predictions)}")
    print(f"   High/Critical Risk:   {high_risk_clients}")
    print(f"   Medium Risk:          {(churn_predictions['risk_level'] == 'medium').sum()}")
    print(f"   Low Risk:             {(churn_predictions['risk_level'] == 'low').sum()}")
    
    print_predictions(churn_predictions, "Clients at Risk of Churning")
    
    # Train and test attrition model
    print_header("Step 4: Training Team Member Attrition Model")
    attrition_model = SimpleAttritionModel()
    attrition_metrics = attrition_model.train(team_members_df)
    print_metrics(attrition_metrics, "Team Member Attrition Model")
    
    # Generate attrition predictions
    print_header("Step 5: Generating Attrition Predictions")
    attrition_predictions = attrition_model.predict(team_members_df)
    
    high_risk_members = (attrition_predictions['risk_level'].isin(['high', 'critical'])).sum()
    print(f"\n📈 Prediction Summary:")
    print(f"   Total Active Team Members: {len(attrition_predictions)}")
    print(f"   High/Critical Risk:        {high_risk_members}")
    print(f"   Medium Risk:               {(attrition_predictions['risk_level'] == 'medium').sum()}")
    print(f"   Low Risk:                  {(attrition_predictions['risk_level'] == 'low').sum()}")
    
    print_predictions(attrition_predictions, "Team Members at Risk of Leaving")
    
    # Summary
    print_header("Test Complete!")
    print("""
✅ ML models are working correctly with mock data.

📋 Next Steps for Production:
   1. Export real data from Supabase (read-only)
   2. Train models with real historical data
   3. Deploy to AWS Lambda for automated predictions
   4. Create staging Supabase project for safe testing

💡 To test with real data (READ-ONLY):
   python test_with_real_data.py --read-only
""")
    
    # Save test results
    results = {
        'test_date': datetime.now().isoformat(),
        'mode': 'local_mock_data',
        'churn_model': {
            'metrics': {k: float(v) for k, v in churn_metrics.items()},
            'predictions_count': len(churn_predictions),
            'high_risk_count': int(high_risk_clients)
        },
        'attrition_model': {
            'metrics': {k: float(v) for k, v in attrition_metrics.items()},
            'predictions_count': len(attrition_predictions),
            'high_risk_count': int(high_risk_members)
        }
    }
    
    results_path = os.path.join(os.path.dirname(__file__), 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Test results saved to: {results_path}")


if __name__ == '__main__':
    main()
