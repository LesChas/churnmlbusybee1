"""
BusyBee ML - SHAP Explainability
=================================
Generates per-client explanations for churn predictions using SHAP.
100% local - no database connections, no live data touched.

Usage:
    python shap_explain.py

Output:
    - shap_results.json (per-client top factors + global feature importance)
    - Console summary of "Why They Might Leave" for each at-risk client
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    import shap
except ImportError:
    print("❌ SHAP not installed. Run: pip install shap")
    sys.exit(1)


# Human-readable labels for each feature
FEATURE_LABELS = {
    'tenure_months': 'Short client tenure',
    'health_score': 'Low satisfaction/health score',
    'days_since_activity': 'No recent engagement',
    'industry_encoded': 'Industry risk factor',
    'location_encoded': 'Regional risk pattern',
}

# Descriptions for negative vs positive contribution
FEATURE_REASONS = {
    'tenure_months': {
        'high_risk': 'Client is relatively new (short tenure)',
        'low_risk': 'Long-standing client relationship',
    },
    'health_score': {
        'high_risk': 'Satisfaction/health score is low',
        'low_risk': 'Strong satisfaction/health score',
    },
    'days_since_activity': {
        'high_risk': 'No recent platform activity',
        'low_risk': 'Actively engaged recently',
    },
    'industry_encoded': {
        'high_risk': 'Industry has higher churn tendency',
        'low_risk': 'Industry is stable',
    },
    'location_encoded': {
        'high_risk': 'Region shows higher churn patterns',
        'low_risk': 'Region is stable',
    },
}


def generate_mock_clients(n: int = 50) -> pd.DataFrame:
    """Generate realistic mock client data."""
    np.random.seed(42)

    industries = ['HEALTHCARE', 'TECHNOLOGY', 'FINANCE', 'RETAIL', 'EDUCATION', 'MANUFACTURING']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia']
    health_options = ['excellent', 'good', 'fair', 'poor']

    clients = []
    for i in range(n):
        start_date = datetime.now() - timedelta(days=np.random.randint(30, 730))
        health = np.random.choice(health_options, p=[0.25, 0.40, 0.25, 0.10])
        tenure_days = (datetime.now() - start_date).days

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
            'updated_at': (datetime.now() - timedelta(days=np.random.randint(0, 60))).isoformat(),
        })

    return pd.DataFrame(clients)


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and return (X, feature_names)."""
    features = pd.DataFrame()
    reference_date = pd.to_datetime('today')

    df = df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')

    features['tenure_months'] = ((reference_date - df['start_date']).dt.days / 30).fillna(0).astype(int)
    health_mapping = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
    features['health_score'] = df['health'].map(health_mapping).fillna(2)
    features['days_since_activity'] = ((reference_date - df['updated_at']).dt.days).fillna(30).clip(0, 365)
    features['industry_encoded'] = pd.factorize(df['industry_type'].fillna('OTHER'))[0]
    features['location_encoded'] = pd.factorize(df['client_location'].fillna('Unknown'))[0]

    return features, features.columns.tolist()


def get_risk_level(prob: float) -> str:
    if prob >= 0.75:
        return 'critical'
    if prob >= 0.50:
        return 'high'
    if prob >= 0.25:
        return 'medium'
    return 'low'


def explain_client(shap_values_row: np.ndarray, feature_names: List[str], top_n: int = 3) -> List[Dict]:
    """Get top N reasons driving a client's risk, with human-readable labels."""
    contributions = list(zip(feature_names, shap_values_row))
    # Sort by absolute SHAP value (biggest impact first)
    contributions.sort(key=lambda x: -abs(x[1]))

    reasons = []
    for feat_name, shap_val in contributions[:top_n]:
        direction = 'high_risk' if shap_val > 0 else 'low_risk'
        reason_text = FEATURE_REASONS.get(feat_name, {}).get(direction, FEATURE_LABELS.get(feat_name, feat_name))
        reasons.append({
            'feature': feat_name,
            'label': FEATURE_LABELS.get(feat_name, feat_name),
            'reason': reason_text,
            'impact': round(float(shap_val), 4),
            'direction': 'increases_risk' if shap_val > 0 else 'decreases_risk',
        })

    return reasons


def main():
    print("=" * 60)
    print("  BusyBee ML - SHAP Explainability (Local)")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("🔒 Mode: 100% LOCAL - no database connections\n")

    # --- Generate data & train model ---
    print("📦 Generating mock client data...")
    clients_df = generate_mock_clients(50)
    print(f"   {len(clients_df)} clients ({(clients_df['is_active']).sum()} active, {(~clients_df['is_active']).sum()} churned)\n")

    X_all, feature_names = prepare_features(clients_df)
    y_all = (clients_df['attrition_date'].notna() | (clients_df['status'] == 'inactive')).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    print("🏋️ Training XGBoost model...")
    model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    )
    model.fit(X_train, y_train, verbose=False)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"   Accuracy: {acc:.1%} | AUC-ROC: {auc:.3f}\n")

    # --- SHAP Explainability ---
    print("🔍 Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)

    # Predict on active clients only
    active_mask = clients_df['is_active'] == True
    active_df = clients_df[active_mask].reset_index(drop=True)
    X_active, _ = prepare_features(active_df)

    shap_values = explainer.shap_values(X_active)
    probabilities = model.predict_proba(X_active)[:, 1]

    print(f"   ✅ SHAP values computed for {len(active_df)} active clients\n")

    # --- Global Feature Importance (mean |SHAP|) ---
    global_importance = []
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    for feat, importance in sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1]):
        global_importance.append({
            'feature': feat,
            'label': FEATURE_LABELS.get(feat, feat),
            'mean_abs_shap': round(float(importance), 4),
        })

    print("📊 Global Feature Importance (mean |SHAP|):")
    print("-" * 50)
    for item in global_importance:
        bar = "█" * int(item['mean_abs_shap'] * 50)
        print(f"   {item['label']:<35} {item['mean_abs_shap']:.4f} {bar}")

    # --- Per-client explanations ---
    print("\n" + "=" * 60)
    print("  WHY THEY MIGHT LEAVE - Top At-Risk Clients")
    print("=" * 60)

    client_explanations = []
    for i in range(len(active_df)):
        prob = probabilities[i]
        risk_level = get_risk_level(prob)
        reasons = explain_client(shap_values[i], feature_names, top_n=3)

        entry = {
            'client_id': active_df.iloc[i]['id'],
            'client_name': active_df.iloc[i]['name'],
            'risk_score': round(float(prob), 4),
            'risk_level': risk_level,
            'top_factors': reasons,
        }
        client_explanations.append(entry)

    # Sort by risk (highest first)
    client_explanations.sort(key=lambda x: -x['risk_score'])

    # Print top 10 at-risk
    risk_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
    for entry in client_explanations[:10]:
        emoji = risk_emoji.get(entry['risk_level'], '⚪')
        print(f"\n{emoji} {entry['client_name']} — {entry['risk_level'].upper()} ({entry['risk_score']:.0%})")
        # Only show risk-increasing factors
        risk_reasons = [r for r in entry['top_factors'] if r['direction'] == 'increases_risk']
        if not risk_reasons:
            risk_reasons = entry['top_factors'][:2]
        for reason in risk_reasons:
            print(f"   → {reason['reason']} (impact: {reason['impact']:+.4f})")

    # --- Save results ---
    output = {
        'generated_at': datetime.now().isoformat(),
        'mode': 'local_mock_data',
        'model_performance': {'accuracy': round(acc, 4), 'auc_roc': round(auc, 4)},
        'global_feature_importance': global_importance,
        'client_explanations': client_explanations,
        'summary': {
            'total_active_clients': len(active_df),
            'critical_risk': sum(1 for c in client_explanations if c['risk_level'] == 'critical'),
            'high_risk': sum(1 for c in client_explanations if c['risk_level'] == 'high'),
            'medium_risk': sum(1 for c in client_explanations if c['risk_level'] == 'medium'),
            'low_risk': sum(1 for c in client_explanations if c['risk_level'] == 'low'),
        },
    }

    output_path = os.path.join(os.path.dirname(__file__), 'shap_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\n💾 Results saved to: {output_path}")
    print(f"\n📋 Summary:")
    print(f"   🔴 Critical: {output['summary']['critical_risk']}")
    print(f"   🟠 High:     {output['summary']['high_risk']}")
    print(f"   🟡 Medium:   {output['summary']['medium_risk']}")
    print(f"   🟢 Low:      {output['summary']['low_risk']}")
    print(f"\n✅ Done! Use shap_results.json to power the 'Why They Might Leave' dashboard section.")


if __name__ == '__main__':
    main()
