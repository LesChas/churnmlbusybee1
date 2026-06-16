"""
BusyBee ML - SHAP Explainability (Real Model)
==============================================
Loads the actual trained churn_model_latest.pkl and generates per-client
explanations using SHAP TreeExplainer with all 12 production features.

100% local - no database connections, no live data touched.
Uses realistic mock data that matches the trained model's feature schema.

Usage:
    python shap_explain.py

Output:
    - shap_results.json (per-client top factors + global feature importance)
    - Console summary of "Why They Might Leave" for each at-risk client
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

try:
    import shap
except ImportError:
    print("❌ SHAP not installed. Run: pip install shap")
    sys.exit(1)


# The 12 features used by the real trained model (from metadata)
MODEL_FEATURES = [
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
    'geo_encoded',
]

# Human-readable labels for each feature
FEATURE_LABELS = {
    'current_seat_count': 'Client seat count',
    'no_replacement': 'No replacement lined up',
    'replacement_urgency': 'Replacement urgency/timeline',
    'client_tenure_at_termination': 'Client tenure length',
    'termination_type_encoded': 'Termination type',
    'checkin_frequency_encoded': 'Check-in frequency',
    'days_since_checkin': 'Days since last check-in',
    'days_since_communication': 'Days since last communication',
    'offered_discount': 'Discount/relief offered',
    'geo_encoded': 'Geo-location',
    'health_score': 'Client health score',
    'had_pip': 'PIP was implemented',
}

# Descriptions for SHAP direction
FEATURE_REASONS = {
    'current_seat_count': {
        'high_risk': 'Larger seat count — higher revenue at stake',
        'low_risk': 'Smaller engagement — lower exposure',
    },
    'no_replacement': {
        'high_risk': 'No replacement is being lined up',
        'low_risk': 'Replacement is in progress',
    },
    'replacement_urgency': {
        'high_risk': 'Urgent replacement timeline',
        'low_risk': 'No urgency on replacement',
    },
    'client_tenure_at_termination': {
        'high_risk': 'Short client tenure — not yet embedded',
        'low_risk': 'Long-standing relationship — more invested',
    },
    'termination_type_encoded': {
        'high_risk': 'Termination type indicates higher risk',
        'low_risk': 'Termination type is routine/planned',
    },
    'checkin_frequency_encoded': {
        'high_risk': 'Low check-in frequency',
        'low_risk': 'Regular check-in cadence maintained',
    },
    'days_since_checkin': {
        'high_risk': 'No recent check-in with client',
        'low_risk': 'Recent check-in completed',
    },
    'days_since_communication': {
        'high_risk': 'No recent communication',
        'low_risk': 'Active communication maintained',
    },
    'offered_discount': {
        'high_risk': 'Had to offer discount to retain (still at risk)',
        'low_risk': 'No discount needed — organically satisfied',
    },
    'geo_encoded': {
        'high_risk': 'Region shows higher churn patterns',
        'low_risk': 'Region is stable',
    },
    'health_score': {
        'high_risk': 'Client health score is low',
        'low_risk': 'Strong client health score',
    },
    'had_pip': {
        'high_risk': 'PIP was implemented — performance concerns',
        'low_risk': 'No PIP history — clean record',
    },
}

# Check-in frequency encoding reference
CHECKIN_FREQ = {'weekly': 4, 'bi-monthly': 3, 'monthly': 2, 'quarterly': 1, 'none': 0}
# Termination type encoding reference
TERM_TYPES = {'resignation': 0, 'replacement': 1, 'termination': 2, 'immediate': 3}
# Geo encoding reference
GEOS = {'US': 0, 'Philippines': 1, 'South Africa': 2, 'India': 3, 'Other': 4}


def load_trained_model():
    """Load the real trained model from models/ directory."""
    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / "churn_model_latest.pkl"
    metadata_path = model_dir / "churn_model_latest_metadata.json"

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("   Run training first or ensure churn_model_latest.pkl exists.")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    # The pkl may contain just the model or a dict with model + extras
    if isinstance(model_data, dict):
        model = model_data.get('model', model_data)
    else:
        model = model_data

    features = metadata.get('features', MODEL_FEATURES)
    print(f"   ✅ Loaded model: {model_path.name}")
    print(f"   📊 Trained on {metadata.get('metrics', {}).get('training_records', '?')} records")
    print(f"   🎯 AUC-ROC: {metadata.get('metrics', {}).get('auc_roc', '?')}")
    print(f"   🔧 Features: {len(features)}")

    return model, metadata, features


def generate_realistic_mock_data(n: int = 50) -> pd.DataFrame:
    """
    Generate mock client data matching the real model's 12-feature schema.
    Simulates realistic patterns from form responses.
    """
    np.random.seed(42)

    clients = []
    for i in range(n):
        # Simulate realistic distributions based on form data patterns
        seat_count = np.random.choice([1, 2, 3, 5, 8, 10, 15, 20, 30], p=[0.20, 0.20, 0.15, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03])
        tenure_days = np.random.randint(30, 1200)
        health = np.random.choice([1, 2, 3, 4], p=[0.10, 0.20, 0.40, 0.30])
        checkin_freq = np.random.choice([0, 1, 2, 3, 4], p=[0.05, 0.10, 0.30, 0.30, 0.25])
        days_comm = int(max(1, min(90, np.random.exponential(14))))
        days_checkin = int(max(1, min(60, np.random.exponential(10))))
        no_replacement = np.random.choice([0, 1], p=[0.65, 0.35])
        replacement_urgency = np.random.choice([0, 1, 2, 3], p=[0.55, 0.20, 0.15, 0.10]) if no_replacement == 0 else 0
        had_pip = np.random.choice([0, 1], p=[0.85, 0.15])
        offered_discount = np.random.choice([0, 1], p=[0.80, 0.20])
        term_type = np.random.choice([0, 1, 2, 3], p=[0.30, 0.35, 0.25, 0.10])
        geo = np.random.choice([0, 1, 2, 3, 4], p=[0.35, 0.25, 0.20, 0.10, 0.10])

        # Determine churn label based on realistic signal combinations
        churn_score = 0
        if no_replacement == 1:
            churn_score += 3
        if health <= 2:
            churn_score += 2
        if days_comm > 30:
            churn_score += 1
        if days_checkin > 21:
            churn_score += 1
        if seat_count >= 10:
            churn_score += 1
        if term_type >= 2:
            churn_score += 1

        churned = 1 if (churn_score >= 4 or (churn_score >= 3 and np.random.random() < 0.6)) else 0
        if churn_score <= 1:
            churned = 0

        clients.append({
            'client_id': f'client-{i:04d}',
            'client_name': f'Client {i} Corp',
            'days_since_communication': int(days_comm),
            'days_since_checkin': int(days_checkin),
            'health_score': int(health),
            'checkin_frequency_encoded': int(checkin_freq),
            'no_replacement': int(no_replacement),
            'replacement_urgency': int(replacement_urgency),
            'had_pip': int(had_pip),
            'offered_discount': int(offered_discount),
            'current_seat_count': int(seat_count),
            'client_tenure_at_termination': int(tenure_days),
            'termination_type_encoded': int(term_type),
            'geo_encoded': int(geo),
            'churned': int(churned),
        })

    return pd.DataFrame(clients)


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
    print("  BusyBee ML - SHAP Explainability (Real Model)")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("🔒 Mode: 100% LOCAL - no database connections")
    print("🧠 Using: churn_model_latest.pkl (12 features)\n")

    # --- Load real trained model ---
    print("📦 Loading trained model...")
    model, metadata, feature_names = load_trained_model()

    # --- Generate mock data matching real feature schema ---
    print("\n📦 Generating realistic mock client data (12 features)...")
    data_df = generate_realistic_mock_data(50)
    active_df = data_df[data_df['churned'] == 0].reset_index(drop=True)
    print(f"   {len(data_df)} total clients ({len(active_df)} active, {data_df['churned'].sum()} churned)\n")

    # --- Prepare feature matrix ---
    X_active = active_df[feature_names].copy()

    # --- Run predictions with real model ---
    print("🎯 Running predictions with trained model...")
    probabilities = model.predict_proba(X_active)[:, 1]
    print(f"   ✅ Predictions generated for {len(active_df)} active clients\n")

    # --- SHAP Explainability ---
    print("🔍 Computing SHAP values (TreeExplainer on real model)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_active)
    print(f"   ✅ SHAP values computed for {len(active_df)} clients\n")

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
    print("-" * 60)
    max_shap = max(item['mean_abs_shap'] for item in global_importance) if global_importance else 1
    for item in global_importance:
        bar_len = int((item['mean_abs_shap'] / max_shap) * 30) if max_shap > 0 else 0
        bar = "█" * bar_len
        print(f"   {item['label']:<40} {item['mean_abs_shap']:.4f} {bar}")

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
            'client_id': active_df.iloc[i]['client_id'],
            'client_name': active_df.iloc[i]['client_name'],
            'risk_score': round(float(prob), 4),
            'risk_level': risk_level,
            'top_factors': reasons,
            'features': {feat: int(active_df.iloc[i][feat]) for feat in feature_names},
        }
        client_explanations.append(entry)

    # Sort by risk (highest first)
    client_explanations.sort(key=lambda x: -x['risk_score'])

    # Print top 10 at-risk
    risk_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
    for entry in client_explanations[:10]:
        emoji = risk_emoji.get(entry['risk_level'], '⚪')
        print(f"\n{emoji} {entry['client_name']} — {entry['risk_level'].upper()} ({entry['risk_score']:.0%})")
        risk_reasons = [r for r in entry['top_factors'] if r['direction'] == 'increases_risk']
        if not risk_reasons:
            risk_reasons = entry['top_factors'][:2]
        for reason in risk_reasons:
            print(f"   → {reason['reason']} (impact: {reason['impact']:+.4f})")

    # --- Save results ---
    output = {
        'generated_at': datetime.now().isoformat(),
        'mode': 'local_with_real_model',
        'model_info': {
            'file': 'churn_model_latest.pkl',
            'features_count': len(feature_names),
            'features': feature_names,
            'training_records': metadata.get('metrics', {}).get('training_records', 'unknown'),
            'auc_roc': metadata.get('metrics', {}).get('auc_roc', 'unknown'),
        },
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
    print(f"\n🔧 Model uses {len(feature_names)} features from form responses:")
    for feat in feature_names:
        print(f"   • {FEATURE_LABELS.get(feat, feat)}")
    print(f"\n✅ Done! Use shap_results.json to power the 'Why They Might Leave' dashboard section.")


if __name__ == '__main__':
    main()
