"""
Local test script: Train the precision-optimized churn model on sample data
and compare metrics vs the old configuration.

Usage:
    python test_precision_training.py

This does NOT touch any database - uses generated sample data only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_models import ChurnPredictionModel, AttritionPredictionModel, generate_sample_data


def main():
    print("=" * 60)
    print("  LOCAL TEST: Precision-Optimized Churn Model Training")
    print("  Using SAMPLE DATA only - no database connection")
    print("=" * 60)
    
    # Generate sample data (same as --use-sample-data flag)
    clients_df, team_members_df = generate_sample_data(n_clients=500, n_team_members=1000)
    
    print(f"\nGenerated {len(clients_df)} sample clients, {len(team_members_df)} sample team members")
    print(f"Client churn rate: {(clients_df['status'] == 'inactive').mean():.1%}")
    print(f"Team member attrition rate: {(team_members_df['status'] == 'inactive').mean():.1%}")
    
    # ─── Train Churn Model (precision-optimized) ───
    print("\n" + "─" * 60)
    churn_model = ChurnPredictionModel(model_version="test_precision_v2")
    churn_metrics = churn_model.train(clients_df)
    
    # ─── Train Attrition Model ───
    print("\n" + "─" * 60)
    attrition_model = AttritionPredictionModel(model_version="test_precision_v2")
    attrition_metrics = attrition_model.train(team_members_df)
    
    # ─── Generate test predictions ───
    print("\n" + "─" * 60)
    print("\nGenerating predictions on full dataset...")
    
    churn_predictions = churn_model.predict(clients_df)
    
    # Count by risk level (using new thresholds)
    risk_counts = churn_predictions['risk_level'].value_counts()
    
    print(f"\n  CHURN PREDICTION DISTRIBUTION (new thresholds: 0.85/0.65/0.40):")
    print(f"  {'─' * 45}")
    for level in ['critical', 'high', 'medium', 'low']:
        count = risk_counts.get(level, 0)
        pct = count / len(churn_predictions) * 100
        bar = '█' * int(pct / 2)
        print(f"  {level:10s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"\n  Total flagged (medium+): {risk_counts.get('critical', 0) + risk_counts.get('high', 0) + risk_counts.get('medium', 0)}")
    print(f"  Total low-risk:          {risk_counts.get('low', 0)}")
    
    # ─── Summary comparison ───
    print("\n" + "=" * 60)
    print("  METRICS COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<15} {'Old Model':<12} {'New Model':<12} {'Change':<10}")
    print(f"  {'─' * 49}")
    
    # Old metrics from mock data
    old_metrics = {'precision': 0.55, 'recall': 0.786, 'f1': 0.647, 'auc_roc': 0.967}
    
    for metric in ['precision', 'recall', 'f1', 'auc_roc']:
        old_val = old_metrics.get(metric, 0)
        new_val = churn_metrics.get(metric, 0)
        change = new_val - old_val
        arrow = '↑' if change > 0 else '↓' if change < 0 else '─'
        print(f"  {metric:<15} {old_val:<12.3f} {new_val:<12.3f} {arrow} {abs(change):.3f}")
    
    if 'f0.5' in churn_metrics:
        print(f"  {'f0.5':<15} {'N/A':<12} {churn_metrics['f0.5']:<12.3f} (NEW - precision-weighted)")
    
    print(f"\n  Decision threshold: {churn_metrics.get('decision_threshold', 0.5)}")
    print(f"  Calibrated: {churn_metrics.get('calibrated', False)}")
    
    # Save models locally for inspection
    os.makedirs('./models_test', exist_ok=True)
    churn_model.save('./models_test/churn_model_precision.pkl')
    attrition_model.save('./models_test/attrition_model.pkl')
    
    print(f"\n  Models saved to: ./models_test/")
    print("=" * 60)
    print("  TEST COMPLETE - No database was touched")
    print("=" * 60)


if __name__ == '__main__':
    main()
