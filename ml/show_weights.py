"""Extract and display model weights from the trained model."""
import pickle
import numpy as np

with open('./models_real/churn_model_real_precision.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
features = data['feature_columns']

print('=' * 60)
print('  MODEL WEIGHTS & PARAMETERS')
print('=' * 60)

# Feature importance (gain-based)
print('\n--- Feature Importance (Gain) ---')
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f'  {feat:<28s} {imp:.4f}')

# Booster params
print('\n--- XGBoost Parameters ---')
config = model.get_params()
for k, v in sorted(config.items()):
    if v is not None and k not in ('callbacks', 'kwargs', 'missing'):
        print(f'  {k:<25s} = {v}')

# Tree count
print(f'\n--- Model Structure ---')
print(f'  Number of trees: {model.n_estimators}')
print(f'  Max depth: {model.max_depth}')
print(f'  Base score: {model.base_score}')

# Dump first 3 trees
print(f'\n--- First 3 Tree Structures ---')
booster = model.get_booster()
trees = booster.get_dump(dump_format='text')
for i, tree in enumerate(trees[:3]):
    print(f'\n  Tree {i}:')
    lines = tree.strip().split('\n')
    for line in lines[:10]:
        print(f'    {line}')
    if len(lines) > 10:
        print(f'    ... ({len(lines)} nodes total)')

# Decision threshold
threshold = data.get('decision_threshold', 0.7)
print(f'\n--- Decision Threshold ---')
print(f'  Optimal threshold: {threshold}')
print(f'  (probabilities >= {threshold} -> flagged as churn risk)')

# Scaler parameters
print(f'\n--- StandardScaler Weights (mean / std per feature) ---')
scaler = data['scaler']
print(f'  {"Feature":<28s} {"Mean":>10s} {"Std":>10s}')
print(f'  {"-"*50}')
for feat, mean, std in zip(features, scaler.mean_, scaler.scale_):
    print(f'  {feat:<28s} {mean:>10.4f} {std:>10.4f}')

# Calibration info
print(f'\n--- Calibration ---')
print(f'  Method: Platt scaling (sigmoid)')
cal = data.get('calibrated_model')
if cal:
    print(f'  Calibrator type: {type(cal).__name__}')
    print(f'  Base estimators: {len(cal.calibrated_classifiers_)}')

print(f'\n--- Label Encoders ---')
for name, le in data.get('label_encoders', {}).items():
    print(f'  {name}: {list(le.classes_)}')
