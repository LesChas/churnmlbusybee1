# ML Scripts

These scripts handle the ML workflow using a **local SQLite database** (separate from production Supabase).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL (SQLite)                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Import CSV   │───▶│   Train      │───▶│   Predict    │  │
│  │ Terminations │    │   Model      │    │   Churn      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌────────────────────────────────────────────────────┐   │
│  │         ml/data/ml_training.db (SQLite)            │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ sync_predictions.py
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION (Supabase)                      │
│  ┌────────────────────────────────────────────────────┐   │
│  │         client_churn_predictions table             │   │
│  │              (Dashboard reads from here)            │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
cd ml
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Import Historical Data

Export your termination spreadsheet as CSV, then:

```bash
python scripts/import_terminations.py data/historical_terminations.csv
```

Supported CSV columns (from your Google Form):
- Timestamp
- Client Name
- Last Date of General Communication With The Client
- Cadence of Check Ins With The Client
- Last Date of Official Weekly/Bi-Monthly/Monthly Check In
- Specify the Client Health In The Last Month
- Will There Be A Replacement
- Have we lost the Client? ← **Important: This is the target variable**
- etc.

### 3. Train the Model

```bash
python scripts/train_churn_model.py
```

Options:
- `--test-size 0.3` - Use 30% of data for testing (default: 20%)
- `--n-estimators 200` - More trees for potentially better accuracy
- `--seed 42` - Random seed for reproducibility

### 4. Review Results

The script outputs:
- Accuracy, Precision, Recall, F1, AUC-ROC scores
- Feature importance ranking
- Confusion matrix

Model files saved to `ml/models/`:
- `churn_model_latest.pkl` - Trained model
- `churn_model_latest_metadata.json` - Metrics & feature importance

### 5. Sync to Dashboard

Only when ready to push to production:

```bash
python scripts/sync_predictions.py
```

This is the **only** script that writes to Supabase.

## Key Files

| File | Purpose |
|------|---------|
| `database/ml_db.py` | SQLite database wrapper |
| `scripts/import_terminations.py` | Import CSV data |
| `scripts/train_churn_model.py` | Train XGBoost model |
| `scripts/sync_predictions.py` | Push to Supabase |
| `data/ml_training.db` | SQLite database (gitignored) |
| `models/churn_model_latest.pkl` | Trained model |

## Features Used

The model uses these features (in order of typical importance):

1. **days_since_communication** - Days since last client communication
2. **days_since_checkin** - Days since last official check-in
3. **health_score** - Client health (4=Excellent, 3=Good, 2=Fair, 1=Poor)
4. **no_replacement** - Whether replacement was requested
5. **replacement_urgency** - How urgent the replacement need
6. **had_pip** - Performance Improvement Plan implemented
7. **offered_discount** - Relief package offered
8. **current_seat_count** - Number of seats
9. **client_tenure_at_termination** - How long client has been active
10. **checkin_frequency_encoded** - Weekly=1, Bi-Monthly=2, Monthly=3

## Database Independence

- All training data stored in local SQLite (`ml/data/ml_training.db`)
- Supabase is only touched by `sync_predictions.py`
- Safe to experiment without affecting production
- Can rebuild/retrain without Supabase access
