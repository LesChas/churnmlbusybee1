# BusyBee ML Predictions Pipeline

## Overview

This module implements ML-powered predictive analytics for:
- **Client Churn Prediction**: Predicts which clients are likely to leave
- **Team Member Attrition**: Predicts which employees may leave

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BusyBee ML Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌─────────────┐    ┌───────────────────┐  │
│  │   Supabase   │───▶│ AWS Lambda  │───▶│ Predictions Table │  │
│  │   Database   │    │ (XGBoost)   │    │   (Supabase)      │  │
│  └──────────────┘    └─────────────┘    └───────────────────┘  │
│         │                   ▲                     │             │
│         │            ┌──────┴──────┐              │             │
│         │            │  S3 Bucket  │              │             │
│         │            │  (Models)   │              ▼             │
│         │            └─────────────┘    ┌───────────────────┐  │
│         │                               │   React Frontend  │  │
│         └──────────────────────────────▶│   Dashboard       │  │
│                                         └───────────────────┘  │
│                                                                  │
│  Trigger: EventBridge (Daily at 2 AM UTC)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
ml/
├── database/
│   └── ml_db.py             # SQLite database for local training
├── scripts/
│   ├── import_terminations.py  # Import historical CSV data
│   ├── train_churn_model.py    # Train model locally
│   ├── predict_churn.py        # Run predictions locally
│   └── sync_predictions.py     # Push to Supabase (only DB write)
├── training/
│   └── train_models.py      # Model training script
├── lambda/
│   └── prediction_handler.py # Lambda inference handler
├── utils/
│   └── export_training_data.py # Data export utility
├── aws/
│   └── cloudformation-template.yaml # AWS infrastructure
├── migrations/
│   └── create_prediction_tables.sql # Database schema
├── data/                     # Local SQLite DB & CSVs (git-ignored)
├── models/                   # Trained model files (git-ignored)
├── requirements.txt          # Python dependencies
├── deploy.sh                 # Deployment script
└── README.md                 # This file

## Local Training Option (NEW)

For training without affecting Supabase, use the scripts/ folder:

```bash
# 1. Import historical termination data
python scripts/import_terminations.py data/terminations.csv

# 2. Train the model locally
python scripts/train_churn_model.py

# 3. Run predictions locally
python scripts/predict_churn.py

# 4. Only when ready, push to Supabase
python scripts/sync_predictions.py
```

See `scripts/README.md` for full documentation.
```

## Quick Start

### 1. Install Dependencies

```bash
cd ml
pip install -r requirements.txt
```

### 2. Export Training Data

```bash
# Set environment variables
export SUPABASE_URL="your-supabase-url"
export SUPABASE_SERVICE_KEY="your-service-key"

# Export data
python utils/export_training_data.py --output-path ./data
```

### 3. Train Models

```bash
# Train with real data
python training/train_models.py --data-path ./data --output-path ./models

# Or use sample data for testing
python training/train_models.py --use-sample-data --output-path ./models
```

### 4. Deploy to AWS

```bash
# Set credentials
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"
export SUPABASE_URL="your-supabase-url"
export SUPABASE_SERVICE_KEY="your-service-key"

# Deploy
chmod +x deploy.sh
./deploy.sh production
```

### 5. Apply Database Migrations

Run in Supabase SQL Editor:
```sql
-- Copy contents of migrations/create_prediction_tables.sql
```

## Models

### Client Churn Model (XGBoost)

**Features:**
| Feature | Description |
|---------|-------------|
| `tenure_months` | How long the client has been active |
| `health_score` | Client health rating (1-4) |
| `days_since_activity` | Days since last interaction |
| `industry_encoded` | Industry type (encoded) |
| `location_encoded` | Geographic location (encoded) |
| `is_active` | Current active status |
| `start_month` | Month client started (seasonality) |
| `start_quarter` | Quarter client started |

**Target:** Binary classification (churned: 1, active: 0)

### Team Member Attrition Model (XGBoost)

**Features:**
| Feature | Description |
|---------|-------------|
| `tenure_months` | Employment duration |
| `role_encoded` | Job role (encoded) |
| `days_since_activity` | Days since last activity |
| `has_client` | Whether assigned to a client |
| `tenure_bucket` | Tenure category (0-3m, 3-6m, etc.) |
| `is_new_employee` | First 90 days flag |
| `start_month` | Start month (seasonality) |
| `start_day_of_week` | Day of week started |

**Target:** Binary classification (left: 1, active: 0)

## API Reference

### Database Tables

#### `client_churn_predictions`
Stores daily churn predictions for each client.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `client_id` | UUID | Reference to clients table |
| `prediction_date` | TIMESTAMP | When prediction was made |
| `churn_probability` | DECIMAL | Probability (0-1) |
| `risk_level` | TEXT | low/medium/high/critical |
| `risk_score` | INTEGER | Score (0-100) |
| `features` | JSONB | Feature values used |
| `top_risk_factors` | JSONB | Contributing factors |

#### `team_member_attrition_predictions`
Stores daily attrition predictions for each team member.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `team_member_id` | UUID | Reference to team_members |
| `prediction_date` | TIMESTAMP | When prediction was made |
| `attrition_probability` | DECIMAL | Probability (0-1) |
| `risk_level` | TEXT | low/medium/high/critical |
| `predicted_days_to_attrition` | INTEGER | Estimated days to departure |
| `recommended_actions` | JSONB | Suggested interventions |

### Views

- `latest_client_churn_predictions` - Most recent prediction per client
- `latest_team_member_attrition_predictions` - Most recent prediction per team member

### Functions

- `get_high_risk_clients_summary()` - Summary of client risk distribution
- `get_high_risk_team_members_summary()` - Summary of team member risk distribution

## Lambda Invocation

### Manual Trigger

```bash
aws lambda invoke \
    --function-name busybee-ml-predictions-production \
    --payload '{"model_type": "all", "model_source": "s3"}' \
    response.json
```

### Event Payload Options

```json
{
    "model_type": "all" | "churn" | "attrition",
    "model_source": "s3" | "local"
}
```

## Monitoring

### CloudWatch Logs

```bash
aws logs tail /aws/lambda/busybee-ml-predictions-production --follow
```

### Check Prediction History

```sql
SELECT * FROM ml_prediction_runs 
ORDER BY run_date DESC 
LIMIT 10;
```

## Frontend Integration

The React dashboard is available at `/ml-predictions` and displays:
- Overview with risk distribution charts
- Client churn predictions table
- Team member attrition predictions table
- Detailed risk factors and recommendations

## Troubleshooting

### No Predictions Appearing

1. Check Lambda logs for errors
2. Verify Supabase credentials are correct
3. Ensure models are uploaded to S3
4. Check RLS policies on prediction tables

### Model Performance Issues

1. Retrain with more recent data
2. Check for class imbalance in training data
3. Review feature importance to identify key predictors

### Lambda Timeout

1. Increase Lambda timeout (max 15 min)
2. Consider batching large datasets
3. Optimize feature engineering

## Cost Estimation

| Component | Estimated Monthly Cost |
|-----------|----------------------|
| Lambda (daily runs) | ~$2-5 |
| S3 (model storage) | ~$1 |
| CloudWatch Logs | ~$1-2 |
| **Total** | **~$5-10** |

## Future Improvements

- [ ] Add SHAP explanations for feature importance
- [ ] Implement model retraining pipeline
- [ ] Add A/B testing for model versions
- [ ] Integrate with notification system for high-risk alerts
- [ ] Add real-time predictions via API Gateway
