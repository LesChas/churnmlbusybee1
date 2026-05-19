# BusyBee ML Churn Prediction - Developer Handoff Guide
## Integration & Testing Instructions

**Date:** May 13, 2026  
**Module:** Client Churn Prediction (AI/ML)  
**Status:** Ready for Integration Testing

---

## 📋 Summary

A machine learning model has been built to predict client churn based on historical termination form data. The model is trained, tested locally, and ready for production integration.

| Metric | Value |
|--------|-------|
| Model Type | XGBoost Classifier |
| Training Records | 1,355 |
| Accuracy | 95.6% |
| AUC-ROC | 96.7% |
| Detection Rate | 78.6% |

---

## 📁 Files Created/Modified

### New Files

```
ml/
├── database/
│   └── ml_db.py                    # SQLite database wrapper (MODIFIED - added prediction tables)
├── scripts/
│   ├── train_churn_model.py        # Model training script
│   ├── predict_churn.py            # Prediction generator
│   ├── sync_predictions.py         # Push predictions to Supabase
│   └── test_predictions_local.py   # Local testing script (NEW)
├── models/
│   ├── churn_model_latest.pkl      # Trained model file
│   └── churn_model_latest_metadata.json  # Model metrics
├── data/
│   ├── ml_training.db              # Local SQLite database
│   └── historical_terminations.csv # Training data (from Google Sheets)
├── migrations/
│   ├── create_prediction_tables.sql      # Original Supabase migration
│   └── create_prediction_tables_safe.sql # Safe/idempotent version (NEW)
└── .env.local                      # Safety config (blocks production writes)

src/components/ml/
└── MLPredictionsDashboard.tsx      # Frontend dashboard component (EXISTS)

src/pages/
└── MLPredictionsPage.tsx           # Dashboard page route (EXISTS)

docs/presentations/
├── churn_prediction_executive_brief.md   # 7-slide executive presentation
├── churn_model_architecture_simple.md    # Non-technical architecture guide
└── churn_prediction_video_script.md      # Video explainer script
```

### Routes Already Configured

| Route | Component | Status |
|-------|-----------|--------|
| `/ml-predictions` | MLPredictionsDashboard | ✅ Exists in App.tsx |
| `/enhanced-client-churn-dashboard` | EnhancedClientChurnDashboard | ✅ Exists in App.tsx |

---

## 🔧 Setup Steps for Developer

### Step 1: Pull Latest Code

```bash
git pull origin main  # or feature branch
```

### Step 2: Create Python Virtual Environment (if not exists)

```bash
cd ml
python -m venv venv_ml
.\venv_ml\Scripts\activate  # Windows
# or: source venv_ml/bin/activate  # Mac/Linux

pip install pandas numpy scikit-learn xgboost python-dotenv
```

### Step 3: Run Supabase Migration

Open Supabase SQL Editor and run:

```sql
-- Copy contents of: ml/migrations/create_prediction_tables_safe.sql
```

This creates:
- `client_churn_predictions` table
- `team_member_attrition_predictions` table
- `ml_model_metrics` table
- `ml_prediction_runs` table
- `latest_client_churn_predictions` view
- RLS policies for security

### Step 4: Verify Migration

```sql
SELECT table_name FROM information_schema.tables 
WHERE table_name LIKE '%prediction%' OR table_name LIKE 'ml_%';
```

Should return 4 tables.

### Step 5: Enable Prediction Sync (When Ready)

Edit `ml/.env.local`:
```env
ML_MODE=production
ML_SKIP_EXTERNAL_CONNECTIONS=false
```

### Step 6: Run Initial Predictions

```bash
cd ml
.\venv_ml\Scripts\activate
python scripts/sync_predictions.py
```

---

## 🧪 Testing Checklist

### Local Testing (Already Complete)

- [x] Model trained on historical data
- [x] Local SQLite predictions working
- [x] Test clients scored successfully
- [x] Database schema verified

### Integration Testing (Developer)

- [ ] Run Supabase migration
- [ ] Verify tables created
- [ ] Run `sync_predictions.py` to populate data
- [ ] Verify predictions appear in `client_churn_predictions` table
- [ ] Test dashboard at `/ml-predictions`
- [ ] Verify RLS policies (different roles see correct data)

### User Acceptance Testing

- [ ] CSP can view predictions for their clients
- [ ] Admin can view all predictions
- [ ] Risk levels display correctly (Critical/High/Medium/Low)
- [ ] Risk factors explanation is clear
- [ ] Navigation to dashboard works

---

## 🔌 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION DATA FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

  1. TERMINATION FORM               2. DAILY PREDICTIONS           3. DASHBOARD
     (CSP submits)                     (6 AM CAT cron)                (User views)

┌─────────────────────┐          ┌─────────────────────┐        ┌─────────────────────┐
│  BusyBee App        │          │  Python ML Pipeline │        │  MLPredictionsDash  │
│  (Termination Form) │          │                     │        │  board.tsx          │
│                     │          │  1. Fetch clients   │        │                     │
│  • Seat count       │          │  2. Engineer feats  │        │  • Risk scores      │
│  • Replacement?     │──────────│  3. XGBoost predict │───────►│  • Risk factors     │
│  • Health rating    │  Supabase│  4. Classify risk   │ Supabase• Action items      │
│  • Communication    │          │  5. Write to DB     │        │  • Trends           │
│                     │          │                     │        │                     │
└─────────────────────┘          └─────────────────────┘        └─────────────────────┘
         │                                │                              │
         ▼                                ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SUPABASE TABLES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  termination_requests    │  client_churn_predictions  │  clients            │
│  (source data)           │  (ML output)                │  (joined for view) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Features (12 inputs)

| Feature | Source | Description |
|---------|--------|-------------|
| `current_seat_count` | termination_requests | Active team members |
| `no_replacement` | termination_requests | Client not replacing TM |
| `replacement_urgency` | termination_requests | How urgent is replacement |
| `client_tenure_at_termination` | clients.start_date | Days as client |
| `termination_type_encoded` | termination_requests | Type of termination |
| `checkin_frequency_encoded` | termination_requests | Check-in cadence |
| `days_since_checkin` | termination_requests | Days since last check-in |
| `days_since_communication` | termination_requests | Days since contact |
| `offered_discount` | termination_requests | Was discount offered |
| `geo_encoded` | clients.client_location | Geographic region |
| `health_score` | termination_requests | Client health rating |
| `had_pip` | termination_requests | Had performance plan |

---

## 🎯 Risk Level Thresholds

| Level | Score Range | Recommended Action |
|-------|-------------|-------------------|
| 🔴 Critical | 75-100% | Executive call within 48 hours |
| 🟠 High | 50-74% | CSP outreach within 1 week |
| 🟡 Medium | 25-49% | Increase check-in frequency |
| 🟢 Low | 0-24% | Continue standard engagement |

---

## 🚀 Production Deployment Steps

### Phase 1: Database Setup
1. Run `create_prediction_tables_safe.sql` in Supabase
2. Verify tables and views created
3. Test RLS policies with different user roles

### Phase 2: Initial Data Load
1. Set `ML_SKIP_EXTERNAL_CONNECTIONS=false` in `ml/.env.local`
2. Run `python scripts/sync_predictions.py`
3. Verify data in Supabase dashboard

### Phase 3: Dashboard Access
1. Add navigation link (if not already visible)
2. Test at `/ml-predictions`
3. Verify role-based access

### Phase 4: Automation (Optional)
1. Set up cron job for daily predictions (6 AM CAT)
2. Configure email alerts for high-risk clients
3. Set up quarterly model retraining schedule

---

## 📞 Support

### Files to Share with Developer

1. **This document** - `ML_HANDOFF_GUIDE.md`
2. **Migration SQL** - `ml/migrations/create_prediction_tables_safe.sql`
3. **Model files** - `ml/models/churn_model_latest.pkl` + `*_metadata.json`
4. **Training data** - `ml/data/historical_terminations.csv` (if needed for retraining)

### Key Environment Variables Needed

```env
# Already in .env
VITE_SUPABASE_URL=<your-supabase-url>
VITE_SUPABASE_ANON_KEY=<your-anon-key>

# For ML sync (use service role for writes)
SUPABASE_SERVICE_KEY=<service-role-key>
```

### Common Issues

| Issue | Solution |
|-------|----------|
| "Port in use" error | Kill process or use different port |
| "No model found" | Run `train_churn_model.py` first |
| Predictions not showing | Check RLS policies, verify sync ran |
| Import errors | Activate venv, install dependencies |

---

## ✅ Ready for Handoff

- [x] Model trained and validated
- [x] Local testing complete
- [x] Migration scripts ready
- [x] Dashboard components exist
- [x] Documentation complete
- [ ] **Pending: Developer runs migration and sync**
