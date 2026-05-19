-- =====================================================
-- BusyBee ML Predictions Database Schema (SAFE VERSION)
-- Run this in Supabase SQL Editor
-- =====================================================

-- =====================================================
-- STEP 1: Create prediction tables
-- =====================================================

-- Table to store client churn predictions
CREATE TABLE IF NOT EXISTS client_churn_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    prediction_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    churn_probability DECIMAL(5,4) NOT NULL CHECK (churn_probability >= 0 AND churn_probability <= 1),
    risk_level TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    risk_score INTEGER NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    
    -- Feature values used for prediction (for explainability)
    features JSONB DEFAULT '{}',
    
    -- Top contributing factors
    top_risk_factors JSONB DEFAULT '[]',
    
    -- Model metadata
    model_version TEXT NOT NULL DEFAULT '1.0.0',
    model_name TEXT NOT NULL DEFAULT 'xgboost_churn_v1',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add unique constraint if not exists (handles idempotency)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'client_churn_predictions_client_id_prediction_date_key'
    ) THEN
        ALTER TABLE client_churn_predictions 
        ADD CONSTRAINT client_churn_predictions_client_id_prediction_date_key 
        UNIQUE (client_id, (prediction_date::DATE));
    END IF;
EXCEPTION WHEN others THEN
    -- Constraint might already exist with different name, ignore
    NULL;
END $$;

-- Table to store team member attrition predictions
CREATE TABLE IF NOT EXISTS team_member_attrition_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_member_id UUID NOT NULL REFERENCES team_members(id) ON DELETE CASCADE,
    prediction_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    attrition_probability DECIMAL(5,4) NOT NULL CHECK (attrition_probability >= 0 AND attrition_probability <= 1),
    risk_level TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    risk_score INTEGER NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    
    -- Predicted timeframe
    predicted_days_to_attrition INTEGER,
    
    -- Feature values used for prediction
    features JSONB DEFAULT '{}',
    
    -- Top contributing factors
    top_risk_factors JSONB DEFAULT '[]',
    
    -- Recommended actions
    recommended_actions JSONB DEFAULT '[]',
    
    -- Model metadata
    model_version TEXT NOT NULL DEFAULT '1.0.0',
    model_name TEXT NOT NULL DEFAULT 'xgboost_attrition_v1',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table to store model performance metrics
CREATE TABLE IF NOT EXISTS ml_model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    metric_date DATE NOT NULL DEFAULT CURRENT_DATE,
    
    -- Performance metrics
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_roc DECIMAL(5,4),
    
    -- Dataset info
    training_samples INTEGER,
    test_samples INTEGER,
    positive_rate DECIMAL(5,4),
    
    -- Model artifacts location
    model_s3_path TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table to track prediction batch runs
CREATE TABLE IF NOT EXISTS ml_prediction_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    
    -- Run statistics
    total_predictions INTEGER DEFAULT 0,
    successful_predictions INTEGER DEFAULT 0,
    failed_predictions INTEGER DEFAULT 0,
    
    -- Risk distribution
    high_risk_count INTEGER DEFAULT 0,
    medium_risk_count INTEGER DEFAULT 0,
    low_risk_count INTEGER DEFAULT 0,
    
    -- Timing
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    
    -- Status
    status TEXT CHECK (status IN ('running', 'completed', 'failed')) DEFAULT 'running',
    error_message TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- STEP 2: Create indexes
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_client_churn_predictions_client_id 
    ON client_churn_predictions(client_id);
CREATE INDEX IF NOT EXISTS idx_client_churn_predictions_date 
    ON client_churn_predictions(prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_client_churn_predictions_risk 
    ON client_churn_predictions(risk_level);

CREATE INDEX IF NOT EXISTS idx_team_member_attrition_predictions_tm_id 
    ON team_member_attrition_predictions(team_member_id);
CREATE INDEX IF NOT EXISTS idx_team_member_attrition_predictions_date 
    ON team_member_attrition_predictions(prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_team_member_attrition_predictions_risk 
    ON team_member_attrition_predictions(risk_level);

-- =====================================================
-- STEP 3: Create views (FIXED: handles NULL is_active)
-- =====================================================

-- View to get latest predictions for each client
CREATE OR REPLACE VIEW latest_client_churn_predictions AS
SELECT DISTINCT ON (ccp.client_id)
    ccp.*,
    c.name as client_name,
    c.industry_type,
    c.client_location,
    c.health,
    c.status as client_status
FROM client_churn_predictions ccp
JOIN clients c ON c.id = ccp.client_id
WHERE c.is_active = true OR (c.is_active IS NULL AND c.status = 'active')
ORDER BY ccp.client_id, ccp.prediction_date DESC;

-- View to get latest predictions for each team member
CREATE OR REPLACE VIEW latest_team_member_attrition_predictions AS
SELECT DISTINCT ON (tmap.team_member_id)
    tmap.*,
    tm.name as team_member_name,
    tm.email,
    c.name as client_name
FROM team_member_attrition_predictions tmap
JOIN team_members tm ON tm.id = tmap.team_member_id
LEFT JOIN clients c ON c.id = tm.client_id
WHERE tm.status = 'active'
ORDER BY tmap.team_member_id, tmap.prediction_date DESC;

-- =====================================================
-- STEP 4: Create summary functions
-- =====================================================

CREATE OR REPLACE FUNCTION get_high_risk_clients_summary()
RETURNS TABLE (
    total_clients INTEGER,
    high_risk_count INTEGER,
    medium_risk_count INTEGER,
    low_risk_count INTEGER,
    avg_churn_probability DECIMAL,
    clients_needing_attention INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_clients,
        COUNT(*) FILTER (WHERE risk_level IN ('high', 'critical'))::INTEGER as high_risk_count,
        COUNT(*) FILTER (WHERE risk_level = 'medium')::INTEGER as medium_risk_count,
        COUNT(*) FILTER (WHERE risk_level = 'low')::INTEGER as low_risk_count,
        AVG(churn_probability)::DECIMAL as avg_churn_probability,
        COUNT(*) FILTER (WHERE churn_probability > 0.5)::INTEGER as clients_needing_attention
    FROM latest_client_churn_predictions;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- STEP 5: Enable RLS
-- =====================================================

ALTER TABLE client_churn_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_member_attrition_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_model_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_prediction_runs ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- STEP 6: Create RLS Policies (with safe DROP IF EXISTS)
-- =====================================================

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS "Allow authenticated read client predictions" ON client_churn_predictions;
DROP POLICY IF EXISTS "Allow authenticated read team member predictions" ON team_member_attrition_predictions;
DROP POLICY IF EXISTS "Allow authenticated read model metrics" ON ml_model_metrics;
DROP POLICY IF EXISTS "Allow authenticated read prediction runs" ON ml_prediction_runs;
DROP POLICY IF EXISTS "Allow service role write client predictions" ON client_churn_predictions;
DROP POLICY IF EXISTS "Allow service role write team member predictions" ON team_member_attrition_predictions;
DROP POLICY IF EXISTS "Allow service role write model metrics" ON ml_model_metrics;
DROP POLICY IF EXISTS "Allow service role write prediction runs" ON ml_prediction_runs;

-- Read policies (all authenticated users)
CREATE POLICY "Allow authenticated read client predictions" 
    ON client_churn_predictions FOR SELECT 
    TO authenticated 
    USING (true);

CREATE POLICY "Allow authenticated read team member predictions" 
    ON team_member_attrition_predictions FOR SELECT 
    TO authenticated 
    USING (true);

CREATE POLICY "Allow authenticated read model metrics" 
    ON ml_model_metrics FOR SELECT 
    TO authenticated 
    USING (true);

CREATE POLICY "Allow authenticated read prediction runs" 
    ON ml_prediction_runs FOR SELECT 
    TO authenticated 
    USING (true);

-- Write policies (service role only - for ML pipeline)
CREATE POLICY "Allow service role write client predictions" 
    ON client_churn_predictions FOR ALL 
    TO service_role 
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow service role write team member predictions" 
    ON team_member_attrition_predictions FOR ALL 
    TO service_role 
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow service role write model metrics" 
    ON ml_model_metrics FOR ALL 
    TO service_role 
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow service role write prediction runs" 
    ON ml_prediction_runs FOR ALL 
    TO service_role 
    USING (true)
    WITH CHECK (true);

-- =====================================================
-- STEP 7: Grant permissions
-- =====================================================

GRANT SELECT ON latest_client_churn_predictions TO authenticated;
GRANT SELECT ON latest_team_member_attrition_predictions TO authenticated;
GRANT EXECUTE ON FUNCTION get_high_risk_clients_summary TO authenticated;

-- =====================================================
-- VERIFICATION (run after migration)
-- =====================================================

-- Check tables were created
SELECT 'Tables created:' as status, table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'client_churn_predictions', 
    'team_member_attrition_predictions',
    'ml_model_metrics',
    'ml_prediction_runs'
);
