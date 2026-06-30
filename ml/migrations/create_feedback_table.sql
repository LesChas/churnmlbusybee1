-- ============================================================
-- Priority #6: Feedback Loop Table for Tracking False Positives
-- ============================================================
-- DO NOT RUN AGAINST PRODUCTION without approval.
-- This creates a table to track whether flagged clients actually churned.
-- Used for model retraining and continuous precision improvement.
-- ============================================================

CREATE TABLE IF NOT EXISTS ml_prediction_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    prediction_date DATE NOT NULL,
    actually_churned BOOLEAN NOT NULL,
    feedback_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT DEFAULT '',
    recorded_by UUID REFERENCES auth.users(id),
    
    -- One feedback per client per prediction date
    CONSTRAINT unique_feedback_per_prediction UNIQUE (client_id, prediction_date)
);

-- Index for querying recent feedback
CREATE INDEX idx_feedback_date ON ml_prediction_feedback(feedback_date DESC);
CREATE INDEX idx_feedback_client ON ml_prediction_feedback(client_id);

-- View: False positive rate summary (last 90 days)
CREATE OR REPLACE VIEW ml_false_positive_summary AS
SELECT
    COUNT(*) AS total_feedback,
    SUM(CASE WHEN actually_churned THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN NOT actually_churned THEN 1 ELSE 0 END) AS false_positives,
    ROUND(
        SUM(CASE WHEN NOT actually_churned THEN 1 ELSE 0 END)::NUMERIC / NULLIF(COUNT(*), 0),
        3
    ) AS false_positive_rate,
    ROUND(
        SUM(CASE WHEN actually_churned THEN 1 ELSE 0 END)::NUMERIC / NULLIF(COUNT(*), 0),
        3
    ) AS actual_precision
FROM ml_prediction_feedback
WHERE feedback_date >= NOW() - INTERVAL '90 days';

COMMENT ON TABLE ml_prediction_feedback IS 
    'Tracks whether clients flagged as churn risks actually churned. Used to measure and improve model precision.';
