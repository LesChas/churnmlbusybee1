"""
BusyBee ML Training Database (SQLite)
=====================================
Completely separate from production Supabase.
Stores historical termination data for ML model training.

Usage:
    from ml.database.ml_db import MLDatabase
    
    db = MLDatabase()  # Creates/connects to local SQLite file
    db.import_terminations_csv('data/historical_terminations.csv')
    training_data = db.get_training_data()
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

# Database file location (in ml/data folder, gitignored)
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "ml_training.db"


class MLDatabase:
    """SQLite database for ML training data - separate from Supabase production."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Historical terminations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_terminations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                
                -- Timestamps
                form_timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                
                -- People
                client_success_partner TEXT,
                csp_email TEXT,
                sales_person TEXT,
                team_member_name TEXT,
                team_member_email TEXT,
                team_member_phone TEXT,
                staff_sign_off_names TEXT,
                
                -- Client Info
                client_name TEXT NOT NULL,
                client_id TEXT,
                client_start_date TEXT,
                current_seat_count INTEGER,
                client_timezone TEXT,
                client_working_hours TEXT,
                geo_location TEXT,
                
                -- Termination Details
                terminating_entity_name TEXT,
                termination_notice_date TEXT NOT NULL,
                team_member_last_date TEXT,
                termination_reason TEXT,
                termination_type TEXT,
                additional_comments TEXT,
                
                -- Communication Signals (KEY FOR CHURN PREDICTION)
                last_general_communication_date TEXT,
                check_in_cadence TEXT,
                last_official_checkin_date TEXT,
                client_health_last_month TEXT,
                
                -- Replacement Info
                will_have_replacement INTEGER, -- 0/1 boolean
                replacement_timeframe TEXT,
                client_requested_interview INTEGER,
                requested_replacement_names TEXT,
                temporary_replacement_period TEXT,
                sme_seated_until_replacement INTEGER,
                
                -- Outcome (TARGET VARIABLE FOR ML)
                client_lost INTEGER DEFAULT 0, -- 0/1 boolean
                no_longer_replacing_reason TEXT,
                
                -- Team Member Details
                team_member_position TEXT,
                team_member_start_date TEXT,
                team_member_skillset TEXT,
                team_member_pms_applications TEXT,
                team_member_workstation TEXT,
                team_member_rehirable_status TEXT,
                rehirable_concerns TEXT,
                team_member_geo_experience TEXT,
                
                -- Process Tracking
                pip_implemented INTEGER DEFAULT 0,
                pip_link TEXT,
                relief_package_offered INTEGER DEFAULT 0,
                relief_package_details TEXT,
                client_informed INTEGER DEFAULT 0,
                
                -- Department Notifications
                notified_finance INTEGER DEFAULT 0,
                notified_team_experience INTEGER DEFAULT 0,
                notified_talent_acquisition INTEGER DEFAULT 0,
                notified_gts INTEGER DEFAULT 0,
                notified_client_solutions INTEGER DEFAULT 0,
                notified_rcm INTEGER DEFAULT 0,
                
                -- Other
                rcm_seat INTEGER DEFAULT 0,
                byod_access INTEGER DEFAULT 0,
                project_based_dates TEXT,
                score REAL,
                other_specify TEXT,
                documentation_files TEXT
            )
        """)
        
        # Computed features table (for caching)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS computed_features (
                termination_id INTEGER PRIMARY KEY,
                days_since_communication INTEGER,
                days_since_checkin INTEGER,
                team_member_tenure_days INTEGER,
                client_tenure_at_termination INTEGER,
                health_score INTEGER,
                checkin_frequency_encoded INTEGER,
                no_replacement INTEGER,
                replacement_urgency INTEGER,
                had_pip INTEGER,
                offered_discount INTEGER,
                computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (termination_id) REFERENCES historical_terminations(id)
            )
        """)
        
        # Model metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                trained_at TEXT DEFAULT CURRENT_TIMESTAMP,
                training_records INTEGER,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                auc_roc REAL,
                feature_importance TEXT, -- JSON
                hyperparameters TEXT, -- JSON
                notes TEXT
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_term_client ON historical_terminations(client_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_term_date ON historical_terminations(termination_notice_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_term_lost ON historical_terminations(client_lost)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_term_health ON historical_terminations(client_health_last_month)")
        
        # =====================================================
        # PREDICTION TABLES (for local testing before Supabase)
        # =====================================================
        
        # Client churn predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_churn_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT,
                client_name TEXT NOT NULL,
                prediction_date TEXT DEFAULT CURRENT_TIMESTAMP,
                churn_probability REAL NOT NULL CHECK (churn_probability >= 0 AND churn_probability <= 1),
                risk_level TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
                risk_score INTEGER NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
                features TEXT, -- JSON
                top_risk_factors TEXT, -- JSON
                model_version TEXT NOT NULL DEFAULT '1.0.0',
                model_name TEXT NOT NULL DEFAULT 'xgboost_churn_v1',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Prediction runs tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                high_risk_count INTEGER DEFAULT 0,
                medium_risk_count INTEGER DEFAULT 0,
                low_risk_count INTEGER DEFAULT 0,
                status TEXT CHECK (status IN ('running', 'completed', 'failed')) DEFAULT 'completed',
                error_message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexes for predictions
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_client ON client_churn_predictions(client_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_date ON client_churn_predictions(prediction_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_risk ON client_churn_predictions(risk_level)")
        
        conn.commit()
        conn.close()
        print(f"✓ ML Database initialized at: {self.db_path}")
    
    def import_terminations_csv(self, csv_path: str, column_mapping: Optional[Dict[str, str]] = None) -> int:
        """
        Import historical terminations from CSV.
        
        Args:
            csv_path: Path to CSV file
            column_mapping: Optional dict mapping CSV columns to DB columns
            
        Returns:
            Number of records imported
        """
        # Default mapping from your spreadsheet columns to DB columns
        default_mapping = {
            'Timestamp': 'form_timestamp',
            'Client Success Partner': 'client_success_partner',
            'Email Address': 'csp_email',
            'Client Name': 'client_name',
            'Name of Client Terminating': 'terminating_entity_name',
            'Date of Receiving Termination/Replacement/Resignation Notice': 'termination_notice_date',
            'Reason for Termination': 'termination_reason',
            'Last Date for Team Member': 'team_member_last_date',
            'Last Date of General Communication With The Client': 'last_general_communication_date',
            'Cadence of Check Ins With The Client': 'check_in_cadence',
            'Last Date of Official Weekly/Bi-Monthly/Monthly Check In': 'last_official_checkin_date',
            'Will There Be A Replacement': 'will_have_replacement',
            'Other Please Specify': 'other_specify',
            'Specify the Client Health In The Last Month': 'client_health_last_month',
            'Sales Person': 'sales_person',
            'Team Member Re-Hirable Status': 'team_member_rehirable_status',
            'Departments Notified [Finance]': 'notified_finance',
            'Departments Notified [Team Experience]': 'notified_team_experience',
            'Departments Notified [Talent Acquistion]': 'notified_talent_acquisition',
            'Departments Notified [GTS]': 'notified_gts',
            'Staff Sign Off Names (Confirmation By Staff Member)': 'staff_sign_off_names',
            'Additional Comments or Reason For Termination/Resignation/Replacement': 'additional_comments',
            'Score': 'score',
            'Departments Notified [Client Solutions]': 'notified_client_solutions',
            'Team Member Skillset': 'team_member_skillset',
            'PMS & Applications Team Member is Familiar With?': 'team_member_pms_applications',
            'Team Member Workstation Number': 'team_member_workstation',
            'Reasons/Concerns for Re-Hirable Status': 'rehirable_concerns',
            'Team Member Name': 'team_member_name',
            'Client Requested to Interview Replacement': 'client_requested_interview',
            'Timeframe for Replacement': 'replacement_timeframe',
            'Client Time Zone & Working Hours': 'client_timezone',
            'Team Member Position': 'team_member_position',
            'Termination/Resignation Type': 'termination_type',
            'In the case of Immediate Termination/Resignation was an SME seated until replacement is found?': 'sme_seated_until_replacement',
            'Requested Replacements Name(s) for TA:': 'requested_replacement_names',
            'Has Client Been Informed?': 'client_informed',
            'Departments Notified [RCM]': 'notified_rcm',
            'If Temporary Replacement Specify Time Period': 'temporary_replacement_period',
            'Geo - Location': 'geo_location',
            'Was a Performance Improvement Plan Implemented?': 'pip_implemented',
            'Please share the Performance Improvement Plan Link below': 'pip_link',
            'Business Issue Termination - Was the relief package/discounts offered?': 'relief_package_offered',
            'Team Member  Start Date': 'team_member_start_date',
            'Client Start Date:': 'client_start_date',
            'If answered "yes" what was offered?': 'relief_package_details',
            'If no longer replacing, explain below:': 'no_longer_replacing_reason',
            'Current Client Seat Count': 'current_seat_count',
            'Insert Files for any Additional Documentation Here:': 'documentation_files',
            'If Selected Other, Please Specify': 'other_specify',
            'Team Member Email': 'team_member_email',
            'Team Member Phone Number': 'team_member_phone',
            'Have we lost the Client?': 'client_lost',
            'Which geo location(s) does the team member have experience working in?': 'team_member_geo_experience',
            'If Selected Project Based seat, Please specify start and end date of project.': 'project_based_dates',
            'RCM Seat?': 'rcm_seat',
            'BYOD(Bring your own device) Access?': 'byod_access',
            'Time Frame for Replacement': 'replacement_timeframe',
        }
        
        mapping = column_mapping or default_mapping
        
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"Read {len(df)} rows from {csv_path}")
        
        # Rename columns
        df = df.rename(columns=mapping)
        
        # Convert boolean-like columns
        bool_columns = [
            'will_have_replacement', 'client_requested_interview', 'client_lost',
            'pip_implemented', 'relief_package_offered', 'client_informed',
            'notified_finance', 'notified_team_experience', 'notified_talent_acquisition',
            'notified_gts', 'notified_client_solutions', 'notified_rcm',
            'rcm_seat', 'byod_access', 'sme_seated_until_replacement'
        ]
        
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_boolean)
        
        # Remove rows with missing required fields
        if 'client_name' in df.columns:
            original_count = len(df)
            df = df[df['client_name'].notna() & (df['client_name'].str.strip() != '')]
            removed = original_count - len(df)
            if removed > 0:
                print(f"  Filtered out {removed} rows with empty client names")
        
        # Filter to only columns that exist in our table
        valid_columns = self._get_table_columns('historical_terminations')
        df = df[[c for c in df.columns if c in valid_columns]]
        
        # Insert into database
        conn = self._get_connection()
        df.to_sql('historical_terminations', conn, if_exists='append', index=False)
        conn.close()
        
        # Compute features for new records
        self._compute_features()
        
        print(f"✓ Imported {len(df)} termination records")
        return len(df)
    
    def _parse_boolean(self, value) -> int:
        """Convert various boolean representations to 0/1."""
        if pd.isna(value):
            return 0
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return 1 if value else 0
        value_str = str(value).lower().strip()
        return 1 if value_str in ('yes', 'true', '1', 'y', 'x') else 0
    
    def _get_table_columns(self, table_name: str) -> List[str]:
        """Get list of column names for a table."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row['name'] for row in cursor.fetchall()]
        conn.close()
        return columns
    
    def _compute_features(self):
        """Compute ML features for all terminations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get terminations without computed features
        cursor.execute("""
            SELECT t.* FROM historical_terminations t
            LEFT JOIN computed_features cf ON t.id = cf.termination_id
            WHERE cf.termination_id IS NULL
        """)
        
        records = cursor.fetchall()
        
        for record in records:
            features = self._calculate_features(dict(record))
            cursor.execute("""
                INSERT INTO computed_features (
                    termination_id, days_since_communication, days_since_checkin,
                    team_member_tenure_days, client_tenure_at_termination,
                    health_score, checkin_frequency_encoded, no_replacement,
                    replacement_urgency, had_pip, offered_discount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record['id'],
                features['days_since_communication'],
                features['days_since_checkin'],
                features['team_member_tenure_days'],
                features['client_tenure_at_termination'],
                features['health_score'],
                features['checkin_frequency_encoded'],
                features['no_replacement'],
                features['replacement_urgency'],
                features['had_pip'],
                features['offered_discount']
            ))
        
        conn.commit()
        conn.close()
        print(f"✓ Computed features for {len(records)} records")
    
    def _calculate_features(self, record: Dict) -> Dict:
        """Calculate ML features from a termination record."""
        def parse_date(date_str):
            if not date_str:
                return None
            try:
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        return datetime.strptime(str(date_str).split()[0], fmt)
                    except:
                        continue
            except:
                return None
            return None
        
        notice_date = parse_date(record.get('termination_notice_date'))
        last_comm_date = parse_date(record.get('last_general_communication_date'))
        last_checkin_date = parse_date(record.get('last_official_checkin_date'))
        tm_start_date = parse_date(record.get('team_member_start_date'))
        tm_last_date = parse_date(record.get('team_member_last_date'))
        client_start_date = parse_date(record.get('client_start_date'))
        
        # Health score encoding
        health_map = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        health = str(record.get('client_health_last_month', '')).lower()
        health_score = health_map.get(health, 0)
        
        # Checkin frequency encoding
        cadence_map = {'weekly': 1, 'bi-weekly': 2, 'bi-monthly': 2, 'monthly': 3}
        cadence = str(record.get('check_in_cadence', '')).lower()
        checkin_freq = cadence_map.get(cadence, 4)
        
        # Replacement urgency
        urgency_map = {'immediate': 3, 'asap': 3, '1-2 weeks': 2, '2-4 weeks': 1, '1 week': 2}
        timeframe = str(record.get('replacement_timeframe', '')).lower()
        urgency = 0
        for key, val in urgency_map.items():
            if key in timeframe:
                urgency = val
                break
        
        return {
            'days_since_communication': (notice_date - last_comm_date).days if notice_date and last_comm_date else 999,
            'days_since_checkin': (notice_date - last_checkin_date).days if notice_date and last_checkin_date else 999,
            'team_member_tenure_days': (tm_last_date - tm_start_date).days if tm_last_date and tm_start_date else 0,
            'client_tenure_at_termination': (notice_date - client_start_date).days if notice_date and client_start_date else 0,
            'health_score': health_score,
            'checkin_frequency_encoded': checkin_freq,
            'no_replacement': 0 if record.get('will_have_replacement') else 1,
            'replacement_urgency': urgency,
            'had_pip': record.get('pip_implemented', 0),
            'offered_discount': record.get('relief_package_offered', 0)
        }
    
    def get_training_data(self) -> pd.DataFrame:
        """
        Get processed training data for ML model.
        
        Returns:
            DataFrame with features and target variable (client_lost)
        """
        conn = self._get_connection()
        
        df = pd.read_sql_query("""
            SELECT 
                -- Target variable
                COALESCE(t.client_lost, 0) AS client_lost,
                
                -- Computed features
                cf.days_since_communication,
                cf.days_since_checkin,
                cf.team_member_tenure_days,
                cf.client_tenure_at_termination,
                cf.health_score,
                cf.checkin_frequency_encoded,
                cf.no_replacement,
                cf.replacement_urgency,
                cf.had_pip,
                cf.offered_discount,
                
                -- Raw features for additional encoding
                t.current_seat_count,
                t.termination_reason,
                t.termination_type,
                t.geo_location,
                t.client_name,
                t.termination_notice_date
                
            FROM historical_terminations t
            JOIN computed_features cf ON t.id = cf.termination_id
            WHERE t.termination_notice_date IS NOT NULL
        """, conn)
        
        conn.close()
        
        print(f"✓ Retrieved {len(df)} training records")
        return df
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM historical_terminations")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM historical_terminations WHERE client_lost = 1")
        churned = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT client_name) FROM historical_terminations")
        unique_clients = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT client_health_last_month, COUNT(*) 
            FROM historical_terminations 
            GROUP BY client_health_last_month
        """)
        health_dist = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_records': total,
            'churned_clients': churned,
            'churn_rate': churned / total if total > 0 else 0,
            'unique_clients': unique_clients,
            'health_distribution': health_dist
        }
    
    def _convert_to_native(self, obj):
        """Convert numpy types to Python native for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        return str(obj)
    
    def log_model_run(self, model_name: str, version: str, metrics: Dict, 
                      feature_importance: Dict = None, hyperparameters: Dict = None,
                      notes: str = None) -> int:
        """Log a model training run."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Convert numpy types to native Python types
        if feature_importance:
            feature_importance = self._convert_to_native(feature_importance)
        if metrics:
            metrics = self._convert_to_native(metrics)
        
        cursor.execute("""
            INSERT INTO model_runs (
                model_name, model_version, training_records,
                accuracy, precision_score, recall_score, f1_score, auc_roc,
                feature_importance, hyperparameters, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name, version, metrics.get('training_records', 0),
            metrics.get('accuracy'), metrics.get('precision'),
            metrics.get('recall'), metrics.get('f1'), metrics.get('auc_roc'),
            json.dumps(feature_importance) if feature_importance else None,
            json.dumps(hyperparameters) if hyperparameters else None,
            notes
        ))
        
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✓ Logged model run #{run_id}")
        return run_id

    # =====================================================
    # PREDICTION METHODS (for local testing)
    # =====================================================
    
    def save_predictions(self, predictions: pd.DataFrame, model_version: str, 
                         model_name: str = 'xgboost_churn_v1') -> int:
        """
        Save churn predictions to local SQLite database.
        
        Args:
            predictions: DataFrame with columns: client_id, client_name, 
                        churn_probability, risk_level, risk_score, 
                        features (dict), top_risk_factors (list)
            model_version: Version string of the model
            model_name: Name of the model
            
        Returns:
            Number of predictions saved
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        saved = 0
        for _, row in predictions.iterrows():
            try:
                features = row.get('features', {})
                risk_factors = row.get('top_risk_factors', [])
                
                cursor.execute("""
                    INSERT INTO client_churn_predictions (
                        client_id, client_name, churn_probability,
                        risk_level, risk_score, features, top_risk_factors,
                        model_version, model_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get('client_id', ''),
                    row.get('client_name', ''),
                    float(row.get('churn_probability', 0)),
                    row.get('risk_level', 'low'),
                    int(row.get('risk_score', 0)),
                    json.dumps(self._convert_to_native(features)) if features else '{}',
                    json.dumps(self._convert_to_native(risk_factors)) if risk_factors else '[]',
                    model_version,
                    model_name
                ))
                saved += 1
            except Exception as e:
                print(f"⚠️ Error saving prediction for {row.get('client_name')}: {e}")
        
        # Log the prediction run
        high_risk = len(predictions[predictions['risk_level'].isin(['high', 'critical'])]) if 'risk_level' in predictions.columns else 0
        medium_risk = len(predictions[predictions['risk_level'] == 'medium']) if 'risk_level' in predictions.columns else 0
        low_risk = len(predictions[predictions['risk_level'] == 'low']) if 'risk_level' in predictions.columns else 0
        
        cursor.execute("""
            INSERT INTO prediction_runs (
                model_name, model_version, total_predictions,
                high_risk_count, medium_risk_count, low_risk_count, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (model_name, model_version, saved, high_risk, medium_risk, low_risk, 'completed'))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Saved {saved} predictions to local database")
        return saved
    
    def get_latest_predictions(self) -> pd.DataFrame:
        """Get the most recent prediction for each client."""
        conn = self._get_connection()
        
        query = """
            SELECT 
                client_id, client_name, churn_probability,
                risk_level, risk_score, features, top_risk_factors,
                model_version, prediction_date
            FROM client_churn_predictions
            WHERE prediction_date = (
                SELECT MAX(prediction_date) 
                FROM client_churn_predictions p2 
                WHERE p2.client_name = client_churn_predictions.client_name
            )
            ORDER BY risk_score DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_high_risk_clients(self, threshold: float = 0.5) -> pd.DataFrame:
        """Get clients with churn probability above threshold."""
        conn = self._get_connection()
        
        query = f"""
            SELECT DISTINCT
                client_name, MAX(churn_probability) as churn_probability,
                risk_level, risk_score, MAX(prediction_date) as prediction_date
            FROM client_churn_predictions
            WHERE churn_probability >= {threshold}
            GROUP BY client_name
            ORDER BY churn_probability DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_prediction_runs(self, limit: int = 10) -> pd.DataFrame:
        """Get recent prediction runs."""
        conn = self._get_connection()
        
        query = f"""
            SELECT * FROM prediction_runs
            ORDER BY run_date DESC
            LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_prediction_stats(self) -> Dict:
        """Get prediction statistics summary."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM client_churn_predictions")
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT client_name) FROM client_churn_predictions")
        unique_clients = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT risk_level, COUNT(*) 
            FROM (
                SELECT client_name, risk_level
                FROM client_churn_predictions
                WHERE prediction_date = (
                    SELECT MAX(prediction_date) FROM client_churn_predictions p2 
                    WHERE p2.client_name = client_churn_predictions.client_name
                )
            )
            GROUP BY risk_level
        """)
        risk_dist = dict(cursor.fetchall())
        
        cursor.execute("SELECT COUNT(*) FROM prediction_runs")
        total_runs = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'unique_clients_scored': unique_clients,
            'risk_distribution': risk_dist,
            'total_prediction_runs': total_runs
        }


# Convenience function
def get_db(db_path: Optional[str] = None) -> MLDatabase:
    """Get ML database instance."""
    path = Path(db_path) if db_path else None
    return MLDatabase(path)


if __name__ == "__main__":
    # Test database creation
    db = MLDatabase()
    print("\nDatabase stats:", db.get_stats())
