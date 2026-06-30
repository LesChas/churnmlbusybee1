"""
BusyBee Lambda Inference Handler
================================
AWS Lambda function for running daily batch predictions.
Loads trained models from S3 and stores predictions in Supabase.

Deployment:
    1. Package with dependencies: pip install -t ./package xgboost scikit-learn pandas numpy
    2. Add this file and model files to package
    3. Zip and upload to Lambda
    4. Set environment variables: SUPABASE_URL, SUPABASE_SERVICE_KEY, S3_BUCKET
"""

import os
import json
import pickle
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Import ML dependencies
import pandas as pd
import numpy as np

# Import Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase client not available. Install with: pip install supabase")


class PredictionService:
    """Service class for running predictions."""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.s3_bucket = os.environ.get('S3_BUCKET', 'busybee-ml-models')
        self.supabase: Optional[Client] = None
        
        # Initialize Supabase if available
        if SUPABASE_AVAILABLE:
            supabase_url = os.environ.get('SUPABASE_URL')
            supabase_key = os.environ.get('SUPABASE_SERVICE_KEY')
            
            if supabase_url and supabase_key:
                self.supabase = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized")
            else:
                logger.warning("Supabase credentials not found in environment")
        
        self.churn_model = None
        self.attrition_model = None
        
    def load_model_from_s3(self, model_key: str):
        """Load a pickle model from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=model_key)
            model_data = pickle.loads(response['Body'].read())
            logger.info(f"Model loaded from S3: {model_key}")
            return model_data
        except Exception as e:
            logger.error(f"Error loading model from S3: {e}")
            raise
    
    def load_model_from_local(self, model_path: str):
        """Load a pickle model from local filesystem (for Lambda layer)."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            logger.info(f"Model loaded from local: {model_path}")
            return model_data
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            raise
    
    def fetch_clients(self) -> pd.DataFrame:
        """Fetch active clients from Supabase."""
        if not self.supabase:
            raise ValueError("Supabase client not initialized")
        
        response = self.supabase.table('clients').select(
            'id, name, industry_type, client_location, start_date, '
            'status, is_active, health, attrition_date, created_at, updated_at'
        ).eq('is_active', True).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Fetched {len(df)} active clients")
            return df
        else:
            logger.warning("No clients found")
            return pd.DataFrame()
    
    def fetch_team_members(self) -> pd.DataFrame:
        """Fetch active team members from Supabase."""
        if not self.supabase:
            raise ValueError("Supabase client not initialized")
        
        response = self.supabase.table('team_members').select(
            'id, first_name, last_name, email, role, status, '
            'start_date, client_id, created_at, updated_at'
        ).eq('status', 'active').execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Fetched {len(df)} active team members")
            return df
        else:
            logger.warning("No team members found")
            return pd.DataFrame()
    
    def run_churn_predictions(self, model_data: Dict) -> Dict[str, Any]:
        """Run churn predictions for all active clients."""
        try:
            # Fetch data
            clients_df = self.fetch_clients()
            if clients_df.empty:
                return {'success': True, 'predictions': 0, 'message': 'No clients to predict'}
            
            # Prepare features
            features = self._prepare_churn_features(clients_df, model_data)
            
            # Scale features
            features_scaled = model_data['scaler'].transform(features)
            
            # Use calibrated model if available (priority #4)
            predictor = model_data.get('calibrated_model') or model_data['model']
            
            # Get predictions
            probabilities = predictor.predict_proba(features_scaled)[:, 1]
            
            # Priority #2: Fetch previous predictions for two-strike confirmation
            previous_predictions = self._fetch_previous_predictions('client_churn_predictions', 'client_id')
            
            # Create predictions
            predictions = []
            for idx, row in clients_df.iterrows():
                prob = probabilities[idx]
                risk_score = int(prob * 100)
                
                # Determine risk level (raised thresholds to reduce false positives)
                if prob >= 0.85:
                    risk_level = 'critical'
                elif prob >= 0.65:
                    risk_level = 'high'
                elif prob >= 0.40:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
                
                # Priority #2: Two-strike confirmation
                # Only confirm 'critical' if previous run also scored high/critical
                # This eliminates transient spikes from noisy data
                prev = previous_predictions.get(row['id'])
                confirmed_risk_level = risk_level
                if risk_level == 'critical' and prev:
                    if prev.get('risk_level') not in ('critical', 'high'):
                        confirmed_risk_level = 'high'  # Downgrade until confirmed
                        logger.info(f"Two-strike: {row['id']} downgraded critical→high (prev was {prev.get('risk_level')})")
                elif risk_level == 'high' and prev:
                    if prev.get('risk_level') == 'low':
                        confirmed_risk_level = 'medium'  # Don't jump from low to high in one run
                
                risk_level = confirmed_risk_level
                
                # Get feature values for explainability
                feature_values = features.iloc[idx].to_dict()
                
                # Get risk factors
                risk_factors = self._get_churn_risk_factors(feature_values, prob)
                
                predictions.append({
                    'client_id': row['id'],
                    'prediction_date': datetime.now().isoformat(),
                    'churn_probability': float(prob),
                    'risk_level': risk_level,
                    'risk_score': risk_score,
                    'features': feature_values,
                    'top_risk_factors': risk_factors,
                    'model_version': model_data['model_version'],
                    'model_name': model_data['model_name']
                })
            
            # Store predictions in Supabase
            if self.supabase and predictions:
                # Upsert predictions (one per client per day)
                for pred in predictions:
                    # Convert numpy types to Python types for JSON
                    pred['features'] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                       for k, v in pred['features'].items()}
                    
                self.supabase.table('client_churn_predictions').upsert(
                    predictions,
                    on_conflict='client_id,prediction_date::DATE'
                ).execute()
                logger.info(f"Stored {len(predictions)} churn predictions")
            
            return {
                'success': True,
                'predictions': len(predictions),
                'high_risk_count': sum(1 for p in predictions if p['risk_level'] in ['high', 'critical']),
                'medium_risk_count': sum(1 for p in predictions if p['risk_level'] == 'medium'),
                'low_risk_count': sum(1 for p in predictions if p['risk_level'] == 'low')
            }
            
        except Exception as e:
            logger.error(f"Error running churn predictions: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_attrition_predictions(self, model_data: Dict) -> Dict[str, Any]:
        """Run attrition predictions for all active team members."""
        try:
            # Fetch data
            team_members_df = self.fetch_team_members()
            if team_members_df.empty:
                return {'success': True, 'predictions': 0, 'message': 'No team members to predict'}
            
            # Prepare features
            features = self._prepare_attrition_features(team_members_df, model_data)
            
            # Scale features
            features_scaled = model_data['scaler'].transform(features)
            
            # Use calibrated model if available (priority #4)
            predictor = model_data.get('calibrated_model') or model_data['model']
            
            # Get predictions
            probabilities = predictor.predict_proba(features_scaled)[:, 1]
            
            # Create predictions
            predictions = []
            for idx, row in team_members_df.iterrows():
                prob = probabilities[idx]
                risk_score = int(prob * 100)
                
                # Determine risk level (raised thresholds to reduce false positives)
                if prob >= 0.85:
                    risk_level = 'critical'
                elif prob >= 0.65:
                    risk_level = 'high'
                elif prob >= 0.40:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
                
                # Get feature values
                feature_values = features.iloc[idx].to_dict()
                
                # Get risk factors and recommendations
                risk_factors = self._get_attrition_risk_factors(feature_values, prob)
                recommendations = self._get_attrition_recommendations(feature_values, prob)
                
                # Estimate days to attrition
                predicted_days = int(max(7, 180 * (1 - prob))) if prob > 0.3 else None
                
                predictions.append({
                    'team_member_id': row['id'],
                    'prediction_date': datetime.now().isoformat(),
                    'attrition_probability': float(prob),
                    'risk_level': risk_level,
                    'risk_score': risk_score,
                    'predicted_days_to_attrition': predicted_days,
                    'features': feature_values,
                    'top_risk_factors': risk_factors,
                    'recommended_actions': recommendations,
                    'model_version': model_data['model_version'],
                    'model_name': model_data['model_name']
                })
            
            # Store predictions in Supabase
            if self.supabase and predictions:
                for pred in predictions:
                    pred['features'] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                       for k, v in pred['features'].items()}
                    
                self.supabase.table('team_member_attrition_predictions').upsert(
                    predictions,
                    on_conflict='team_member_id,prediction_date::DATE'
                ).execute()
                logger.info(f"Stored {len(predictions)} attrition predictions")
            
            return {
                'success': True,
                'predictions': len(predictions),
                'high_risk_count': sum(1 for p in predictions if p['risk_level'] in ['high', 'critical']),
                'medium_risk_count': sum(1 for p in predictions if p['risk_level'] == 'medium'),
                'low_risk_count': sum(1 for p in predictions if p['risk_level'] == 'low')
            }
            
        except Exception as e:
            logger.error(f"Error running attrition predictions: {e}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_churn_features(self, df: pd.DataFrame, model_data: Dict) -> pd.DataFrame:
        """Prepare features for churn model."""
        features = pd.DataFrame()
        reference_date = pd.to_datetime('today')
        
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        
        start_dates = df['start_date'].fillna(df['created_at'])
        
        features['tenure_months'] = ((reference_date - start_dates).dt.days / 30).fillna(0).astype(int)
        
        health_mapping = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'unknown': 2}
        features['health_score'] = df['health'].fillna('unknown').str.lower().map(health_mapping).fillna(2)
        
        features['days_since_activity'] = ((reference_date - df['updated_at']).dt.days).fillna(30).clip(0, 365)
        
        # Handle label encoding for industry
        le = model_data['label_encoders'].get('industry_type')
        if le:
            known_labels = set(le.classes_)
            industry_values = df['industry_type'].fillna('OTHER').astype(str)
            industry_values = industry_values.apply(lambda x: x if x in known_labels else 'OTHER')
            features['industry_encoded'] = le.transform(industry_values)
        else:
            features['industry_encoded'] = 0
        
        # Handle label encoding for location
        le = model_data['label_encoders'].get('client_location')
        if le:
            known_labels = set(le.classes_)
            location_values = df['client_location'].fillna('Unknown').astype(str)
            location_values = location_values.apply(lambda x: x if x in known_labels else 'Unknown')
            features['location_encoded'] = le.transform(location_values)
        else:
            features['location_encoded'] = 0
        
        features['is_active'] = df['is_active'].fillna(True).astype(int)
        features['start_month'] = start_dates.dt.month.fillna(1).astype(int)
        features['start_quarter'] = start_dates.dt.quarter.fillna(1).astype(int)
        features['days_since_start'] = (reference_date - start_dates).dt.days.fillna(0).clip(0, 3650)
        
        # Ensure column order matches training
        expected_columns = model_data['feature_columns']
        for col in expected_columns:
            if col not in features.columns:
                features[col] = 0
        features = features[expected_columns]
        
        return features
    
    def _prepare_attrition_features(self, df: pd.DataFrame, model_data: Dict) -> pd.DataFrame:
        """Prepare features for attrition model."""
        features = pd.DataFrame()
        reference_date = pd.to_datetime('today')
        
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        
        start_dates = df['start_date'].fillna(df['created_at'])
        
        features['tenure_months'] = ((reference_date - start_dates).dt.days / 30).fillna(0).astype(int)
        
        # Handle role encoding
        le = model_data['label_encoders'].get('role')
        if le:
            known_labels = set(le.classes_)
            role_values = df['role'].fillna('team_member').astype(str)
            role_values = role_values.apply(lambda x: x if x in known_labels else 'team_member')
            features['role_encoded'] = le.transform(role_values)
        else:
            features['role_encoded'] = 0
        
        features['days_since_activity'] = ((reference_date - df['updated_at']).dt.days).fillna(7).clip(0, 365)
        features['has_client'] = df['client_id'].notna().astype(int)
        
        features['tenure_bucket'] = pd.cut(
            features['tenure_months'],
            bins=[-1, 3, 6, 12, 24, 999],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        features['is_new_employee'] = (features['tenure_months'] <= 3).astype(int)
        features['start_month'] = start_dates.dt.month.fillna(1).astype(int)
        features['start_day_of_week'] = start_dates.dt.dayofweek.fillna(0).astype(int)
        
        # Ensure column order matches training
        expected_columns = model_data['feature_columns']
        for col in expected_columns:
            if col not in features.columns:
                features[col] = 0
        features = features[expected_columns]
        
        return features
    
    def _fetch_previous_predictions(self, table: str, id_column: str) -> Dict[str, Dict]:
        """
        Priority #2: Fetch the most recent prediction for each entity.
        Used for two-strike confirmation - prevents single-run false alarms.
        Returns dict of {entity_id: {risk_level, probability, date}}.
        """
        if not self.supabase:
            return {}
        
        try:
            response = self.supabase.table(table).select(
                f'{id_column}, risk_level, churn_probability, prediction_date'
            ).order('prediction_date', desc=True).limit(500).execute()
            
            if not response.data:
                return {}
            
            # Keep only the most recent prediction per entity
            latest = {}
            for row in response.data:
                eid = row[id_column]
                if eid not in latest:
                    latest[eid] = {
                        'risk_level': row.get('risk_level'),
                        'probability': row.get('churn_probability'),
                        'date': row.get('prediction_date')
                    }
            
            logger.info(f"Fetched {len(latest)} previous predictions for two-strike check")
            return latest
            
        except Exception as e:
            logger.warning(f"Could not fetch previous predictions (non-fatal): {e}")
            return {}
    
    def record_feedback(self, client_id: str, prediction_date: str, 
                       actually_churned: bool, feedback_notes: str = ''):
        """
        Priority #6: Record whether a flagged client actually churned.
        This feedback is used for model retraining and tracking false positive rate.
        """
        if not self.supabase:
            logger.warning("Cannot record feedback - Supabase not available")
            return
        
        try:
            feedback_data = {
                'client_id': client_id,
                'prediction_date': prediction_date,
                'actually_churned': actually_churned,
                'feedback_date': datetime.now().isoformat(),
                'notes': feedback_notes
            }
            
            self.supabase.table('ml_prediction_feedback').upsert(
                feedback_data,
                on_conflict='client_id,prediction_date'
            ).execute()
            
            logger.info(f"Feedback recorded for {client_id}: churned={actually_churned}")
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
    
    def get_false_positive_rate(self, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Priority #6: Calculate false positive rate from feedback data.
        Returns metrics on model accuracy based on real outcomes.
        """
        if not self.supabase:
            return {'error': 'Supabase not available'}
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
            
            response = self.supabase.table('ml_prediction_feedback').select(
                'actually_churned, prediction_date'
            ).gte('feedback_date', cutoff_date).execute()
            
            if not response.data:
                return {'feedback_count': 0, 'message': 'No feedback data yet'}
            
            total = len(response.data)
            actually_churned = sum(1 for r in response.data if r['actually_churned'])
            false_positives = total - actually_churned
            
            return {
                'feedback_count': total,
                'true_positives': actually_churned,
                'false_positives': false_positives,
                'false_positive_rate': false_positives / total if total > 0 else 0,
                'precision_actual': actually_churned / total if total > 0 else 0,
                'lookback_days': lookback_days
            }
        except Exception as e:
            logger.error(f"Error calculating FP rate: {e}")
            return {'error': str(e)}

    def _get_churn_risk_factors(self, features: Dict, probability: float) -> List[Dict]:
        """Get risk factors for churn prediction."""
        risk_factors = []
        
        if features.get('health_score', 3) <= 2:
            risk_factors.append({
                'factor': 'Poor Health Score',
                'impact': 'high',
                'description': 'Client health is rated as fair or poor'
            })
        
        if features.get('days_since_activity', 0) > 30:
            risk_factors.append({
                'factor': 'Low Recent Activity',
                'impact': 'medium',
                'description': f"No activity in {int(features.get('days_since_activity', 0))} days"
            })
        
        if features.get('tenure_months', 12) < 3:
            risk_factors.append({
                'factor': 'New Client',
                'impact': 'medium',
                'description': 'Client is in critical first 3 months'
            })
        
        return risk_factors[:3]
    
    def _get_attrition_risk_factors(self, features: Dict, probability: float) -> List[Dict]:
        """Get risk factors for attrition prediction."""
        risk_factors = []
        
        if features.get('is_new_employee', 0) == 1:
            risk_factors.append({
                'factor': 'New Employee Period',
                'impact': 'high',
                'description': 'First 90 days are a critical retention period'
            })
        
        if features.get('days_since_activity', 0) > 14:
            risk_factors.append({
                'factor': 'Low Engagement',
                'impact': 'medium',
                'description': f"No recent activity in {int(features.get('days_since_activity', 0))} days"
            })
        
        if features.get('has_client', 1) == 0:
            risk_factors.append({
                'factor': 'No Client Assignment',
                'impact': 'medium',
                'description': 'Team member not assigned to a client'
            })
        
        return risk_factors[:3]
    
    def _get_attrition_recommendations(self, features: Dict, probability: float) -> List[Dict]:
        """Get recommendations for attrition risk."""
        recommendations = []
        
        if probability > 0.7:
            recommendations.append({
                'action': 'Immediate 1:1 Meeting',
                'priority': 'critical',
                'description': 'Schedule urgent conversation to understand concerns'
            })
        
        if features.get('is_new_employee', 0) == 1:
            recommendations.append({
                'action': 'Onboarding Check-in',
                'priority': 'high',
                'description': 'Ensure smooth onboarding experience'
            })
        
        if probability > 0.5:
            recommendations.append({
                'action': 'Career Development Discussion',
                'priority': 'medium',
                'description': 'Discuss growth opportunities'
            })
        
        return recommendations[:3]
    
    def log_prediction_run(self, model_name: str, model_version: str, 
                          results: Dict, start_time: datetime, end_time: datetime):
        """Log prediction run to database."""
        if not self.supabase:
            return
        
        duration = int((end_time - start_time).total_seconds())
        
        run_data = {
            'run_date': datetime.now().isoformat(),
            'model_name': model_name,
            'model_version': model_version,
            'total_predictions': results.get('predictions', 0),
            'successful_predictions': results.get('predictions', 0) if results.get('success') else 0,
            'failed_predictions': 0 if results.get('success') else 1,
            'high_risk_count': results.get('high_risk_count', 0),
            'medium_risk_count': results.get('medium_risk_count', 0),
            'low_risk_count': results.get('low_risk_count', 0),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'status': 'completed' if results.get('success') else 'failed',
            'error_message': results.get('error')
        }
        
        self.supabase.table('ml_prediction_runs').insert(run_data).execute()


def lambda_handler(event, context):
    """
    AWS Lambda handler for batch predictions.
    
    Triggered by EventBridge (CloudWatch Events) on a daily schedule.
    
    Event structure:
    {
        "model_type": "all" | "churn" | "attrition",
        "model_source": "s3" | "local"
    }
    """
    logger.info(f"Lambda invoked with event: {json.dumps(event)}")
    
    start_time = datetime.now()
    
    # Parse event
    model_type = event.get('model_type', 'all')
    model_source = event.get('model_source', 'local')
    
    # Initialize service
    service = PredictionService()
    
    results = {
        'churn': None,
        'attrition': None
    }
    
    try:
        # Run churn predictions
        if model_type in ['all', 'churn']:
            logger.info("Running churn predictions...")
            
            if model_source == 's3':
                churn_model_data = service.load_model_from_s3('models/churn_model.pkl')
            else:
                churn_model_data = service.load_model_from_local('/opt/ml/models/churn_model.pkl')
            
            churn_start = datetime.now()
            results['churn'] = service.run_churn_predictions(churn_model_data)
            churn_end = datetime.now()
            
            service.log_prediction_run(
                'xgboost_churn_v1',
                churn_model_data.get('model_version', '1.0.0'),
                results['churn'],
                churn_start,
                churn_end
            )
        
        # Run attrition predictions
        if model_type in ['all', 'attrition']:
            logger.info("Running attrition predictions...")
            
            if model_source == 's3':
                attrition_model_data = service.load_model_from_s3('models/attrition_model.pkl')
            else:
                attrition_model_data = service.load_model_from_local('/opt/ml/models/attrition_model.pkl')
            
            attrition_start = datetime.now()
            results['attrition'] = service.run_attrition_predictions(attrition_model_data)
            attrition_end = datetime.now()
            
            service.log_prediction_run(
                'xgboost_attrition_v1',
                attrition_model_data.get('model_version', '1.0.0'),
                results['attrition'],
                attrition_start,
                attrition_end
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'body': {
                'success': True,
                'results': results,
                'duration_seconds': duration,
                'timestamp': end_time.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        return {
            'statusCode': 500,
            'body': {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        }


# For local testing
if __name__ == '__main__':
    # Simulate Lambda event
    test_event = {
        'model_type': 'all',
        'model_source': 'local'
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2, default=str))
