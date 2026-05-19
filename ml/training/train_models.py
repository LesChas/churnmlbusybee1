"""
BusyBee ML Model Training Script
================================
Trains XGBoost models for client churn and team member attrition prediction.
Exports models in pickle format for Lambda deployment.

Usage:
    python train_models.py --data-path ./data --output-path ./models
"""

import os
import json
import pickle
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# For SHAP explanations (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Feature importance will use XGBoost built-in.")


class ChurnPredictionModel:
    """XGBoost model for client churn prediction."""
    
    def __init__(self, model_version: str = "1.0.0"):
        self.model_version = model_version
        self.model_name = "xgboost_churn_v1"
        self.model = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw client data.
        
        Expected columns:
        - id, name, industry_type, client_location, start_date, 
        - status, is_active, health, attrition_date, created_at, updated_at
        """
        features = pd.DataFrame()
        
        # Calculate tenure in months
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        reference_date = pd.to_datetime('today')
        
        start_dates = df['start_date'].fillna(df['created_at'])
        features['tenure_months'] = ((reference_date - start_dates).dt.days / 30).fillna(0).astype(int)
        
        # Health score (encode categorical)
        health_mapping = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'unknown': 2}
        features['health_score'] = df['health'].fillna('unknown').str.lower().map(health_mapping).fillna(2)
        
        # Days since last activity
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        features['days_since_activity'] = ((reference_date - df['updated_at']).dt.days).fillna(30).clip(0, 365)
        
        # Industry type (encode)
        if 'industry_type' not in self.label_encoders:
            self.label_encoders['industry_type'] = LabelEncoder()
            features['industry_encoded'] = self.label_encoders['industry_type'].fit_transform(
                df['industry_type'].fillna('OTHER').astype(str)
            )
        else:
            # Handle unseen labels
            known_labels = set(self.label_encoders['industry_type'].classes_)
            industry_values = df['industry_type'].fillna('OTHER').astype(str)
            industry_values = industry_values.apply(lambda x: x if x in known_labels else 'OTHER')
            features['industry_encoded'] = self.label_encoders['industry_type'].transform(industry_values)
        
        # Location (encode)
        if 'client_location' not in self.label_encoders:
            self.label_encoders['client_location'] = LabelEncoder()
            features['location_encoded'] = self.label_encoders['client_location'].fit_transform(
                df['client_location'].fillna('Unknown').astype(str)
            )
        else:
            known_labels = set(self.label_encoders['client_location'].classes_)
            location_values = df['client_location'].fillna('Unknown').astype(str)
            location_values = location_values.apply(lambda x: x if x in known_labels else 'Unknown')
            features['location_encoded'] = self.label_encoders['client_location'].transform(location_values)
        
        # Is active flag
        features['is_active'] = df['is_active'].fillna(True).astype(int)
        
        # Month of start (seasonality)
        features['start_month'] = start_dates.dt.month.fillna(1).astype(int)
        
        # Quarter of start
        features['start_quarter'] = start_dates.dt.quarter.fillna(1).astype(int)
        
        # Days since start (another tenure measure)
        features['days_since_start'] = (reference_date - start_dates).dt.days.fillna(0).clip(0, 3650)
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable (churned = 1, active = 0)."""
        # Churned if attrition_date is set OR status is inactive OR is_active is False
        churned = (
            df['attrition_date'].notna() |
            (df['status'].str.lower() == 'inactive') |
            (df['is_active'] == False)
        )
        return churned.astype(int)
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """Train the XGBoost model."""
        print(f"\n{'='*50}")
        print("Training Client Churn Prediction Model")
        print(f"{'='*50}")
        
        # Prepare data
        X = self.prepare_features(df)
        y = self.prepare_target(df)
        
        print(f"Total samples: {len(y)}")
        print(f"Churned: {y.sum()} ({100*y.mean():.1f}%)")
        print(f"Active: {len(y) - y.sum()} ({100*(1-y.mean()):.1f}%)")
        
        # Handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = dict(zip(np.unique(y), class_weights))
        sample_weights = y.map(weight_dict)
        
        # Split data
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # XGBoost parameters optimized for small datasets
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 4,  # Shallow to prevent overfitting on small data
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'use_label_encoder': False
        }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train_scaled, y_train,
            sample_weight=w_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'training_samples': len(y_train),
            'test_samples': len(y_test),
            'positive_rate': float(y.mean())
        }
        
        print(f"\nModel Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.3f}")
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_columns, importance))
        
        print(f"\nTop Features:")
        for feat, imp in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feat}: {imp:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'client_id': df['id'],
            'churn_probability': probabilities,
            'risk_score': (probabilities * 100).astype(int),
            'risk_level': pd.cut(
                probabilities,
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['low', 'medium', 'high', 'critical']
            )
        })
        
        # Add top risk factors for each prediction
        predictions['top_risk_factors'] = predictions.apply(
            lambda row: self._get_risk_factors(X.iloc[row.name], row['churn_probability']),
            axis=1
        )
        
        # Add features used
        predictions['features'] = X.apply(lambda row: row.to_dict(), axis=1)
        
        return predictions
    
    def _get_risk_factors(self, features: pd.Series, probability: float) -> List[Dict]:
        """Get top risk factors for a prediction."""
        risk_factors = []
        
        # Check each feature against thresholds
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
        
        if features.get('tenure_months', 12) > 24 and probability > 0.5:
            risk_factors.append({
                'factor': 'Long-term Client at Risk',
                'impact': 'high',
                'description': 'Established client showing churn signals'
            })
        
        # Sort by impact and return top 3
        impact_order = {'high': 0, 'medium': 1, 'low': 2}
        risk_factors.sort(key=lambda x: impact_order.get(x['impact'], 3))
        
        return risk_factors[:3]
    
    def save(self, path: str):
        """Save model to pickle file."""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version,
            'model_name': self.model_name,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ChurnPredictionModel':
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_version=model_data['model_version'])
        instance.model = model_data['model']
        instance.label_encoders = model_data['label_encoders']
        instance.scaler = model_data['scaler']
        instance.feature_columns = model_data['feature_columns']
        instance.feature_importance = model_data['feature_importance']
        instance.model_name = model_data['model_name']
        
        return instance


class AttritionPredictionModel:
    """XGBoost model for team member attrition prediction."""
    
    def __init__(self, model_version: str = "1.0.0"):
        self.model_version = model_version
        self.model_name = "xgboost_attrition_v1"
        self.model = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw team member data.
        
        Expected columns:
        - id, first_name, last_name, email, role, status,
        - start_date, client_id, created_at, updated_at
        """
        features = pd.DataFrame()
        
        # Calculate tenure in months
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        reference_date = pd.to_datetime('today')
        
        start_dates = df['start_date'].fillna(df['created_at'])
        features['tenure_months'] = ((reference_date - start_dates).dt.days / 30).fillna(0).astype(int)
        
        # Role encoding
        if 'role' not in self.label_encoders:
            self.label_encoders['role'] = LabelEncoder()
            features['role_encoded'] = self.label_encoders['role'].fit_transform(
                df['role'].fillna('team_member').astype(str)
            )
        else:
            known_labels = set(self.label_encoders['role'].classes_)
            role_values = df['role'].fillna('team_member').astype(str)
            role_values = role_values.apply(lambda x: x if x in known_labels else 'team_member')
            features['role_encoded'] = self.label_encoders['role'].transform(role_values)
        
        # Days since last activity
        df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        features['days_since_activity'] = ((reference_date - df['updated_at']).dt.days).fillna(7).clip(0, 365)
        
        # Has client assignment
        features['has_client'] = df['client_id'].notna().astype(int)
        
        # Tenure buckets (critical periods)
        features['tenure_bucket'] = pd.cut(
            features['tenure_months'],
            bins=[-1, 3, 6, 12, 24, 999],
            labels=[0, 1, 2, 3, 4]  # 0-3m, 3-6m, 6-12m, 12-24m, 24m+
        ).astype(int)
        
        # Is new employee (first 90 days - high risk period)
        features['is_new_employee'] = (features['tenure_months'] <= 3).astype(int)
        
        # Start month (seasonality)
        features['start_month'] = start_dates.dt.month.fillna(1).astype(int)
        
        # Day of week started (potential indicator)
        features['start_day_of_week'] = start_dates.dt.dayofweek.fillna(0).astype(int)
        
        self.feature_columns = features.columns.tolist()
        return features
    
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable (left = 1, active = 0)."""
        left = (df['status'].str.lower() != 'active')
        return left.astype(int)
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """Train the XGBoost model."""
        print(f"\n{'='*50}")
        print("Training Team Member Attrition Prediction Model")
        print(f"{'='*50}")
        
        # Prepare data
        X = self.prepare_features(df)
        y = self.prepare_target(df)
        
        print(f"Total samples: {len(y)}")
        print(f"Left: {y.sum()} ({100*y.mean():.1f}%)")
        print(f"Active: {len(y) - y.sum()} ({100*(1-y.mean()):.1f}%)")
        
        # Handle class imbalance
        if len(np.unique(y)) > 1:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            weight_dict = dict(zip(np.unique(y), class_weights))
            sample_weights = y.map(weight_dict)
        else:
            sample_weights = pd.Series([1.0] * len(y))
        
        # Split data
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=test_size, random_state=42, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'use_label_encoder': False
        }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train_scaled, y_train,
            sample_weight=w_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'training_samples': len(y_train),
            'test_samples': len(y_test),
            'positive_rate': float(y.mean())
        }
        
        print(f"\nModel Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.3f}")
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_columns, importance))
        
        print(f"\nTop Features:")
        for feat, imp in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feat}: {imp:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'team_member_id': df['id'],
            'attrition_probability': probabilities,
            'risk_score': (probabilities * 100).astype(int),
            'risk_level': pd.cut(
                probabilities,
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['low', 'medium', 'high', 'critical']
            )
        })
        
        # Estimate days to potential attrition (based on probability)
        predictions['predicted_days_to_attrition'] = predictions['attrition_probability'].apply(
            lambda p: int(max(7, 180 * (1 - p))) if p > 0.3 else None
        )
        
        # Add risk factors and recommendations
        predictions['top_risk_factors'] = predictions.apply(
            lambda row: self._get_risk_factors(X.iloc[row.name], row['attrition_probability']),
            axis=1
        )
        
        predictions['recommended_actions'] = predictions.apply(
            lambda row: self._get_recommendations(X.iloc[row.name], row['attrition_probability']),
            axis=1
        )
        
        # Add features used
        predictions['features'] = X.apply(lambda row: row.to_dict(), axis=1)
        
        return predictions
    
    def _get_risk_factors(self, features: pd.Series, probability: float) -> List[Dict]:
        """Get top risk factors for a prediction."""
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
        
        tenure_months = features.get('tenure_months', 12)
        if 6 <= tenure_months <= 12:
            risk_factors.append({
                'factor': '6-12 Month Window',
                'impact': 'medium',
                'description': 'Common attrition period for evaluating long-term fit'
            })
        
        impact_order = {'high': 0, 'medium': 1, 'low': 2}
        risk_factors.sort(key=lambda x: impact_order.get(x['impact'], 3))
        
        return risk_factors[:3]
    
    def _get_recommendations(self, features: pd.Series, probability: float) -> List[Dict]:
        """Get recommended actions based on risk factors."""
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
                'description': 'Ensure smooth onboarding experience and address early concerns'
            })
        
        if features.get('has_client', 1) == 0:
            recommendations.append({
                'action': 'Client Assignment Review',
                'priority': 'medium',
                'description': 'Consider assigning to an appropriate client project'
            })
        
        if probability > 0.5:
            recommendations.append({
                'action': 'Career Development Discussion',
                'priority': 'medium',
                'description': 'Discuss growth opportunities and career path'
            })
        
        return recommendations[:3]
    
    def save(self, path: str):
        """Save model to pickle file."""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version,
            'model_name': self.model_name,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AttritionPredictionModel':
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_version=model_data['model_version'])
        instance.model = model_data['model']
        instance.label_encoders = model_data['label_encoders']
        instance.scaler = model_data['scaler']
        instance.feature_columns = model_data['feature_columns']
        instance.feature_importance = model_data['feature_importance']
        instance.model_name = model_data['model_name']
        
        return instance


def generate_sample_data(n_clients: int = 100, n_team_members: int = 300) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample training data for development/testing."""
    np.random.seed(42)
    
    # Generate client data
    industries = ['HEALTHCARE', 'TECHNOLOGY', 'FINANCE', 'RETAIL', 'EDUCATION', 'OTHER']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Unknown']
    health_options = ['excellent', 'good', 'fair', 'poor']
    
    clients = []
    for i in range(n_clients):
        start_date = datetime.now() - timedelta(days=np.random.randint(30, 730))
        is_churned = np.random.random() < 0.15  # 15% churn rate
        
        clients.append({
            'id': f'client-{i}',
            'name': f'Client {i}',
            'industry_type': np.random.choice(industries),
            'client_location': np.random.choice(locations),
            'start_date': start_date.isoformat(),
            'status': 'inactive' if is_churned else 'active',
            'is_active': not is_churned,
            'health': np.random.choice(health_options, p=[0.3, 0.4, 0.2, 0.1]),
            'attrition_date': (start_date + timedelta(days=np.random.randint(60, 365))).isoformat() if is_churned else None,
            'created_at': start_date.isoformat(),
            'updated_at': (datetime.now() - timedelta(days=np.random.randint(0, 60))).isoformat()
        })
    
    clients_df = pd.DataFrame(clients)
    
    # Generate team member data
    roles = ['team_member', 'team_lead', 'manager', 'senior', 'junior']
    
    team_members = []
    for i in range(n_team_members):
        start_date = datetime.now() - timedelta(days=np.random.randint(30, 730))
        is_left = np.random.random() < 0.20  # 20% attrition rate
        
        team_members.append({
            'id': f'tm-{i}',
            'first_name': f'First{i}',
            'last_name': f'Last{i}',
            'email': f'team{i}@company.com',
            'role': np.random.choice(roles),
            'status': 'inactive' if is_left else 'active',
            'start_date': start_date.isoformat(),
            'client_id': f'client-{np.random.randint(0, n_clients)}' if np.random.random() > 0.1 else None,
            'created_at': start_date.isoformat(),
            'updated_at': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat()
        })
    
    team_members_df = pd.DataFrame(team_members)
    
    return clients_df, team_members_df


def main():
    parser = argparse.ArgumentParser(description='Train BusyBee ML Models')
    parser.add_argument('--data-path', type=str, default='./data', help='Path to training data')
    parser.add_argument('--output-path', type=str, default='./models', help='Path to save models')
    parser.add_argument('--use-sample-data', action='store_true', help='Use generated sample data')
    parser.add_argument('--model-version', type=str, default='1.0.0', help='Model version')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load or generate data
    if args.use_sample_data:
        print("Using generated sample data for training...")
        clients_df, team_members_df = generate_sample_data()
    else:
        # Load from CSV files
        clients_path = os.path.join(args.data_path, 'clients.csv')
        team_members_path = os.path.join(args.data_path, 'team_members.csv')
        
        if os.path.exists(clients_path):
            clients_df = pd.read_csv(clients_path)
        else:
            print(f"Warning: {clients_path} not found. Using sample data.")
            clients_df, _ = generate_sample_data()
        
        if os.path.exists(team_members_path):
            team_members_df = pd.read_csv(team_members_path)
        else:
            print(f"Warning: {team_members_path} not found. Using sample data.")
            _, team_members_df = generate_sample_data()
    
    # Train and save churn model
    churn_model = ChurnPredictionModel(model_version=args.model_version)
    churn_metrics = churn_model.train(clients_df)
    churn_model.save(os.path.join(args.output_path, 'churn_model.pkl'))
    
    # Train and save attrition model
    attrition_model = AttritionPredictionModel(model_version=args.model_version)
    attrition_metrics = attrition_model.train(team_members_df)
    attrition_model.save(os.path.join(args.output_path, 'attrition_model.pkl'))
    
    # Helper to convert numpy types to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Save metrics
    metrics_output = {
        'churn_model': {
            'model_name': churn_model.model_name,
            'model_version': args.model_version,
            'metrics': convert_to_serializable(churn_metrics),
            'feature_importance': convert_to_serializable(churn_model.feature_importance)
        },
        'attrition_model': {
            'model_name': attrition_model.model_name,
            'model_version': args.model_version,
            'metrics': convert_to_serializable(attrition_metrics),
            'feature_importance': convert_to_serializable(attrition_model.feature_importance)
        },
        'trained_at': datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_path, 'training_metrics.json'), 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    print(f"Models saved to: {args.output_path}")
    print(f"  - churn_model.pkl")
    print(f"  - attrition_model.pkl")
    print(f"  - training_metrics.json")


if __name__ == '__main__':
    main()
