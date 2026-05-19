"""
BusyBee Data Export Utility
===========================
Exports data from Supabase to CSV/JSON for model training.
Can be run locally or as a Lambda function.

Usage:
    python export_training_data.py --output-path ./data
"""

import os
import json
import argparse
from datetime import datetime
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.error("pandas not installed. Run: pip install pandas")

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.error("supabase not installed. Run: pip install supabase")


class DataExporter:
    """Export data from Supabase for ML training."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        if not SUPABASE_AVAILABLE:
            raise ImportError("supabase package not installed")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized")
    
    def export_clients(self, include_inactive: bool = True) -> pd.DataFrame:
        """Export all clients data."""
        logger.info("Exporting clients data...")
        
        query = self.supabase.table('clients').select(
            'id, name, industry_type, client_location, client_success_manager, '
            'sales_person, start_date, status, is_active, health, '
            'attrition_date, attrition_reason, created_at, updated_at'
        )
        
        if not include_inactive:
            query = query.eq('is_active', True)
        
        response = query.execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Exported {len(df)} clients")
            return df
        else:
            logger.warning("No clients found")
            return pd.DataFrame()
    
    def export_team_members(self, include_inactive: bool = True) -> pd.DataFrame:
        """Export all team members data."""
        logger.info("Exporting team members data...")
        
        query = self.supabase.table('team_members').select(
            'id, first_name, last_name, email, role, status, '
            'start_date, client_id, created_at, updated_at'
        )
        
        if not include_inactive:
            query = query.eq('status', 'active')
        
        response = query.execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Exported {len(df)} team members")
            return df
        else:
            logger.warning("No team members found")
            return pd.DataFrame()
    
    def export_attrition_history(self) -> pd.DataFrame:
        """Export historical attrition data."""
        logger.info("Exporting attrition history...")
        
        # Get client attrition
        client_response = self.supabase.table('client_attrition').select('*').execute()
        
        # Get team member attrition  
        tm_response = self.supabase.table('team_member_attrition').select('*').execute()
        
        client_attrition = pd.DataFrame(client_response.data) if client_response.data else pd.DataFrame()
        tm_attrition = pd.DataFrame(tm_response.data) if tm_response.data else pd.DataFrame()
        
        # Add entity type column
        if not client_attrition.empty:
            client_attrition['entity_type'] = 'client'
        if not tm_attrition.empty:
            tm_attrition['entity_type'] = 'team_member'
        
        # Combine
        combined = pd.concat([client_attrition, tm_attrition], ignore_index=True)
        logger.info(f"Exported {len(combined)} attrition records")
        
        return combined
    
    def export_attendance_data(self, days: int = 90) -> pd.DataFrame:
        """Export attendance tracking data."""
        logger.info(f"Exporting last {days} days of attendance data...")
        
        from datetime import timedelta
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        response = self.supabase.table('attendance_tracking').select(
            'id, team_member_id, date, clock_in, clock_out, status, '
            'hours_worked, is_late, created_at'
        ).gte('date', start_date).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Exported {len(df)} attendance records")
            return df
        else:
            logger.warning("No attendance records found")
            return pd.DataFrame()
    
    def export_all(self, output_path: str, format: str = 'csv'):
        """Export all data for training."""
        os.makedirs(output_path, exist_ok=True)
        
        datasets = {
            'clients': self.export_clients(include_inactive=True),
            'team_members': self.export_team_members(include_inactive=True),
            'attrition_history': self.export_attrition_history(),
            'attendance': self.export_attendance_data(days=180)
        }
        
        for name, df in datasets.items():
            if not df.empty:
                if format == 'csv':
                    filepath = os.path.join(output_path, f'{name}.csv')
                    df.to_csv(filepath, index=False)
                elif format == 'json':
                    filepath = os.path.join(output_path, f'{name}.json')
                    df.to_json(filepath, orient='records', date_format='iso')
                else:
                    filepath = os.path.join(output_path, f'{name}.parquet')
                    df.to_parquet(filepath, index=False)
                
                logger.info(f"Saved {name} to {filepath}")
        
        # Save export metadata
        metadata = {
            'exported_at': datetime.now().isoformat(),
            'datasets': {
                name: {
                    'rows': len(df),
                    'columns': list(df.columns) if not df.empty else []
                }
                for name, df in datasets.items()
            }
        }
        
        with open(os.path.join(output_path, 'export_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Export complete. Files saved to: {output_path}")
        return metadata


def main():
    parser = argparse.ArgumentParser(description='Export BusyBee data for ML training')
    parser.add_argument('--output-path', type=str, default='./data', 
                       help='Output directory for exported data')
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'parquet'], 
                       default='csv', help='Output format')
    parser.add_argument('--supabase-url', type=str, 
                       default=os.environ.get('SUPABASE_URL'),
                       help='Supabase project URL')
    parser.add_argument('--supabase-key', type=str,
                       default=os.environ.get('SUPABASE_SERVICE_KEY'),
                       help='Supabase service role key')
    args = parser.parse_args()
    
    if not args.supabase_url or not args.supabase_key:
        logger.error("Supabase credentials required. Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables.")
        return
    
    exporter = DataExporter(args.supabase_url, args.supabase_key)
    exporter.export_all(args.output_path, args.format)


if __name__ == '__main__':
    main()
