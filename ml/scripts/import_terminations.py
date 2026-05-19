#!/usr/bin/env python3
"""
BusyBee ML - Import Historical Terminations
============================================
Import your spreadsheet data into the local SQLite database.

Usage:
    python import_terminations.py path/to/your/terminations.csv
    
    # Or from Google Sheets export:
    python import_terminations.py "Historical Terminations Export.csv"
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.ml_db import MLDatabase


def main():
    if len(sys.argv) < 2:
        print("Usage: python import_terminations.py <csv_file_path>")
        print("\nExample:")
        print("  python import_terminations.py data/historical_terminations.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("BusyBee ML - Historical Terminations Import")
    print(f"{'='*60}\n")
    
    # Initialize database
    db = MLDatabase()
    
    # Import CSV
    print(f"\nImporting from: {csv_path}")
    count = db.import_terminations_csv(csv_path)
    
    # Show stats
    print(f"\n{'='*60}")
    print("Database Statistics After Import:")
    print(f"{'='*60}")
    
    stats = db.get_stats()
    print(f"  Total records:      {stats['total_records']}")
    print(f"  Churned clients:    {stats['churned_clients']}")
    print(f"  Churn rate:         {stats['churn_rate']:.1%}")
    print(f"  Unique clients:     {stats['unique_clients']}")
    print(f"\n  Health Distribution:")
    for health, count in stats['health_distribution'].items():
        print(f"    {health or 'Unknown'}: {count}")
    
    print(f"\n✓ Import complete! Database: ml/data/ml_training.db")
    print(f"\nNext step: Run 'python train_churn_model.py' to train the model")


if __name__ == "__main__":
    main()
