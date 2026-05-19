#!/usr/bin/env python3
"""
Initialize ML Database - Quick verification script
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.ml_db import MLDatabase

if __name__ == "__main__":
    print("Initializing ML Database...")
    db = MLDatabase()
    print(f"\nDatabase location: {db.db_path}")
    print(f"Stats: {db.get_stats()}")
    print("\n✓ Database ready!")
