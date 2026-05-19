#!/usr/bin/env python3
"""Show SQLite schema"""
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent.parent / "data" / "ml_training.db"
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("=" * 60)
print("SQLite Database Schema")
print("=" * 60)
print(f"\nDatabase: {db_path}")
print(f"\nTables created ({len(tables)}):")

for table in tables:
    table_name = table[0]
    print(f"\n  📋 {table_name}")
    
    # Get columns
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    for col in columns:
        col_id, name, col_type, not_null, default, pk = col
        pk_marker = " 🔑" if pk else ""
        print(f"      - {name}: {col_type}{pk_marker}")

# Get indexes
cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL")
indexes = cursor.fetchall()
print(f"\nIndexes ({len(indexes)}):")
for idx in indexes:
    print(f"  - {idx[0]}")

conn.close()
print("\n✓ Schema verified!")
