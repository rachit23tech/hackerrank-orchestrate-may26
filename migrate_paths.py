import os
from pathlib import Path

# Resolves to the root of your HackerRank repository
REPO_ROOT = Path(__file__).resolve().parent

old_dir = REPO_ROOT / "support_tickets"
new_dir = REPO_ROOT / "support_issues"

if old_dir.exists():
    print(f"Found {old_dir.name}/ directory. Starting migration...")
    
    # Rename the files inside the directory
    file_renames = [
        ("support_tickets.csv", "support_issues.csv"),
        ("sample_support_tickets.csv", "sample_support_issues.csv")
    ]
    
    for old_name, new_name in file_renames:
        old_file = old_dir / old_name
        if old_file.exists():
            old_file.rename(old_dir / new_name)
            print(f"  -> Renamed {old_name} to {new_name}")
            
    # Rename the directory itself
    old_dir.rename(new_dir)
    print(f"  -> Renamed directory to {new_dir.name}/")
    print("\n✅ Migration complete! The filesystem now matches the judge's evaluation environment.")
else:
    print(f"The {old_dir.name}/ directory was not found. It may have already been renamed.")