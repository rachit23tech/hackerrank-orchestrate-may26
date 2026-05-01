import zipfile
import shutil
from pathlib import Path

def package_code():
    repo_root = Path(__file__).resolve().parent
    code_dir = repo_root / "code"
    zip_path = repo_root / "code.zip"
    
    if not code_dir.exists():
        print(f"❌ Error: Could not find {code_dir}")
        return

    print(f"📦 Packaging {code_dir.name}/ into {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in code_dir.rglob('*'):
            # Skip cache directories to keep the zip small and clean
            if '__pycache__' in file_path.parts or '.cache' in file_path.parts:
                continue
            
            if file_path.is_file():
                arcname = file_path.relative_to(repo_root)
                zipf.write(file_path, arcname)
                print(f"  Added {arcname}")

    # Attempt to copy log.txt from the home directory to the repo root
    log_source = Path.home() / "hackerrank_orchestrate" / "log.txt"
    log_dest = repo_root / "log.txt"
    
    if log_source.exists():
        print(f"\n📝 Found chat transcript at {log_source}")
        shutil.copy2(log_source, log_dest)
        print(f"  -> Copied to {log_dest.name} for easy submission.")
    else:
        print(f"\n⚠️  Could not automatically locate log.txt at {log_source}")
        print("   Please ensure you locate your chat transcript manually.")

    print(f"\n✅ Success! You can now submit these 3 files to HackerRank:")
    print(f"   1. {zip_path.name}\n   2. support_issues/output.csv\n   3. log.txt")

if __name__ == "__main__":
    package_code()