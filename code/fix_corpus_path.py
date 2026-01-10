# Quick fix for Bangla corpus path
# Run this BEFORE training

from pathlib import Path
import shutil

# Paths
curated_path = Path('data/bangla/bangla_curated.txt')
default_path = Path('data/bangla/bangla_slm.txt')

# Ensure directory exists
default_path.parent.mkdir(parents=True, exist_ok=True)

# Copy curated corpus to default location
if curated_path.exists():
    print(f"Copying {curated_path} → {default_path}")
    shutil.copy(curated_path, default_path)
    print("✓ Done! Training can now find the corpus.")
elif default_path.exists():
    print("✓ Corpus already at default location")
else:
    print("❌ ERROR: No corpus found!")
    print(f"   Expected at: {curated_path}")
    print("   Run the corpus download cell first!")
