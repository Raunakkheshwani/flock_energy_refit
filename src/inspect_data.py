
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIRS = [
    PROJECT_ROOT / "data/raw/extracted",
    PROJECT_ROOT / "data/cleaned/extracted",
]

for directory in DATA_DIRS:
    print(f"\n{'=' * 60}")
    print(f"Directory: {directory}")

    if not directory.exists():
        print("Directory does not exist yet.")
        continue

    csv_files = sorted(directory.rglob("*.csv"))

    print(f"CSV files found: {len(csv_files)}")

    for file_path in csv_files[:5]:
        print(f"\nFile: {file_path.name}")
        print(f"Size: {file_path.stat().st_size / (1024**2):.2f} MB")

        try:
            sample = pd.read_csv(file_path, nrows=5)

            print("Columns:")
            print(sample.columns.tolist())

            print("\nSample rows:")
            print(sample.head(3).to_string(index=False))

        except Exception as error:
            print(f"Could not inspect file: {error}")
