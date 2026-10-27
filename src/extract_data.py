
from pathlib import Path
import py7zr

PROJECT_ROOT = Path(__file__).resolve().parent.parent

archives = [
    (
        PROJECT_ROOT / "data/raw/REFIT_RAW_081116.7z",
        PROJECT_ROOT / "data/raw/extracted",
    ),
    (
        PROJECT_ROOT / "data/cleaned/CLEAN_REFIT_081116.7z",
        PROJECT_ROOT / "data/cleaned/extracted",
    ),
]

for archive, destination in archives:
    if not archive.exists():
        print(f"Archive not found: {archive}")
        continue

    destination.mkdir(parents=True, exist_ok=True)

    print(f"Extracting: {archive.name}")

    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=destination)

    print(f"Extracted to: {destination}")

print("Extraction check complete.")
