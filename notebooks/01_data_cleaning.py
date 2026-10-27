"""
Notebook 1: Data Cleaning — Raw House 1 Data
=============================================
This script replicates exactly what the notebook does, in plain Python.
Run it directly: python notebooks/01_data_cleaning.py
All figures are saved to outputs/figures/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import (
    PROJECT_ROOT, RAW_DIR, OUTPUTS_DIR,
    HOUSE1_LABELS,
    load_raw_house1, basic_clean, resample_1min, detect_gaps,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1.1 — Load raw data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Section 1: Basic Inspection & Cleaning — House 1 (Raw)")
print("=" * 60)

print("\n[1.1] Loading raw House 1 data (Parts 1 & 2)...")
df_raw = load_raw_house1()
print(f"  Shape: {df_raw.shape}")
print(f"  Columns: {df_raw.columns.tolist()}")
print(f"  Date range (before parsing): {df_raw['Time'].min()} → {df_raw['Time'].max()}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1.2 — Initial inspection findings
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1.2] Inspection findings:")

# Sampling intervals
df_raw["Time"] = pd.to_datetime(df_raw["Time"])
df_sorted = df_raw.sort_values("Time")
intervals = df_sorted["Time"].diff().dt.total_seconds().dropna()

print(f"\n  Sampling interval statistics (seconds):")
print(f"    Median: {intervals.median():.0f}s")
print(f"    Mean:   {intervals.mean():.1f}s")
print(f"    Min:    {intervals.min():.0f}s")
print(f"    Max:    {intervals.max():.0f}s")
print(f"    Pct within [7,9]s: {((intervals >= 7) & (intervals <= 9)).mean()*100:.1f}%")

# Duplicate timestamps
n_dupes = df_raw.duplicated(subset="Time").sum()
print(f"\n  Duplicate timestamps: {n_dupes:,} ({n_dupes/len(df_raw)*100:.2f}%)")

# Missing / NaN check
power_cols = ["Aggregate", "Appliance1", "Appliance2", "Appliance3",
              "Appliance4", "Appliance5", "Appliance6", "Appliance7",
              "Appliance8", "Appliance9"]
for col in power_cols[:3]:  # print first 3 for brevity
    nan_pct = df_raw[col].isna().mean() * 100
    neg_pct = (df_raw[col] < 0).sum() / len(df_raw) * 100
    print(f"  {col}: NaN={nan_pct:.2f}%, Negative={neg_pct:.2f}%")

# Obvious impossible values
n_extreme = (df_raw["Aggregate"] > 25000).sum()
print(f"\n  Readings > 25,000W (impossible): {n_extreme}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1.3 — Apply cleaning pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1.3] Applying cleaning pipeline...")

df_clean = basic_clean(df_raw)
print(f"  After deduplication: {len(df_clean):,} rows (removed {len(df_raw)-len(df_clean):,})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1.4 — Resample to 1-minute
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1.4] Resampling to 1-minute intervals...")
df_1min = resample_1min(df_clean)
print(f"  1-minute rows: {len(df_1min):,}")
print(f"  Date range: {df_1min.index.min()} → {df_1min.index.max()}")

# Detect long gaps
gaps = detect_gaps(df_1min, threshold_min=30)
print(f"\n  Gaps > 30 minutes: {len(gaps)}")
if len(gaps) > 0:
    print(gaps.head(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1.5 — Plot: Before cleaning (raw 8-second data, 1 week sample)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1.5] Creating plots...")

# Use a 1-week window for visibility
sample_start = "2014-01-06"
sample_end   = "2014-01-13"

# Raw sample
raw_sample = df_raw.copy()
raw_sample["Time"] = pd.to_datetime(raw_sample["Time"])
raw_sample = raw_sample.set_index("Time").sort_index()
raw_sample = raw_sample.loc[sample_start:sample_end]

# 1-min sample
min1_sample = df_1min.loc[sample_start:sample_end]

fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col")
fig.suptitle("House 1 — Raw vs Cleaned (1-week sample: 6–13 Jan 2014)", fontsize=14, fontweight="bold")

channels = ["Aggregate", "Appliance1", "Appliance4"]  # Aggregate, Fridge, Tumble Dryer
labels   = ["Aggregate Power (W)", "Appliance 1 — Fridge (W)", "Appliance 4 — Tumble Dryer (W)"]
colors   = ["#2196F3", "#4CAF50", "#FF9800"]

for i, (ch, lab, col) in enumerate(zip(channels, labels, colors)):
    # Before
    ax_before = axes[i, 0]
    if ch in raw_sample.columns:
        ax_before.plot(raw_sample.index, pd.to_numeric(raw_sample[ch], errors="coerce"),
                       linewidth=0.4, color=col, alpha=0.8)
    ax_before.set_ylabel(lab, fontsize=8)
    ax_before.set_ylim(bottom=0)
    if i == 0:
        ax_before.set_title("Before Cleaning (raw ~8s)", fontweight="bold")
    ax_before.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # After
    ax_after = axes[i, 1]
    if ch in min1_sample.columns:
        ax_after.plot(min1_sample.index, min1_sample[ch],
                      linewidth=0.8, color=col, alpha=0.9)
    ax_after.set_ylim(bottom=0)
    if i == 0:
        ax_after.set_title("After Cleaning (1-minute means)", fontweight="bold")
    ax_after.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

plt.tight_layout()
fig_path = OUTPUTS_DIR / "01_before_after_cleaning.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1.6 — Summary statistics table
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1.6] Summary statistics (1-minute resampled):")
summary = df_1min[["Aggregate", "Appliance1", "Appliance4"]].describe().round(1)
print(summary.to_string())

print("\n✅  Notebook 1 complete. Figure saved to outputs/figures/")
