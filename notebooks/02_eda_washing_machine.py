"""
Notebook 2: EDA — Washing Machine Usage Across Houses
=====================================================
Uses the official CLEANED REFIT data.
Houses analysed: 1, 2, 3, 5  (all confirmed to have a washing machine channel)
Run: python notebooks/02_eda_washing_machine.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from utils import (
    CLEAN_DIR, OUTPUTS_DIR,
    load_clean_house, resample_1min, detect_wm_cycles,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Washing machine column per house (from REFIT documentation Table 3) ───
# Verified against power ranges: max ~2000-2500W, low duty cycle
WM_COL = {
    1: "Appliance5",
    2: "Appliance2",
    3: "Appliance6",
    5: "Appliance3",
}
ON_THRESHOLD_W = 20   # W — power above this → machine is running
MIN_CYCLE_MIN  = 5    # minutes — minimum duration to call it a cycle

HOUSES = [1, 2, 3, 5]
COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800", 5: "#E91E63"}

print("=" * 60)
print("Section 2: EDA — Washing Machine Usage")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Load & resample data for each house
# ─────────────────────────────────────────────────────────────────────────────
house_data = {}   # {house_id: {"df": df_1min, "wm": wm_series, "cycles": [...]}}

print("\n[2.1] Loading and resampling cleaned data...")
for h in HOUSES:
    print(f"  House {h}...", end=" ")
    df = load_clean_house(h)

    # Drop the Issues column and Unix if present
    df = df.drop(columns=[c for c in ["Unix", "Issues"] if c in df.columns])

    # Resample to 1-minute
    df_1min = resample_1min(df)

    wm = df_1min[WM_COL[h]].fillna(0).clip(lower=0)
    cycles = detect_wm_cycles(wm, on_threshold_w=ON_THRESHOLD_W, min_duration_min=MIN_CYCLE_MIN)

    house_data[h] = {"df": df_1min, "wm": wm, "cycles": cycles}
    print(f"  {len(df_1min):,} rows, {len(cycles)} WM cycles detected")

# ─────────────────────────────────────────────────────────────────────────────
# 2.2  Data quality summary per house
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2.2] Data quality summary:")
quality_rows = []
for h in HOUSES:
    wm = house_data[h]["wm"]
    df = house_data[h]["df"]
    missing_pct  = df[WM_COL[h]].isna().mean() * 100
    zero_pct     = (wm <= ON_THRESHOLD_W).mean() * 100
    n_cycles     = len(house_data[h]["cycles"])
    date_start   = df.index.min().date()
    date_end     = df.index.max().date()
    quality_rows.append({
        "House": h,
        "Start": date_start,
        "End": date_end,
        "Missing %": round(missing_pct, 1),
        "Off/idle %": round(zero_pct, 1),
        "WM cycles detected": n_cycles,
    })

quality_df = pd.DataFrame(quality_rows)
print(quality_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Plot (i): 24-hour aggregate + WM trace — House 1
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2.3] Creating plots...")

# Pick a representative day with WM activity for House 1
h1_wm = house_data[1]["wm"]
# Find a date where WM ran for at least 30 minutes
daily_wm = h1_wm.resample("D").sum() / 60  # Wh
representative_day = daily_wm[daily_wm > 50].index[5]  # 6th such day
day_str = representative_day.strftime("%Y-%m-%d")

h1_day_agg = house_data[1]["df"]["Aggregate"].loc[day_str]
h1_day_wm  = house_data[1]["df"][WM_COL[1]].loc[day_str].fillna(0).clip(lower=0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle(f"House 1 — 24-hour Profile: {day_str}", fontsize=13, fontweight="bold")

ax1.fill_between(h1_day_agg.index, h1_day_agg, alpha=0.4, color="#2196F3")
ax1.plot(h1_day_agg.index, h1_day_agg, color="#2196F3", linewidth=1.2, label="Aggregate")
ax1.set_ylabel("Aggregate Power (W)")
ax1.legend(loc="upper right")
ax1.set_ylim(bottom=0)

ax2.fill_between(h1_day_wm.index, h1_day_wm, alpha=0.5, color="#E91E63",
                 label="Washing Machine")
ax2.axhline(y=ON_THRESHOLD_W, color="gray", linestyle="--", linewidth=0.8,
            label=f"On threshold ({ON_THRESHOLD_W}W)")
ax2.set_ylabel("WM Power (W)")
ax2.set_xlabel("Time of day")
ax2.legend(loc="upper right")
ax2.set_ylim(bottom=0)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

plt.tight_layout()
fig_path = OUTPUTS_DIR / "02a_24hr_trace_house1.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.4  Plot (ii): Detected WM cycles — House 1
# ─────────────────────────────────────────────────────────────────────────────
cycles_h1 = house_data[1]["cycles"]
# Show first 3 cycles
n_show = min(3, len(cycles_h1))
fig, axes = plt.subplots(n_show, 1, figsize=(14, 3.5 * n_show))
if n_show == 1:
    axes = [axes]
fig.suptitle("House 1 — Detected Washing-Machine Cycles", fontsize=13, fontweight="bold")

for i, cyc in enumerate(cycles_h1[:n_show]):
    ax = axes[i]
    # 10-min buffer around cycle
    t_start = cyc["start"] - pd.Timedelta(minutes=10)
    t_end   = cyc["end"]   + pd.Timedelta(minutes=10)
    seg = house_data[1]["wm"].loc[t_start:t_end]

    ax.fill_between(seg.index, seg, alpha=0.6, color="#E91E63")
    ax.plot(seg.index, seg, color="#C2185B", linewidth=1.2)
    ax.axhline(y=ON_THRESHOLD_W, color="gray", linestyle="--", linewidth=0.8)
    ax.axvspan(cyc["start"], cyc["end"], alpha=0.15, color="yellow")
    ax.set_ylabel("WM Power (W)")
    ax.set_title(
        f"Cycle {i+1}: {cyc['start'].strftime('%Y-%m-%d %H:%M')} | "
        f"Duration: {cyc['duration_min']:.0f} min | "
        f"Peak: {cyc['peak_power_w']:.0f} W | "
        f"Energy: {cyc['energy_wh']:.0f} Wh",
        fontsize=9
    )
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

plt.tight_layout()
fig_path = OUTPUTS_DIR / "02b_detected_cycles_house1.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.5  Usage patterns: Hour-of-day and day-of-week
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Washing Machine — Usage Patterns by Hour & Day of Week", fontsize=13, fontweight="bold")

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

for idx, h in enumerate(HOUSES):
    wm = house_data[h]["wm"]
    ax_hour = axes[idx // 2, idx % 2]

    # Hour of day — average power when active
    hour_avg = wm.groupby(wm.index.hour).mean()
    bars = ax_hour.bar(hour_avg.index, hour_avg.values, color=COLORS[h], alpha=0.8, width=0.8)
    ax_hour.set_title(f"House {h} — Avg WM Power by Hour", fontsize=10)
    ax_hour.set_xlabel("Hour of day")
    ax_hour.set_ylabel("Avg Power (W)")
    ax_hour.set_xticks(range(0, 24, 2))
    ax_hour.set_ylim(bottom=0)

plt.tight_layout()
fig_path = OUTPUTS_DIR / "02c_hourly_usage_patterns.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# Day-of-week
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Washing Machine — Day-of-Week Usage Frequency", fontsize=13, fontweight="bold")

for idx, h in enumerate(HOUSES):
    wm = house_data[h]["wm"]
    ax = axes[idx // 2, idx % 2]
    dow_avg = wm.groupby(wm.index.dayofweek).mean()
    ax.bar(range(7), dow_avg.values, color=COLORS[h], alpha=0.8)
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_title(f"House {h} — Avg WM Power by Day", fontsize=10)
    ax.set_ylabel("Avg Power (W)")
    ax.set_ylim(bottom=0)

plt.tight_layout()
fig_path = OUTPUTS_DIR / "02d_dayofweek_usage_patterns.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.6  Plot (iii): Cross-house comparison — cycle characteristics
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Cross-House Washing Machine Cycle Comparison", fontsize=13, fontweight="bold")

metrics = ["duration_min", "peak_power_w", "energy_wh"]
metric_labels = ["Cycle Duration (min)", "Peak Power (W)", "Energy per Cycle (Wh)"]

for ax, metric, label in zip(axes, metrics, metric_labels):
    data_per_house = []
    house_labels = []
    for h in HOUSES:
        vals = [c[metric] for c in house_data[h]["cycles"]]
        if vals:
            data_per_house.append(vals)
            house_labels.append(f"H{h}\n(n={len(vals)})")

    bp = ax.boxplot(data_per_house, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, h in zip(bp["boxes"], HOUSES):
        patch.set_facecolor(COLORS[h])
        patch.set_alpha(0.7)

    ax.set_xticklabels(house_labels)
    ax.set_ylabel(label)
    ax.set_title(label, fontsize=10)

plt.tight_layout()
fig_path = OUTPUTS_DIR / "02e_cross_house_comparison.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.7  Print cycle statistics summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2.7] Cycle statistics summary:")
for h in HOUSES:
    cycles = house_data[h]["cycles"]
    if not cycles:
        print(f"  House {h}: No cycles detected")
        continue
    durations = [c["duration_min"] for c in cycles]
    peaks     = [c["peak_power_w"] for c in cycles]
    energies  = [c["energy_wh"] for c in cycles]
    print(f"  House {h}: {len(cycles)} cycles | "
          f"Median duration: {np.median(durations):.0f} min | "
          f"Median peak: {np.median(peaks):.0f}W | "
          f"Median energy: {np.median(energies):.0f}Wh")

print("\n✅  Notebook 2 complete. Figures saved to outputs/figures/")
