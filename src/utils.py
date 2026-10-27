"""
utils.py — Shared helper functions for the REFIT pipeline.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw/extracted"
CLEAN_DIR = PROJECT_ROOT / "data/cleaned/extracted"
OUTPUTS_DIR = PROJECT_ROOT / "outputs/figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ── REFIT House metadata ───────────────────────────────────────────────────
# Appliance labels for House 1 (from REFIT README)
HOUSE1_LABELS = {
    "Aggregate":   "Aggregate",
    "Appliance1":  "Fridge",
    "Appliance2":  "Chest Freezer",
    "Appliance3":  "Upright Freezer",
    "Appliance4":  "Tumble Dryer",
    "Appliance5":  "Washing Machine",
    "Appliance6":  "Dishwasher",
    "Appliance7":  "Computer",
    "Appliance8":  "TV",
    "Appliance9":  "Microwave",
}

# Which appliance column holds the Washing Machine per house
# Based on REFIT README / CLEAN_README documentation
WM_COLUMN = {
    1: "Appliance5",
    2: "Appliance2",
    3: "Appliance2",
    5: "Appliance3",
}


# ── Loading functions ──────────────────────────────────────────────────────

def load_raw_house1() -> pd.DataFrame:
    """
    Load and concatenate both parts of raw House 1 data.
    Returns a DataFrame with a parsed datetime index (not yet cleaned).
    """
    parts = []
    for part in [1, 2]:
        fp = RAW_DIR / f"RAW_House1_Part{part}.csv"
        df = pd.read_csv(fp, parse_dates=["Time"])
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    return df


def load_clean_house(house_id: int) -> pd.DataFrame:
    """
    Load the official cleaned CSV for a given house number.
    Returns a DataFrame with a parsed datetime index at original resolution.
    """
    fp = CLEAN_DIR / f"CLEAN_House{house_id}.csv"
    # The cleaned files start directly with the header row (no comment lines)
    df = pd.read_csv(fp, parse_dates=["Time"])
    df = df.sort_values("Time").set_index("Time")
    return df


# ── Cleaning helpers ───────────────────────────────────────────────────────

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the basic cleaning pipeline to a raw REFIT DataFrame.
    Steps (in order):
      1. Parse & sort by Time
      2. Remove duplicate timestamps (keep last)
      3. Clip physically impossible values (negatives → NaN, >25000W → NaN)
      4. Set Time as index
    Returns the sorted, deduplicated, clipped DataFrame.
    """
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")

    # Keep last reading for duplicate timestamps
    df = df.drop_duplicates(subset="Time", keep="last")
    df = df.set_index("Time")

    # Drop Unix column if present
    if "Unix" in df.columns:
        df = df.drop(columns=["Unix"])

    # Power columns: clip negatives and extreme outliers to NaN
    power_cols = [c for c in df.columns if c != "Unix"]
    for col in power_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
        df.loc[df[col] > 25_000, col] = np.nan  # >25kW is physically impossible

    return df


def resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a cleaned DataFrame to 1-minute intervals using mean aggregation.
    Short gaps (≤5 min) are linearly interpolated.
    Longer gaps are left as NaN (flagged, not filled).
    """
    df_1min = df.resample("1min").mean()

    # Identify which 1-min buckets were originally NaN (true gaps)
    gap_mask = df_1min.isna()

    # Interpolate only short gaps (limit=5 minutes)
    df_1min = df_1min.interpolate(method="linear", limit=5, limit_direction="forward")

    # Re-apply NaN where gap was longer than 5 minutes
    long_gap = gap_mask & df_1min.notna()
    # (The 5-min interpolation already leaves long gaps as NaN — no extra step needed)

    return df_1min


def detect_gaps(df: pd.DataFrame, threshold_min: int = 30) -> pd.DataFrame:
    """
    Return a DataFrame listing gaps longer than `threshold_min` minutes.
    Columns: gap_start, gap_end, duration_min
    """
    idx = df.index
    diffs = idx.to_series().diff().dt.total_seconds() / 60  # minutes
    long_gaps = diffs[diffs > threshold_min]
    records = []
    for ts, dur in long_gaps.items():
        records.append({
            "gap_start": ts - pd.Timedelta(minutes=dur),
            "gap_end": ts,
            "duration_min": round(dur, 1),
        })
    return pd.DataFrame(records)


# ── Cycle detection ────────────────────────────────────────────────────────

def detect_wm_cycles(
    series: pd.Series,
    on_threshold_w: float = 20.0,
    min_duration_min: int = 5,
) -> list[dict]:
    """
    Detect washing-machine operating cycles in a 1-minute power series.

    Rule (plain English):
        A cycle starts when power exceeds `on_threshold_w` Watts and
        ends when power drops back below that threshold.
        Only cycles lasting at least `min_duration_min` minutes are kept.

    Returns a list of dicts with keys:
        start, end, duration_min, peak_power_w, energy_wh
    """
    is_on = (series.fillna(0) > on_threshold_w).astype(int)
    transitions = is_on.diff().fillna(0)

    cycle_starts = series.index[transitions == 1].tolist()
    cycle_ends = series.index[transitions == -1].tolist()

    # Handle edge cases: series starts ON or ends ON
    if is_on.iloc[0] == 1:
        cycle_starts = [series.index[0]] + cycle_starts
    if is_on.iloc[-1] == 1:
        cycle_ends = cycle_ends + [series.index[-1]]

    cycles = []
    for start, end in zip(cycle_starts, cycle_ends):
        segment = series.loc[start:end]
        duration = (end - start).total_seconds() / 60
        if duration < min_duration_min:
            continue
        energy = segment.mean() * (duration / 60)  # Wh
        cycles.append({
            "start": start,
            "end": end,
            "duration_min": round(duration, 1),
            "peak_power_w": round(float(segment.max()), 1),
            "energy_wh": round(float(energy), 1),
        })
    return cycles


# ── Feature engineering ────────────────────────────────────────────────────

def make_features(agg: pd.Series) -> pd.DataFrame:
    """
    Build a feature matrix from the aggregate power series for disaggregation.

    Features (all derived from aggregate only — no appliance info leaked):
        - agg:           raw aggregate power (W)
        - roll_mean_5:   5-minute rolling mean
        - roll_std_5:    5-minute rolling std
        - roll_mean_15:  15-minute rolling mean
        - roll_std_15:   15-minute rolling std
        - delta:         first difference (change from previous minute)
        - lag_1, lag_5, lag_10: lagged values
        - hour_sin/cos:  hour of day encoded as sin/cos (cyclical)
        - dow_sin/cos:   day of week encoded as sin/cos (cyclical)
    """
    df = pd.DataFrame({"agg": agg})

    df["roll_mean_5"] = agg.rolling(5, min_periods=1).mean()
    df["roll_std_5"] = agg.rolling(5, min_periods=1).std().fillna(0)
    df["roll_mean_15"] = agg.rolling(15, min_periods=1).mean()
    df["roll_std_15"] = agg.rolling(15, min_periods=1).std().fillna(0)

    df["delta"] = agg.diff().fillna(0)
    df["lag_1"] = agg.shift(1).bfill()
    df["lag_5"] = agg.shift(5).bfill()
    df["lag_10"] = agg.shift(10).bfill()

    hour = agg.index.hour
    dow = agg.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df.fillna(0)
