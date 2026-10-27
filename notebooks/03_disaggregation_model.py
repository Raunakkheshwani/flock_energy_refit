"""
Notebook 3: Appliance Disaggregation Model
==========================================
Goal: Estimate washing-machine power from the whole-house aggregate signal.
Uses official cleaned REFIT data, 1-minute resolution.
Train: Houses 1, 2, 3  |  Test: House 5 (held-out household)

Run: python notebooks/03_disaggregation_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import (
    CLEAN_DIR, OUTPUTS_DIR,
    load_clean_house, resample_1min, make_features,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Config ────────────────────────────────────────────────────────────────────
WM_COL = {1: "Appliance5", 2: "Appliance2", 3: "Appliance6", 5: "Appliance3"}
TRAIN_HOUSES = [1, 2, 3]
TEST_HOUSE   = 5
ON_THRESHOLD = 20.0   # W — used for binary F1 evaluation

print("=" * 60)
print("Section 3: Appliance Disaggregation Model")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Load and prepare data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3.1] Loading and preparing data...")

def prepare_house(house_id: int) -> tuple[pd.DataFrame, pd.Series]:
    """Load, resample, build features and return (X, y)."""
    df = load_clean_house(house_id)
    df = df.drop(columns=[c for c in ["Unix", "Issues"] if c in df.columns])
    df_1min = resample_1min(df)

    agg = df_1min["Aggregate"].fillna(df_1min["Aggregate"].median()).clip(lower=0)
    wm  = df_1min[WM_COL[house_id]].fillna(0).clip(lower=0)

    X = make_features(agg)
    y = wm

    # Drop rows with any NaN
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]

train_X_parts, train_y_parts = [], []
for h in TRAIN_HOUSES:
    print(f"  House {h} (train)...", end=" ")
    X, y = prepare_house(h)
    train_X_parts.append(X)
    train_y_parts.append(y)
    print(f"{len(X):,} rows")

# Test house
print(f"  House {TEST_HOUSE} (test)...", end=" ")
test_X, test_y = prepare_house(TEST_HOUSE)
print(f"{len(test_X):,} rows")

X_train = pd.concat(train_X_parts)
y_train = pd.concat(train_y_parts)
X_test  = test_X
y_test  = test_y

print(f"\n  Train set: {len(X_train):,} samples (Houses {TRAIN_HOUSES})")
print(f"  Test set:  {len(X_test):,} samples (House {TEST_HOUSE})")
print(f"\n  Split strategy: Houses 1,2,3 → train | House 5 → test")
print(f"  Why: Cross-household split avoids data leakage.")
print(f"  Overlapping time windows from the same house would let the model")
print(f"  memorise household-specific patterns rather than learn general features.")

# ─────────────────────────────────────────────────────────────────────────────
# 3.2  Baselines
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3.2] Computing baselines...")

# Baseline 1: Always-off (predict 0W at all times)
pred_always_off = np.zeros(len(y_test))

# Baseline 2: Usage-profile baseline
# For each (hour, day-of-week) bin, predict the mean WM power from training data
# Build the profile on training data
y_train_df = y_train.to_frame("wm")
y_train_df["hour"] = y_train_df.index.hour
y_train_df["dow"]  = y_train_df.index.dayofweek
profile = y_train_df.groupby(["hour", "dow"])["wm"].mean()

def apply_profile(index, profile):
    preds = []
    for ts in index:
        key = (ts.hour, ts.dayofweek)
        preds.append(profile.get(key, 0.0))
    return np.array(preds)

pred_profile = apply_profile(y_test.index, profile)

def evaluate(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Binary F1
    true_on = (y_true > ON_THRESHOLD).astype(int)
    pred_on = (np.array(y_pred) > ON_THRESHOLD).astype(int)
    tp = ((true_on == 1) & (pred_on == 1)).sum()
    fp = ((true_on == 0) & (pred_on == 1)).sum()
    fn = ((true_on == 1) & (pred_on == 0)).sum()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"  {name:<35} MAE={mae:6.1f}W  RMSE={rmse:6.1f}W  F1={f1:.3f}")
    return {"name": name, "MAE": mae, "RMSE": rmse, "F1": f1}

results = []
results.append(evaluate(y_test, pred_always_off, "Always-off baseline"))
results.append(evaluate(y_test, pred_profile, "Usage-profile baseline"))

# ─────────────────────────────────────────────────────────────────────────────
# 3.3  Random Forest model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3.3] Training Random Forest Regressor...")
print("  (n_estimators=100, max_depth=12 — chosen to balance accuracy vs speed)")

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
pred_rf = np.clip(pred_rf, 0, None)  # power cannot be negative

results.append(evaluate(y_test, pred_rf, "Random Forest (our model)"))

# Metrics table
print("\n[3.4] Results comparison:")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n  Metrics explanation:")
print("  - MAE  (Mean Absolute Error): average error in Watts. Easy to interpret.")
print("  - RMSE (Root Mean Squared Error): penalises large errors more heavily.")
print("    Important here because failing to detect an active WM cycle is costly.")
print("  - F1   (binary 'is WM on?'): measures detection accuracy as a classifier.")

# ─────────────────────────────────────────────────────────────────────────────
# 3.4  Feature importance
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3.5] Feature importances:")
feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feat_imp.round(4).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
feat_imp.plot(kind="bar", ax=ax, color="#2196F3", alpha=0.8)
ax.set_title("Random Forest — Feature Importance", fontweight="bold")
ax.set_ylabel("Importance")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
fig_path = OUTPUTS_DIR / "03a_feature_importance.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.5  Actual vs. Predicted plot (zoomed to show cycles)
# ─────────────────────────────────────────────────────────────────────────────
# Find a week with WM activity in test set
pred_series = pd.Series(pred_rf, index=y_test.index)
daily_pred  = pred_series.resample("D").sum()
active_days = daily_pred[daily_pred > 500].index

if len(active_days) >= 7:
    zoom_start = active_days[0]
    zoom_end   = active_days[6]
else:
    zoom_start = y_test.index[0]
    zoom_end   = y_test.index[min(10080, len(y_test)-1)]  # 1 week

y_zoom    = y_test.loc[zoom_start:zoom_end]
pred_zoom = pred_series.loc[zoom_start:zoom_end]

fig, ax = plt.subplots(figsize=(16, 5))
ax.fill_between(y_zoom.index, y_zoom, alpha=0.5, color="#4CAF50", label="Actual WM Power")
ax.plot(y_zoom.index, pred_zoom, color="#E91E63", linewidth=1.2,
        alpha=0.9, label="Random Forest Prediction")
ax.axhline(y=ON_THRESHOLD, color="gray", linestyle="--", linewidth=0.8,
           label=f"On threshold ({ON_THRESHOLD}W)")
ax.set_title(f"House 5 (Test) — Actual vs Predicted WM Power\n"
             f"{zoom_start.date()} → {zoom_end.date()}",
             fontweight="bold")
ax.set_ylabel("Washing Machine Power (W)")
ax.set_xlabel("Time")
ax.legend(loc="upper right")
ax.set_ylim(bottom=-10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
plt.tight_layout()
fig_path = OUTPUTS_DIR / "03b_actual_vs_predicted.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# Scatter: actual vs predicted (sampled for speed)
n_sample = min(10000, len(y_test))
idx_sample = np.random.choice(len(y_test), n_sample, replace=False)
y_samp = y_test.values[idx_sample]
p_samp = pred_rf[idx_sample]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_samp, p_samp, alpha=0.1, s=5, color="#2196F3")
max_val = max(y_samp.max(), p_samp.max())
ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual WM Power (W)")
ax.set_ylabel("Predicted WM Power (W)")
ax.set_title("Actual vs Predicted — Scatter Plot\n(sampled 10,000 points)", fontweight="bold")
ax.legend()
plt.tight_layout()
fig_path = OUTPUTS_DIR / "03c_scatter_actual_vs_predicted.png"
plt.savefig(fig_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.6  Failure cases
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3.6] Known failure cases:")
print("""
  1. Overlapping appliances with similar power signatures
     ─────────────────────────────────────────────────────
     When a kettle (1800–2200W) or tumble dryer runs simultaneously with the
     washing machine, the aggregate signal cannot be decomposed without
     individual sub-meter readings. The model is likely to over-predict WM power
     during these moments.

  2. Low-power standby / spin detection
     ─────────────────────────────────────
     Modern washing machines draw only 3–10W during standby and can dip low
     during the spin cycle ramp-up. The model may classify these as "off"
     (false negatives), under-estimating the true cycle count and energy used.
""")

print("✅  Notebook 3 complete. Figures saved to outputs/figures/")
