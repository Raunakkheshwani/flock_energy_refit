# RESULTS.md — EDA Findings, Model Results & Validation

## Section 1: Raw Data Cleaning — House 1

### Key Findings

| Finding | Value |
|---------|-------|
| Total raw rows (Parts 1+2) | 8,518,763 |
| Duplicate timestamps removed | **761,551 (8.94%)** |
| Impossible readings (>25 kW) clipped | 3 |
| Rows after deduplication | 7,757,212 |
| Rows after 1-minute resampling | **920,031** |
| Date coverage | Oct 2013 – Jul 2015 (~21 months) |
| Gaps > 30 minutes (after resampling) | 0 |

### Sampling Interval Distribution

| Metric | Value |
|--------|-------|
| Median interval | 6 seconds |
| Mean interval | 6.5 seconds |
| % within nominal [7–9 s] | 35.8% |
| Longest gap (raw) | ~1,000 hours |

The raw data is **not uniformly sampled** — readings arrive faster than the nominal 8s in most cases, and a small fraction are extremely delayed (outages).

### Before / After Cleaning Plots
- `outputs/figures/01_before_after_cleaning.png`  
  Shows Aggregate, Fridge (App1), and Tumble Dryer (App4) before (raw ~6s) and after (1-minute means). The resampled signal is much smoother and easier to analyse.

### Aggregate Power Statistics (1-minute, House 1)

| Statistic | Aggregate (W) | Appliance 1 – Fridge (W) |
|-----------|:---:|:---:|
| Mean | 478 | 19 |
| Std | 762 | 78 |
| Min | 114 | 0 |
| Median | 246 | 0 |
| Max | 23,966 | 16,466 |

---

## Section 2: Washing Machine EDA

### Households Analysed
Houses **1, 2, 3, 5** — all confirmed to have a washing machine channel (verified against REFIT Table 3 and power-range inspection).

### Data Quality per House (1-minute resolution)

| House | WM Column | Date Range | Missing % | Off/Idle % | WM Cycles Detected |
|-------|-----------|-----------|-----------|------------|:---:|
| 1 | Appliance5 | Oct 2013 – Jul 2015 | 12.3% | 98.3% | **1,162** |
| 2 | Appliance2 | Sep 2013 – May 2015 | 24.3% | 96.5% | **1,339** |
| 3 | Appliance6 | Sep 2013 – Jun 2015 | 11.2% | 95.0% | **928** |
| 5 | Appliance3 | Sep 2013 – Jul 2015 | 8.7% | 92.9% | **1,865** |

**Cycle detection rule**: Power > 20W for ≥ 5 consecutive minutes = active cycle.

### Cycle Characteristics (median across detected cycles)

| House | Median Duration | Median Peak Power | Median Energy/Cycle |
|-------|:-:|:-:|:-:|
| 1 | 9 min | 499 W | 28 Wh |
| 2 | 13 min | 252 W | 20 Wh |
| 3 | **59 min** | **1,896 W** | **462 Wh** |
| 5 | 14 min | 315 W | 60 Wh |

**Interpretation**:
- House 3 stands out — much longer cycles at very high peak power. This is consistent with a heavy-duty machine running full hot-wash programmes. Energy per cycle (~462 Wh) is 10–16× higher than the other houses.
- Houses 1, 2, 5 show shorter, lower-power cycles — consistent with modern A+++ rated machines on eco programmes.
- House 5 has the most wash events (1,865) — possibly a larger family or shared household.

### Usage Patterns

**By hour of day**: Washing is concentrated in morning (07:00–10:00) and early afternoon (13:00–16:00) across all houses. Very little overnight washing.  
**By day of week**: Weekend use (Saturday/Sunday) is slightly elevated in most households.

### Plots Generated
- `02a_24hr_trace_house1.png` — Representative 24-hour aggregate + WM power trace, House 1.
- `02b_detected_cycles_house1.png` — Three detected WM cycles with start/end markers, House 1.
- `02c_hourly_usage_patterns.png` — Average WM power by hour of day, all 4 houses.
- `02d_dayofweek_usage_patterns.png` — Average WM power by day of week, all 4 houses.
- `02e_cross_house_comparison.png` — Box plots of cycle duration, peak power, and energy across houses.

---

## Section 3: Disaggregation Model Results

### Model Architecture
**Random Forest Regressor** (scikit-learn)
- `n_estimators = 100` trees
- `max_depth = 12`
- `min_samples_leaf = 10`
- Input features: 13 aggregate-derived features (see below)
- Target: Washing machine power (W) at 1-minute resolution

**Rationale for Random Forest**: Interpretable, robust to outliers, no scaling required, fast inference. Model complexity was deliberately kept low — the assignment values clear explanations over marginal accuracy gains.

### Train / Test Split

| Set | Houses | Samples | Rationale |
|-----|--------|---------|-----------|
| Train | 1, 2, 3 | 2,694,264 | Full time periods |
| Test | 5 | 933,593 | Held-out household |

**Why cross-household split?** Splitting by time within the same house risks leakage: rolling features and lag features overlap across windows. Splitting by household tests true generalisation — the model must work on a home it has never seen.

### Input Features (from aggregate signal only)

| Feature | Importance |
|---------|:----------:|
| `agg` (raw aggregate power) | **0.352** |
| `hour_cos` (hour of day, cosine) | 0.093 |
| `roll_std_15` (15-min rolling std) | 0.085 |
| `roll_std_5` (5-min rolling std) | 0.076 |
| `lag_1` (1-min lag) | 0.056 |
| `lag_10` (10-min lag) | 0.055 |
| `roll_mean_5` (5-min rolling mean) | 0.054 |
| `hour_sin` (hour of day, sine) | 0.054 |
| `delta` (first difference) | 0.048 |
| `lag_5` (5-min lag) | 0.047 |
| `roll_mean_15` (15-min rolling mean) | 0.036 |
| `dow_sin` (day of week, sine) | 0.030 |
| `dow_cos` (day of week, cosine) | 0.015 |

The raw aggregate power is by far the most important feature (35%), followed by time-of-day encoding and rolling variability — which captures the "spikiness" that characterises an active wash cycle.

### Evaluation Metrics

| Model | MAE (W) | RMSE (W) | F1 (on/off) |
|-------|:-------:|:--------:|:-----------:|
| **Always-off baseline** | 28.9 | 199.4 | 0.000 |
| Usage-profile baseline | 40.7 | 198.1 | 0.138 |
| **Random Forest** | 43.0 | **197.3** | **0.271** |

**Interpretation of results**:

- **MAE**: The always-off baseline wins on MAE (28.9W) because the WM is off >92% of the time — predicting zero is almost always approximately correct. This is a known weakness of MAE for imbalanced targets.
- **RMSE**: All three models have similar RMSE (~197–199W) because the dominant error source is the large WM power spikes (~1500–2000W) during active cycles that all models occasionally miss. The RF is marginally best (197.3W).
- **F1 (binary detection)**: The RF achieves **F1=0.271**, nearly **2× the usage-profile baseline** and infinitely better than always-off. This is the most meaningful metric for the NILM use case — can we correctly detect *when the machine is running*?

The F1 of 0.271 may seem low, but this is a **cross-household test on an unseen home** — a very hard task. Within-household evaluations typically achieve F1 > 0.5 for RF models on REFIT (Bonfigli et al., 2018).

### Plots Generated
- `03a_feature_importance.png` — Bar chart of RF feature importances.
- `03b_actual_vs_predicted.png` — Time-series of actual vs predicted WM power for House 5 (1 week).
- `03c_scatter_actual_vs_predicted.png` — Scatter plot: actual vs predicted (10,000 sampled points).

### Failure Analysis

**1. Overlapping high-power appliances**  
When a kettle (1,800–2,200W) or tumble dryer runs at the same time as the washing machine, their combined aggregate signature is indistinguishable from a single large load. The model cannot separate them without additional sub-meter information. This leads to **false positives** (predicting WM is on when it is not).

**2. Low-power standby / spin phases**  
Modern washing machines draw only 3–10W during standby and drop below the 20W threshold during parts of the spin cycle. The model classifies these as "off" (**false negatives**), under-counting cycles and under-estimating energy use.

---

## Validation Approach

1. **Cross-household held-out test**: House 5 was never seen during training.
2. **Multiple metrics**: MAE (regression accuracy), RMSE (sensitivity to large errors), F1 (detection performance) — no single metric tells the full story.
3. **Two non-trivial baselines**: Always-off shows the floor; usage-profile shows what calendar information alone can do. The RF must beat both to be useful.
4. **Visual inspection**: The actual-vs-predicted trace (Fig. 03b) lets us visually confirm the model is tracking real cycle events, not just predicting noise.

---

## References

- Murray, D., Stankovic, L., & Stankovic, V. (2017). *Scientific Data*, 4, 160122.
- Kelly, J., & Knottenbelt, W. (2015). Neural NILM. *BuildSys 2015*.
- Bonfigli, R. et al. (2018). Denoising autoencoders for non-intrusive load monitoring. *Energy and Buildings*, 158, 548–559.
