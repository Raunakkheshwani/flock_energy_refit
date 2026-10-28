# REFIT Appliance Disaggregation — Flock Energy Take-Home Assignment

## What We Built

A reproducible, end-to-end Python pipeline that:

1. **Cleans raw REFIT smart-meter data** for House 1 — sorts, deduplicates, clips impossible values, resamples from ~8-second to 1-minute intervals, and flags long outages.
2. **Analyses washing-machine usage** across **10 UK households** (Houses 1–10) using the official cleaned REFIT data — detecting cycles, profiling usage patterns by hour and day, and comparing cycle characteristics cross-house.
3. **Builds a Random Forest disaggregation model** trained on 8 households (~3.2M samples) that estimates washing-machine power from the whole-house aggregate signal alone, evaluated on a held-out household (House 5) using both per-minute and per-cycle metrics.

---

## Setup & Run Instructions

### Prerequisites
- Python 3.9+ with [Anaconda](https://www.anaconda.com/) **or** a virtual environment
- ~15 GB free disk space (data archives expand to ~6 GB each)

### 1. Clone the repository
```bash
git clone https://github.com/Raunakkheshwani/flock_energy_refit.git
cd flock_energy_refit
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download data archives
Download both archives from the University of Strathclyde REFIT dataset page:
- **Raw**: `REFIT_RAW_081116.7z` → place in `data/raw/`
- **Cleaned**: `CLEAN_REFIT_081116.7z` → place in `data/cleaned/`

### 4. Extract archives
```bash
python src/extract_data.py
```

### 5. Run the analysis (in order)
```bash
python notebooks/01_data_cleaning.py       # Section 1: Cleaning
python notebooks/02_eda_washing_machine.py # Section 2: EDA
python notebooks/03_disaggregation_model.py# Section 3: Model
```

All figures are saved to `outputs/figures/`.

---

## Key Assumptions & Preprocessing Decisions

| Decision | Rationale |
|----------|-----------|
| Duplicate timestamps: **keep last** | Later readings are more likely to reflect corrected sensor values |
| Negative power → **NaN** | Physically impossible; sensor artefact |
| Power > 25,000 W → **NaN** | No UK household appliance draws this much; clear outlier |
| Resample: **1-minute mean** | Assignment requirement; mean preserves average power faithfully |
| Short gaps (≤5 min): **linear interpolation** | Brief dropout — reasonable to fill with trend |
| Long gaps (>30 min): **flagged, not filled** | Could be a real outage; filling without justification would be dishonest |
| WM on-threshold: **20W** | Eliminates standby noise; confirmed against known WM power profiles |
| Minimum cycle duration: **5 minutes** | Eliminates false positives from brief current spikes |

---

## Validation Design

- **Train/test split by household**: Houses 1, 2, 3 → train; House 5 → test.  
  This avoids temporal data leakage that would occur with overlapping time windows from the same house.
- **Baselines included**: Always-off and usage-profile baselines ensure the model is compared against non-trivial reference points.
- **Metrics**: MAE (interpretable), RMSE (sensitive to missed active cycles), binary F1 (detection accuracy).

---

## Trade-offs

- **Model choice (Random Forest)**: Simple, interpretable, no hyperparameter tuning needed. A neural network (e.g. Seq2Point) would likely score better on RMSE but at the cost of transparency — which the assignment explicitly warns against.
- **Features**: Aggregate-only features (rolling stats, lags, time encoding). We deliberately excluded appliance sub-meters from the feature set to simulate the real-world NILM scenario.
- **Cross-household generalisation**: Tested on a fully unseen house. Results will be noisier than within-household time-split but are a more realistic measure of utility.

---

## AI Tools Used

- **GitHub Copilot / Claude**: Used for code scaffolding (docstrings, boilerplate), not for generating analysis logic or conclusions.
- All analytical decisions (thresholds, model selection, train/test strategy) were made independently and justified in-code.

---

## Possible Improvements

1. **Seq2Point or LSTM** — sequence-to-point neural networks specifically designed for NILM typically outperform RF on this task.
2. **Within-household time-split** (80/20) in addition to cross-household split.
3. **More houses** — include all 12+ REFIT houses with a washing machine to improve generalisation.
4. **Per-cycle evaluation** — in addition to per-minute MAE, measure how well cycle *events* are detected (precision/recall at cycle level).
5. **Temperature & time-of-use tariff features** — could improve off-peak detection.
