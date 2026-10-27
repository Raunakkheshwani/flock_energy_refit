# DATA.md — Raw Data Coverage, Quality Findings & Cleaning Decisions

## Dataset Overview

**Source**: REFIT Electrical Load Measurements, University of Strathclyde  
**Version**: REFIT_RAW_081116 (raw) and CLEAN_REFIT_081116 (official cleaned)  
**Coverage**: 20 UK households, 2013–2015  
**Sampling interval (raw)**: ~8 seconds (nominal), actual varies  
**Units**: Watts (active power)  
**Channels**: `Aggregate` (whole-house) + up to 9 individual appliance monitors (IAMs)

---

## Section 1: House 1 — Raw Data Inspection

### Files Used
| File | Size | Rows |
|------|------|------|
| `RAW_House1_Part1.csv` | ~340 MB | 4,377,877 |
| `RAW_House1_Part2.csv` | ~322 MB | 4,140,886 |
| **Combined** | ~662 MB | **8,518,763** |

### Date Coverage
- **Start**: 2013-10-09 14:06:17
- **End**: 2015-07-10 11:56:32
- **Duration**: ~21 months

### Sampling Interval Analysis
| Metric | Value |
|--------|-------|
| Nominal interval | 8 seconds |
| Median observed | **6 seconds** |
| Mean observed | 6.5 seconds |
| Min (near-simultaneous) | 0 seconds |
| Max (longest gap) | ~1,000 hours (data outage) |
| % within [7, 9] s | 35.8% |

**Findings**: The interval is irregular. Many readings arrive faster than 8 s, and some arrive much later. This is normal for smart-meter hardware. The raw data is not uniformly sampled.

### Data Quality Issues Found

| Issue | Count | % of Data | Action Taken |
|-------|-------|-----------|--------------|
| Duplicate timestamps | 761,551 | **8.94%** | Kept last reading; later value more likely correct |
| NaN in Aggregate | ~189,000 | 2.22% | Handled by resampling (mean of non-NaN) |
| NaN in Appliance1 | ~438,000 | 5.14% | Same |
| Negative power readings | 0 | 0% | — |
| Power > 25,000 W | 3 | < 0.001% | Clipped to NaN (physically impossible) |

### Remaining Concerns
- **Appliance4 max = 16,384 W** in a few spikes — this exceeds a single appliance's capacity and likely reflects a sensor overflow at exactly 2^14. These could be addressed with a house-specific clipping rule, but for simplicity we flag them without removal (they become ~4 W after resampling).
- We cannot verify whether the "keep last" deduplication strategy is optimal without knowing why duplicates occur. An alternative would be keeping the mean.

---

## Section 2: Cleaned Data — Washing Machine EDA (Houses 1, 2, 3, 5)

### Appliance Assignments
| House | WM Column | Verification |
|-------|-----------|--------------|
| 1 | `Appliance5` | Max 2,545 W, 0.6% active time |
| 2 | `Appliance2` | Max 2,422 W, 4.5% active time |
| 3 | `Appliance6` | Max 2,115 W, 1.5% active time |
| 5 | `Appliance3` | Max 2,193 W, 5.1% active time |

Columns were identified by cross-referencing the REFIT paper (Murray et al., 2017) Table 3 and verified against measured power ranges (washing machines draw 0–2,500 W depending on cycle phase).

### Cycle Detection Rule
> **Plain English**: A washing-machine cycle starts when the WM power reading exceeds **20 Watts** and ends when it drops back below 20 Watts. We only count cycles that last at least **5 consecutive minutes** to avoid false positives from brief current spikes.

This threshold is consistent with the literature (e.g., Kelly & Knottenbelt, 2015 use 20W for REFIT).

### Data Quality per House (1-minute resolution)
| House | Date Range | Missing % | Off/Idle % | Cycles Detected |
|-------|-----------|-----------|------------|-----------------|
| 1 | Oct 2013 – Jul 2015 | ~2% | ~99.4% | (see RESULTS.md) |
| 2 | Sep 2013 – Jun 2015 | ~2% | ~95.5% | (see RESULTS.md) |
| 3 | Sep 2013 – Jun 2015 | ~3% | ~98.5% | (see RESULTS.md) |
| 5 | Sep 2013 – Jun 2015 | ~2% | ~94.9% | (see RESULTS.md) |

### Gaps & Outliers (Cleaned Data)
The official REFIT cleaned data has already had major corrections applied (interpolation, outlier removal). Residual gaps in the cleaned data are rare and short; we do not apply additional gap-filling beyond resampling.

---

## References

- Murray, D., Stankovic, L., & Stankovic, V. (2017). An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study. *Scientific Data*, 4, 160122. https://doi.org/10.1038/sdata.2016.122
- Kelly, J., & Knottenbelt, W. (2015). Neural NILM: Deep neural networks applied to energy disaggregation. *BuildSys 2015*.
