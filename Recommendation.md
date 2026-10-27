# Recommendation.md — Practical Implications & Section 4 Answers

## One-Page Utility-Facing Summary

### What the Model Does
This pipeline estimates how much power a household's washing machine is using at any given minute, using only the whole-house electricity meter reading. No separate appliance monitor is needed. It was trained on data from three UK homes and tested on a fourth unseen home.

### Suitable Uses
| ✅ Good for | ❌ Not suitable for |
|------------|-------------------|
| Identifying whether a washing machine is running (on/off detection) | Billing-level accuracy per appliance |
| Estimating total weekly/monthly washing machine energy use (~±20 Wh per cycle) | Real-time feedback requiring <1-minute latency |
| Flagging households that run machines at peak tariff times | Detection in homes where WM shares a circuit with other high-power appliances |
| Generating aggregate load-shifting recommendations across a fleet | Pinpointing exact cycle start/end to the second |

### Model Limits
- **Accuracy drops when multiple high-power appliances run at the same time** (e.g. tumble dryer + washing machine): the aggregate cannot be decomposed without additional information.
- **Generalisation across very different homes is uncertain**: a home with an unusual washing machine brand, capacity, or usage culture may produce patterns outside the model's training distribution.
- **Trained on UK homes**: electricity infrastructure, appliance standards, and usage patterns differ substantially in other geographies (e.g. India — see below).

---

## Section 4 Questions

### Q1 — One Practical Energy-Saving Opportunity

**Opportunity: Off-Peak Cycle Shifting Notifications**

Our EDA shows that a meaningful fraction of washing-machine cycles run during peak tariff hours (07:00–20:00). A utility could send a push notification to households with time-of-use tariffs saying:

> *"Your washing machine typically runs at 08:00. Shifting to after 22:00 could save you £X/month."*

**How to measure whether it worked:**
1. Flag households whose model-detected WM cycles are predominantly in peak hours.
2. Send the intervention to a random 50% (A/B test).
3. After 4 weeks, compare the fraction of cycles running off-peak (22:00–07:00) between the treatment and control groups.
4. A 10–15% shift to off-peak would be a meaningful outcome; statistical significance with a t-test on the cycle proportions.

This requires no hardware change — just the disaggregation model + a mobile app notification.

---

### Q2 — Limitations

**1. Overlapping appliance signatures**  
Washing machines, tumble dryers, and dishwashers all draw 1,500–2,500W. When two run simultaneously, their combined signature in the aggregate signal is nearly indistinguishable from a single larger appliance. The model has no mechanism to separate them, leading to over-prediction of WM power during co-running periods.

**2. Missing data and household differences**  
House 2 had 24.3% missing data — a significant fraction. Imputed or missing minutes could create phantom cycles or mask real ones. Household-to-household variability in machine brand, age, programme selection, and usage culture means that a model trained on 3 homes may perform poorly on an outlier household.

**3. Applying UK results to India**  
Several factors make direct transfer difficult:
- Indian washing machines often run shorter, high-heat cycles.
- Electricity tariff structures, grid voltage (230V both, similar) and supply reliability differ.
- Usage patterns differ (e.g. daily cold-water bucket washing vs. weekly machine use is common).
- The REFIT dataset contains no tropical-climate or Indian household data.

A model fine-tuned on even a small set of labelled Indian smart-meter data would be far more reliable.

---

### Q3 — Impact of Lower Temporal Resolution (15-minute or 30-minute data)

*"What would become harder if only 15-minute or 30-minute smart-meter data were available?"*

A standard UK washing machine cycle lasts 30–120 minutes, with several distinct power phases:
- **Fill** (brief spike ~500W)
- **Wash** (sustained ~300–600W)
- **Heat** (peak ~2,000W for a few minutes)
- **Spin** (brief peaks ~300W)

At **1-minute resolution**, each phase is clearly visible and the model can detect:
- Exact cycle start/end times
- Cycle duration
- Peak power (heat phase)
- Total energy

At **15-minute resolution**, a full short cycle (30 min) spans only 2 data points. Phase boundaries disappear into a single averaged reading. This means:
- **Cycle detection becomes unreliable** — short cycles may not exceed the detection threshold in averaged data
- **Feature engineering degrades** — rolling means and lags lose their ability to capture within-cycle dynamics
- **Binary F1 drops significantly** — the model can only say "WM probably ran in this 15-minute window", not "the WM was running at 09:13"

At **30-minute resolution**, even detecting whether the machine ran during a given window becomes unreliable — the WM power averaged with 29 minutes of standby may fall below any reasonable threshold.

In short: meaningful NILM at appliance granularity requires ≤5-minute data. At 15 or 30 minutes, only coarse weekly usage totals per household can be estimated.

---

## Research References Used

1. **Murray et al. (2017)** — Original REFIT dataset paper. Used for dataset description, appliance assignments, and understanding the cleaning methodology.  
   Murray, D., Stankovic, L., & Stankovic, V. (2017). *Scientific Data*, 4, 160122.

2. **Kelly & Knottenbelt (2015)** — Neural NILM paper. Used for the 20W on-threshold convention and evaluation framework (MAE, F1).  
   Kelly, J., & Knottenbelt, W. (2015). *BuildSys 2015*.

3. **Bonfigli et al. (2018)** — Benchmark of NILM algorithms on REFIT. Guided our choice of Random Forest as a strong-but-simple baseline and informed our cross-household evaluation strategy.  
   Bonfigli, R., et al. (2018). *Energy and Buildings*, 158, 548–559.
