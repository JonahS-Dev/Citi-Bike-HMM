# Gaussian vs. HMM iteration 3, measured against proper baselines (full-year walk-forward)

This compares two models on the same data, splits and information:
- **Iteration 3 HMM:** the clock-aware Poisson HMM in `citibike_hmm_iterations.ipynb`.
- **Multivariate Gaussian:** the model in `citibike_non_hmm_notebook/`.

Both are also scored against simple baselines, so we can see what each model actually adds. Every month from **October 2025 to September 2026** is tested, with each month forecast by models trained only on the 12 months before it (363 test days).

## TL;DR

- **Both models clearly beat the simple baselines** in every comparison: 13–74% lower error on total counts than the hour-of-week mean (significant at 95%). The one exception is the Gaussian's net-flow forecast 12 hours ahead in UES/Midtown, which only ties it.
- **Noon task** (predict hours 12–23 after seeing hours 0–11): the **Gaussian wins**.
  - In R/J zones it wins all 12 months and is 28% more accurate than the HMM.
  - In UES/Midtown it is only 8% better than the HMM and ties the activity-adjusted baseline.
  - The **HMM loses to the simple activity-adjusted baseline** on this task (−15% in R/J zones).
- **Rolling task** (forecast 1–12 hours ahead from any hour):
  - The **Gaussian wins 1 hour ahead** (by 20% in R/J zones; tied in UES/Midtown).
  - The two are **roughly tied around 6 hours** on total counts.
  - The **HMM wins 12 hours ahead** in both setups (by 9% in R/J zones and 23% in UES/Midtown).
- **Net flow** (A→B minus B→A; R→J minus J→R in the R/J setup), the quantity that drains or fills stations:
  - **R/J zones:** the HMM is better from about 6 hours ahead (by 9% at 6 hours and 30% at 12).
  - **UES/Midtown:** tied with the Gaussian 1 hour ahead, better from about 6 hours (by 10% at 6 hours and 17% at 12).
  - Its net-flow error barely grows with the horizon.
- **The hidden state earns its keep only on the rolling task.** There the HMM beats activity-adjusted at every horizon: by 8–26% on total counts and 4–32% on net flow, across both setups.
- **Season matters.** The HMM is weakest in winter (Dec–Feb), when the Gaussian and even activity-adjusted beat it clearly. It is strongest in summer, when it wins the rolling task every month.
- The ranking is the same in both zone setups (see `ZONE_SETUP_IDEAS.md`).

## 1. What a proper baseline is

A baseline is the simplest reasonable method; any real model should beat it. We use three, each testing a different question:

| Baseline | What it predicts | Question it answers |
|---|---|---|
| **Hour-of-week mean** | Training-period average for that weekday and hour (168 slots). This is the baseline named in our proposal. | Does looking at recent data help at all? |
| **Persistence** | Next hour = this hour | Does knowing the clock help? |
| **Activity-adjusted** | Normal hourly profile × today's busyness ratio (trips so far today ÷ normal trips by this time); ratio = 1 when nothing of today has been seen yet | Does a hidden state beat a simple running busyness ratio? |

**Activity-adjusted is the most important baseline for the HMM.** In an earlier single fit (trained Oct 2025–Aug 2026), iteration 3's hidden states turned out to be almost pure volume multipliers: each state applied nearly the same multiplier (×0.14, ×0.38, ×0.68, ×1.00, ×1.33, ×1.67) to every ride type. So the HMM is essentially "clock average × how busy it is right now," which is the activity-adjusted idea with a hidden state added. If it can't beat activity-adjusted, the hidden-state machinery isn't adding anything.

The **Gaussian is not a baseline.** It is a competing probabilistic model with a 48×48 covariance and a tuned shrinkage setting. We present it as a second model, not as something the HMM should trivially beat.

## 2. The two models

| | Multivariate Gaussian | Iteration 3 HMM |
|---|---|---|
| Unit of data | One day = 48 numbers (24 hours × 2 directions) | One hour = 6 ride-type counts (or just AB/BA) |
| Core idea | Learns how each hour co-varies with every other hour, then conditions on the hours seen so far | A hidden "busyness state" scales the hour-of-week average; the state persists from hour to hour |
| Memory | Resets every midnight | Uses roughly the last hour to know the state; that state carries forward with a ~16-hour half-life and fades after ~2 days |
| How the forecast adjusts | A separate adjustment for each future hour and direction | One multiplier per state (per ride type), drifting back toward average the further ahead it looks |
| Distribution | Continuous Gaussian, clipped at 0 | Poisson counts |
| Tuning | Shrinkage α, chosen on validation | Number of states K, chosen on validation |

## 3. How the comparison was run: walk-forward over a full year

| Item | Setting |
|---|---|
| Data | Citi Bike trips, **Oct 2024–Sep 2026**: 91.8M rides, using the notebook's cleaning rules (drop missing stations, dedupe `ride_id`) |
| Test period | **Every month from Oct 2025 to Sep 2026**: 12 folds and 363 test days. The two daylight-saving dates are excluded. |
| Training per fold | **Only the 12 months before the test month.** No future data is ever used. |
| Re-derived inside each fold | Everything is recomputed from that fold's 12 training months: R/J zones, the UES/Midtown station list, the hour-of-week baseline, HMM K and Gaussian α. |
| Hyperparameter selection | Fit on the first 11 training months, validate on the 12th (the month just before the test month), then refit on all 12. **K** uses the notebook's elbow rule on validation log-likelihood and came out 4–6. **α** uses validation RMSE over cutoffs 6/9/12/15/18 and came out mostly 0.05 in R/J and 0.05–0.75 in UES/Midtown. |
| Zone setup 1 | **Iteration R/J zones:** stations labelled by weekday-morning net flow (912–972 R, 515–568 J, depending on fold). AB = R→J, BA = J→R. |
| Zone setup 2 | **UES/Midtown zones:** the non-HMM notebook's rectangles, with the station list rebuilt by its rules (52–57 A, 69–75 B) |
| HMM variants | **6 types**, as in the notebook, and **AB/BA only**, which sees the same information as the Gaussian |
| Metrics | MAE and RMSE over AB and BA cells (trips per hour per direction); **net-flow MAE** = error on (AB − BA) per hour. All errors are pooled over the full year. |
| Improvement | % improvement = 100 × (1 − model error ÷ baseline error). Positive = better than the baseline. |
| Significance | Paired **block bootstrap over weeks** (4,000 resamples). Whole weeks are resampled because consecutive days share weather and aren't independent. |

**Two tasks**, since each model was designed for a different one:
- **Task A (the Gaussian's task):** at noon, predict hours 12–23 from hours 0–11.
- **Task B (the HMM notebook's task):** for every test hour, forecast h = 1…12 hours ahead using everything up to that point.

## 4. Task A: noon, predict hours 12–23 (full year)

### Iteration R/J zones (average afternoon cell ≈ 482 trips/hour)

| Model | MAE | RMSE | Net MAE | MAE vs hour-of-week | MAE vs activity-adj. | Net MAE vs activity-adj. |
|---|---:|---:|---:|---:|---:|---:|
| Hour-of-week mean *(baseline)* | 193.9 | 251.1 | 79.2 | — | −130% | −25% |
| Historical weekday/weekend average | 197.1 | 256.3 | 90.2 | −2% | −134% | −42% |
| Activity-adjusted *(baseline)* | 84.4 | 130.7 | 63.5 | +56% | — | — |
| **Gaussian** (α validation-picked) | **70.4** | **117.6** | **50.6** | **+64%** | **+17%** | **+20%** |
| Gaussian (α = 0.5, original) | 76.7 | 122.5 | 54.5 | +60% | +9% | +14% |
| **Iter 3 HMM** (6 types) | 97.3 | 146.0 | 59.9 | +50% | −15% | +6% |
| Iter 3 HMM (AB/BA only) | 97.6 | 143.0 | 59.8 | +50% | −16% | +6% |

### UES/Midtown zones (average afternoon cell ≈ 35 trips/hour)

| Model | MAE | RMSE | Net MAE | MAE vs hour-of-week | MAE vs activity-adj. | Net MAE vs activity-adj. |
|---|---:|---:|---:|---:|---:|---:|
| Hour-of-week mean *(baseline)* | 15.66 | 22.60 | 10.40 | — | −84% | −18% |
| Historical weekday/weekend average | 15.90 | 23.13 | 11.08 | −2% | −87% | −26% |
| Activity-adjusted *(baseline)* | 8.50 | **13.23** | 8.80 | +46% | — | — |
| **Gaussian** (α validation-picked) | **8.39** | 13.27 | 8.75 | **+46%** | **+1%** | +1% |
| Gaussian (α = 0.5, original) | 8.50 | 13.25 | 8.75 | +46% | 0% | +1% |
| **Iter 3 HMM** (6 types) | 9.13 | 14.50 | 8.92 | +42% | −7% | −1% |
| Iter 3 HMM (AB/BA only) | 8.74 | 13.46 | **8.62** | +44% | −3% | **+2%** |

### Task A by month (MAE)

| Month | R/J: Activity-adj. | R/J: Gaussian | R/J: HMM (6 types) | UES/Midtown: Activity-adj. | UES/Midtown: Gaussian | UES/Midtown: HMM (6 types) |
|---|---:|---:|---:|---:|---:|---:|
| Oct 2025 | 79.7 | **61.9** | 74.8 | 8.97 | 8.51 | **8.38** |
| Nov 2025 | 80.6 | **66.5** | 83.9 | 8.89 | **8.48** | 8.71 |
| Dec 2025 | 53.8 | **40.1** | 86.8 | 5.74 | **5.63** | 7.61 |
| Jan 2026 | 56.0 | **42.1** | 115.6 | **4.77** | 5.41 | 8.89 |
| Feb 2026 | 36.2 | **33.1** | 99.4 | **3.40** | 4.00 | 6.42 |
| Mar 2026 | 67.2 | **58.2** | 73.4 | **6.67** | 6.72 | 7.36 |
| Apr 2026 | 89.6 | **72.3** | 79.6 | 8.55 | **8.20** | 8.64 |
| May 2026 | 106.7 | **83.6** | 105.3 | 9.99 | **9.71** | 10.63 |
| Jun 2026 | 114.2 | **100.4** | 113.1 | 11.22 | 12.53 | **10.61** |
| Jul 2026 | 118.7 | **97.1** | 116.1 | 11.94 | 11.85 | **11.51** |
| Aug 2026 | 115.8 | **100.9** | 107.8 | 11.68 | **10.06** | 10.56 |
| Sep 2026 | 90.3 | **86.6** | 110.3 | 9.72 | **9.25** | 9.94 |
| **Months won** | 0 | **12** | 0 | 3 | **6** | 3 |

**Reading Task A:**
- Both models beat the hour-of-week mean by 42–64%.
- **R/J zones:** the Gaussian wins every month. The HMM loses to activity-adjusted on total counts, and the gap is significant (see section 7).
- **UES/Midtown:** all three are close; the Gaussian and activity-adjusted are statistically tied.
- **The HMM's weak spot is winter.** In Dec–Feb its R/J noon error is 2–3× the Gaussian's. The hour-of-week baseline averages all 12 months, so winter traffic sits far below it, and the HMM has to cover that gap with a few coarse multiplier levels. Activity-adjusted rescales smoothly to any level, and the Gaussian conditions directly on the morning.
- **Over the full year, neither model is biased.** Predicted afternoon totals are within about 1% of actual for every model (R/J: Gaussian 11,479, HMM 11,652, actual 11,559 trips/day). The under-prediction seen in the September-only test was specific to that month.

## 5. Task B: rolling forecast h hours ahead (every hour, full year)

### Total-count MAE (AB and BA cells)

| Setup | Model | h=1 | h=3 | h=6 | h=12 | Mean h=1–12 |
|---|---|---:|---:|---:|---:|---:|
| R/J | Hour-of-week mean *(baseline)* | 141.1 | 141.1 | 141.1 | 141.1 | 141.1 |
| R/J | Persistence *(baseline)* | 116.3 | 264.3 | 380.7 | 453.1 | 351.7 |
| R/J | Activity-adjusted *(baseline)* | 58.0 | 75.2 | 87.2 | 109.8 | 85.9 |
| R/J | **Gaussian** | **36.7** | **56.3** | **74.4** | 98.9 | **72.9** |
| R/J | **Iter 3 HMM** (6 types) | 45.6 | 61.8 | 77.8 | **90.5** | 74.2 |
| R/J | Iter 3 HMM (AB/BA only) | 45.4 | 62.9 | 79.2 | 93.7 | 76.0 |
| UES/Midtown | Hour-of-week mean *(baseline)* | 12.18 | 12.18 | 12.18 | 12.18 | 12.18 |
| UES/Midtown | Persistence *(baseline)* | 12.42 | 23.39 | 31.22 | 38.37 | 29.69 |
| UES/Midtown | Activity-adjusted *(baseline)* | 6.24 | 7.92 | 9.83 | 11.77 | 9.41 |
| UES/Midtown | **Gaussian** | 5.71 | 7.20 | 8.55 | 10.62 | 8.49 |
| UES/Midtown | **Iter 3 HMM** (6 types) | 5.73 | **6.74** | **7.85** | **8.67** | **7.58** |
| UES/Midtown | Iter 3 HMM (AB/BA only) | **5.64** | 6.95 | 7.90 | 9.08 | 7.77 |

### % improvement over activity-adjusted, total-count MAE

| Setup | Model | h=1 | h=3 | h=6 | h=12 | Mean h=1–12 |
|---|---|---:|---:|---:|---:|---:|
| R/J | Gaussian | **+37%** | **+25%** | **+15%** | +10% | **+15%** |
| R/J | Iter 3 HMM (6 types) | +21% | +18% | +11% | **+18%** | +14% |
| UES/Midtown | Gaussian | **+9%** | +9% | +13% | +10% | +10% |
| UES/Midtown | Iter 3 HMM (6 types) | +8% | **+15%** | **+20%** | **+26%** | **+20%** |

### Net-flow MAE (AB − BA)

| Setup | Model | h=1 | h=3 | h=6 | h=12 | Mean h=1–12 |
|---|---|---:|---:|---:|---:|---:|
| R/J | Hour-of-week mean *(baseline)* | 85.2 | 85.2 | 85.2 | 85.2 | 85.2 |
| R/J | Activity-adjusted *(baseline)* | 60.3 | 76.3 | 83.1 | 90.3 | 79.2 |
| R/J | **Gaussian** | **40.1** | **54.6** | 67.2 | 79.3 | 64.7 |
| R/J | **Iter 3 HMM** (6 types) | 47.6 | 54.9 | 61.7 | **61.2** | **57.8** |
| R/J | Iter 3 HMM (AB/BA only) | 45.5 | 55.0 | **61.6** | 61.9 | **57.8** |
| UES/Midtown | Hour-of-week mean *(baseline)* | 9.40 | 9.40 | 9.40 | 9.40 | 9.40 |
| UES/Midtown | Activity-adjusted *(baseline)* | 7.16 | 8.43 | 9.54 | 9.51 | 8.92 |
| UES/Midtown | **Gaussian** | 6.89 | 7.97 | 8.56 | 9.26 | 8.41 |
| UES/Midtown | **Iter 3 HMM** (6 types) | 6.84 | **7.27** | 7.81 | **7.89** | **7.56** |
| UES/Midtown | Iter 3 HMM (AB/BA only) | **6.70** | 7.39 | **7.67** | 8.01 | 7.57 |

### % improvement over hour-of-week mean, net-flow MAE

| Setup | Model | h=1 | h=3 | h=6 | h=12 | Mean h=1–12 |
|---|---|---:|---:|---:|---:|---:|
| R/J | Activity-adjusted | +29% | +10% | +2% | −6% | +7% |
| R/J | Gaussian | **+53%** | **+36%** | +21% | +7% | +24% |
| R/J | Iter 3 HMM (6 types) | +44% | **+36%** | **+28%** | **+28%** | **+32%** |
| UES/Midtown | Activity-adjusted | +24% | +10% | −1% | −1% | +5% |
| UES/Midtown | Gaussian | +27% | +15% | +9% | +2% | +11% |
| UES/Midtown | Iter 3 HMM (6 types) | **+27%** | **+23%** | **+17%** | **+16%** | **+20%** |

### Task B by month (total-count MAE, mean over h = 1–12)

| Month | R/J: Activity-adj. | R/J: Gaussian | R/J: HMM (6 types) | UES/Midtown: Activity-adj. | UES/Midtown: Gaussian | UES/Midtown: HMM (6 types) |
|---|---:|---:|---:|---:|---:|---:|
| Oct 2025 | 78.2 | 63.6 | **59.7** | 9.19 | 8.13 | **7.44** |
| Nov 2025 | 73.9 | **58.8** | 68.5 | 8.17 | 7.41 | **7.33** |
| Dec 2025 | 60.8 | **60.1** | 74.6 | 7.04 | 7.78 | **7.03** |
| Jan 2026 | 65.0 | **61.4** | 90.3 | **6.50** | 7.18 | 7.71 |
| Feb 2026 | **55.8** | 66.8 | 83.7 | **5.49** | 7.43 | 6.30 |
| Mar 2026 | 67.6 | **55.9** | 58.5 | 6.90 | 6.48 | **6.03** |
| Apr 2026 | 86.0 | 69.7 | **66.7** | 9.02 | 7.53 | **7.27** |
| May 2026 | 87.1 | **74.4** | 76.1 | 9.68 | 8.78 | **8.44** |
| Jun 2026 | 143.7 | 112.9 | **82.5** | 18.66 | 12.88 | **8.38** |
| Jul 2026 | 112.6 | 85.2 | **78.1** | 11.58 | 9.60 | **8.45** |
| Aug 2026 | 101.8 | 84.1 | **72.5** | 10.46 | 9.45 | **8.46** |
| Sep 2026 | 95.8 | 81.5 | **79.0** | 9.96 | 9.18 | **7.90** |

Months won at specific horizons (total-count MAE):

| Setup | Horizon | Gaussian | Iter 3 HMM | Activity-adjusted |
|---|---|---:|---:|---:|
| R/J | h=1 | **10** | 2 | 0 |
| R/J | h=6 | 4 | **6** | 2 |
| R/J | h=12 | 0 | **9** | 3 |
| UES/Midtown | h=1 | **7** | 3 | 2 |
| UES/Midtown | h=6 | 2 | **7** | 3 |
| UES/Midtown | h=12 | 0 | **12** | 0 |

**Reading Task B:**
- **Persistence** is only competitive 1 hour ahead and collapses after that, because the clock matters.
- **Activity-adjusted** helps on total counts but adds little or nothing on net flow 6+ hours ahead; it ties or slightly trails the plain hour-of-week mean there. Scaling both directions by the same ratio doesn't help predict the difference between them.
- **The Gaussian is best 1 hour ahead** in R/J zones, when it has seen most of today. It weakens further out because many of those forecasts are issued at night or early morning, when it has seen little of today. Since it resets at midnight, it falls back to an average day.
- **The HMM is best 12 hours ahead**, because its estimated busyness level carries across midnight. On net flow it keeps a ~28% (R/J) and ~16% (UES/Midtown) improvement over the hour-of-week mean even 12 hours out, while the Gaussian falls to 7% and 2%.
- **The seasonal pattern repeats.** The HMM loses the winter months (Dec–Feb) in R/J zones and wins the rolling task every month from June to September in both setups. In June the busyness ratio swings a lot; activity-adjusted error jumps to 144, while the HMM holds at 83.

## 6. Is the difference real? Week-block bootstrap

Positive = the first model is more accurate. The brackets are 95% confidence intervals; an interval that includes 0 means the difference isn't significant.

### Gaussian vs. Iter 3 HMM (6 types)

| Comparison | R/J zones | UES/Midtown |
|---|---|---|
| Task A MAE | **+27.6%** [+20.1, +35.5] | **+8.1%** [+0.6, +15.6] |
| Task A net MAE | **+15.6%** [+7.7, +23.4] | +1.9% [−2.5, +6.4] (tie) |
| Task B h=1 MAE | **+19.5%** [+12.8, +26.3] | +0.4% [−3.6, +4.3] (tie) |
| Task B h=6 MAE | +4.4% [−4.1, +11.8] (tie) | **−9.0%** [−18.0, −2.5] |
| Task B h=12 MAE | **−9.3%** [−16.6, −3.7] | **−22.6%** [−31.1, −16.7] |
| Task B h=1 net MAE | **+15.8%** [+11.1, +20.3] | −0.7% [−3.3, +1.9] (tie) |
| Task B h=6 net MAE | **−8.8%** [−18.9, −0.7] | **−9.7%** [−14.4, −5.2] |
| Task B h=12 net MAE | **−29.5%** [−35.5, −24.0] | **−17.3%** [−20.6, −14.1] |

### Each model vs. the activity-adjusted baseline

| Comparison | HMM, R/J | HMM, UES/Midtown | Gaussian, R/J | Gaussian, UES/Midtown |
|---|---|---|---|---|
| Task A MAE | **−15.2%** [−29.7, −4.2] | −7.5% [−17.3, +0.2] (tie) | **+16.5%** [+13.6, +19.3] | +1.3% [−2.9, +5.0] (tie) |
| Task A net MAE | +5.7% [−3.9, +13.7] (tie) | −1.4% [−6.6, +2.7] (tie) | **+20.4%** [+17.1, +23.3] | +0.6% [−2.4, +3.1] (tie) |
| Task B h=1 MAE | **+21.2%** [+13.2, +27.3] | **+8.1%** [+3.2, +12.4] | **+36.6%** [+33.9, +39.1] | **+8.5%** [+6.2, +10.5] |
| Task B h=6 MAE | +10.8% [−0.4, +20.3] (tie) | **+20.2%** [+9.6, +30.5] | **+14.7%** [+9.7, +18.8] | **+13.0%** [+5.4, +19.0] |
| Task B h=12 MAE | **+17.6%** [+9.8, +24.7] | **+26.4%** [+18.8, +35.3] | **+10.0%** [+6.3, +13.3] | **+9.7%** [+2.9, +16.5] |
| Task B h=6 net MAE | **+25.7%** [+16.9, +33.3] | **+18.2%** [+11.4, +24.9] | **+19.1%** [+11.8, +25.0] | **+10.3%** [+3.6, +15.7] |
| Task B h=12 net MAE | **+32.2%** [+28.4, +35.8] | **+17.0%** [+14.5, +19.6] | **+12.2%** [+9.7, +14.7] | **+2.7%** [+0.3, +4.8] |

Both models beat the **hour-of-week mean** significantly in every comparison but one: the Gaussian's net flow 12 hours ahead in UES/Midtown, which is a tie (+1.5% [−1.5, +4.3]).

## 7. Conclusions

1. Both beat the proposal's hour-of-week baseline significantly almost everywhere, by 13–74% on total counts.
2. **The hidden state earns its keep on the rolling task, not the noon task.**
   - **Rolling task:** the HMM beats activity-adjusted at every horizon, significantly in all but one case (R/J total counts at h=6, a tie). The margin reaches 26% on counts and 32% on net flow.
   - **Noon task:** it loses to activity-adjusted on total counts in R/J zones, and the loss is significant.
3. **The models' strengths complement each other.**
   - **Gaussian:** detailed hour-to-hour relationships within a day. It wins at noon and 1 hour ahead.
   - **HMM:** persistent busyness that carries across hours and days. It wins 12 hours ahead and on net flow from about 6 hours.

   A natural next iteration is an HMM that also knows within-day structure, e.g. hour-dependent transitions, or a Gaussian/regression emission conditioned on earlier hours of the same day.
4. **The HMM's winter weakness points to a concrete fix.** The hour-of-week baseline is a 12-month average, so winter traffic sits far below it, and the HMM has to cover that gap with a few coarse multiplier levels. A **seasonal baseline**, e.g. hour-of-week × month, or a trailing 4–8-week mean, would let the hidden state model day-to-day deviations instead of season.
5. **Seeing 6 ride types barely helps the HMM.** The AB/BA-only version performs about the same, sometimes better. The states are volume levels, so making them carry more than volume would need a different observation design (see `ZONE_SETUP_IDEAS.md`).

## 8. Caveats

- **One test year** (Oct 2025–Sep 2026). The walk-forward design and week-block bootstrap make the results much more reliable than a single month, but a different year could still differ somewhat.
- **No trip-duration filter:** the cached ride data has no end times, so the Gaussian's 60-second to 24-hour filter couldn't be applied.
- **K selection in the notebook:** `citibike_hmm_iterations.ipynb` as written picks K using the **test** month. This comparison picks K on a validation month inside each fold instead. The notebook should be changed to match before reporting its numbers.
- **The original α = 0.5 is close to the tuned value but not the same.** The validation-picked α was mostly 0.05 in R/J zones. It beat α = 0.5 over the full year (R/J noon MAE 70.4 vs 76.7), so the tuned value is the headline number.
