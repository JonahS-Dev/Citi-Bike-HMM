# Trip-Mix HMM: Discovering Citi Bike Usage Modes

Implementation: [`trip_mix_hmm.ipynb`](trip_mix_hmm.ipynb)

## 1. The idea in one paragraph

Earlier designs used **trip counts** (in/out of a station, or between a pair of stations) as the HMM
observation. With counts, the easiest thing for an HMM to learn is *busy vs. quiet*, which mostly
re-learns the clock. We instead remove volume from the observation: every trip is labeled with a
**trip type** (member vs. casual, short vs. long, direction relative to the Manhattan core, round trip),
and each hour of the day is observed as the **mix of trip types** across the whole city. A hidden state
is then a *usage mode*: a characteristic distribution over trip types (e.g. "morning commute" = lots of
short member trips into the core; "leisure" = casual, long, round trips). Because states cannot differ
in volume, they are forced to differ in *how* the city is riding.

## 2. Data

| Item | Choice |
|---|---|
| Source | Citi Bike system data, `https://s3.amazonaws.com/tripdata/` (NYC files; `JC-` Jersey City files are not used) |
| Window | **Oct 2025 – Sep 2026** (12 monthly zips). A full year covers all seasons, winter weather, and every federal holiday. |
| Columns used | `started_at, ended_at, start/end_station_name, start/end_lat/lng, member_casual` |
| Weather | Open-Meteo historical archive, hourly precipitation / snowfall / temperature at Central Park |
| Holidays | `pandas` US federal holiday calendar |

**Cleaning:** drop trips under 1 minute (false starts / re-docks) or over 3 hours (lost or unreturned bikes),
and trips with no end location.

**Service day:** a "day" runs **4 AM → 4 AM**, so a 1 AM ride belongs to the evening before it. Each day is
one HMM sequence of 24 hourly steps.

## 3. Trip types (the observation alphabet)

Each trip gets exactly one type, assigned in this priority order:

| # | Type | Rule | Intuition |
|---|---|---|---|
| 7 | `round_trip` | start station = end station | park loops, joyrides |
| 5 | `casual_short` | casual, < 20 min | tourists / day-pass errands |
| 6 | `casual_long` | casual, ≥ 20 min | sightseeing, leisure |
| 4 | `member_long` | member, ≥ 20 min | long recreational rides by members |
| 0 | `member_into_core` | member, short, starts outside core, ends inside | inbound commute |
| 1 | `member_out_of_core` | member, short, starts inside core, ends outside | outbound commute |
| 2 | `member_within_core` | member, short, both ends in core | midday errands, last mile from transit |
| 3 | `member_outer` | member, short, both ends outside core | neighborhood trips (Brooklyn, Queens, uptown) |

"Core" = Manhattan south of ~59th St, approximated by a 12-vertex lat/lng polygon. The 20-minute
threshold and the core boundary are modeling choices. They are stated, and the ablations check whether
the conclusions survive changing them.

## 4. Model

A standard discrete-state HMM with K hidden states.

- **Hidden state** `z_t ∈ {1..K}`: the city's usage mode in hour t.
- **Transitions** `A[i,j] = P(z_{t+1}=j | z_t=i)`, initial distribution `π`.
- **Emissions:** each state has a distribution `B[i,:]` over the 8 trip types. Hour t's observation is the
  vector of trip-type counts `x_t = (n_{t,0}, ..., n_{t,7})`, and every trip in the hour is treated as an
  independent draw from `B[z_t,:]`:

  ```
  log P(x_t | z_t = i) = Σ_k n_{t,k} · log B[i,k]      (+ a multinomial constant that doesn't depend on i)
  ```

  This is the discrete HMM from class, except each time step emits a *bag* of symbols instead of one.

**Independence assumptions** (to discuss in the report):
1. Markov: the usage mode in hour t+1 depends only on the mode in hour t (not on the clock or weekday).
2. Given the mode, trips in an hour are i.i.d. draws from `B[z_t,:]`. They ignore each other, and they ignore volume.
3. Days are independent sequences. Each day restarts from `π`.

### Tempering (effective sample size)

A busy hour has ~15,000 trips. Treated literally as 15,000 independent draws, the emission
log-likelihood is so large that posteriors are always 0/1 and adding more states always increases
likelihood. Trips within an hour are not really independent (groups ride together, weather affects
everyone), so we **rescale each hour to a fixed effective size**:

```
x_t ← N_eff · x_t / Σ_k x_{t,k}          (default N_eff = 50)
```

That is, "treat each hour's mix as if we had seen 50 independent trips." Hours with very few trips
(overnight) keep the same weight, which is fine: the transitions carry the model through them. `N_eff`
is a hyperparameter and has its own ablation.

### Learning and inference

- **Baum–Welch (EM)** via `hmmlearn`. We subclass `hmmlearn.hmm.MultinomialHMM` (~10 lines) so it accepts
  fractional (tempered) counts. The E-step is unchanged. The emission M-step is
  `B[i,k] ∝ Σ_t γ_t(i) · n_{t,k}`, i.e. expected trip-type counts while in state i.
- **10 random restarts** per configuration, keeping the best training log-likelihood (EM finds local optima).
- **Viterbi** for the most likely state path per day.
- **Forward filtering** for prediction: `P(z_{t+1} | x_{1..t}) = (α_t normalized) · A`, and predicted next-hour
  mix `= P(z_{t+1} | x_{1..t}) · B`.
- **Train/test split by day:** 80% of service days for training, 20% held out (whole days, never split
  within a day).
- **Choosing K:** held-out log-likelihood per hour and BIC on train, for K = 2…9, plus whether the states can be interpreted.
- States are re-ordered by the time of day at which they peak, so state numbering is stable across runs.

## 5. How we show the states are meaningful (not just the clock)

| Check | What would count as success |
|---|---|
| **Emission "lift" heatmap**: `B[i,k] / overall_mix[k]` (log scale) | Each state has a clear signature we can name (inbound commute, outbound commute, leisure, …) |
| **Viterbi calendar** (rows = days, cols = hours) | Visible weekday/weekend banding *without the model ever seeing the day of week* |
| **Conditional mutual information** `I(state ; workday | hour)` | State carries information about the type of day beyond what the hour alone explains. Pure-clock states score ~0. |
| `I(state ; raining | hour, workday)` | State responds to weather |
| **Holiday case studies** (Labor Day, Thanksgiving, July 3, Christmas, …) | Major holidays decode like Sundays; minor ones (Columbus, Veterans) decode like workdays |
| **Rain / temperature** | Leisure states disappear on rainy days; commute states remain |
| **Transition matrix** | Readable daily order, e.g. night → AM commute → midday → PM commute → evening leisure → night |
| **Day-level HMM (stretch)** | A second HMM over days, with observation = hours spent in each hourly state. Recovers day types (workday / weekend / bad-weather) and the weekly rhythm. |

## 6. Prediction (quantitative evaluation)

Task: given today's hours up to t, predict the trip-type mix of hour t+1 on **held-out days**.
Metric: cross-entropy per trip, `−Σ_k p_{t+1,k} log q_{t+1,k}` (lower is better), averaged over trips.

| Predictor | Uses calendar? |
|---|---|
| Overall average mix | no |
| Average mix for this hour of day | clock only |
| Average mix for (hour, workday/non-workday) | clock + calendar |
| Persistence (last hour's mix) | no |
| **HMM forward filter** | **no** (infers the day type from the data seen so far) |

The (hour, workday) baseline is strong because it is *told* the calendar. If the HMM gets close to it
without that information, or beats it on unusual days (holidays, storms), that is the HMM's value.

## 7. Ablations

| Ablation | Question it answers |
|---|---|
| **K** = 2…9 | How many usage modes does the data support? |
| **N_eff** ∈ {5, 20, 50, 200, raw counts} | How does tempering affect confidence, stability, number of state switches, and prediction? |
| **Trip-type alphabet**: (a) full 8 types, (b) no direction split (types 0–3 merged into "member short"), (c) no member/casual split | Which features create the meaningful states? Without direction, do AM and PM commutes merge? |
| **Volume vs. mix**: (a) `PoissonHMM` on total trips per hour, (b) `PoissonHMM` on raw per-type counts, (c) our mix model | Do count-based models collapse into busy/quiet clock states? Compared on `I(state; hour)` vs `I(state; workday | hour)` |

## 8. Results (from `trip_mix_hmm.ipynb`, seed 0)

**Data.** 45.2M trips, Oct 2025 – Sep 2026, giving 362 usable service days (290 train / 72 test). All of the
lost days are snowstorm shutdowns:
- **Jan 25 2026** (~20 cm of snow): 9 zero-trip daytime hours, so it's excluded.
- **Feb 22 2026** (~9 cm): 1 zero-trip hour, so it's kept (under the 2-hour threshold).
- **Feb 23 2026** (~13 cm): no trips at all, so the date is absent from the data.

**Choosing K.** Held-out log-likelihood keeps improving slowly up to K=9, and BIC is lowest at K=8. We use
**K = 6**, where every state can be named. Restarts end hundreds of log-likelihood units apart, so multiple
restarts matter.

**The six states (full year):**

| State | Signature (lift vs. average mix) | When it occurs |
|---|---|---|
| S0 pre-dawn | relatively more casual/short | 4 AM only |
| S1 morning commute | most `member_into_core` (6% vs. 3.5%); almost no casual-long / round trips | workdays 5–10 AM; never on weekends |
| S2 winter members | casual riding nearly disappears | whole days Dec–Mar; 87% of daytime on snowy workdays |
| S3 off-peak mix | close to the average mix | cool-season weekends, late mornings, nights |
| S4 warm workday | slightly more out-of-core and long member trips | warm-season workday daytime and evening |
| S5 leisure | casual short/long and round trips | summer weekends, major holidays, late evenings |

**Transitions** are almost upper-triangular (pre-dawn → morning commute → off-peak → warm workday → leisure),
and leisure is absorbing within a day (`A[S5,S5] = 1.00`). Without being told, the model learned that a day
moves in one direction, like a left-to-right HMM.

**Meaningfulness.**
- `I(state; workday | hour)` = **0.30 bits**. A pure-clock model scores 0, and the maximum is ~0.86.
- **Holidays:** all 6 major holidays decode like weekends. 4 of the 5 minor ones (Columbus, Veterans, MLK,
  Presidents) decode like workdays. The exception, **Juneteenth (Fri Jun 19)**, decodes like a weekend, which
  suggests many riders treated it as a day off. Labor Day's Viterbi path is identical to the Sunday before it.
- **Weather:** on rainy workdays, daytime shifts from S4 warm workday (35.5% → 16.3%) to S3 off-peak
  (18.0% → 36.8%). On snowy workdays S2 winter members takes 87% of daytime.

**Prediction** (next-hour mix, bits per trip, held-out days):

| Predictor | All test days |
|---|---|
| overall mix | 2.4248 |
| hour of day | 2.4093 |
| hour + workday (*given the calendar*) | 2.3983 |
| **HMM forward filter** (no calendar) | **2.3981** |
| persistence (last hour's mix) | 2.3888 |

Without seeing the calendar, the HMM matches the calendar-aware baseline. Persistence is still best: hour-to-hour
mixes are strongly autocorrelated (season, weather), and 6 coarse states can't capture all of that. This is
worth reporting honestly. The states are better at *explaining* the data than at *forecasting* it.

**Ablations.**

| Ablation | Result |
|---|---|
| `N_eff` 5 / 20 / 50 / 200 / raw | `I(workday \| hour)` 0.10 / 0.22 / 0.30 / 0.37 / 0.29. Confidence rises with `N_eff` (0.74 → 0.98 mean max posterior), and raw counts produce jittery paths (5.1 switches/day vs. 2.2 at 50). Prediction is almost flat (2.407 → 2.394). |
| Alphabet | Removing direction drops `I(workday \| hour)` from 0.30 to 0.20. A morning state survives (it's also low-casual), but it's less distinct. Removing member/casual keeps 0.31 but makes the states more clock-like (`I(state; hour)` 0.55 → 0.70). |
| **Volume vs. mix** | Poisson on total trips: `I(hour)` 0.76, `I(workday \| hour)` 0.15. Poisson on per-type counts: 0.79 / 0.17. **Trip mix: 0.55 / 0.30.** The count models are more clock-like and carry about half the day-type information. They do respond more to rain (0.05 vs. 0.03 bits), because rain cuts volume more than it changes the mix. |
| K = 4…8 | `I(workday \| hour)` 0.20, 0.34, 0.30, 0.31, 0.45. The conclusions hold for K ≥ 5. |
| **Warm season only** (May–Sep) | `I(workday \| hour)` 0.57, vs. 0.46 for the full-year model on the same hours. Without season to explain, the states become within-day modes: pre-dawn, weekend-morning neighborhood rides (`member_outer`), morning commute, workday daytime, **workday evening**, weekend leisure. |

**Day-level HMM** (K=4, observation = hours per hourly state). It finds **winter days** (93 days, mean max
2.8 °C), **warm workdays** (138, 24 °C), **cool/shoulder and rainy days** (78, 23% rainy), and **summer
weekends/holidays** (53, 4% workdays). The day-type calendar shows seasons and weekends as blocks.

**Takeaways for the report.**
1. Removing volume from the observation is what makes the states meaningful. Count-based HMMs recover the clock.
2. The strongest hidden structure in a full year is **season × day type**. Within a season, the states
   become usage modes.
3. A natural extension follows from this: seasonal models, or a hierarchical (day-type → hour-mode) HMM.

## 9. Limitations to discuss

- **Season is mixed into the states:** in a full year, two of six states are mostly seasonal regimes. See the
  warm-season ablation.
- **Persistence beats the HMM at forecasting:** the states are coarse summaries, not a forecasting model.
- **No clock in the transitions:** the HMM is time-homogeneous, so part of what the states encode is
  time of day. The CMI metrics measure how much they encode *beyond* the clock.
- **Weekly rhythm isn't first-order Markov:** a first-order day-level HMM can't count "five workdays then two weekend days".
- **Tempering is a modeling choice:** `N_eff` is chosen, not learned.
- **Trip types are hand-designed:** the 20-min / 59th-St thresholds are arbitrary. Ablations partly address this.
- **Member/casual is a pass type, not a purpose:** members also ride for leisure.
- **System-wide aggregation hides local structure:** a Brooklyn commute and a Midtown errand both count as "member short".
- **Data issues:** months with partial uploads, system outages, and DST days (23/25 hours mapped onto 24 bins).
