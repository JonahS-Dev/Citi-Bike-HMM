# Zone setup: current options and ideas for improving them

## Current setups

| | Iteration R/J zones (`citibike_hmm_iterations.ipynb`) | UES/Midtown zones (`citibike_non_hmm_notebook/`) |
|---|---|---|
| How zones are defined | From the data: stations that empty out on weekday mornings (7–10am) are **residential (R)**, stations that fill up are **job centre (J)** | Two fixed latitude/longitude rectangles, drawn in advance |
| Coverage | Whole city: 969 R and 561 J stations, scattered everywhere | Two neighbourhoods: 54 Upper East Side and 70 Midtown stations |
| AB+BA trips per test day (Sep 2026) | ~23,100 | ~1,760 |
| Counts per hour | Hundreds | Tens |

### What each one gets right and wrong

**Iteration R/J zones**
- **For:** they match the project question (residential vs. job-centre flow), cover the whole city, and give large counts, so model differences show up clearly.
- **Against, circular:** zones are defined from morning flows, so "R→J peaks in the morning" is partly true by construction.
- **Against, not actionable:** "R" is ~1,000 stations across Manhattan, Brooklyn and Queens. A rebalancing truck can't act on "net flow into all job centres citywide."

**UES/Midtown zones**
- **For:** fixed in advance and not defined from flows, so no circularity. They match the original 2019 Gaussian study, and two specific neighbourhoods are something you could actually rebalance between.
- **Against:** counts are small and noisy, there's only one pair of neighbourhoods, and "residential" and "commercial" are assumptions about those areas, not measurements.

### Both setups give the same model ranking

We compared iteration 3 (clock-aware HMM) with the multivariate Gaussian on the same year: train Oct 2025–Aug 2026, test September 2026. HMM K and Gaussian α were chosen on an August validation month. Errors below are MAE divided by the average actual count per cell (one hour, one direction), so the two setups are comparable. Lower is better.

| Task | Model | R/J zones | UES/Midtown |
|---|---|---|---|
| Noon, predict hours 12–23 | Gaussian | **13%** | **20%** |
| | Iter 3 HMM | 18% | 22% |
| Rolling, 1 hour ahead | Gaussian | **9%** | **17%** |
| | Iter 3 HMM | 11% | 18% |
| Rolling, 12 hours ahead | Gaussian | 24% | 33% |
| | Iter 3 HMM | **21%** | **25%** |

R/J errors are lower only because counts are bigger: noise is a smaller share of a count of 500 than of 40. The ranking is the same in both setups (the Gaussian wins at noon and in the short term, the HMM wins further ahead), so the conclusion doesn't depend on how zones are drawn.

## Ideas for better zones, from least to most effort

### 1. Quick fix: define R/J zones from a different period
Label stations using data the models never train or test on, for example the first two months, or the previous year.
- **For:** a one-line change to the notebook.
- **Against:** only a partial fix. Commute patterns are stable, so the zones will come out nearly the same.

### 2. Require both a morning and an evening signature
Call a station residential only if it **empties in the morning and fills in the evening**, and job centre only for the reverse.
- **For:** filters out noisy stations near the threshold (tourist spots, stations near transit hubs).
- **Against:** still defined from flows, so still somewhat circular.

### 3. Make zones contiguous neighbourhoods (best for rebalancing)
Group stations into real neighbourhoods instead of scattered individual stations. NYC publishes official boundaries (Neighborhood Tabulation Areas or Community Districts); stations could also be clustered by location.
- Then pick a few high-volume residential→job pairs, e.g. Upper West Side→Midtown, Williamsburg→Financial District.
- **For:** zones are areas a rebalancing truck can actually serve, there are several test pairs instead of one, and counts are larger than UES/Midtown alone.

### 4. Label zones from census data (removes circularity entirely)
Use the Census Bureau's [LODES](https://lehd.ces.census.gov/data/) data, which gives the number of residents and jobs for every census block.
- **Job centre:** many more jobs than residents (e.g. Midtown, Financial District).
- **Residential:** many more residents than jobs.
- **For:** labels come from outside the bike data, so "R→J peaks in the morning" becomes a real finding rather than something built into the definition. This is the strongest choice for the report.
- **Against:** a one-time download (tens of MB for New York) plus a spatial join from stations to census blocks.

## Recommendation
- **Best option: combine 3 and 4.** Neighbourhood-sized zones, labelled residential or job centre from census jobs and residents. This fixes both problems: no circularity, and actionable areas.
- **Threshold check:** whichever setup we use, report results at a few thresholds (e.g. 0.1 / 0.2 / 0.3 for the R/J net-flow share) to show they don't depend on that choice.
- **If we keep the current setups:** use R/J as the main analysis and UES/Midtown as a robustness check. Matching results in both is worth stating in the report.

## Related issue: the "other" ride type
In iteration 3, the HMM observes 6 ride types per hour, and **"other"** (any trip touching a station that is neither R nor J) is most of the traffic. As a result, the fitted hidden states are almost pure volume levels: each state applies nearly the same multiplier (×0.14 up to ×1.67) to every ride type. In the UES/Midtown setup, "other" is nearly all city traffic, so the HMM's state mostly tracks citywide volume rather than those two neighbourhoods.

With neighbourhood zones, replace "other" with something local, such as total trips touching those neighbourhoods, so the hidden state reflects what's happening locally.
