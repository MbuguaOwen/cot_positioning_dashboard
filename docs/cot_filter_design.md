# COTFilter: Multi-Layer COT + Macro Permission Model

## Goal

- Pine is still the execution trigger.
- COT filter acts as permission/regime.
- Reduce low-quality trades (late trend, crowded reversals, macro headwinds, event spikes).
- Expose configurable strictness tiers: `loose`, `balanced`, `strict`, `sniper`.

## Inputs

- Pine signal: `pair`, `signal_direction`, `signal_ts`.
- Weekly COT panel by currency with:
  - `currency`, `report_date`, `release_dt`
  - `net_pctile`, `net_z`, `net_pct_oi` (one or more metrics)
- Optional macro override score.
- Optional event blackout windows (`EventWindow` list).

## Outputs

`COTFilterDecision`:

```json
{
  "allow": false,
  "direction": "none",
  "score": 41.7,
  "strictness_tier": "balanced",
  "reasons": [
    "direction_ok",
    "spread_ok",
    "flow_fail",
    "macro_block"
  ],
  "components": {
    "spread": 14.2,
    "dspread": -1.1,
    "crowded_base": false,
    "crowded_quote": false,
    "macro_score": 0.38
  },
  "risk_multiplier": 0.0
}
```

## Hard gates and soft score

Hard gates (must pass):

1. Directional bias
2. Spread threshold
3. Flow confirmation (tier-dependent)
4. Crowdedness policy (block / override / reduce risk)
5. Macro alignment (for USD pairs; tier-dependent requirement)
6. News blackout (optional, tier-dependent requirement)

Soft score (`0..100`):

- spread strength contribution
- flow alignment contribution
- crowdedness penalty
- macro alignment contribution
- optional news penalty

Score can be advisory or enforced via `enforce_min_score_gate`.

## Strictness ladder

- **Loose**
  - Direction + minimal spread.
  - Higher frequency, weakest drawdown controls.
- **Balanced**
  - Spread + flow confirmation.
  - Crowdedness override allowed only with strong flow.
  - Good default.
- **Strict**
  - Strong spread + flow + crowdedness block.
  - Macro gate required for USD pairs.
  - Lower trade count, typically lower DD.
- **Sniper**
  - Extreme spread + strong flow + acceleration + macro required.
  - Optional mandatory blackout.
  - Lowest frequency, highest selectivity.

## No-lookahead alignment

- Use only the latest `report_date` whose `release_dt <= signal_ts`.
- COT is frozen between releases (no mid-week value changes).
- Prevents report-date leakage in intraday entry logic.

## Pseudocode

```text
resolve effective report by release_dt <= signal timestamp
load base/quote metric values and compute spread

direction gate:
  long: base stronger than quote
  short: quote stronger than base

spread gate:
  long spread >= +X
  short spread <= -X

flow gate (optional by tier):
  dSpread = spread(t) - spread(t-k)
  long requires dSpread > +dMin
  short requires dSpread < -dMin
  optional acceleration: dSpread_now better than dSpread_prev

crowdedness:
  detect base/quote extremes
  apply policy: block OR allow if strong flow OR reduce risk

macro gate:
  if pair has USD, require alignment with USD regime (unless reversal mode override)

news blackout:
  if enabled and event window hit, block new entries when required by tier

combine hard gates -> allow / block
compute soft score 0..100
convert score to risk multiplier
emit decision object with transparent reasons
```

## Backtest / edge verification plan

Run ablation in order:

1. Pine only
2. Pine + directional COT
3. + spread gate
4. + flow gate
5. + crowdedness
6. + macro
7. + blackout (if calendar available)

Track:

- expectancy
- profit factor
- win rate
- average R
- max drawdown
- trade frequency
- time-in-market

Walk-forward:

- Calibrate thresholds on training slice (e.g., 3 years)
- Validate out-of-sample on later slice (e.g., 1 year)
- Roll forward and aggregate OOS metrics and variance

