# COT Positioning Dashboard (FX + Metals)

A small, robust Python project that pulls **weekly** CFTC Commitments of Traders (COT) “comma delimited” datasets, stores them locally (SQLite + Parquet), and computes a **direction bias + reversal risk** dashboard for:

- **FX**: EUR, JPY, GBP, AUD, CHF (from *Traders in Financial Futures* / TFF)
- **Metals**: Gold, Silver (from *Disaggregated* COT)

It is designed to be:
- **Robust**: retries + caching + no fragile web scraping
- **Simple**: one command to fetch, one to compute, one to render dashboard
- **Transparent**: it outputs the raw rows used + the derived metrics

---

## What it computes

For each instrument:
- Net positions (by participant group)
- Net % Open Interest (Net / OI)
- Rolling **3-year percentile** and **3-year z-score** of Net %OI
- **4-week delta** (change over the last 4 reports)
- A simple **Bias** + **Reversal Risk** label

For FX pairs (EURUSD, USDJPY, GBPJPY, AUDUSD, USDCHF):
- A **pair score** built from currency positioning (base – quote where possible; USD-base pairs use the quote currency as proxy)
- Bias + reversal risk label
- **FX Pairs (passed COT filters)**: long-only gate where base is BULLISH with strength MED/HIGH/EXTREME, quote is BEARISH with strength MED/HIGH/EXTREME, and the pair is BULLISH with strength MED/HIGH/EXTREME. Reversal risk is displayed (not filtered) along with a 2-report consistency flag.

---

## Quickstart

### 1) Create a venv and install deps

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Fetch and build the local store (official CFTC files)

```bash
python -m cot_bias fetch --update
```

Use `--force` to refresh the current-year zip and rebuild the store.

If the store is stale for a requested `--as-of`, the dashboard command will
auto-refresh the current year; if refresh fails (offline), the dashboard still
renders with a warning: `AUTO-REFRESH FAILED` + `DATA STALE`.

### 3) Generate an as-of HTML dashboard

```bash
python -m cot_bias dashboard --out outputs\\2026-01-27 --as-of 2026-01-27
```

You will get:
- `outputs\\YYYY-MM-DD\\instruments_latest.csv`
- `outputs\\YYYY-MM-DD\\instruments_history.csv` (truncated to <= as_of)
- `outputs\\YYYY-MM-DD\\dashboard.html`
- `outputs\\YYYY-MM-DD\\run_manifest.json`

### 4) FX Report (any date, no lookahead)

Generate a point-in-time FX report from the official CFTC yearly compressed zips:

```bash
python -m cot_system report --date 2026-01-16 --pairs EURUSD,AUDUSD,USDJPY,GBPJPY --report-type tff --usd-mode basket --out json
```

---

Notes:
- The requested date is resolved to the latest available report_date at or before it (no lookahead).
- USD is computed as a basket proxy by default; `--usd-mode direct` uses the USD index contract when present.
- Output includes `requested_date`, `resolved_report_date`, and the Tuesday/Friday convention note.

---

## FX Bias Engine v2 (Non-COT-Led)

This repo now includes an optional v2 engine where directional bias is driven by:
- price regime (primary)
- rates momentum (secondary)
- risk regime overlay (JPY/CHF)
- carry (structural)
- optional skew confirmation

COT is overlay-only (`confidence/sizing`), never directional.

### Run v2

```bash
python -m cot_bias --config config.yaml fx-bias-v2 \
  --as-of 2026-03-02T21:00:00Z \
  --prices data/fx/prices \
  --rates data/fx/rates_2y.csv \
  --risk data/fx/risk_proxies.csv \
  --carry data/fx/forward_points.csv \
  --skew data/fx/rr25.csv \
  --cot data/fx/cot_flags.csv \
  --out outputs/fx_bias_v2
```

Outputs:
- `outputs/fx_bias_v2/fx_bias_v2_run.json`
- `outputs/fx_bias_v2/fx_bias_v2_dashboard.html`
- `outputs/fx_bias_v2/currency_strength_bias.csv`
- `outputs/fx_bias_v2/pair_bias_trade_gate.csv`
- `outputs/fx_bias_v2/capability_matrix.csv`
- `outputs/fx_bias_v2/currency_polarity_pairs.csv`
- `outputs/fx_bias_v2/diagnostics.json`

Required top-level keys in JSON:
- `RunMeta`
- `CapabilityMatrix`
- `A_Currency_Strength_Bias`
- `B_Pair_Bias_Trade_Gate`
- `C_Diagnostics`
- `D_Currency_Polarity_Pairs`

Currency polarity orientation (operator note):
- Pair direction is **BASE minus QUOTE** (`Spread = BaseBias - QuoteBias`).
- A bullish quote currency makes the pair bearish (negative spread).

Price input convention:
- CSV files in a directory, e.g. `EURUSD_D1.csv`, `EURUSD_H4.csv`, `USDJPY_W1.csv`
- Columns: `ts/open/high/low/close/volume` (or equivalent aliases)
- Loader is recursive (`--prices data\fx\prices` scans nested folders).
- If `D1` is missing but `H1`/`H4` exists, v2 auto-builds `D1` bars (and `W1` fallback) without lookahead.

Optional dataset conventions:
- `rates`: `currency,ts,y2_value`
- `risk`: `asset,ts,value`
- `carry`: `pair,ts,forward_points`
- `skew`: `symbol_or_ccy,ts,rr25[,quality_flags]`
- `cot`: `currency,ts,extreme_flag,persistence_flag`

If a component is missing/stale, v2 sets it to `null`, logs it in `CapabilityMatrix`, and reduces confidence.

Design docs:
- `docs/fx_bias_engine_v2_manual.md`
- `docs/adr/0001-fx-bias-engine-v2-non-cot-led.md`

---

## Configuration

You can optionally create a `config.yaml` (copy from `config.example.yaml`) to:
- change storage locations
- override URLs
- tweak contract matching regex
- choose which participant group drives “bias” per asset

If you don’t provide a config, sane defaults are used.

For a production-oriented multi-layer COT permission model (direction + spread +
flow + crowdedness + macro + optional news blackout), see:
- `docs/cot_filter_design.md`
- `config.example.yaml` (`cot_filter:` block)
- `cot_bias/filters/cot_filter.py`

---

## Notes on interpretation (practical)

- COT is weekly and lagged; use it as a **regime/bias filter**, not a trigger.
- **Bias** is strongest when:
  - Net %OI z-score is meaningfully away from 0 (trend participation)
  - Net %OI percentile is not at an extreme (reversal risk low/moderate)
- **Reversal risk** is highest at positioning extremes + when 4-week delta starts flipping.

---

## Scheduling

Windows Task Scheduler / cron can run weekly:

```bash
# Windows
scripts\run_weekly.bat

# macOS/Linux
bash scripts/run_weekly.sh
```

Both scripts now:
- refresh data,
- build the latest dashboard in `outputs/`,
- and regenerate dated dashboard folders for the last 4 months in `outputs/YYYY-MM-DD/`.

To generate a full calendar year manually (example: all 2025 report dates):

```bash
python scripts/generate_recent_dashboards.py --year 2025 --out outputs
```

---

## Data Sources

This project uses the official CFTC “newcot” comma-delimited datasets:

- Financial Futures: `FinFutWk.txt` (futures only) / `FinComWk.txt` (futures+options)
- Disaggregated: `f_disagg.txt` (futures only) / `c_disagg.txt` (futures+options)

(You can override in config.)

---

## License

MIT (do whatever; keep the notice).


Example //
ppython -m cot_bias fetch --update
python -m cot_bias compute --out outputs
python -m cot_bias dashboard --out outputs --as-of 2026-01-27
