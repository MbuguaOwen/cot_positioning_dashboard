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

## Configuration

You can optionally create a `config.yaml` (copy from `config.example.yaml`) to:
- change storage locations
- override URLs
- tweak contract matching regex
- choose which participant group drives “bias” per asset

If you don’t provide a config, sane defaults are used.

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
