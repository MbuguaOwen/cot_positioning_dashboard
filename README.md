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

### 2) Fetch latest COT data

Default is **combined (Futures + Options)** for maximum completeness.

```bash
python -m cot_bias fetch --combined
```

### 3) Compute dashboard outputs

```bash
python -m cot_bias compute --out outputs
```

### 4) Generate an HTML dashboard

```bash
python -m cot_bias dashboard --out outputs
```

You will get:
- `outputs/instruments_latest.csv`
- `outputs/pairs_latest.csv`
- `outputs/dashboard.html`

### 5) Generate an as-of dashboard (backfill)

```bash
python -m cot_bias dashboard --out outputs --as-at 2026-01-16 --release-time "3:30 p.m. ET"
```

This resolves the release date to the most recent Tuesday report date.

---

## FX Report (any date, no lookahead)

Generate a point-in-time FX report from the official CFTC yearly compressed zips:

```bash
python -m cot_system report --date 2026-01-16 --pairs EURUSD,AUDUSD,USDJPY,GBPJPY --report-type tff --usd-mode basket --out json
```

Notes:
- The requested date is resolved to the latest available Tuesday `report_date` at or before it.
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
python -m cot_bias fetch --combined
python -m cot_bias compute --out outputs
python -m cot_bias dashboard --out outputs --as-at 2026-01-27 --release-time "3:30 p.m. ET"
