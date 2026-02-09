# ADR 0001: FX Bias Engine v2 Non-COT-Led

## Status
Accepted

## Date
2026-02-09

## Context
Legacy engine direction is COT-centric and weekly. Execution now requires daily/intraday directional permission with no directional dependency on COT.

## Decision
Adopt FX Bias Engine v2:
- Directional spine: multi-timeframe price regime (W1/D1/H4/H1)
- Secondary: rates momentum
- Overlay/context: risk regime (JPY/CHF), carry, optional skew
- COT usage restricted to confidence/sizing overlays only

The v2 output contract requires:
- `RunMeta`
- `CapabilityMatrix`
- `A_Currency_Strength_Bias`
- `B_Pair_Bias_Trade_Gate`
- `C_Diagnostics`
- `D_Currency_Polarity_Pairs`

## Consequences
- Runs degrade gracefully when optional components are missing/stale.
- No-lookahead auditing improves through `as_of_ts`, component max timestamps, and config hashing.
- BOT integration consumes `GatingRuleForBOT` at H4 cadence.
