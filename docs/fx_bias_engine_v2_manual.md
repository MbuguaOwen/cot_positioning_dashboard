# FX Bias Engine v2 Manual (Non-COT-Led)

This manual is the implementation target for `cot_bias fx-bias-v2`.

## Core Principles
- No lookahead (`ts <= as_of_ts` everywhere)
- Price-led direction (W1/D1/H4/H1 regimes)
- Rates/risk/carry/skew as secondary/contextual layers
- Graceful degradation with explicit capability matrix
- COT never directional; overlay only for confidence/sizing

## Required Outputs
- Currency table: `A_Currency_Strength_Bias`
- Pair table + BOT gate: `B_Pair_Bias_Trade_Gate`
- Diagnostics: `C_Diagnostics`
- Currency polarity pairs: `D_Currency_Polarity_Pairs`
- Run audit metadata: `RunMeta`
- Data health: `CapabilityMatrix`

## Key Config Block
Use `fx_bias_engine_v2:` in `config.yaml` (see `config.example.yaml`).

## Operator Guide: Currency Polarity Pairs
- This table ranks pairs from currency bias divergence only (not pair trend/gate logic).
- Orientation is always `BASE - QUOTE`:
  - `Spread = BaseBias - QuoteBias`
  - positive spread implies pair-long bias
  - negative spread implies pair-short bias
  - a bullish quote currency pushes pair direction bearish.
- Ranking metric:
  - `Opposition = min(|BaseBias|, |QuoteBias|)`
  - `ConvictionScore = Opposition * sign(Spread)`
  - sorted by `abs(ConvictionScore)` descending.
- Defaults:
  - `spread_threshold = 20`
  - `min_confidence = 60`
  - `top_n = 20`
- Source inputs:
  - `A_Currency_Strength_Bias`
  - configured `tradable_pairs` list under `fx_bias_engine_v2`.

## Change Control
All threshold/weight changes require:
- engine version bump
- config hash update
- ADR update in `docs/adr/`
