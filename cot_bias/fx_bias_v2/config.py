from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "timeframes": ["W1", "D1", "H4", "H1"],
    "tradable_pairs": [
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDCAD",
        "USDCHF",
        "USDJPY",
        "EURJPY",
        "GBPJPY",
        "EURGBP",
        "AUDJPY",
        "NZDJPY",
        "CADJPY",
        "CHFJPY",
    ],
    "freshness_minutes": {
        "prices": 240,
        "rates": 1440,
        "risk": 240,
        "carry": 2880,
        "skew": 4320,
        "cot": 10080,
    },
    "price_regime": {
        "method": "ema200_atr",
        "ema_len": 200,
        "atr_len": 14,
        "slope_k": {"W1": 20, "D1": 20, "H4": 12, "H1": 12},
        "thresholds": {
            "range_strength_lt": 35,
            "slope_abs_min": 0.05,
            "pos_abs_min": 0.20,
        },
    },
    "currency_strength": {
        "horizons_days": [5, 20, 60],
        "zscore_lookback_days": 252,
        "ridge_lambda": 1.0e-4,
        "clip_z": 3.0,
        "min_pair_coverage_ratio": 0.65,
    },
    "weights": {
        "price": 0.50,
        "rates": 0.30,
        "risk": 0.10,
        "carry": 0.10,
        "skew_if_available": {
            "carry": 0.05,
            "skew": 0.05,
        },
    },
    "risk_overlay": {
        "enabled": True,
        "jpy_chf_overlay_abs": 0.60,
    },
    "gate": {
        "t_bias": 15,
        "opposing_strength_block": 60,
    },
    "cot_overlay": {
        "enabled": True,
        "confidence_penalty_if_extreme_and_persistent": 10,
    },
    "currency_polarity_pairs": {
        "spread_threshold": 20.0,
        "min_confidence": 60,
        "top_n": 20,
    },
}


@dataclass(frozen=True)
class FxBiasV2Config:
    raw: Dict[str, Any]

    @property
    def enabled(self) -> bool:
        return bool(self.raw.get("enabled", True))

    @property
    def timeframes(self) -> list[str]:
        vals = self.raw.get("timeframes", ["W1", "D1", "H4", "H1"])
        out = [str(v).strip().upper() for v in vals if str(v).strip()]
        return out or ["W1", "D1", "H4", "H1"]

    @property
    def tradable_pairs(self) -> list[str]:
        vals = self.raw.get("tradable_pairs", DEFAULTS["tradable_pairs"])
        out: list[str] = []
        seen: set[str] = set()
        for v in vals if isinstance(vals, list) else []:
            s = "".join(ch for ch in str(v).upper() if ch.isalpha())
            if len(s) != 6 or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out or list(DEFAULTS["tradable_pairs"])

    @property
    def freshness_minutes(self) -> Dict[str, int]:
        block = self.raw.get("freshness_minutes", {})
        return {
            "prices": int(block.get("prices", 240)),
            "rates": int(block.get("rates", 1440)),
            "risk": int(block.get("risk", 240)),
            "carry": int(block.get("carry", 2880)),
            "skew": int(block.get("skew", 4320)),
            "cot": int(block.get("cot", 10080)),
        }

    @property
    def price_regime(self) -> Dict[str, Any]:
        block = self.raw.get("price_regime", {})
        thresholds = block.get("thresholds", {})
        slope_k = block.get("slope_k", {})
        return {
            "method": str(block.get("method", "ema200_atr")).strip().lower(),
            "ema_len": int(block.get("ema_len", 200)),
            "atr_len": int(block.get("atr_len", 14)),
            "slope_k": {
                "W1": int(slope_k.get("W1", 20)),
                "D1": int(slope_k.get("D1", 20)),
                "H4": int(slope_k.get("H4", 12)),
                "H1": int(slope_k.get("H1", 12)),
            },
            "thresholds": {
                "range_strength_lt": float(thresholds.get("range_strength_lt", 35)),
                "slope_abs_min": float(thresholds.get("slope_abs_min", 0.05)),
                "pos_abs_min": float(thresholds.get("pos_abs_min", 0.20)),
            },
        }

    @property
    def currency_strength(self) -> Dict[str, Any]:
        block = self.raw.get("currency_strength", {})
        horizons = block.get("horizons_days", [5, 20, 60])
        return {
            "horizons_days": [int(v) for v in horizons],
            "zscore_lookback_days": int(block.get("zscore_lookback_days", 252)),
            "ridge_lambda": float(block.get("ridge_lambda", 1.0e-4)),
            "clip_z": float(block.get("clip_z", 3.0)),
            "min_pair_coverage_ratio": float(block.get("min_pair_coverage_ratio", 0.65)),
        }

    @property
    def weights(self) -> Dict[str, Any]:
        block = self.raw.get("weights", {})
        skew_block = block.get("skew_if_available", {})
        return {
            "price": float(block.get("price", 0.50)),
            "rates": float(block.get("rates", 0.30)),
            "risk": float(block.get("risk", 0.10)),
            "carry": float(block.get("carry", 0.10)),
            "skew_if_available": {
                "carry": float(skew_block.get("carry", 0.05)),
                "skew": float(skew_block.get("skew", 0.05)),
            },
        }

    @property
    def risk_overlay(self) -> Dict[str, Any]:
        block = self.raw.get("risk_overlay", {})
        return {
            "enabled": bool(block.get("enabled", True)),
            "jpy_chf_overlay_abs": float(block.get("jpy_chf_overlay_abs", 0.60)),
        }

    @property
    def gate(self) -> Dict[str, Any]:
        block = self.raw.get("gate", {})
        return {
            "t_bias": float(block.get("t_bias", 15)),
            "opposing_strength_block": float(block.get("opposing_strength_block", 60)),
        }

    @property
    def cot_overlay(self) -> Dict[str, Any]:
        block = self.raw.get("cot_overlay", {})
        return {
            "enabled": bool(block.get("enabled", True)),
            "confidence_penalty_if_extreme_and_persistent": int(
                block.get("confidence_penalty_if_extreme_and_persistent", 10)
            ),
        }

    @property
    def currency_polarity_pairs(self) -> Dict[str, Any]:
        block = self.raw.get("currency_polarity_pairs", {})
        return {
            "spread_threshold": float(block.get("spread_threshold", 20.0)),
            "min_confidence": int(block.get("min_confidence", 60)),
            "top_n": int(block.get("top_n", 20)),
        }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_fx_bias_v2_config(config_path: Optional[str]) -> FxBiasV2Config:
    doc: Dict[str, Any] = {}
    if yaml is not None:
        path = Path(config_path) if config_path else Path("config.yaml")
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                doc = loaded
    block = doc.get("fx_bias_engine_v2") if isinstance(doc, dict) else None
    if not isinstance(block, dict):
        block = {}
    merged = _deep_merge(DEFAULTS, block)
    return FxBiasV2Config(raw=merged)


def config_hash(cfg: FxBiasV2Config) -> str:
    payload = repr(cfg.raw).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
