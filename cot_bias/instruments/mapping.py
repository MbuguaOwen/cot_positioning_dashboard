from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..utils import Config


def load_instruments(cfg: Config) -> List[Dict[str, str]]:
    path = Path(__file__).parent / "instruments.yaml"
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load instruments.yaml. Install PyYAML.") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("instruments.yaml must be a list of instrument mappings.")

    out: List[Dict[str, str]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        for key in ["symbol", "human_name", "report_type", "market_match", "driver_group"]:
            if key not in row:
                raise ValueError(f"instruments.yaml missing required field '{key}'")
        # market_code is optional but preferred
        out.append({k: str(row[k]) for k in row if row[k] is not None})
    return out
