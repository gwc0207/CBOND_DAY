from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_daily.core.config import load_config_file, parse_date
from cbond_daily.models.linear_score import run_linear_score, write_score_outputs


def _load_model_config(path: Path) -> dict:
    import json5

    with path.open("r", encoding="utf-8") as handle:
        return json5.load(handle) or {}


def main() -> None:
    paths_cfg = load_config_file("paths")
    model_cfg_path = Path("cbond_daily/config/models/linear_combo_default.json5")
    if len(sys.argv) > 1:
        model_cfg_path = Path(sys.argv[1])
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"model config not found: {model_cfg_path}")
    model_cfg = _load_model_config(model_cfg_path)

    start = parse_date(model_cfg["start"])
    end = parse_date(model_cfg["end"])
    factors = model_cfg.get("factors", [])
    if not factors:
        raise ValueError("model config missing factors")

    result = run_linear_score(
        ods_root=paths_cfg["ods_root"],
        dwd_root=paths_cfg["dwd_root"],
        dws_root=paths_cfg["dws_root"],
        start=start,
        end=end,
        factors=factors,
        label_cfg=model_cfg["label_cfg"],
        regression_cfg=model_cfg.get("regression_cfg", {}),
        weight_source=model_cfg.get("weight_source", "regression"),
        normalize=model_cfg.get("normalize", "zscore"),
    )

    output_cfg = model_cfg.get("output", {})
    score_path = Path(output_cfg.get("score_path", "results/models/scores.csv"))
    weights_path = output_cfg.get("weights_path")
    meta_path = output_cfg.get("meta_path")
    overwrite = bool(model_cfg.get("overwrite", False))
    if weights_path:
        weights_path = Path(weights_path)
    if meta_path:
        meta_path = Path(meta_path)

    meta_payload = {
        "model_id": model_cfg.get("model_id"),
        "model_type": model_cfg.get("model_type", "linear"),
        "start": start,
        "end": end,
        "label_cfg": model_cfg.get("label_cfg"),
        "factors": factors,
        "weight_source": model_cfg.get("weight_source", "regression"),
        "regression_cfg": model_cfg.get("regression_cfg"),
    }

    write_score_outputs(
        result=result,
        score_path=score_path,
        weights_path=weights_path,
        meta_path=meta_path,
        meta_payload=meta_payload,
        overwrite=overwrite,
    )
    print(f"saved: {score_path}")


if __name__ == "__main__":
    main()
