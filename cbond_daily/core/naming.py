from __future__ import annotations


def build_factor_col(name: str, params: dict | None) -> str:
    safe_name = _safe_part(name)
    if not params:
        return f"{safe_name}_"
    parts = [_safe_part(str(params[key])) for key in sorted(params)]
    return f"{safe_name}_" + "_".join(parts) + "_"


def _safe_part(value: str) -> str:
    return value.replace(" ", "").replace("/", "_").replace("\\", "_").replace(":", "_")
