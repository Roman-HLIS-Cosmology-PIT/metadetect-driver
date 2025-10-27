from collections.abc import Iterable
from copy import deepcopy
from typing import Optional, Union

import yaml

# ---- Allowed values ----
ALLOWED_BANDS = ["R062", "Z087", "Y106", "J129", "H158", "F184", "K213", "W146"]

with open("../config/driver_default.yaml", "r") as file:
    DEFAULT_DRIVER_CFG = yaml.safe_load(file)


# ---- Helpers ----
def _coerce_list(val, elem_type, key_name):
    """
    If val is None -> None.
    If val is elem_type -> [val].
    If val is iterable (not str/bytes) -> list(val) with element type checks.
    Else -> TypeError.
    """
    if val is None:
        return None
    if isinstance(val, elem_type):
        out = [val]
    elif isinstance(val, (list, tuple)):
        out = list(val)
    elif isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        out = list(val)
    else:
        raise TypeError(
            f"'{key_name}' must be {elem_type.__name__} or an iterable of {elem_type.__name__}"
        )
    # element-wise checks
    for x in out:
        if isinstance(x, bool) and elem_type is int:
            raise TypeError(
                f"'{key_name}' elements must be {elem_type.__name__}, got bool"
            )
        if not isinstance(x, elem_type):
            raise TypeError(f"'{key_name}' elements must be {elem_type.__name__}")
    return out


def _validate_bands(
    val: Optional[Union[str, Iterable[str]]], key_name: str
) -> Optional[list[str]]:
    """
    Coerce to list[str] (if provided) and validate against ALLOWED_BANDS.
    """
    if val is None:
        return None
    bands = _coerce_list(val, str, key_name)
    bad = [b for b in bands if b not in ALLOWED_BANDS]
    if bad:
        raise ValueError(
            f"Invalid entries in '{key_name}': {bad}. Allowed: {ALLOWED_BANDS}"
        )
    return bands


# ---- Main parser ----
def parse_driver_cfg(driver_cfg: Optional[dict]) -> dict:
    """
    Parse/validate an driver_cfg dict:
      - Fill defaults for missing keys from DEFAULT_DRIVER_CFG.
      - Coerce scalars to lists for det_bands, shear_bands (only if provided).
      - Validate band names, numeric ranges, and types.
      - keepcols is always a list; if user passes None/empty, fallback to
        default.
    Returns a NEW dict.
    """
    cfg = deepcopy(DEFAULT_DRIVER_CFG)
    if driver_cfg:
        cfg.update(driver_cfg)

    # Coerce list-like keys (only when provided)
    cfg["det_bands"] = _validate_bands(cfg.get("det_bands"), "det_bands")
    cfg["shear_bands"] = _validate_bands(cfg.get("shear_bands"), "shear_bands")

    # keepcols: always ensure list, fallback to default if None/empty
    keep = _coerce_list(cfg.get("keepcols"), str, "keepcols")
    cfg["keepcols"] = (
        keep if (keep and len(keep) > 0) else list(DEFAULT_DRIVER_CFG["keepcols"])
    )

    # ---- Validation ----
    # sizes
    if not (isinstance(cfg["psf_img_size"], int) and cfg["psf_img_size"] > 0):
        raise ValueError("'psf_img_size' must be a positive int")
    if not (isinstance(cfg["bound_size"], int) and cfg["bound_size"] >= 0):
        raise ValueError("'bound_size' must be a non-negative int")

    # seed
    if not isinstance(cfg["mdet_seed"], int):
        raise ValueError("'mdet_seed' must be an int")

    # layer
    if cfg["layer"] is not None and not isinstance(cfg["layer"], str):
        raise ValueError("'layer' must be a string")
    # outdir
    if cfg["outdir"] is not None and not isinstance(cfg["outdir"], str):
        raise ValueError("'outdir' must be a string")

    # executor params
    mw = cfg.get("max_workers")
    if mw is not None:
        if not (isinstance(mw, int) and mw >= 1):
            raise ValueError("'max_workers' must be None or an int >= 1")

    ch = cfg.get("chunksize")
    if not (isinstance(ch, int) and ch >= 1):
        raise ValueError("'chunksize' must be an int >= 1")

    return cfg
