import importlib.resources
import logging
from typing import Iterable, Optional, Union

import yaml


logger = logging.getLogger(__name__)


# ---- Allowed values ----
_ALLOWED_BANDS = [
    "R062",
    "Z087",
    "Y106",
    "J129",
    "H158",
    "F184",
    "K213",
    "W146",
]


_DEFAULT_DRIVER_CONFIG = (
    importlib.resources.files(__package__).parent / "config" / "driver_default.yaml"
)


def _load_default_driver_config():
    with open(_DEFAULT_DRIVER_CONFIG, "r") as file:
        config = yaml.safe_load(file)

    return config


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
    bad = [b for b in bands if b not in _ALLOWED_BANDS]
    if bad:
        raise ValueError(
            f"Invalid entries in '{key_name}': {bad}. Allowed: {_ALLOWED_BANDS}"
        )
    return bands


# ---- Main parser ----
def parse_driver_config(driver_cfg: Optional[dict]) -> dict:
    """
    Parse/validate a driver_cfg dict:
      - Fill defaults for missing keys from _DEFAULT_DRIVER_CONFIG.
      - Coerce scalars to lists for det_bands, shear_bands (only if provided).
      - Validate band names, numeric ranges, and types.
      - keepcols is always a list; if user passes None/empty, fallback to default.
    """
    logger.info("Parsing metadetect driver config")

    cfg = _load_default_driver_config()
    if driver_cfg:
        cfg.update(driver_cfg)

    # Coerce list-like keys (only when provided)
    cfg["det_bands"] = _validate_bands(cfg.get("det_bands"), "det_bands")
    logger.debug(f"config det_bands: {cfg['det_bands']}")
    cfg["shear_bands"] = _validate_bands(cfg.get("shear_bands"), "shear_bands")
    logger.debug(f"config shear_bands: {cfg['shear_bands']}")

    # # keepcols: always ensure list, fallback to default if None/empty
    # keep = _coerce_list(cfg.get("keepcols"), str, "keepcols")
    # cfg["keepcols"] = keep if (keep and len(keep) > 0) else list(DEFAULT_DRIVER_CFG["keepcols"])
    # logger.debug(f"config keepcols: {cfg['keepcols']}")

    # ---- Validation ----
    # sizes
    if not (isinstance(cfg["psf_img_size"], int) and cfg["psf_img_size"] > 0):
        raise ValueError("'psf_img_size' must be a positive int")
    logger.debug(f"config psf_img_size: {cfg['psf_img_size']}")

    if not (isinstance(cfg["bound_size"], int) and cfg["bound_size"] >= 0):
        raise ValueError("'bound_size' must be a non-negative int")
    logger.debug(f"config bound_size: {cfg['bound_size']}")

    # seed
    if not isinstance(cfg["mdet_seed"], int):
        raise ValueError("'mdet_seed' must be an int")
    logger.debug(f"config mdet_seed: {cfg['mdet_seed']}")

    # layer
    if cfg["layer"] is not None and not isinstance(cfg["layer"], str):
        raise ValueError("'layer' must be a string")
    logger.debug(f"config layer: {cfg['layer']}")

    # # outdir
    # if cfg["outdir"] is not None and not isinstance(cfg["outdir"], str):
    #     raise ValueError("'outdir' must be a string")
    # logger.debug(f"config outdir: {cfg['outdir']}")

    # executor params
    mw = cfg.get("max_workers")
    if mw is not None:
        if not (isinstance(mw, int) and mw >= 1):
            raise ValueError("'max_workers' must be None or an int >= 1")
    logger.debug(f"config max_workers: {cfg['max_workers']}")

    ch = cfg.get("chunksize")
    if not (isinstance(ch, int) and ch >= 1):
        raise ValueError("'chunksize' must be an int >= 1")
    logger.debug(f"config chunksize: {cfg['chunksize']}")

    return cfg
