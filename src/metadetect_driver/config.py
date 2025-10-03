import logging
from copy import deepcopy
from typing import Iterable, Optional, Union

from .defaults import DRIVER_DEFAULTS

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


def _parse_driver_config(config: Optional[dict]) -> dict:
    """
    Parse/validate a config dict:
      - Fill defaults for missing keys from _DEFAULT_DRIVER_CONFIG.
      - Coerce scalars to lists for det_bands, shear_bands (only if provided).
      - Validate band names, numeric ranges, and types.
      - keepcols is always a list; if user passes None/empty, fallback to default.
    """
    logger.info("Parsing metadetect driver config")

    _config = (
        deepcopy(config)
        if config is not None
        else deepcopy(DRIVER_DEFAULTS)
    )

    # Coerce list-like keys (only when provided)
    _config["det_bands"] = _validate_bands(_config.get("det_bands"), "det_bands")
    logger.debug(f"config det_bands: {_config['det_bands']}")
    _config["shear_bands"] = _validate_bands(_config.get("shear_bands"), "shear_bands")
    logger.debug(f"config shear_bands: {_config['shear_bands']}")

    # sizes
    if not (isinstance(_config["psf_image_size"], int) and _config["psf_image_size"] > 0):
        raise ValueError("'psf_image_size' must be a positive int")
    logger.debug(f"config psf_image_size: {_config['psf_image_size']}")

    if not (isinstance(_config["bound_size"], int) and _config["bound_size"] >= 0):
        raise ValueError("'bound_size' must be a non-negative int")
    logger.debug(f"config bound_size: {_config['bound_size']}")

    # seed
    if not isinstance(_config["mdet_seed"], int):
        raise ValueError("'mdet_seed' must be an int")
    logger.debug(f"config mdet_seed: {_config['mdet_seed']}")

    # layer
    if _config["layer"] is not None and not isinstance(_config["layer"], str):
        raise ValueError("'layer' must be a string")
    logger.debug(f"config layer: {_config['layer']}")

    return _config
