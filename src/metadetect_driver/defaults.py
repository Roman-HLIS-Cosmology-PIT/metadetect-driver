DRIVER_DEFAULTS = {
    "psf_img_size": 151,
    "bound_size": 100,
    "mdet_seed": 42,
    "det_bands": None,
    "shear_bands": None,
    "layer": "SCI",
}

_METACAL_DEFAULTS = {
    "psf": "fitgauss",
    "types": ["noshear", "1p", "1m", "2p", "2m"],
}

_MEDS_DEFAULTS = {
    "min_box_size": 32,
    "max_box_size": 600,
    "box_type": "iso_radius",
    "rad_min": 4,
    "rad_fac": 2,
    "box_padding": 2,
}

_SX_DEFAULTS = {
    "detect_thresh": 5,
    "deblend_cont": 1.0e-05,
    "minarea": 4,
    "filter_type": "conv",
    "filter_kernel": [
        [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],
        [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],
        [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],
        [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],
        [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],
        [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],
        [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],
    ],
}

METADETECT_DEFAULTS = {
    "model": "wmom",
    "weight": {"fwhm": 1.2},
    "metacal": _METACAL_DEFAULTS,
    "meds": _MEDS_DEFAULTS,
    "sx": _SX_DEFAULTS,
    "bmask_flags": 1073741824,
    "nodet_flags": 1,
}
