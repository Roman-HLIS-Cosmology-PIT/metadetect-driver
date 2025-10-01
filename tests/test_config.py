import metadetect_driver
import pytest


def test_validate():
    _config = {
        "psf_img_size": None,
    }
    with pytest.raises(Exception):
        metadetect_driver.config.parse_driver_config(_config)

    _config = {
        "bound_size": None,
    }
    with pytest.raises(Exception):
        metadetect_driver.config.parse_driver_config(_config)

    _config = {
        "mdet_seed": None,
    }
    with pytest.raises(Exception):
        metadetect_driver.config.parse_driver_config(_config)

    _config = {
        "layer": False,
    }
    with pytest.raises(Exception):
        metadetect_driver.config.parse_driver_config(_config)
