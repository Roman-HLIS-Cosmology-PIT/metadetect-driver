import metadetect_driver
import pytest


def test_validate():
    with pytest.raises(Exception):
        _config = {
            "psf_img_size": None,
        }
        metadetect_driver.config.parse_driver_config(_config)

    with pytest.raises(Exception):
        _config = {
            "bound_size": None,
        }
        metadetect_driver.config.parse_driver_config(_config)

    with pytest.raises(Exception):
        _config = {
            "mdet_seed": None,
        }
        metadetect_driver.config.parse_driver_config(_config)

    with pytest.raises(Exception):
        _config = {
            "layer": False,
        }
        metadetect_driver.config.parse_driver_config(_config)
