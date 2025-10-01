import pytest
import metadetect_driver


def test_default():
    _default_driver_config = metadetect_driver.config._load_default_driver_config()
    parsed_driver_cfg = metadetect_driver.config.parse_driver_config(None)

    assert parsed_driver_cfg == _default_driver_config


def test_validate():
    _config = {
        "psf_img_size": None,
    }
    with pytest.raises(Exception) as e_info:
        parsed_driver_cfg = metadetect_driver.config.parse_driver_config(_config)

    _config = {
        "bound_size": None,
    }
    with pytest.raises(Exception) as e_info:
        parsed_driver_cfg = metadetect_driver.config.parse_driver_config(_config)

    _config = {
        "mdet_seed": None,
    }
    with pytest.raises(Exception) as e_info:
        parsed_driver_cfg = metadetect_driver.config.parse_driver_config(_config)

    _config = {
        "layer": False,
    }
    with pytest.raises(Exception) as e_info:
        parsed_driver_cfg = metadetect_driver.config.parse_driver_config(_config)
