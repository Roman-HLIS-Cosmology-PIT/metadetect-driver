import pytest

import metadetect_driver


def test_validate():
    """
    Test that the driver config validation raises errors for invalid configs.
    """
    with pytest.raises(Exception):  # noqa
        _config = {
            "psf_img_size": None,
        }
        metadetect_driver.config.parse_driver_config(_config)

    with pytest.raises(Exception):  # noqa
        _config = {
            "bound_size": None,
        }
        metadetect_driver.config.parse_driver_config(_config)

    with pytest.raises(Exception):  # noqa
        _config = {
            "mdet_seed": None,
        }
        metadetect_driver.config.parse_driver_config(_config)

    with pytest.raises(Exception):  # noqa
        _config = {
            "layer": False,
        }
        metadetect_driver.config.parse_driver_config(_config)
