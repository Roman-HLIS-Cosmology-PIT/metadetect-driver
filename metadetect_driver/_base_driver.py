import yaml
import numpy as np
import metadetect
from copy import deepcopy

__all__ = ["BaseDriver"]


class BaseDriver(object):
    """Driver class to run metadetecion"""

    def __init__(self, config_file):
        """The initializer for metadetect driver

        Args:
            config_file (str): path to the configuration file
        """
        self._parse_config(config_file)

    def _parse_config(self, config_file):
        """Parser for the YAML configuration file

        Args:
            config_file (str): path to the configuration file
        """
        with open(config_file, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.config["METADETECT_CONFIG"]["bmask_flags"] = eval(
            self.config["METADETECT_CONFIG"]["bmask_flags"]
        )
        self.config["METADETECT_CONFIG"]["nodet_flags"] = eval(
            self.config["METADETECT_CONFIG"]["nodet_flags"]
        )

    def run_metadetect(self, mbobs, seed=42):
        """Engine function for running Metadetection.

        This function serves as the main entry point to execute the Metadetection
        pipeline.

        Args:
            mbobs (Object): ngmix.MultiBandObsList
            seed (int, optional): random seed. Defaults to 42.

        Returns:
            dict: the result dict
        """
        res = metadetect.do_metadetect(
            deepcopy(self.config["METADETECT_CONFIG"]),
            mbobs=mbobs,
            rng=np.random.RandomState(seed),
        )
        return res
