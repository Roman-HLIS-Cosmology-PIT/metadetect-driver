import yaml
import numpy as np
import metadetect
from copy import deepcopy


class BaseDriver(object):
    def __init__(self, config_file):
        self._parse_config(config_file)

    def _parse_config(self, config_file):
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
        print(self.config)
        # print(self.config["METADETECT_CONFIG"]["sx"]["filter_kernel"])

    def run_metadetect(self, mbobs, seed=42):
        res = metadetect.do_metadetect(
            # deepcopy(METADETECT_CONFIG),
            deepcopy(self.config["METADETECT_CONFIG"]),
            mbobs=mbobs,
            rng=np.random.RandomState(seed),
        )
        return res


if __name__ == "__main__":
    # driver = BaseDriver()
    # driver.parse_config(
    #     "/hpc/home/yf194/Work/projects/metadetect-driver/config/config_imcom_sci.yaml"
    # )
    driver = BaseDriver(
        config_file="/hpc/home/yf194/Work/projects/metadetect-driver/config/config_imcom_sci.yaml"
    )
