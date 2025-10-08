import os
import s3fs
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize

import ngmix
from metadetect_driver import IMCOMDriver

band = "H158"
row_ind = 0

driver = IMCOMDriver(
    "config/imcom_sci.yaml"
)

# Setup the filesystem
fs = s3fs.S3FileSystem(anon=True)
# Path to the IMCOM coadds on AWS S3
coadd_dir = (
    "nasa-irsa-simulations/openuniverse2024/roman/preview/RomanWAS/images/coadds"
)
band_dir = os.path.join(coadd_dir, band)
subdirs = fs.ls(band_dir)
img_dir = subdirs[row_ind]
imgs = fs.ls(img_dir)
image = "s3://" + imgs[0]

obs, img = driver.get_obs_imcom(filepath=image, layer=0)

obslist = ngmix.ObsList()
obslist.append(obs)
mbobs = ngmix.MultiBandObsList()
mbobs.append(obslist)

res = driver.run_metadetect(mbobs=mbobs)

print("number of detected objects: {}".format(res["noshear"].size))

buff_mask = (
    (res["noshear"]["sx_col"] > driver.config["BOUND_SIZE"])
    & (
        res["noshear"]["sx_col"]
        < driver.config["IMG_SIZE"] - driver.config["BOUND_SIZE"]
    )
    & (res["noshear"]["sx_row"] > driver.config["BOUND_SIZE"])
    & (
        res["noshear"]["sx_row"]
        < driver.config["IMG_SIZE"] - driver.config["BOUND_SIZE"]
    )
)
m_good = (
    (res["noshear"]["wmom_s2n"] > 10)
    & (res["noshear"]["wmom_T_ratio"] > 0.5)
    & (res["noshear"]["wmom_flags"] == 0)
    & buff_mask
)

plt.figure(figsize=(10, 10))
m_img = np.mean(img)
s_img = np.std(img)

interval = ZScaleInterval()
vmin, vmax = interval.get_limits(img)
norm = ImageNormalize(img, interval=interval)
plt.imshow(img, origin="lower", cmap="viridis", norm=norm)

plt.plot(res["noshear"]["sx_col"], res["noshear"]["sx_row"], "b.", label="All")
plt.plot(
    res["noshear"]["sx_col"][m_good],
    res["noshear"]["sx_row"][m_good],
    "r.",
    label="Good",
)
plt.colorbar()
plt.legend()
plt.savefig("metadetect_imcom_sci_test.png")
