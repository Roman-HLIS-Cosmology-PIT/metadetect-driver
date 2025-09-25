import os
from pathlib import Path

# import s3fs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyarrow.compute as pc

from pyimcom.analysis import OutImage
from metadetect_driver import MetaDetectRunner

band = "H158"
# row_ind = 0

# # Setup the filesystem
# fs = s3fs.S3FileSystem(anon=True)
# # Path to the IMCOM coadds on AWS S3
# coadd_dir = (
#     "nasa-irsa-simulations/openuniverse2024/roman/preview/RomanWAS/images/coadds"
# )
# band_dir = os.path.join(coadd_dir, band)
# subdirs = fs.ls(band_dir)
# img_dir = subdirs[row_ind]
# imgs = fs.ls(img_dir)
# image_path = "s3://" + imgs[0]

image_path = str(Path("IMCOM_data") / "images" / band / "output_00_00.fits")

outimage = OutImage(image_path)

mdet_runner = MetaDetectRunner(outimage)
catalogs, block_indices = mdet_runner.make_catalogs()

res = catalogs[0]

is_primary = pc.field("is_primary") == True
is_quality = (
    (pc.field("pgauss_s2n") > 10)
    & (pc.field("pgauss_T_ratio") > 0.5)
    & (pc.field("pgauss_flags") == 0)
)

fig, axs = plt.subplots()

img = outimage.get_coadded_layer(mdet_runner.driver_cfg.get("layer", "SCI"))

norm = mpl.colors.CenteredNorm(0)

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "grey_diverging", ["black", "white", "black"]
)
axs.imshow(np.arcsinh(img), origin="lower", cmap=cmap, norm=norm)

axs.scatter(
    res["noshear"]["sx_col"],
    res["noshear"]["sx_row"],
    c="r",
    marker="x",
    s=72,
    label="Detection",
)
axs.scatter(
    res["noshear"].filter(is_primary)["sx_col"],
    res["noshear"].filter(is_primary)["sx_row"],
    ec="b",
    fc="none",
    marker="s",
    s=72,
    label="Primary",
)
axs.scatter(
    res["noshear"].filter(is_quality)["sx_col"],
    res["noshear"].filter(is_quality)["sx_row"],
    ec="m",
    fc="none",
    marker="D",
    s=72,
    label="Quality",
)
fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax=axs)
axs.legend()
plt.show()

print(res["noshear"].schema)
