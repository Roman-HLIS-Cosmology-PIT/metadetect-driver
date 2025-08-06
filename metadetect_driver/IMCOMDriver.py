import ngmix
import galsim
import sep
import numpy as np
from astropy.io import fits

from BaseDriver import BaseDriver

import os
import s3fs
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize


class IMCOMDriver(BaseDriver):
    def __init__(self, config_file):
        super().__init__(config_file)

    def _read_imcom(self, filepath, layer=0):
        with fits.open(filepath, fsspec_kwargs={"anon": True}) as hdul:
            f = hdul[0].section[0, layer]
            h = hdul[0].header
        h.pop("NAXIS3")
        h.pop("NAXIS4")
        h["NAXIS"] = 2

        # [TODO] needs to find a way to figure out band from imcom product
        band = "H158"
        return f, h, band

    def get_obs_imcom(self, filepath, layer=0, psf_img=None, rng=None):
        f, h, band = self._read_imcom(filepath=filepath, layer=layer)

        w = galsim.AstropyWCS(header=h)
        img_jacobian = w.jacobian(image_pos=galsim.PositionD(h["CRPIX1"], h["CRPIX2"]))

        # Make noise
        # rng = np.random.RandomState(42)
        # noise_sigma = 1e-4
        # noise = rng.normal(size=(IMG_SIZE, IMG_SIZE)) * noise_sigma

        # Make PSF
        if not psf_img:
            psf = galsim.Gaussian(fwhm=self.config["PSF_FWHM"][band])
            psf_img = psf.drawImage(
                nx=self.config["PSF_IMG_SIZE"],
                ny=self.config["PSF_IMG_SIZE"],
                wcs=w,
            ).array

        psf_cen = (self.config["PSF_IMG_SIZE"] - 1) / 2
        img_cen = (np.array([self.config["IMG_SIZE"], self.config["IMG_SIZE"]]) - 1) / 2

        psf_jac = ngmix.Jacobian(
            row=psf_cen,
            col=psf_cen,
            wcs=img_jacobian,
        )
        img_jac = ngmix.Jacobian(
            row=img_cen[0],
            col=img_cen[1],
            wcs=img_jacobian,
        )

        psf_obs = ngmix.Observation(
            image=psf_img,
            jacobian=psf_jac,
        )

        bkg = sep.Background(f.astype(f.dtype.newbyteorder("=")))
        print(bkg.globalback)
        print(bkg.globalrms)
        # plt.figure()
        # plt.imshow(bkg_image, interpolation="nearest", cmap="gray", origin="lower")
        # plt.colorbar()

        obs = ngmix.Observation(
            # image= f - bkg + noise,
            image=f - bkg,
            jacobian=img_jac,
            # weight=np.ones((IMG_SIZE, IMG_SIZE), dtype=float) / noise_sigma**2,
            weight=np.ones(
                (self.config["IMG_SIZE"], self.config["IMG_SIZE"]), dtype=float
            ),
            psf=psf_obs,
            ormask=np.zeros(
                (self.config["IMG_SIZE"], self.config["IMG_SIZE"]), dtype=np.int32
            ),
            bmask=np.zeros(
                (self.config["IMG_SIZE"], self.config["IMG_SIZE"]), dtype=np.int32
            ),
        )

        return obs, f


if __name__ == "__main__":
    # driver = BaseDriver()
    # driver.parse_config(
    #     "/hpc/home/yf194/Work/projects/metadetect-driver/config/config_imcom_sci.yaml"
    # )
    driver = IMCOMDriver(
        config_file="/hpc/home/yf194/Work/projects/metadetect-driver/config/config_imcom_sci.yaml"
    )

    band = "H158"
    row_ind = 0

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

    print(res["noshear"].size)

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
    # plt.imshow(img, origin='lower', cmap='gray', vmin=max(np.min(img), m_img - 5 * s_img), vmax=m_img + 5 * s_img)

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
