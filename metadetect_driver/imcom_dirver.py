import ngmix
import galsim
import sep
import numpy as np
from astropy.io import fits

from .base_driver import BaseDriver


class IMCOMDriver(BaseDriver):
    """Driver class to run metadetect on IMCOM product

    Args:
        BaseDriver (class): Derived from BaseDriver class
    """

    def __init__(self, config_file):
        """The initializer for metadetect driver on IMCOM layers

        Args:
            config_file (str): path to the configuration file
        """
        super().__init__(config_file)

    def _read_imcom(self, filepath, layer=0):
        """Read IMCOM block file

        Args:
            filepath (str): path to the IMCOM block file
            layer (int, optional): only read in this layer. Defaults to 0.
        Returns:
            _type_: _description_
        """
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
        """Convert a block of IMCOM input to a ngmix.Observation

        Args:
            filepath (str): path to the IMCOM block file
            layer (int, optional): only read in this layer. Defaults to 0.
            psf_img (_type_, optional): psf image. Defaults to None.
            rng (_type_, optional): random number generator. Defaults to None.

        Returns:
            _type_: _description_
        """
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

        obs = ngmix.Observation(
            # image= f + noise,
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

        # [TODO] Returning image f for sanity check; might remove later.
        return obs, f
