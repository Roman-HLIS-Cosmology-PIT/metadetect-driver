import importlib.metadata
import logging
import sys
from copy import deepcopy

import galsim
import galsim.roman as roman
import metadetect
import ngmix
import numpy as np
import pyarrow as pa
import sep
from astropy import wcs
from pyimcom.config import Settings

from .config import _parse_driver_config
from .defaults import METADETECT_DEFAULTS

logger = logging.getLogger(__name__)


_NATIVE_PIX = Settings.pixscale_native / Settings.arcsec


def _get_package_metadata():
    return {
        f"{__package__} version": importlib.metadata.version(__package__),
        "python version": sys.version,
        "asdf version": importlib.metadata.version("asdf"),
        "astropy version": importlib.metadata.version("astropy"),
        "galsim version": importlib.metadata.version("galsim"),
        "metadetect version": importlib.metadata.version("metadetect"),
        "ngmix version": importlib.metadata.version("ngmix"),
        "numpy version": importlib.metadata.version("numpy"),
        "pyarrow version": importlib.metadata.version("pyarrow"),
        "pyimcom version": importlib.metadata.version("pyimcom"),
        "scipy version": importlib.metadata.version("scipy"),
        "sep version": importlib.metadata.version("sep"),
    }


def from_imcom_flux(flux, dtheta):
    """
    Convert IMCOM flux units to e-/cm^2/s.

    Parameters
    ----------
    flux : np.array
        Array of fluxes

    Returns
    -------
    flux_converted: np.array

    Notes
    -----
    IMCOM flux unit is e- / (0.11 arcsec)^2 / exposure.
    We convert to e-/cm^2/s using Roman collecting area and exposure,
    and correct for the coadd oversampling relative to native pixels.
    The (_NATIVE_PIX**2/oversample_pix**2) takes into account that the
    coadds are oversampled and not in the native Roman pixel scale.
    AB magnitude can be calculated using galsim.roman zeropoints.
    """
    # coadd pixel scale in arcsec (PyIMCOM stores in degrees)
    # FIXME why is there a radian conversion here?
    # oversample_pix = dtheta * 3600.0  # deg --> arcsec
    oversample_pix = dtheta / Settings.arcsec
    norm = roman.exptime * roman.collecting_area * (_NATIVE_PIX**2 / oversample_pix**2)
    return flux / norm


def get_imcom_wcs(outimage):
    """
    Construct WCS for a outimage from its configuration.

    Parameters
    ----------
    outimage : OutImage
        PyIMCOM outimage

    Returns
    -------
    outwcs : astropy.WCS object
        Output WCS from coadded image

    Notes
    -----
    Code mirrors:
    https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/blob/main/coadd.py
    """
    cfg = outimage.cfg
    ibx, iby = outimage.ibx, outimage.iby
    outwcs = wcs.WCS(naxis=2)
    outwcs.wcs.crpix = [
        (cfg.NsideP + 1) / 2.0 - cfg.Nside * (ibx - (cfg.nblock - 1) / 2.0),
        (cfg.NsideP + 1) / 2.0 - cfg.Nside * (iby - (cfg.nblock - 1) / 2.0),
    ]
    outwcs.wcs.cdelt = [-cfg.dtheta, cfg.dtheta]
    outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
    outwcs.wcs.crval = [cfg.ra, cfg.dec]
    outwcs.wcs.lonpole = cfg.lonpole
    return outwcs


def get_imcom_psf(cfg):
    """
    Build a GalSim PSF from the outimage configuration.

    Parameters
    ----------
    cfg : PyIMCOM Config

    Returns
    -------
    psf : galsim object
        Output PSF object

    Notes
    -----
    - PyIMCOM output PSF can be GAUSSIAN, AIRY (obscured/unobscured).
    - The AIRY kernels are also be convolved with a Gaussian.
    """
    # Base Gaussian width: cfg.sigmatarget is in native pixels; convert to arcsec
    sigma = cfg.sigmatarget * _NATIVE_PIX
    psf = galsim.Gaussian(sigma=sigma)

    # Optional Airy with/without obscuration, then convolve with Gaussian.
    if cfg.outpsf in ("AIRYOBSC", "AIRYUNOBSC"):
        obsc = Settings.obsc if cfg.outpsf == "AIRYOBSC" else 0.0
        # PyIMCOM settings stores the lambda over diameter factor for every band in units of native pixel,
        # so we multiply by roman native pixel scale (0.11) to convert to arcsec
        lam_over_diam = Settings.QFilterNative[cfg.use_filter] * _NATIVE_PIX  # arcsec
        airy = galsim.Airy(lam_over_diam=lam_over_diam, obscuration=obsc)
        psf = galsim.Convolve([airy, psf])

    return psf


def run_metadetect(outimages, driver_config=None, metadetect_config=None):
    """
    Run metadetect on multi-band coadd images.

    Parameters
    ----------
    outimages : list of OutImage
        PyIMCOM output objects to process. Each element corresponds to the
        coadd of the same block in different bands
    driver_config : dict, optional
        Driver configuration dictionary. If None, uses parsed DEFAULT_EXTRA_CFG. [default : None]
    metadetect_config : dict, optional
        Metadetection configuration dictionary. If None, uses default METADETECT_CONFIG. [default : None]

    Returns
    -------
    dict of pyarrow Table
        metadetect catalogs for each shear type
    """
    runner = MetadetectRunner(
        outimages, driver_config=driver_config, metadetect_config=metadetect_config,
    )
    return runner.run()


class MetadetectRunner:
    """
    Class to run Metadetection on PyIMCOM coadds (OutImage objects).
    Stores the input coadds, Metadetection config, and driver config, and provides
    methods to build catalogs from the multi-band imaging.
    """

    def __init__(self, outimages, driver_config=None, metadetect_config=None):
        """
        Initialize the MetadetectRunner.

        Parameters
        ----------
        outimages : list of OutImage
            PyIMCOM output objects to process. Each element corresponds to the
            coadd of the same block in different bands
        driver_config : dict, optional
            Driver configuration dictionary. If None, uses DRIVER_DEFAULTS. [default : None]
        metadetect_config : dict, optional
            Metadetection configuration dictionary. If None, uses METADETECT_DEFAULTS. [default : None]
        """
        logger.info("Instantiating MetadetectRunner")
        logger.debug(f"Driver config: {driver_config}")
        logger.debug(f"Metadetect config: {metadetect_config}")

        self.driver_config = _parse_driver_config(driver_config)
        self.metadetect_config = (
            deepcopy(metadetect_config)
            if metadetect_config is not None
            else deepcopy(METADETECT_DEFAULTS)
        )

        # Ensure each outimage corresponds to the same block
        _block_ids = set((outimage.ibx, outimage.iby) for outimage in outimages)
        _block_idx, _block_idy = _block_ids.pop()
        assert len(_block_ids) == 0

        # Ensure that each outimage corresponds to a different filter
        _filters = set(outimage.cfg.use_filter for outimage in outimages)
        assert len(_filters) == len(outimages)

        # TODO ensure that the configs are consistent
        self.imcom_config = outimages[0].cfg

        self.outimages = outimages

        self.block_id = (_block_idx, _block_idy)
        self.block_ra = self.imcom_config.ra
        self.block_dec = self.imcom_config.dec
        self.block_lonpole = self.imcom_config.lonpole

        # parameters (e.g.location center, number of blocks) will be the same.
        # TODO it would be nice to have some way to validate consistency...
        self.bands = MetadetectRunner.get_bands(self.outimages)
        self.shear_types = self.get_shear_types()
        self.metacal_step = self.get_metacal_step()
        self.det_combs = self.get_det_combs()
        self.shear_combs = self.get_shear_combs()

    def get_metacal_step(self):
        return self.metadetect_config["metacal"].get("step", ngmix.metacal.DEFAULT_STEP)

    def get_shear_types(self):
        return self.metadetect_config["metacal"].get(
            "types", ngmix.metacal.METACAL_MINIMAL_TYPES
        )

    @staticmethod
    def get_bands(outimages):
        """
        Get band names for provided blocks.
        PyIMCOM has a certain ordering of the filters (e.g. filter 2 is H158).
        So, we get the appropiate band name from Settings.RomanFilters (Settings is PyIMCOM settings)

        Returns
        -------
        list[str]
            Band labels matching each block in `blocks`.
        """
        band_list = []
        for outimage in outimages:
            band = Settings.RomanFilters[outimage.cfg.use_filter]
            band_list.append(band)
        return band_list

    @staticmethod
    def get_shear(shear_type, metacal_step):
        match shear_type:
            case "noshear":
                _shear = galsim.Shear(g1=0.0, g2=0.0)
            case "1p":
                _shear = galsim.Shear(g1=metacal_step, g2=0)
            case "1m":
                _shear = galsim.Shear(g1=-metacal_step, g2=0)
            case "2p":
                _shear = galsim.Shear(g1=0.0, g2=metacal_step)
            case "2m":
                _shear = galsim.Shear(g1=0.0, g2=-metacal_step)
            case _:
                raise ValueError(f"{shear_type} is an invalid shear type!")

        return _shear

    def get_det_combs(self):
        det_bands = self.driver_config["det_bands"]
        if det_bands is not None:
            # Select only detection and shear bands from bands in blocks provided.
            det_idx = np.arange(len(self.bands))[np.isin(self.bands, det_bands)]
            det_combs = [det_idx]
        else:
            det_combs = None

        return det_combs

    def get_shear_combs(self):
        shear_bands = self.driver_config["shear_bands"]

        if shear_bands is not None:
            shear_idx = np.arange(len(self.bands))[np.isin(self.bands, shear_bands)]
            shear_combs = [shear_idx]
        else:
            shear_combs = None

        return shear_combs

    @staticmethod
    def get_shear_jacobian(shear_type, metacal_step):
        # cf. https://github.com/GalSim-developers/GalSim/blob/releases/2.7/galsim/gsobject.py#L909-L939
        _shear = MetadetectRunner.get_shear(shear_type, metacal_step)
        return _shear.getMatrix()

    def run(self):
        """
        Main driver to run Metadetection and produce a catalog.

        Parameters
        ----------

        Returns
        -------
        pyarrow Table
            The final combined catalog from all processed blocks for each
            shear type
        """
        mbobs = self.make_mbobs()
        res = self._run_metadetect(mbobs)

        wcs = None
        # Ensure that all blocks have the same WCS
        for outimage in self.outimages:
            if wcs is None:
                wcs = get_imcom_wcs(outimage)
            else:
                _wcs = get_imcom_wcs(outimage)
                # TODO is there a better way to check for WCS consistency?
                assert wcs.wcs == _wcs.wcs
        logger.debug(f"WCS is {wcs}")

        return self.construct_table(wcs, res)

    def make_mbobs(self):
        """
        Build an ngmix MultiBandObsList from a list of outimages.

        Parameters
        ----------
        outimages : list of OutImage objects

        Returns
        -------
        mbobs : ngmix MultiBandObservation
        """
        mbobs = ngmix.MultiBandObsList()
        for outimage in self.outimages:
            obslist = self.make_ngmix_obs(outimage)
            mbobs.append(obslist)
        return mbobs

    def make_ngmix_obs(self, outimage):
        """
        Create an ngmix ObsList for a single outimage.

        Parameters
        ----------
        outimage : OutImage
            PyIMCOM outimage

        Returns
        -------
        obslist : ngmix Observation
        """
        image, jacobian, psf_image, noise_image, noise_sigma = self._get_ngmix_data(outimage)

        # Centers
        psf_image_center = (psf_image.shape[0] - 1) / 2.0
        image_center = (np.array([image.shape[0], image.shape[1]]) - 1) / 2.0

        # ngmix Jacobians
        psf_image_jacobian = ngmix.Jacobian(row=psf_image_center, col=psf_image_center, wcs=jacobian)
        image_jacobian = ngmix.Jacobian(row=image_center[0], col=image_center[1], wcs=jacobian)

        # Observations
        psf_obs = ngmix.Observation(image=psf_image, jacobian=psf_image_jacobian)
        obs = ngmix.Observation(
            image=image,
            noise=noise_image,
            jacobian=image_jacobian,
            weight=np.full(image.shape, 1 / noise_sigma**2, dtype=float),
            psf=psf_obs,
            ormask=np.zeros(image.shape, dtype=np.int32),
            bmask=np.zeros(image.shape, dtype=np.int32),
        )
        obslist = ngmix.ObsList()
        obslist.append(obs)
        return obslist

    def _get_ngmix_data(self, outimage):
        """
        Generate inputs needed to make ngmix Observation for a single outimage.

         Parameters
        ----------
        outimage : OutImage object representing a single outimage in one band.

        Returns
        -------
        image : np.ndarray
            Coadded image for the requested layer.
        jacobian : ngmix.Jacobian
            Image-plane Jacobian derived from WCS at the reference pixel.
        psf_image : np.ndarray
            PSF image.
        noise_image : np.ndarray
            Coadded image for the noise layer.
        noise_sigma : float
            Global RMS of the image background.
        """
        image = outimage.get_coadded_layer(self.driver_config.get("layer", "SCI"))
        # noise_image = outimage.get_coadded_layer(self.driver_config.get("noise_layer", "whitenoise10"))  # FIXME renormalize
        noise_image = None

        # Build GalSim WCS and Jacobian
        _wcs = get_imcom_wcs(outimage)
        wcs = galsim.AstropyWCS(wcs=_wcs)
        jacobian = wcs.jacobian(
            image_pos=galsim.PositionD(wcs.wcs.wcs.crpix[0], wcs.wcs.wcs.crpix[1])
        )

        # Estimate background RMS using SEP
        bkg = sep.Background(image.astype(image.dtype.newbyteorder("=")))
        noise_sigma = bkg.globalrms

        # Draw PSF image
        psf_image = self.get_psf(outimage, wcs)

        return image, jacobian, psf_image, noise_image, noise_sigma

    def _run_metadetect(self, mbobs):
        """
        Run metadetect on the provided MultiBandObsList.

        Parameters
        ----------
        mbobs : ngmix.MultiBandObsList
            Observations across one or more bands.

        Returns
        -------
        res : dict
            Metadetect results.
        """
        _metadetect_config = deepcopy(self.metadetect_config)
        _rng = np.random.RandomState(seed=self.driver_config.get("mdet_seed"))

        return metadetect.do_metadetect(
            _metadetect_config,
            mbobs=mbobs,
            rng=_rng,
            det_band_combs=self.det_combs,
            shear_band_combs=self.shear_combs,
        )

    def get_psf(self, outimage, wcs):
        """
        Draw a PSF image for a given outimage.

        Parameters
        ----------
        outimage: OutImage
            PyIMCOM outimageblock.
        wcs : galsim.BaseWCS
            GalSim WCS instance.

        Returns
        -------
        psf_image: np.ndarray
            PSF image array with shape (psf_image_size, psf_image_size).
        """
        psf = get_imcom_psf(outimage.cfg)
        return psf.drawImage(
            nx=self.driver_config["psf_image_size"],
            ny=self.driver_config["psf_image_size"],
            wcs=wcs,
        ).array

    @staticmethod
    def det_bound_from_padding(cfg):
        """
        Derive a suitable edge bound from the block's padding.
        In this case it sets the bound size to exclude entire padding region.

        Returns
        -------
        int
            Bound size in pixels.
        """
        pad = cfg.postage_pad  # number of postage stamps of padding
        npix_post_stamp = cfg.n2  # pixels per postage stamp
        return int(pad * npix_post_stamp)

    def get_bounded_region(self, res, shear_type):
        """
        Identify detections too close to image edges where detections can be bad.

        Parameters
        ----------
        res : dict
            Metadetect result dict.

        Returns
        -------
        np.ndarray (bool)
            True for detections kept; False for excluded.
        """
        # 'bound_size' sets the maximum distance (in pixels) a detection can be
        # from the edge of the image. If 'bound_size' is None, the boundsize is
        # set to be the padded region from the coadded image.
        if self.driver_config["bound_size"] is None:
            bound_size = MetadetectRunner.det_bound_from_padding(self.imcom_config)
        else:
            bound_size = self.driver_config["bound_size"]

        image_size = self.imcom_config.NsideP  # (ny, nx)
        x = res[shear_type]["sx_col"]
        y = res[shear_type]["sx_row"]
        return (
            (x > bound_size)
            & (x < image_size - bound_size)
            & (y > bound_size)
            & (y < image_size - bound_size)
        )

    def _get_metadata(self):
        _packages = _get_package_metadata()
        # TODO it might be confusing to have `metacal_step` in the metadata
        # if we try to read in the entire catalog through one interface
        # (e.g., via a pyarrow Dataset)
        _meta = {
            "det_band_combs": self.det_combs or "null",
            "shear_band_combs": self.shear_combs or "null",
            "metacal_step": str(self.get_metacal_step()),
        }
        return _packages | _meta

    def construct_table(self, wcs, res):
        """
        Convert metadetect results into pyarrow Tables.
        Converts IMCOM fluxes to e-/cm^2/s,
        computes RA/Dec for detections, and
        checks whether detections are primary (in bounds) or not.

        Parameters
        ----------
        wcs : galsim.WCS
            WCS of coadded image

        res : dict
            Metadetect result dict.

        Returns
        -------
        dict
            dict of pyarrow Tables for each Metadetect catalog

        """
        _metadata = self._get_metadata()
        results = {}
        for shear_type, catalog in res.items():
            _shear_jacobian = MetadetectRunner.get_shear_jacobian(
                shear_type, self.metacal_step
            )

            # World coordinates
            w = galsim.AstropyWCS(wcs=wcs)
            x, y = catalog["sx_col"], catalog["sx_row"]
            ra_pos, dec_pos = w.toWorld(x, y, units="deg")

            is_primary = self.get_bounded_region(res, shear_type)

            _results = {}

            _results["is_primary"] = is_primary.tolist()
            _results["ra"] = ra_pos.tolist()
            _results["dec"] = dec_pos.tolist()

            for name in res[shear_type].dtype.names:
                data = catalog[name]
                if ("flux" in name) and ("flags" not in name):
                    data = from_imcom_flux(data, self.imcom_config.dtheta)

                _results[name] = data.tolist()

            _results["block_id"] = [self.block_id for _ in range(len(x))]
            _results["projection_center_ra"] = [self.block_ra for _ in range(len(x))]
            _results["projection_center_dec"] = [self.block_dec for _ in range(len(x))]
            _results["projection_center_lonpole"] = [
                self.block_lonpole for _ in range(len(x))
            ]
            _results["shear_jacobian"] = [
                _shear_jacobian.tolist() for _ in range(len(x))
            ]
            _results["shear_type"] = [shear_type for _ in range(len(x))]

            results[shear_type] = pa.Table.from_pydict(_results, metadata=_metadata)

        return results
