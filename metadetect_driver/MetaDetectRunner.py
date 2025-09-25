import importlib.metadata
import importlib.resources
import logging
import sys
import warnings
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import galsim
import galsim.roman as roman
import metadetect
import ngmix
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sep
import yaml
from astropy import wcs

from pyimcom.analysis import OutImage, Mosaic
from pyimcom.config import Settings

from .config import parse_driver_config


logger = logging.getLogger(__name__)


_DEFAULT_METADETECT_CONFIG = (
    importlib.resources.files(__package__).parent
    / "config"
    / "metadetect_default.yaml"
)


def _get_metadata():
    return {
        f"{__package__} version": importlib.metadata.version(__package__),
        "python version": sys.version,
        "asdf version": importlib.metadata.version("asdf"),
        "astropy version": importlib.metadata.version("astropy"),
        "galsim version": importlib.metadata.version("galsim"),
        "metadetect version": importlib.metadata.version("metadetect"),
        "ngmix version": importlib.metadata.version("ngmix"),
        "numpy version": importlib.metadata.version("numpy"),
        "pyimcom version": importlib.metadata.version("pyimcom"),
        "sep version": importlib.metadata.version("sep"),
        "pyarrow version": importlib.metadata.version("pyarrow"),
    }


def _load_default_metadetect_config():
    with open(_DEFAULT_METADETECT_CONFIG, "r") as file:
        config = yaml.safe_load(file)

    return config


class MetaDetectRunner:
    """
    Class to run MetaDetection on PyIMCOM coadds (Mosaic or OutImage objects).
    Stores the input coadds, MetaDetection config, and driver config, and provides
    methods to build catalogs from the multi-band imaging.
    """

    NATIVE_PIX = 0.11  # arcsec/pixel (Roman native pixel scale)

    def __init__(self, coadds, meta_cfg=None, driver_cfg=None):
        """
        Initialize the MetaDetectRunner.

        Parameters
        ----------
        coadds : Mosaic, OutImage, or list of Mosaic/OutImage
            PyIMCOM output objects to process. Can be a single object or a list
            of the same field in different bands.
        meta_cfg : dict, optional
            MetaDetection configuration dictionary. If None, uses default METADETECT_CONFIG. [default : None]
        driver_cfg : dict, optional
            Driver configuration dictionary. If None, uses parsed DEFAULT_EXTRA_CFG. [default : None]
        """
        logger.info("Instantiating MetaDetectRunner")
        logger.debug(f"MetaDetect config: {meta_cfg}")
        logger.debug(f"Driver config: {driver_cfg}")

        self.coadds = (
            coadds if isinstance(coadds, (list, np.ndarray)) else [coadds]
        )  # convert to list if not already given as list
        # determine if user input were Mosaic or OutImage objects

        self.input_type = self._determine_input_type()
        if self.input_type == "unrecognized":
            raise TypeError(
                "Coadds must be PyIMCOM Mosaic or OutImage objects."
            )

        _default_config = _load_default_metadetect_config()
        self.meta_cfg = (
            deepcopy(meta_cfg) if meta_cfg is not None else _default_config
        )
        # parse driver config
        self.driver_cfg = parse_driver_config(driver_cfg)
        # Set the PyIMCOM config used to make images. The config will vary between bands, but some
        # parameters (e.g.location center, number of blocks) will be the same.
        self.cfg = self.coadds[0].cfg
        # get the bands corresponding to the input images.
        self.bands = self.get_bands()
        _bands = " ".join(np.unique(self.bands))

        self.shear_steps = self.get_shear_steps()

        logger.info(
            f"Processing {len(self.coadds)} {self.input_type} coadds for {_bands}"
        )

    def get_shear_steps(self):
        return self.meta_cfg["metacal"].get("types", None)

    def _determine_input_type(self):
        """
        Determine if coadds are Mosaic or OutImage objects, or other (unrecognized). OutImage objects hold PyIMCOM
        blocks, so we will call OutImage objects "blocks".

        Returns
        -------
        str
        """
        if all(isinstance(coadd, Mosaic) for coadd in self.coadds):
            return "mosaic"
        elif all(isinstance(coadd, OutImage) for coadd in self.coadds):
            return "block"
        else:
            return "unrecognized"

    # ----------------------------
    # User functions and helpers
    # ----------------------------

    def make_catalogs(
        self,
        block_ids=None,
        block_rows=None,
        block_cols=None,
    ):
        """
        Main driver to run MetaDetection and produce a catalog.
        The parameters block_ids, block_rows, and block cols determines
        what blocks within the Mosaic are ran. Either block_ids, or block_rows
        and block_cols, should be given. They can be single integers or lists of integers.
        block_ids represents the block within the Mosaic as used in PyIMCOM, and can range
        between [0, nblocks^2 -1], where nblocks is the number of blocks on each side of the mosaic.
        So if a Mosaic is 12x12 blocks, nblocks would be 12.
        block_rows and block_cols represents the same concept but given as an index location.
        So block_rows and block_cols can range between [0,nblocks - 1].
        The two input types are related by: block_cols, block_rows = divmod (block_ids, nblocks).
        If only one of block_rows or block_cols is passed, all blocks in those rows or columns are ran.
        Example, to run blocks 5_6 and 7_3, you should pass block_rows = [6,3], block_cols = [5,7].
        Note that in the PyIMCOM convention, the column is the first number and row the last.
        If you wanted to run all blocks in row 7 and 8, set block_rows = [7,8], block_cols = None.
        If none are passed for either input, all blocks in a mosaic are ran.
        These arguments only apply if a Mosaic object is passed when creating the MetaDetectRunner object.
        If a block is passed instead, these variables are ignored as the block to run is already explicitly given.

        Parameters
        ----------
        block_ids : int, list of ints or None, optional
            Block indices to process. Only blocks with these indices will be ran.
            If given block_rows and block_cols should be None. [default : None]
        block_rows : int, list of ints or None, optional
            Block rows to process. If passed and block_cols is not None, this serves
            as a list of block positions to run. If given and block_cols is None, all
            blocks in the specified rows will be ran.
            If given, block_ids should be None. [default : None]
        block_cols :  int, list of ints or None, optional
            Block columns to process. If passed and block_rows is not None, this serves
            as a list of block positions to run. If given and block_rows is None, all
            blocks in the specified columns will be ran.
            If given, block_ids should be None. [default : None]
        save: bool, optional
            Whether or not to save the combined final catalog, composed of all processed blocks,
            to the output directory in the driver config (self.driver_cfg) [default : False]
        save_blocks: bool, optional
            Whether or not to save the catalog from all processed blocks individually,
            to the output directory in the driver config (self.driver_cfg). The outputs
            will be in a directory "BlockCatalogs", each catalog in subdirectories ordered
            by row number. [default : False]
        return_cat: bool, optional
            Whether or not to return the final combined catalog when calling this function.
            [default : True]

        Returns
        -------
        pyarrow Table or None
            The final combined catalog from all processed blocks if return_cat = True. Otherwise
            it return None.
        """
        # do some sanity checks on the block index inputs, and convert block_ids to block_rows, block_cols
        block_indices = self._block_inputs(
            block_ids, block_rows, block_cols
        )  # block_rows, block_cols stored as tuple
        if block_indices is not None:
            logger.info(f"Processing blocks {block_indices}")

        ## If the inputs are mosaics or single blocks changes where we start processing.
        if self.input_type == "mosaic":
            catalogs = self._make_cat_mosaic(block_indices)
        elif self.input_type == "block":
            catalogs = [
                self._make_cat_block(self.coadds)
            ]  # make into list since its only one catalog (see _save_outputs)

        return catalogs, block_indices

    def _block_inputs(self, block_ids, block_rows, block_cols):
        """
        Sanity checks on the input block positions. Checks if inputs given
        are integers, converts integers to lists when needed, and checks if blocks
        are within Mosaic. It also converts block_ids (if not None) to their
        corresponding row and column positions in mosaic.

        Parameters
        ----------
        block_ids : int, list of ints, or None
        block_rows : int, list of ints
        block_cols :  int, list of ints

        Returns
        -------
        block_rows : list or None
        block_cols : list or None

        """
        if (
            block_ids is not None
            or block_rows is not None
            or block_cols is not None
        ) and self.input_type == "block":
            warnings.warn(
                "Ignoring input block id/row/col since input images represent a single block or multi-band block already."
            )
            return None, None
        if block_ids is not None and (
            block_rows is not None or block_cols is not None
        ):
            raise ValueError(
                "If specifying block_id, then do not specify block_rows or block_cols, and vice versa."
            )

        if block_rows is not None:
            block_rows = (
                block_rows
                if (
                    isinstance(block_rows, (list, np.ndarray))
                    or block_rows is None
                )
                else [block_rows]
            )
            if not all(
                isinstance(block_row, (int, np.integer))
                for block_row in block_rows
            ):
                raise ValueError(
                    "block_rows must be an integer or list of integers."
                )
            # check all block_rows are less than the number of blocks on each side of mosaic
            if not all(
                block_row < self.cfg.nblock for block_row in block_rows
            ):
                raise ValueError(
                    "Elements in block_rows must be less than Mosaic number of blocks"
                )
        if block_cols is not None:
            block_cols = (
                block_cols
                if (
                    isinstance(block_cols, (list, np.ndarray))
                    or block_cols is None
                )
                else [block_cols]
            )
            if not all(
                isinstance(block_col, (int, np.integer))
                for block_col in block_cols
            ):
                raise ValueError(
                    "block_cols must be an integer or list of integers."
                )
            # check all block_cols are less than the number of blocks on each side of mosaic
            if not all(
                block_col < self.cfg.nblock for block_col in block_cols
            ):
                raise ValueError(
                    "Elements in block_cols must be less than Mosaic number of blocks."
                )
        if block_rows is not None and block_cols is not None:
            if len(block_rows) != len(block_cols):
                raise ValueError(
                    "If both block_rows and block cols are given, arrays must be the same length."
                )

        if block_ids is not None:
            block_ids = (
                block_ids
                if isinstance(block_ids, (list, np.ndarray))
                else [block_ids]
            )
            # check all block_ids passed are integers
            if not all(
                isinstance(block_id, (int, np.integer))
                for block_id in block_ids
            ):
                raise ValueError(
                    "block_ids must be an integer or list of integers."
                )
            # check all block_ids are less than the total number of blocks in a mosaic
            if not all(
                block_id < self.cfg.nblock**2 for block_id in block_ids
            ):
                raise ValueError(
                    "Elements in block_ids must be less than Mosaic number of blocks squared."
                )

            block_rows, block_cols = [], []
            for block_id in block_ids:
                col, row = divmod(block_id, self.cfg.nblock)
                block_rows.append(row)
                block_cols.append(col)

        return block_rows, block_cols

    def write_catalogs(self, output_dir, catalogs, block_indices):
        blocks_ran = self._get_block_pairs(block_indices)

        # _output_dir = self.driver_cfg["outdir"]
        output_path = Path(output_dir)

        logger.info(f"Writing catalogs to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        # output_file = output_path / "metadetect_catalog.parquet"

        # FIXME
        _schema = catalogs[0]["noshear"].schema

        parquet_writers = {}
        for shear_step in self.shear_steps:
            output_file = (
                output_path / f"metadetect_catalog_{shear_step}.parquet"
            )
            logger.debug(
                f"Opening parquet writer for {shear_step} at {output_file}"
            )
            parquet_writers[shear_step] = pq.ParquetWriter(
                output_file, schema=_schema
            )

        for catalog, block_idx in zip(catalogs, blocks_ran):
            logger.info(f"Writing block {block_idx[0]:02d}_{block_idx[1]:02d}")
            for shear_step in catalog.keys():
                parquet_writers[shear_step].write(catalog[shear_step])

        for shear_step, parquet_writer in parquet_writers.items():
            logger.debug(f"Closing parquet writer for {shear_step}")
            parquet_writer.close()

        logger.info("Writing finished")

    # ----------------------------
    # Mosaic-level functions
    # ----------------------------

    def _make_cat_mosaic(self, block_indices):
        """
        Run MetaDetection in parallel over all blocks in the mosaics.

        Parameters
        ----------
        block_indices : tuple

        Returns
        -------
        list of pyarrow Tables
            Every element of the list is the resulting catalog from every processed block.
        """
        # get what blocks within the mosaic to run
        block_to_run = self._get_block_pairs(block_indices)
        # Run blocks in parallel
        with ProcessPoolExecutor(
            max_workers=self.driver_cfg["max_workers"]
        ) as ex:
            return list(
                ex.map(
                    self._run_block,
                    block_to_run,
                    chunksize=self.driver_cfg["chunksize"],
                )
            )

    def _get_block_pairs(self, block_indices):
        """
        Decide which (ibx, iby) blocks to run from the mosaic grid.

        Parameters
        ----------
        block_indices : tuple

        Returns
        -------
        list of tuples
            Column and row indices to process. Default is all blocks.
        """
        block_rows, block_cols = block_indices
        # if specific set of rows and columns are provided
        if block_cols is not None and block_rows is not None:
            args = [(ibx, iby) for ibx, iby in zip(block_cols, block_rows)]
            return args
        if block_cols is None:
            block_cols = np.arange(
                self.cfg.nblock
            )  # the defaults to running all columns
        if block_rows is None:
            block_rows = np.arange(
                self.cfg.nblock
            )  # the defaults to running all rows
        args = [(ibx, iby) for ibx in block_cols for iby in block_rows]
        return args

    # ----------------------------
    # Block-level functions
    # ----------------------------

    def _run_block(self, block_to_run):
        """
        Run processing for a single multi-band block

        Parameters
        ----------
        block_to_run : tuple of lists

        Returns
        -------
        pyarrow Table
            Catalog catalog for the block.
        """
        ibx, iby = block_to_run
        # make multi-band list of blocks
        blocks = [mosaic.outimages[iby][ibx] for mosaic in self.coadds]
        return self._make_cat_block(
            blocks
        )  # run metadetection and produce catalog

    def _make_cat_block(self, blocks):
        """
        Run MetaDetection over a single block or list of blocks. Each block in
        list represents a different band.

         Parameters
        ----------
        blocks : list of OutImage objects (multi-band)

        Returns
        -------
        pyarrow Table
            Catalog catalog for the block.
        """

        mbobs = self.make_mbobs(blocks)
        res = self.run_metadetect(mbobs)
        return self.construct_table(blocks, res)

    # ----------------------------
    # ngmix observation builders
    # ----------------------------

    def make_mbobs(self, blocks):
        """
        Build an ngmix MultiBandObsList from a list of blocks (each a different band).

        Parameters
        ----------
        blocks : list of OutImage objects

        Returns
        -------
        mbobs : ngmix MultiBandObservation
        """
        mbobs = ngmix.MultiBandObsList()
        for block in (
            blocks if isinstance(blocks, list) else [blocks]
        ):  # loop over blocks of different bands
            obslist = self.make_ngmix_obs(block)
            mbobs.append(obslist)
        return mbobs

    def make_ngmix_obs(self, block):
        """
        Create an ngmix ObsList for a single block image.

        Parameters
        ----------
        block : OutImage
            PyIMCOM block

        Returns
        -------
        obslist : ngmix Observation
        """
        img, img_jacobian, psf_img, noise_sigma = self.get_ngmix_data(block)

        # Centers
        psf_cen = (psf_img.shape[0] - 1) / 2.0
        img_cen = (np.array([img.shape[0], img.shape[1]]) - 1) / 2.0

        # ngmix Jacobians
        psf_jac = ngmix.Jacobian(row=psf_cen, col=psf_cen, wcs=img_jacobian)
        img_jac = ngmix.Jacobian(
            row=img_cen[0], col=img_cen[1], wcs=img_jacobian
        )

        # Observations
        psf_obs = ngmix.Observation(image=psf_img, jacobian=psf_jac)
        obs = ngmix.Observation(
            image=img,
            jacobian=img_jac,
            weight=np.full(img.shape, 1 / noise_sigma**2, dtype=float),
            psf=psf_obs,
            ormask=np.zeros(img.shape, dtype=np.int32),
            bmask=np.zeros(img.shape, dtype=np.int32),
        )
        obslist = ngmix.ObsList()
        obslist.append(obs)
        return obslist

    def get_ngmix_data(self, block):
        """
        Generate inputs needed to make ngmix Observation for a single block.

         Parameters
        ----------
        block : OutImage object representing a single block (one band).

        Returns
        -------
        image : np.ndarray
            Coadded image for the requested layer.
        img_jacobian : ngmix.Jacobian
            Image-plane Jacobian derived from WCS at the reference pixel.
        psf_img : np.ndarray
            PSF image.
        noise_sigma : float
            Global RMS of the image background.
        """
        image = block.get_coadded_layer(self.driver_cfg["layer"])

        # Build GalSim WCS and Jacobian
        w = galsim.AstropyWCS(wcs=self.get_wcs(block))
        img_jacobian = w.jacobian(
            image_pos=galsim.PositionD(w.wcs.wcs.crpix[0], w.wcs.wcs.crpix[1])
        )

        # Estimate background RMS using SEP
        bkg = sep.Background(image.astype(image.dtype.newbyteorder("=")))
        noise_sigma = bkg.globalrms

        # Draw PSF image
        psf_img = self.get_psf(block, w)
        return image, img_jacobian, psf_img, noise_sigma

    # ----------------------------
    # Running metadetect
    # ----------------------------
    def run_metadetect(self, mbobs):
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
        det_bands = self.driver_cfg["det_bands"]
        shear_bands = self.driver_cfg["shear_bands"]

        det_combs = None
        shear_combs = None
        if det_bands is not None:
            # Select only detection and shear bands from bands in coadds provided.
            det_idx = np.arange(len(self.bands))[
                np.isin(self.bands, det_bands)
            ]
            det_combs = [det_idx]
        if shear_bands is not None:
            shear_idx = np.arange(len(self.bands))[
                np.isin(self.bands, shear_bands)
            ]
            shear_combs = [shear_idx]

        # Run metadetect
        res = metadetect.do_metadetect(
            deepcopy(self.meta_cfg),
            mbobs=mbobs,
            rng=np.random.RandomState(seed=self.driver_cfg["mdet_seed"]),
            det_band_combs=det_combs,
            shear_band_combs=shear_combs,
        )
        return res

    # ----------------------------
    # Band helpers
    # ----------------------------
    def get_bands(self):
        """
        Get band names for provided blocks.
        PyIMCOM has a certain ordering of the filters (e.g. filter 2 is H158).
        So, we get the appropiate band name from Settings.RomanFilters (Settings is PyIMCOM settings)

        Returns
        -------
        list[str]
            Band labels matching each block in `blocks`.
        """
        # if self.input_type == 'block':
        #    blocks = self.coadds
        # elif self.input_type == 'mosaic':
        # for mosaics simply use one of the
        #    blocks = [mosaic.outimages[0][0] for mosaic in self.coadds]

        band_list = []
        for coadd in self.coadds:
            band = Settings.RomanFilters[coadd.cfg.use_filter]
            band_list.append(band)
        return band_list

    # ----------------------------
    # WCS / PSF helpers
    # ----------------------------
    @staticmethod
    def get_wcs(block):
        """
        Construct WCS for a block from its configuration.

        Parameters
        ----------
        block : OutImage
            PyIMCOM block

        Returns
        -------
        outwcs : galsim.WCS object
            Output WCS from coadded image

        Notes
        -----
        Code mirrors:
        https://github.com/Roman-HLIS-Cosmology-PIT/pyimcom/blob/main/coadd.py
        """
        cfg = block.cfg
        ibx, iby = block.ibx, block.iby
        outwcs = wcs.WCS(naxis=2)
        outwcs.wcs.crpix = [
            (cfg.NsideP + 1) / 2.0
            - cfg.Nside * (ibx - (cfg.nblock - 1) / 2.0),
            (cfg.NsideP + 1) / 2.0
            - cfg.Nside * (iby - (cfg.nblock - 1) / 2.0),
        ]
        outwcs.wcs.cdelt = [-cfg.dtheta, cfg.dtheta]
        outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        outwcs.wcs.crval = [cfg.ra, cfg.dec]
        outwcs.wcs.lonpole = cfg.lonpole
        return outwcs

    @staticmethod
    def get_psf_obj(block):
        """
        Build a GalSim PSF from the block configuration.

        Parameters
        ----------
        block : OutImage
            PyIMCOM block

        Returns
        -------
        psf : galsim object
            Output PSF object

        Notes
        -----
        - PyIMCOM output PSF can be GAUSSIAN, AIRY (obscured/unobscured).
        - The AIRY kernels are also be convolved with a Gaussian.
        """
        cfg = block.cfg

        # Base Gaussian width: cfg.sigmatarget is in native pixels; convert to arcsec then to FWHM.
        fwhm = cfg.sigmatarget * MetaDetectRunner.NATIVE_PIX * 2.355
        psf = galsim.Gaussian(fwhm=fwhm)

        # Optional Airy with/without obscuration, then convolve with Gaussian.
        if cfg.outpsf in ("AIRYOBSC", "AIRYUNOBSC"):
            obsc = Settings.obsc if cfg.outpsf == "AIRYOBSC" else 0.0
            # PyIMCOM settings stores the lambda over diameter factor for every band in units of native pixel,
            # so we multiply by roman native pixel scale (0.11) to convert to arcsec
            lam_over_diam = (
                Settings.QFilterNative[cfg.use_filter]
                * MetaDetectRunner.NATIVE_PIX
            )  # arcsec
            airy = galsim.Airy(lam_over_diam=lam_over_diam, obscuration=obsc)
            psf = galsim.Convolve([airy, psf])

        return psf

    def get_psf(self, block, w):
        """
        Draw a PSF image for a block given.

        Parameters
        ----------
        block : OutImage
            PyIMCOM block.
        w : galsim.BaseWCS
            GalSim WCS instance.

        Returns
        -------
        psf_img: np.ndarray
            PSF image array with shape (psf_img_size, psf_img_size).
        """
        psf = self.get_psf_obj(block)
        psf_img = psf.drawImage(
            nx=self.driver_cfg["psf_img_size"],
            ny=self.driver_cfg["psf_img_size"],
            wcs=w,
        ).array
        return psf_img

    # ----------------------------
    # Unit conversions
    # ----------------------------
    def imcom_flux_conv(self, flux):
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
        The (NATIVE_PIX**2/oversample_pix**2) takes into account that the
        coadds are oversampled and not in the native Roman pixel scale.
        AB magnitude can be calculated using galsim.roman zeropoints
        """
        # coadd pixel scale in arcsec (PyIMCOM stores in degrees)
        oversample_pix = (
            self.cfg.dtheta * (180.0 / np.pi) * 3600.0
        )  # deg --> arcsec
        norm_fact = (
            roman.exptime
            * roman.collecting_area
            * (MetaDetectRunner.NATIVE_PIX**2 / oversample_pix**2)
        )
        flux_converted = flux / norm_fact
        return flux_converted

    # ----------------------------
    # Catalog filtering / region selection
    # ----------------------------
    def det_bound_from_padding(self):
        """
        Derive a suitable edge bound from the block's padding.
        In this case it sets the bound size to exlude entire padding region.

        Returns
        -------
        int
            Bound size in pixels.
        """
        pad = self.cfg.postage_pad  # number of postage stamps of padding
        npix_post_stamp = self.cfg.n2  # pixels per postage stamp
        return int(pad * npix_post_stamp)

    def get_bounded_region(self, res, shear_step):
        """
        Build a mask that excludes detections too close to image edges.
        Determine what detections from the image to exclude from catalog.
        This is applied after Metadetect is run. This avoids including objects
        too close to the edge of the image, where detections can be bad.


        Parameters
        ----------
        res : dict
            Metadetect result dict.

        Returns
        -------
        np.ndarray (bool)
            True for detections kept; False for excluded.
        """
        #'bound_size' sets the maximum distance (in pixels) a detection can be from the edge of the image
        # If 'bound_size' is None, the boundsize is set to be the padded region from the coadded image.
        if self.driver_cfg["bound_size"] is None:
            bound_size = self.det_bound_from_padding()
        else:
            bound_size = self.driver_cfg["bound_size"]

        img_size = self.cfg.NsideP  # (ny, nx)
        x = res[shear_step]["sx_col"]
        y = res[shear_step]["sx_row"]
        keep = (
            (x > bound_size)
            & (x < img_size - bound_size)
            & (y > bound_size)
            & (y < img_size - bound_size)
        )
        return keep

    # ----------------------------
    # Results construction
    # ----------------------------
    def construct_table(self, blocks, res):
        """
        Convert metadetect results into a catalog pyarrow Table.
        Also converts IMCOM fluxes to e-/cm^2/s, and computes RA/DEC for detections.

        Parameters
        ----------
        blocks : list of OutImage objects

        res : dict
            Metadetect result dict.

        Returns
        -------
        pyarrow Table
            Catalog of detected objects.

        """
        metadata = _get_metadata()
        results = {}
        for shear_step in res.keys():
            # World coordinates
            w = galsim.AstropyWCS(wcs=self.get_wcs(blocks[0]))
            x, y = res[shear_step]["sx_col"], res[shear_step]["sx_row"]
            ra_pos, dec_pos = w.toWorld(x, y, units="deg")

            # get masked region. All detections outside the bounded region are excluded from catalog
            keep_mask = self.get_bounded_region(res, shear_step)

            resultdict = {}

            for name in res[shear_step].dtype.names:
                data = res[shear_step][name]
                if "flux" in name:
                    # for flux columns, first convert units. See imcom_flux_conv for why we do this.
                    data = self.imcom_flux_conv(data)

                    # if cf.ndim == 1:
                    #     cf = cf[:, None]  # make it (N, 1) instead of (N,)
                    # for i, band in enumerate(self.bands):
                    #     # flux is stored as a (N_det, N_band) array if more than one band
                    #     resultdict[f"{model}_{band}_{col}"] = cf[:, i]

                resultdict[name] = data.tolist()

            resultdict["ra"] = ra_pos
            resultdict["dec"] = dec_pos

            resultdict["is_primary"] = keep_mask.tolist()

            # # Select requested columns; convert flux-like columns
            # # for col in self.driver_cfg["keepcols"]:
            # for col in res[shear_step].dtype.names:
            #     models = []
            #     if "fitters" in self.meta_cfg:
            #         for fitter_config in self.meta_cfg["fitters"]:
            #             # https://github.com/esheldon/metadetect/blob/master/metadetect/metadetect.py#L240
            #             model = fitter_config.get("model", "wmom")
            #             models.append(model)
            #     else:
            #         model = fitter_config.get("model", "wmom")
            #         models.append(model)

            #     for model in models:
            #         key = f"{model}_{col}"
            #         if "flux" in col:
            #             # for flux columns, first convert units. See imcom_flux_conv for why we do this.
            #             flux = self.imcom_flux_conv(res[shear_step][key])
            #             cf = np.asarray(flux)
            #             if cf.ndim == 1:
            #                 cf = cf[:, None]  # make it (N, 1) instead of (N,)
            #             for i, band in enumerate(self.bands):
            #                 # flux is stored as a (N_det, N_band) array if more than one band
            #                 resultdict[f"{model}_{band}_{col}"] = cf[:, i][keep_mask]
            #         else:
            #             resultdict[key] = res[shear_step][key][keep_mask]

            results[shear_step] = pa.Table.from_pydict(
                resultdict, metadata=metadata
            )

        return results
