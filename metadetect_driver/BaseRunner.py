from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import galsim
import ngmix
import metadetect
import galsim.roman as roman
import sep
from astropy import wcs
import os, sys

#from pyimcom.analysis import OutImage, Mosaic
from pyimcom.config import Settings as Stn

from .config import parse_driver_cfg
import warnings
import yaml

# load default metadetect config file
with open('../config/metadetect_default.yaml', 'r') as file:
        METADETECT_CONFIG= yaml.safe_load(file)



class OutImage:
    
    def __init__(self, image, psf, wcs,pix_scale, band = None):
        self.image = image
        self.psf = psf
        self.wcs = wcs
        self.band = band
        self.pix_scale = pix_scale

class Mosaic:
    def __init__(self, blocks):
        self.outimages = blocks if isinstance(blocks, (list, np.ndarray)) else [blocks]
        self.nblock = len(blocks)
        self.band = blocks[0].band
        

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
        self.coadds = coadds if isinstance(coadds, (list, np.ndarray)) else [coadds] # convert to list if not already given as list
        # determine if user input were Mosaic or OutImage objects
        self.input_type = self._determine_input_type()
        if self.input_type == "unrecognized":
            raise TypeError("Coadds must be Mosaic or OutImage objects.")
            
        self.meta_cfg = deepcopy(meta_cfg) if meta_cfg is not None else deepcopy(METADETECT_CONFIG)
        # parse driver config
        self.driver_cfg = parse_driver_cfg(driver_cfg)
        # Set the PyIMCOM config used to make images. The config will vary between bands, but some
        # parameters (e.g.location center, number of blocks) will be the same. 
        #self.cfg = self.coadds[0].cfg
        # get the bands corresponding to the input images.
        #self.bands = self.get_bands()
        self.bands = [mosaic.band for mosaic in self.coadds]


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

    def make_catalog(self, block_ids = None, save = False, save_blocks = False, return_cat = True):
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
        pandas DataFrame or None
            The final combined catalog from all processed blocks if return_cat = True. Otherwise
            it return None.
        """
        # do some sanity checks on the block index inputs, and convert block_ids to block_rows, block_cols
        block_indices = self._block_inputs (block_ids) # block_rows, block_cols stored as tuple

        ## If the inputs are mosaics or single blocks changes where we start processing.
        if self.input_type == "mosaic":
            catalog = self._make_cat_mosaic(block_indices)
        elif self.input_type == "block":
            catalog = [self._make_cat_block(self.coadds)] # make into list since its only one catalog (see _save_outputs)

        # get combined final catalog from observations and save to disk if needed.
        catalog = self._save_outputs (catalog, block_indices, save, save_blocks)
        if return_cat:
            return catalog
            
        
            
    def _block_inputs (self, block_ids):
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
        if (block_ids is not None) and self.input_type == 'block':
            warnings.warn("Ignoring input block id/row/col since input images represent a single block or multi-band block already.")
            return None, None
            
                
        if block_ids is not None:
            block_ids =  block_ids if isinstance(block_ids, (list, np.ndarray)) else [block_ids]
            # check all block_ids passed are integers
            if not all(isinstance(block_id,  (int, np.integer)) for block_id in block_ids):
                raise ValueError("block_ids must be an integer or list of integers.")

        return block_ids

    def _save_outputs (self, catalog, block_indices, save, save_blocks):
        """
        Makes final catalog of all processed blocks by concatinating the catalogs
        from each block. If specified, the catalog from each block is also saved
        as an output. If specified, the final combined output is saved as an output.

        Parameters
        ----------
        block_indices : tuple
        save : bool
        save_blocks :  bool

        Returns
        -------
        pandas DataFrame
            The final combined catalog.
        """
        blocks_ran = self._get_block_pairs(block_indices)
            
        if save_blocks:
            block_dir = os.path.join(self.driver_cfg['outdir'], 'BlockCatalogs') 
            # make new directory in output directory to store individual catalogs
            os.makedirs(block_dir, exist_ok=True) 
            for cat, block_idx in zip(catalog, blocks_ran):
                block_row_dir = os.path.join(block_dir, str(block_idx[1]))
                # make new directory for each row
                os.makedirs(block_row_dir, exist_ok=True) 
                block_file = os.path.join(block_row_dir, f'Catalog_{block_idx[0]:02d}_{block_idx[1]:02d}.parquet')
                cat.to_parquet(block_file, engine='pyarrow', compression=None)
        # Concatenate all blocks into one catalog. If only one block is passed, we made it into a list in make_catalog function
        catalog = pd.concat(catalog, ignore_index=True)
        if save:
            outfile =os.path.join(self.driver_cfg['outdir'], 'MetaDetect_Catalog.parquet')
            catalog.to_parquet(outfile, engine='pyarrow', compression=None)
        return catalog
             

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
        list of pandas DataFrames
            Every element of the list is the resulting catalog from every processed block.
        """
        # get what blocks within the mosaic to run
        block_to_run = self._get_block_pairs(block_indices)
        # Run blocks in parallel
        with ProcessPoolExecutor(max_workers=self.driver_cfg['max_workers']) as ex:
            return list(ex.map(self._run_block, block_to_run, chunksize=self.driver_cfg['chunksize']))

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
        # if specific set of rows and columns are provided
        if block_indices is not None:
            args = block_indices
        elif self.input_type == "block":
            args = [0]
        else:
            args = np.arange(self.coadds[0].nblock) # the defaults to running all columns
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
        pandas DataFrame
            Catalog catalog for the block.
        """
        # make multi-band list of blocks
        blks = [mosaic.outimages[block_to_run] for mosaic in self.coadds]
        return self._make_cat_block(blks) # run metadetection and produce catalog
        
    def _make_cat_block(self, blks):
        """
        Run MetaDetection over a single block or list of blocks. Each block in 
        list represents a different band.

         Parameters
        ----------
        blks : list of OutImage objects (multi-band)
    
        Returns
        -------
        pandas DataFrame
            Catalog catalog for the block.
        """

        # Make ngmix multi-band observation object
        mbobs = self.make_mbobs(blks)
        # Run Metadetection
        res = self.run_metadetect(mbobs)
        return self.construct_dataframe(blks, res) # Convert Metadetection results into a catalog


    # ----------------------------
    # ngmix observation builders
    # ----------------------------

    def make_mbobs(self, blks):
        """
        Build an ngmix MultiBandObsList from a list of blocks (each a different band).

        Parameters
        ----------
        blks : list of OutImage objects
        
        Returns
        -------
        mbobs : ngmix MultiBandObservation
        """
        mbobs = ngmix.MultiBandObsList()
        for blk in (blks if isinstance(blks, list) else [blks]): # loop over blocks of different bands
            obslist = self.make_ngmix_obs(blk)
            mbobs.append(obslist)
        return mbobs

    def make_ngmix_obs(self, blk):
        """
        Create an ngmix ObsList for a single block image.

        Parameters
        ----------
        blk : OutImage
            PyIMCOM block
        
        Returns
        -------
        obslist : ngmix Observation
        """
        img, img_jacobian, psf_img, noise_sigma = self.get_ngmix_data(blk)

        # Centers
        psf_cen = (psf_img.shape[0] - 1) / 2.0
        img_cen = (np.array([img.shape[0], img.shape[1]]) - 1) / 2.0

        # ngmix Jacobians
        psf_jac = ngmix.Jacobian(row=psf_cen, col=psf_cen, wcs=img_jacobian)
        img_jac = ngmix.Jacobian(row=img_cen[0], col=img_cen[1], wcs=img_jacobian)

        # Observations
        psf_obs = ngmix.Observation(image=psf_img, jacobian=psf_jac)
        obs = ngmix.Observation(
            image=img,
            jacobian=img_jac,
            weight=np.ones(img.shape, dtype=float) / noise_sigma**2,
            psf=psf_obs,
            ormask=np.zeros(img.shape, dtype=np.int32),
            bmask=np.zeros(img.shape, dtype=np.int32),
        )
        obslist = ngmix.ObsList()
        obslist.append(obs)
        return obslist
        
    def get_ngmix_data(self, blk):
        """
        Generate inputs needed to make ngmix Observation for a single block.

         Parameters
        ----------
        blk : OutImage object representing a single block (one band).
        
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
        image = blk.image

        # Build GalSim WCS and Jacobian 
        w = galsim.AstropyWCS(wcs=blk.wcs)
        img_jacobian = w.jacobian(image_pos=galsim.PositionD(w.wcs.wcs.crpix[0], w.wcs.wcs.crpix[1]))

        # Estimate background RMS using SEP
        bkg = sep.Background(image.astype(image.dtype.newbyteorder('=')))
        noise_sigma = bkg.globalrms

        # Draw PSF image
        psf_img = self.get_psf(blk, w)
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
        det_bands = self.driver_cfg['det_bands']
        shear_bands = self.driver_cfg['shear_bands']

        det_combs = None
        shear_combs = None
        if det_bands is not None:
            # Select only detection and shear bands from bands in coadds provided.
            det_idx = np.arange(len(self.bands))[np.isin(self.bands, det_bands)]
            det_combs = [det_idx]
        if shear_bands is not None:
            shear_idx = np.arange(len(self.bands))[np.isin(self.bands, shear_bands)]
            shear_combs = [shear_idx]

        # Run metadetect
        res = metadetect.do_metadetect(
            deepcopy(self.meta_cfg),
            mbobs=mbobs,
            rng=np.random.RandomState(seed=self.driver_cfg['mdet_seed']),
            det_band_combs=det_combs,
            shear_band_combs=shear_combs,
        )
        return res

    def run_shapes(self, mbobs):
        """
        CODE TO DEVELOP, NOT FUNCTIONAL RIGHT NOW. Add arguments to function as well
        Shapes using a different shape measurement algorithm.

        Parameters
        ----------
        mbobs : ngmix.MultiBandObsList
            Observations across one or more bands.

        Returns
        -------
        res : dict
            Metadetect results.
        """
        ## From config what are the detection/measurement bands
        det_bands = self.driver_cfg['det_bands']
        shear_bands = self.driver_cfg['shear_bands']

        ## This does some checks to only select detection/measurement bands from the available ones provided
        det_combs = None
        shear_combs = None
        if det_bands is not None:
            # Select only detection and shear bands from bands in coadds provided.
            det_idx = np.arange(len(self.bands))[np.isin(self.bands, det_bands)]
            det_combs = [det_idx]
        if shear_bands is not None:
            shear_idx = np.arange(len(self.bands))[np.isin(self.bands, shear_bands)]
            shear_combs = [shear_idx]

        # Run metacal on all bands
        odict  = ngmix.metacal.get_all_metacal(
                mbobs,  rng=mbobs.rng
                )
        
        reslists = {}
        ## Not sure how to change this for detection in some bands and measurement in others, I am used to metadetect which
        ## deals with it in the do_metadetect function.
        for key in odict:
            medsifier_det = sim_det.get_medsifier(obs=odict[key])
    
            cat_k = medsifier_det.cat
            seg_k = medsifier_det.seg
            ### Your shape measurement/fitting code here

            ###
    
        return res


    # ----------------------------
    # WCS / PSF helpers
    # ----------------------------

    def get_psf(self, blk, w):
        """
        Draw a PSF image for a block given.

        Parameters
        ----------
        blk : OutImage
            PyIMCOM block.
        w : galsim.BaseWCS
            GalSim WCS instance.

        Returns
        -------
        psf_img: np.ndarray
            PSF image array with shape (psf_img_size, psf_img_size).
        """
        psf = blk.psf
        psf_img = psf.drawImage(
            nx=self.driver_cfg['psf_img_size'],
            ny=self.driver_cfg['psf_img_size'],
            wcs=w,
        ).array
        return psf_img

    
    # ----------------------------
    # Catalog filtering / region selection
    # ----------------------------

    def get_bounded_region(self, blks, res):
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
        if self.driver_cfg['bound_size'] is None:
            bound_size = 0
        else:
            bound_size = self.driver_cfg['bound_size']

        img_size_x, img_size_y = blks[0].image.shape # (ny, nx)
        x = res["noshear"]["sx_col"]
        y = res["noshear"]["sx_row"]
        keep = (x > bound_size) & (x < img_size_x - bound_size) & (y > bound_size) & (y < img_size_y - bound_size)
        return keep

    # ----------------------------
    # Results construction
    # ----------------------------
    def construct_dataframe(self, blks, res):
        """
        Convert metadetect results into a catalog pandas DataFrame.
        Keeps only columns requested in driver_cfg['keepcols'] and applies edge mask.
        Also converts IMCOM fluxes to e-/cm^2/s, and computes RA/DEC for detections.

        Parameters
        ----------
        blks : list of OutImage objects
        
        res : dict
            Metadetect result dict.

        Returns
        -------
        pandas DataFrame
            Catalog of detected objects.

        """
        # World coordinates
        w = galsim.AstropyWCS(wcs=blks[0].wcs)
        x, y = res["noshear"]["sx_col"], res["noshear"]["sx_row"]
        ra_pos, dec_pos = w.toWorld(x, y, units='deg')

        # get masked region. All detections outside the bounded region are excluded from catalog
        keep_mask = self.get_bounded_region(blks,res)

        resultdict = {
            'ra_meta': ra_pos[keep_mask],
            'dec_meta': dec_pos[keep_mask],
        }

        # Select requested columns; convert flux-like columns
        for col in self.driver_cfg['keepcols']:
            key = f"{self.meta_cfg['model']}_{col}"
            if 'flux' in col:
                # for flux columns, first convert units. See imcom_flux_conv for why we do this.
                flux = res['noshear'][key]
                cf = np.asarray(flux)
                if cf.ndim == 1:
                    cf = cf[:, None] # make it (N, 1) instead of (N,)
                for i, band in enumerate(self.bands):
                    # flux is stored as a (N_det, N_band) array if more than one band
                    resultdict[f"{self.meta_cfg['model']}_{band}_{col}"] = cf[:, i][keep_mask]
            else:
                resultdict[key] = res['noshear'][key][keep_mask]

        return pd.DataFrame(resultdict)

    

    



