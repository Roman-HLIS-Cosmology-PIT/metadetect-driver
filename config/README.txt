Quick explanation of the configuration files read above

1. `METADETECT_CONFIG`: This is the configuration file that goes into Metadetect. For general info about how metadetect works, see the metadetect tutorial: https://github.com/Roman-HLIS-Cosmology-PIT/tutorial-metadetect/tree/main

2. `DEFAULT_DRIVER_CFG`: This is the configuration file for the driver we have made here. The arguments are the following:
 
    - `psf_img_size`: Size of the PSF image to make. The PSF is made inside the driver from the PyIMCOM configuration. [default = 151]
    - `bound_size`: Sets the maximum number of pixels from the edge for which we mask objects. Example, if bound_size = 100, all detections                   that are 100 pixels from the edge of the image are discarded from the catalog. [default = 100]
    - `mdet_seed`: Random seed that goes into Metadetect
    - `keepcols`: What columns from the Metadetect outputs to keep. Given as a list. [default = ['flags', 's2n', 'band_flux', 'band_flux_flags', 'T', 'psf_T']]
    - `det_bands`: What bands to use for detection. The default is to use all bands given. [default = None]
    - `shear_bands`: What bands to use for shape measurement. The default is to use all bands given. [default = None]
    - `layer`: What layer from the coadds to use (as a string).  [default = 'SCI']
    - `max_workers`: Maximum number of workers for parallelization of multiple block. Driver code allows to parallelize if multiple blocks                     need to be processed.[default = None] See https://docs.python.org/3/library/concurrent.futures.html for more details.
    - `chunksize`: Chunksize for the parallelization. See https://docs.python.org/3/library/concurrent.futures.html for more details.
    - `outdir`: The output directory to ave catalogs if you want them saved. [default = 1]