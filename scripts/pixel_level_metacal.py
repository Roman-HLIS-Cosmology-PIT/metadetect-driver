import numpy as np

import anacal
import galsim

def make_data_metacal_simple(model="gauss", hlr=0.5,
	g1=0.0, g2=0.0, psf_model="gauss", psf_fwhm=0.6,
	do_shift=False, image_size=50, pixel_scale=0.2):
	if model == "gauss":
		gal = galsim.Gaussian(fwhm=hlr)
	elif model == "exp":
		gal = galsim.Exponential(fwhm=hlr)

	gal = gal.shear(g1=g1, g2=g2)
	if do_shift:
		rng = np.random.RandomState(42)
		dy, dx = rng.uniform(
			low=-pixel_scale / 2, high=pixel_scale / 2, size=2)
	else:
		dx = 0
		dy = 0
	gal = gal.shift(dx=dx, dy=dy)

	if psf_model == "gauss":
		psf = galsim.Gaussian(fwhm=psf_fwhm)
	elif psf_model == "moffat":
		psf = galsim.Moffat(fwhm=psf_fwhm, beta=2.5)
	elif psf_model is None:
		psf = galsim.DeltaFunction()

	obj = galsim.Convolve([gal, psf])

	img = obj.drawImage(
		nx=image_size,
		ny=image_size,
		scale=pixel_scale).array

	if psf_model is None:
		psf_img = None
	else:
		psf_img = psf.drawImage(
			nx=image_size,
			ny=image_size,
			scale=pixel_scale,
		).array

	return img, psf_img, (dx, dy)


def make_data_metacal_catalog(gal_image,
	psf_fwhm, pixel_scale, image_size,
	g1=0.0, g2=0.0,
	psf_model="gauss"):

	if psf_model == "gauss":
		psf = galsim.Gaussian(fwhm=psf_fwhm)
	elif psf_model == "moffat":
		psf = galsim.Moffat(fwhm=psf_fwhm, beta=2.5)
	elif psf_model is None:
		psf = galsim.DeltaFunction()

	# 1. Create interpolated image
	gal_interp = galsim.InterpolatedImage(galsim.Image(gal_image, scale=pixel_scale))

	# 2. Shear the image
	shear = galsim.Shear(g1=g1, g2=g2)
	gal_shear = gal_interp.shear(shear)

	# 3. Convolve the image
	gal_conv = galsim.Convolve([gal_shear, psf])

	# Draw PSF
	psf = psf.drawImage(nx=image_size[1], ny=image_size[0], 
						 scale=pixel_scale).array

	# Since the interpolated image already includes the pixelisation we use `no_pixel` when drawing it
	gal = gal_conv.drawImage(nx=image_size[1], ny=image_size[0], 
							   scale=pixel_scale, method='no_pixel').array

	return gal, psf



def create_perf_images(gal_image, noise, config, psf_obs):
	image_size = config['image_size']
	pixel_scale = config['pixel_scale']
	hlr = config['hlr']
	USE_MAKE_DATA_METACAL_SIMPLE = config['USE_MAKE_DATA_METACAL_SIMPLE']

	gal_mcal_perf, psf_mcal_perf = {}, {}

	for key in CF_dict.keys():
		g1, g2 = CF_dict[key]

		shear = galsim.Shear(g1=g1, g2=g2)
		dilation = 1.0 + 2.0*np.sqrt(g1**2 + g2**2)

		if USE_MAKE_DATA_METACAL_SIMPLE:
			this_gal_mcal_perf, this_psf_mcal_perf, _ = make_data_metacal_simple(
				g1=g1, g2=g2,
				hlr=hlr,
				image_size=image_size,
				pixel_scale=pixel_scale,
				psf_fwhm=psf_obs*dilation,
				do_shift=False)

		else:
			this_gal_mcal_perf, this_psf_mcal_perf = make_data_metacal_catalog(gal_image, psf_obs*dilation,
																	  pixel_scale=pixel_scale, image_size=image_size,
																	  g1=g1, g2=g2)

		gal_mcal_perf[key] = this_gal_mcal_perf + noise
		psf_mcal_perf[key] = this_psf_mcal_perf

	return gal_mcal_perf, psf_mcal_perf


def create_CF_images(gal_image, noise, config, psf_obs_fwhm, psf_deconv_fwhm, psf_deconv_model):
	USE_MAKE_DATA_METACAL_SIMPLE = config['USE_MAKE_DATA_METACAL_SIMPLE']
	hlr = config['hlr']
	pixel_scale = config['pixel_scale']
	image_size = config['image_size']

	gal_mcal_all = {}
	for key in CF_dict.keys():

		if USE_MAKE_DATA_METACAL_SIMPLE:
			
			gal, _, _ = make_data_metacal_simple(
				hlr=hlr,
				psf_fwhm=psf_obs_fwhm,
				pixel_scale=pixel_scale,
				image_size=image_size)

			# Get the wrong PSF model
			_, psf, _ = make_data_metacal_simple(
				hlr=hlr,
				psf_fwhm=psf_deconv_fwhm,
				psf_model=psf_deconv_model,
				pixel_scale=pixel_scale,
				image_size=image_size)

		else:
			gal, _ = make_data_metacal_catalog(gal_image, psf_obs_fwhm,
												pixel_scale=pixel_scale, image_size=image_size,
												g1=g1, g2=g2)

			_, psf = make_data_metacal_catalog(gal_image, psf_deconv_fwhm, psf_deconv_model,
												pixel_scale=pixel_scale, image_size=image_size,
												g1=g1, g2=g2)

		####------------------------ 1. Deconvolve Image ------------------------####
		
		# Interpolate the galaxy image and the PSF
		gal_interp = galsim.InterpolatedImage(galsim.Image(gal, scale=pixel_scale))
		psf_interp = galsim.InterpolatedImage(galsim.Image(psf, scale=pixel_scale))

		# Deconvolve the galaxy image from the PSF
		psf_deconv = galsim.Deconvolve(psf_interp)
		gal_deconv = galsim.Convolve(gal_interp, psf_deconv)
		
		####------------------------ 2. Get dilated PSF ------------------------####
		# Get pixelization
		pixel = galsim.Pixel(pixel_scale)

		# New PSF
		psf_reconv = galsim.Gaussian(fwhm=psf_deconv_fwhm)

		# Dilate the PSF
		g1, g2 = CF_dict[key]
		shear = galsim.Shear(g1=g1, g2=g2)
		g = np.sqrt(shear.g1**2 + shear.g2**2)
		dilation = 1.0 + 2.0*g
		psf_reconv = psf_reconv.dilate(dilation)
		
		# Convolve the new PSF with the pixelization
		psf_reconv_pixel = galsim.Convolve(psf_reconv, pixel)

		# Dilate the PSF

		psf_mcal = psf_reconv_pixel.drawImage(
		nx=psf.shape[1],
		ny=psf.shape[0],
		scale=pixel_scale,
		method='no_pixel').array
		
		####------------------------ 3. Shear Image and reconvolve ------------------------####
		g1, g2 = CF_dict[key]
		shear = galsim.Shear(g1=g1, g2=g2)

		# Apply Shear
		gal_shear = gal_deconv.shear(shear)
		
		# Reconvolve
		gal_new = galsim.Convolve(gal_shear, psf_reconv_pixel)
		
		# Draw the object in 'no_pixel' mode as it would be included in the PSF already
		gal_mcal = gal_new.drawImage(
			nx=gal.shape[1],
			ny=gal.shape[0],
			scale=pixel_scale,
			method='no_pixel',
		).array

		gal_mcal_all[key] = gal_mcal + noise

	return gal_mcal_all, psf


def get_config(pixel_scale, gal_image=None, noise_sigma=1e-5, image_size=50, hlr=0.5):
	if gal_image is not None:
		image_size = gal_image.shape
		hlr = None
		USE_MAKE_DATA_METACAL_SIMPLE = False

	else:
		USE_MAKE_DATA_METACAL_SIMPLE = True

	return {
		'image_size': image_size,
		'pixel_scale': pixel_scale,
		'noise_sigma': noise_sigma,
		'hlr': hlr,
		'USE_MAKE_DATA_METACAL_SIMPLE': USE_MAKE_DATA_METACAL_SIMPLE
	}

def run_anacal(gal_img, psf_img, scale, sigma_arcsec, noise_variance = 1e-5):
	npix = len(psf_img[0])
	noise = np.zeros((npix, npix))
	fpTask = anacal.fpfs.FpfsTask(npix = npix, psf_array = psf_img, pixel_scale = scale, 
								  noise_variance = noise_variance, sigma_arcsec=sigma_arcsec)
	mms =  fpTask.run(gal_array = gal_img, psf= psf_img)
	return mms

def get_shear_bias(config, CF_images, perfect_images):
	pixel_scale = config['pixel_scale']
	delta_e1 = {}
	delta_e2 = {}

	for key in CF_dict.keys():
		diff_image =  CF_images[key] - perfect_images[key]
		mms = run_anacal(diff_image, psf_image, pixel_scale, CF_PSF_FWHM)

		delta_e1[key] = mms['data']['m22c'][0]
		delta_e2[key] = mms['data']['m22s'][0]
	
	R11 = (delta_e1['1p'] - delta_e1['1m'])/2/g
	R22 = (delta_e2['2p'] - delta_e2['2m'])/2/g

	print(f'Delta R_11: {R11}')
	print(f'Delta R_22: {R22}')

	C1 = (delta_e1['1p'] + delta_e1['1m'])/2
	C2 = (delta_e2['2p'] + delta_e2['2m'])/2

	print(f'Delta C_g1: {C1}')
	print(f'Delta C_g2: {C2}')

def get_bias_analytic(PSF_obs, PSF_reconv, gal_size):
	return 2* (PSF_reconv-PSF_obs)/PSF_obs * (PSF_obs/gal_size)**2


def get_noise_realization(config):
	image_size = config['image_size']
	noise_sigma = config['noise_sigma']

	seed = 42
	rng = np.random.RandomState(seed)
	noise = rng.normal(size=(image_size, image_size)) * noise_sigma
	# noise = noise.drawImage(nx=image_size, ny=image_size, scale=pixel_scale).array

	return noise


# Shears for making counterfactual images
CF_dict = {'1p': (0.01, 0), 
			'1m': (-0.01, 0), 
			'2p': (0, 0.01),
			'2m': (0, -0.01)}

# Make sure |g|=0.01 for the above combinations
shear = galsim.Shear(g1=0.01, g2=0.0)
g = np.sqrt(shear.g1**2 + shear.g2**2)
dilation = 1.0 + 2.0*g

####################################################
###----------------- User Defined ---------------###
TRUE_PSF_MODEL = 'gauss'
TRUE_PSF_FWHM = 0.5

# PSF for generating counterfactual images
# Assumed to be a (round) Gaussian
CF_PSF_MODEL = 'gauss'
# CF_PSF_MODEL = 'moffat'
CF_PSF_FWHM = 0.55
####################################################
CONFIG = get_config(pixel_scale=0.2)
noise = get_noise_realization(CONFIG)

perfect_images, _ = create_perf_images(gal_image=None, noise=noise, config=CONFIG, psf_obs=TRUE_PSF_FWHM)
CF_images, psf_image = create_CF_images(gal_image=None, noise=noise, config=CONFIG, 
										psf_obs_fwhm=TRUE_PSF_FWHM, psf_deconv_fwhm=CF_PSF_FWHM,
										psf_deconv_model=CF_PSF_MODEL)
get_shear_bias(CONFIG, CF_images, perfect_images)

print(get_bias_analytic(TRUE_PSF_FWHM, CF_PSF_FWHM, CONFIG['image_size']))