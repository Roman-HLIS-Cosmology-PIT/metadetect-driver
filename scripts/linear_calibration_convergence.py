import galsim
import numpy as np
import math
import matplotlib.pyplot as plt

import ipdb

real_galaxy_path = "/opt/homebrew/Caskroom/miniforge/base/envs/general/lib/python3.13/site-packages/galsim/share/COSMOS_25.2_training_sample.tar.gz"

def draw_blended_scene(catalog, parameters, shear_step=(0.01,0.0), rng=None ):
    # pick two galaxies from the catalog.
    gal1Pars = parameters['galaxy1']
    gal2Pars = parameters['galaxy2']
    imPars = parameters['image']

    gal1 = catalog.makeGalaxy(index=gal1Pars['index'],gal_type=gal1Pars['type']).withFlux(gal1Pars['flux'])
    theta1 = 2.*math.pi * rng() * galsim.radians
    gal1 = gal1.rotate(theta1)

    gal2 = catalog.makeGalaxy(index=gal2Pars['index'],gal_type=gal2Pars['type']).withFlux(gal2Pars['flux'])
    theta2 = 2.*math.pi * rng() * galsim.radians
    gal2 = gal2.rotate(theta2)

    mixture_unsheared = gal1.shift(-gal1Pars['offset'],0.) + gal2.shift(gal2Pars['offset'],0.0)
    mixture_sheared = gal1.shear(g1=shear_step[0],g2=shear_step[1]).shift(-gal1Pars['offset'],0.) + gal2.shift(gal2Pars['offset'],0.0)

    # Then draw this.
    mixture_unsheared_convolved = galsim.Convolve([mixture_unsheared,imPars['psf']])
    mixture_sheared_convolved = galsim.Convolve([mixture_sheared,imPars['psf']])
    image_sheared = mixture_sheared_convolved.drawImage(scale=imPars['pixelscale'])
    image_unsheared = mixture_unsheared_convolved.drawImage(scale=imPars['pixelscale'])
    return {'sheared':image_sheared,
            'unsheared':image_unsheared}



def main(nsim = 1000,random_seed=314):
    
    parameters = {'galaxy1':{'index':0,
                             'flux':100.0,
                             'type':'parametric',
                             'offset':0.0},
                  'galaxy2':{'index':10,
                             'flux':5,
                             'type':'parametric',
                             'offset':1.0},
                  'image':{'psf':galsim.Gaussian(sigma=0.2),
                           'pixelscale':0.1}}

    rgc  = galsim.RealGalaxyCatalog()
    catalog = galsim.COSMOSCatalog(file_name=rgc.getFileName())
    rng = galsim.UniformDeviate(random_seed)
    scene_images = draw_blended_scene(catalog, parameters, shear_step=(0.01,0.0),rng=rng)

    from matplotlib.colors import SymLogNorm
    vmin,vmax = np.min(scene_images['sheared'].array),np.max(scene_images['sheared'].array)
    norm = SymLogNorm(linthresh=1e-3,vmin=vmin,vmax=vmax)
    fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(18,6))
    im = ax1.imshow(scene_images['sheared'].array,origin='lower',norm=norm)
    ax2.imshow(scene_images['unsheared'].array,origin='lower',norm=norm)
    ax3.imshow((scene_images['sheared'] - scene_images['unsheared']).array,origin='lower',norm=norm)
    fig.colorbar(im, ax=[ax1, ax2, ax3])
    #plt.tight_layout()
    plt.show()
    ipdb.set_trace()

if __name__ == '__main__':
    main()

    