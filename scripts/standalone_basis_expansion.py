import numpy as np
from math import factorial  # Corrected import
from scipy.special import eval_hermitenorm, genlaguerre
import galsim as gs  # Assume GalSim is always installed

# --- Helper Function ---

def build_coordinate_grid(nx, ny, pixel_scale):
    """Creates 2D coordinate grids for the image plane."""
    x_ticks = np.linspace(
        -(nx - 1) / 2. * pixel_scale, (nx - 1) / 2. * pixel_scale, num=nx
    )
    y_ticks = np.linspace(
        -(ny - 1) / 2. * pixel_scale, (ny - 1) / 2. * pixel_scale, num=ny
    )
    X, Y = np.meshgrid(x_ticks, y_ticks, indexing='xy')
    return X, Y

# --- Basis Class Definition ---

class ShapeletBasis:
    """
    A unified class for Cartesian and Polar Shapelet basis functions,
    with optional PSF convolution.
    """
    def __init__(self, beta, nmax, mode='cartesian', psf=None):
        """
        Initializes the Shapelet basis.

        Args:
            beta (float): The characteristic scale of the basis functions.
            nmax (int): The maximum quantum number.
            mode (str): The basis mode, 'cartesian' or 'polar'.
            psf (galsim.GSObject, optional): A GalSim object representing the
                Point Spread Function. If provided, the final rendered image
                will be convolved with this PSF. Defaults to None.
        """
        if not isinstance(beta, (int, float)) or beta <= 0:
            raise ValueError('beta must be a positive number.')
        if not isinstance(nmax, int) or nmax < 0:
            raise ValueError('nmax must be a non-negative integer.')
        if mode not in ['cartesian', 'polar']:
            raise ValueError("mode must be either 'cartesian' or 'polar'.")
        if psf and not isinstance(psf, gs.GSObject):
            raise TypeError("psf must be a galsim.GSObject.")

        self.beta = beta
        self.nmax = nmax
        self.mode = mode
        self.psf = psf
        self.N = (nmax + 1) * (nmax + 2) // 2
        
        if self.mode == 'cartesian':
            self._setup_cartesian_grid()
        else:
            self._setup_polar_grid()

    ## --------------- Setup and Indexing Methods ---------------
    def _setup_cartesian_grid(self):
        """Creates a mapping from index 'k' to (nx, ny)."""
        self._ngrid = -1 * np.ones((self.nmax + 1, self.nmax + 1), dtype=int)
        k = 0
        for n_sum in range(self.nmax + 1):
            for nx in range(n_sum + 1):
                ny = n_sum - nx
                self._ngrid[nx, ny] = k
                k += 1

    def _setup_polar_grid(self):
        """Creates a mapping from index 'k' to (n, m)."""
        self._polar_map = []
        for n in range(self.nmax + 1):
            for m in range(-n, n + 1, 2):
                self._polar_map.append({'n': n, 'm': m})

    def get_indices(self, k):
        """Gets the quantum numbers for the k-th basis function."""
        if not (0 <= k < self.N):
            raise IndexError(f'Index k must be between 0 and {self.N - 1}.')
        if self.mode == 'cartesian':
            Nx, Ny = np.where(self._ngrid == k)
            return int(Nx[0]), int(Ny[0])
        else:
            return self._polar_map[k]

    ## --------------- Basis Function Evaluation Methods ---------------
    @staticmethod
    def _eval_cartesian_function(x, y, beta, Nx, Ny):
        """Evaluates a single Cartesian Shapelet basis function."""
        norm_factor = 1. / (beta * np.sqrt(2**(Nx + Ny) * np.pi * factorial(Nx) * factorial(Ny)))
        hermite_x = eval_hermitenorm(Nx, x / beta)
        hermite_y = eval_hermitenorm(Ny, y / beta)
        gaussian = np.exp(-(x**2 + y**2) / (2 * beta**2))
        return norm_factor * hermite_x * hermite_y * gaussian

    @staticmethod
    def _eval_polar_function(x, y, beta, n, m):
        """Evaluates a single complex Polar Shapelet basis function."""
        p = int((n - abs(m)) / 2)
        norm = np.sqrt(factorial(p) / (np.pi * beta**2 * factorial(p + abs(m))))
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        r_scaled_sq = (r / beta)**2
        laguerre = genlaguerre(p, abs(m))(r_scaled_sq)
        radial_part = norm * (r / beta)**abs(m) * laguerre * np.exp(-r_scaled_sq / 2)
        angular_part = np.exp(1j * m * phi)
        return radial_part * angular_part

    ## --------------- Convolution and Rendering ---------------
    def _convolve_image(self, image_array, image_config):
        """Convolves a numpy image array with the stored PSF."""
        pixel_scale = image_config['pixel_scale']
        
        # 1. Convert the numpy array to a GalSim InterpolatedImage
        model_image = gs.Image(image_array, scale=pixel_scale)
        model_obj = gs.InterpolatedImage(model_image)
        
        # 2. Convolve with the PSF
        convolved_obj = gs.Convolve([self.psf, model_obj])
        
        # 3. Draw the convolved object back to a new image
        return convolved_obj.drawImage(
            scale=pixel_scale,
            nx=image_config['nx'],
            ny=image_config['ny']
        ).array

    def render_image(self, coefficients, image_config):
        """Renders an image from a linear combination of basis functions."""
        if len(coefficients) != self.N:
            raise ValueError(f'Expected {self.N} coefficients, but received {len(coefficients)}.')

        X, Y = build_coordinate_grid(image_config['nx'], image_config['ny'], image_config['pixel_scale'])
        unconvolved_image = np.zeros((image_config['ny'], image_config['nx']), dtype=float)

        # Step 1: Build the unconvolved model image
        if self.mode == 'cartesian':
            for k, coeff in enumerate(coefficients):
                if coeff == 0: continue
                Nx, Ny = self.get_indices(k)
                basis_func = self._eval_cartesian_function(X, Y, self.beta, Nx, Ny)
                unconvolved_image += coeff * basis_func
        else: # Polar mode
            for k, coeff in enumerate(coefficients):
                if coeff == 0: continue
                indices = self.get_indices(k)
                complex_basis_func = self._eval_polar_function(X, Y, self.beta, indices['n'], indices['m'])
                unconvolved_image += coeff * complex_basis_func.real
        
        # Step 2: If a PSF is provided, convolve the final image
        if self.psf:
            return self._convolve_image(unconvolved_image, image_config)
        else:
            return unconvolved_image

# --- Main User-Facing Function ---

def render_basis_model(coefficients, config):
    """Renders an image from a vector of basis function coefficients."""
    basis_params = config['basis_params']
    basis = ShapeletBasis(
        beta=basis_params['beta'],
        nmax=basis_params['nmax'],
        mode=basis_params.get('mode', 'cartesian'),
        psf=basis_params.get('psf', None) # Pass PSF to the constructor
    )
    return basis.render_image(coefficients, config['image_params'])

# --- Example Usage ---

if __name__ == '__main__':
    # =========================================================
    # == Example: Cartesian Mode with and without PSF         ==
    # =========================================================
    print("--- Running Polar Shapelet Example ---")
    
    # Create a simple PSF model (a Gaussian)
    # The size (fwhm) should be in the same units as pixel_scale (e.g., arcsec)
    psf_model = gs.Gaussian(fwhm=.50)
    print(f"Created a sample PSF: Gaussian with FWHM = {psf_model.fwhm:.2f}")

    # --- Configuration ---
    # We will use the same config, but add the 'psf' key for the convolved version
    config = {
        'image_params': { 'nx': 128, 'ny': 128, 'pixel_scale': 0.1 },
        'basis_params': { 'mode': 'polar', 'nmax': 10, 'beta': 0.3 }
    }
    
    nmax = config['basis_params']['nmax']
    total_funcs = (nmax + 1) * (nmax + 2) // 2
    coeffs = np.zeros(total_funcs)
    coeffs[0] = 0.0   # (nx=0, ny=0) Gaussian core
    coeffs[60] = 3.0  # (nx=4, ny=0) Some higher-order structure
    
    # --- Rendering ---
    # 1. Render the raw, unconvolved model
    print("Rendering unconvolved model...")
    unconvolved_model = render_basis_model(coeffs, config)

    # 2. Render the PSF-convolved model
    print("Rendering PSF-convolved model...")
    config['basis_params']['psf'] = psf_model
    convolved_model = render_basis_model(coeffs, config)

    # =========================================================
    # == Plotting                                           ==
    # =========================================================
    try:
        import matplotlib.pyplot as plt
        print("\nDisplaying the results...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # Plot Unconvolved
        im0 = axes[0].imshow(unconvolved_model, origin='lower', cmap='viridis')
        fig.colorbar(im0, ax=axes[0], label='Intensity', shrink=0.8)
        axes[0].set_title("Unconvolved Model")
        axes[0].set_xlabel("X pixel"); axes[0].set_ylabel("Y pixel")
        
        # Plot PSF
        psf_image = psf_model.drawImage(nx=32, ny=32, scale=config['image_params']['pixel_scale']).array
        im1 = axes[1].imshow(psf_image, origin='lower', cmap='viridis')
        fig.colorbar(im1, ax=axes[1], label='Intensity', shrink=0.8)
        axes[1].set_title("PSF Kernel")
        axes[1].set_xlabel("X pixel"); axes[1].set_ylabel("Y pixel")

        # Plot Convolved
        vmax = np.max(unconvolved_model) # Use same color scale for comparison
        im2 = axes[2].imshow(convolved_model, origin='lower', cmap='viridis', vmax=vmax)
        fig.colorbar(im2, ax=axes[2], label='Intensity', shrink=0.8)
        axes[2].set_title("Convolved Model")
        axes[2].set_xlabel("X pixel"); axes[2].set_ylabel("Y pixel")
            
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping image display.")