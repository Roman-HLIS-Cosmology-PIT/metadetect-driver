import numpy as np
from scipy.special import eval_hermitenorm, factorial

# --- Helper Function ---

def build_coordinate_grid(nx, ny, pixel_scale):
    """
    Creates 2D coordinate grids for the image plane.

    Args:
        nx (int): Number of pixels along the x-axis.
        ny (int): Number of pixels along the y-axis.
        pixel_scale (float): The scale of each pixel (e.g., arcseconds/pixel).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the X and Y
                                       coordinate grids.
    """
    x_ticks = np.linspace(
        -(nx - 1) / 2. * pixel_scale,
        (nx - 1) / 2. * pixel_scale,
        num=nx
    )
    y_ticks = np.linspace(
        -(ny - 1) / 2. * pixel_scale,
        (ny - 1) / 2. * pixel_scale,
        num=ny
    )
    # Note: 'xy' indexing means the output X grid has shape (ny, nx)
    X, Y = np.meshgrid(x_ticks, y_ticks, indexing='xy')
    return X, Y

# --- Basis Class Definition ---

class ShapeletBasis:
    """
    A simplified class for Cartesian Shapelet basis functions.
    See: https://arxiv.org/abs/astro-ph/0105178
    """
    def __init__(self, beta, nmax):
        """
        Initializes the Shapelet basis.

        Args:
            beta (float): The characteristic scale of the basis functions.
            nmax (int): The maximum order of the basis, where n = nx + ny.
        """
        if not isinstance(beta, (int, float)) or beta <= 0:
            raise ValueError('beta must be a positive number.')
        if not isinstance(nmax, int) or nmax < 0:
            raise ValueError('nmax must be a non-negative integer.')

        self.beta = beta
        self.nmax = nmax
        # Total number of basis functions for a given nmax
        self.N = (nmax + 1) * (nmax + 2) // 2
        self._setup_nxny_grid()

    def _setup_nxny_grid(self):
        """
        Creates a mapping from a single index 'n' to a pair of quantum
        numbers (nx, ny).
        """
        self.ngrid = -1 * np.ones((self.nmax + 1, self.nmax + 1), dtype=int)
        k = 0
        for n_sum in range(self.nmax + 1):
            for nx in range(n_sum + 1):
                ny = n_sum - nx
                self.ngrid[nx, ny] = k
                k += 1

    def n_to_NxNy(self, n):
        """Converts a single index n to its corresponding (nx, ny) pair."""
        if not (0 <= n < self.N):
            raise ValueError(f'Index n must be between 0 and {self.N - 1}.')
        Nx, Ny = np.where(self.ngrid == n)
        return int(Nx[0]), int(Ny[0])

    @staticmethod
    def _eval_single_function(x, y, beta, Nx, Ny):
        """
        Evaluates a single Cartesian Shapelet basis function of order (Nx, Ny)
        at all points in the provided (x, y) coordinate grids.
        """
        # Normalization constants
        norm_factor = 1. / (beta * np.sqrt(2**(Nx + Ny) * np.pi * factorial(Nx) * factorial(Ny)))

        # Hermite polynomials
        hermite_x = eval_hermitenorm(Nx, x / beta)
        hermite_y = eval_hermitenorm(Ny, y / beta)

        # Gaussian envelope
        gaussian = np.exp(-(x**2 + y**2) / (2 * beta**2))

        return norm_factor * hermite_x * hermite_y * gaussian

    def render_image(self, coefficients, image_config):
        """
        Renders an image from a linear combination of basis functions.

        Args:
            coefficients (list or np.ndarray): A vector of coefficients. The
                length must match the number of basis functions (self.N).
            image_config (dict): Dictionary specifying image format.
                Expected keys: 'nx', 'ny', 'pixel_scale'.

        Returns:
            np.ndarray: A 2D numpy array representing the rendered image.
        """
        if len(coefficients) != self.N:
            raise ValueError(
                f'Expected {self.N} coefficients for nmax={self.nmax}, '
                f'but received {len(coefficients)}.'
            )

        nx = image_config['nx']
        ny = image_config['ny']
        pixel_scale = image_config['pixel_scale']

        # 1. Create the coordinate grid for the image
        X, Y = build_coordinate_grid(nx, ny, pixel_scale)

        # 2. Initialize an empty image array
        #    The shape is (ny, nx) to match standard image/numpy row, col format
        final_image = np.zeros((ny, nx), dtype=float)

        # 3. Sum the weighted basis functions
        for n, coeff in enumerate(coefficients):
            if coeff == 0:
                continue
            # Get the (nx, ny) orders for the nth basis function
            Nx, Ny = self.n_to_NxNy(n)
            # Evaluate the basis function on the grid and add it to the image
            basis_func = self._eval_single_function(X, Y, self.beta, Nx, Ny)
            final_image += coeff * basis_func

        return final_image

# --- Main User-Facing Function ---

def render_basis_model(coefficients, config):
    """
    Renders an image from a vector of basis function coefficients using a
    specified configuration.

    This is the primary function for the user.

    Args:
        coefficients (list or np.ndarray): The vector of weights for the
            linear combination of basis functions.
        config (dict): A configuration dictionary with two keys:
            'image_params': A dict with 'nx', 'ny', 'pixel_scale'.
            'basis_params': A dict with 'type', 'beta', 'nmax'.

    Returns:
        np.ndarray: The rendered 2D image as a numpy array.
    """
    basis_type = config['basis_params'].get('type', 'shapelets').lower()

    if basis_type == 'shapelets':
        basis = ShapeletBasis(
            beta=config['basis_params']['beta'],
            nmax=config['basis_params']['nmax']
        )
    else:
        raise NotImplementedError(f"Basis type '{basis_type}' is not supported.")

    # Call the render method on the instantiated basis object
    image = basis.render_image(coefficients, config['image_params'])

    return image

# --- Example Usage ---

if __name__ == '__main__':
    # 1. Define the configuration for the image and basis set
    #    This dictionary is the central control point.
    config = {
        'image_params': {
            'nx': 100,          # Image width in pixels
            'ny': 100,          # Image height in pixels
            'pixel_scale': 0.1  # e.g., arcseconds per pixel
        },
        'basis_params': {
            'type': 'shapelets',
            'nmax': 4,          # Maximum order, n = nx + ny
            'beta': 0.8         # Characteristic scale in same units as pixel_scale
        }
    }

    # 2. Define the coefficients for the basis functions.
    #    Let's create an image with a non-zero coefficient for the first
    #    (Gaussian) and fifth (a "donut" shape) basis functions.
    total_funcs = (config['basis_params']['nmax'] + 1) * (config['basis_params']['nmax'] + 2) // 2
    coeffs = np.zeros(total_funcs)
    coeffs[0] = 0.0   # Weight for the (nx=0, ny=0) Gaussian component
    coeffs[7] = -2.0  # Weight for the (nx=1, ny=1) component

    # 3. Call the main function to generate the image
    model_image = render_basis_model(coeffs, config)

    # 4. (Optional) Print and display the result
    print(f"Successfully generated an image with shape: {model_image.shape}")
    print(f"Image data type: {model_image.dtype}")

    try:
        import matplotlib.pyplot as plt
        print("Displaying the generated image...")
        plt.figure(figsize=(8, 6))
        plt.imshow(model_image, origin='lower', cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.title(f"Rendered Shapelet Model (nmax={config['basis_params']['nmax']})")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping image display.")
        print("To visualize the output, please install it: pip install matplotlib")