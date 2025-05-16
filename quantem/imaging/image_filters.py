from collections.abc import Sequence
from typing import List, Optional, Union, Tuple, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import os

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.utils.imaging_utils import fourier_cropping
from quantem.core.visualization import show_2d

class FourierFilter(AutoSerialize):
    """
    FourierFilter provides Fourier-space filtering for a single 2D image, enabling q-vector specific
    filtering for crystalline or periodic structures.

    This class supports filtering around specific q-vectors in Fourier space, which is useful
    for isolating specific periodicities in materials science images.

    Features
    --------
    - Load data from arrays or files
    - Filter around specific q-vectors with adjustable radius and filter shape
    - Customize filter parameters including filter power and radius
    - Visualize results in both spatial and frequency domains
    - Serialize state with `.save()` and restore with `.load()`

    Parameters (via `from_data` or `from_file`)
    ----------
    image : 2D array or Dataset2d
        The image to filter.
    q_vectors : list of tuples, default None
        List of q-vectors to filter around, specified as (qx, qy) coordinates in Fourier space.
        If None, no specific q-vector filtering is applied.
    filter_radius : float, default 6.0
        Radius of the filter in Fourier space.
    filter_power : float, default 2.0
        Power parameter controlling the steepness of the filter transition.
    filter_sign : int, default 1
        Sign of the filter (1 for positive, -1 for negative).
    apply_tuckey_window : bool, default True
        Whether to apply a Tukey window before FFT to reduce edge effects.
    """

    _token = object()

    def __init__(
        self,
        image: Dataset2d,
        q_vectors: List[Tuple[float, float]],
        filter_radius: float,
        filter_power: float,
        filter_sign: int,
        apply_tuckey_window: bool,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use FourierFilter.from_data() or .from_file() to instantiate this class."
            )

        self._image = image
        self._q_vectors = q_vectors
        self._filter_radius = filter_radius
        self._filter_power = filter_power
        self._filter_sign = filter_sign
        self._apply_tuckey_window = apply_tuckey_window
        
        # Initialize filtered_images as None until apply_filter is called
        self.filtered_images = None
        self.filter_masks = None
        self.fourier_coords = None

    @classmethod
    def from_file(
        cls,
        file_path: str,
        q_vectors: List[Tuple[float, float]] = None,
        filter_radius: float = 6.0,
        filter_power: float = 2.0,
        filter_sign: int = 1,
        apply_tuckey_window: bool = True,
        file_type: str | None = None,
    ) -> "FourierFilter":
        image = Dataset2d.from_file(file_path, file_type=file_type)
        return cls.from_data(
            image, q_vectors, filter_radius, filter_power, filter_sign, apply_tuckey_window
        )

    @classmethod
    def from_data(
        cls,
        image: Union[Dataset2d, NDArray],
        q_vectors: List[Tuple[float, float]] = None,
        filter_radius: float = 6.0,
        filter_power: float = 2.0,
        filter_sign: int = 1,
        apply_tuckey_window: bool = True,
    ) -> "FourierFilter":
        # Convert numpy array to Dataset2d if needed
        if isinstance(image, np.ndarray):
            image = Dataset2d.from_array(image)
        elif not isinstance(image, Dataset2d):
            raise TypeError("Image must be a Dataset2d or a 2D numpy array")
        
        # Default q_vectors if None
        if q_vectors is None:
            q_vectors = []

        return cls(
            image=image,
            q_vectors=q_vectors,
            filter_radius=filter_radius,
            filter_power=filter_power,
            filter_sign=filter_sign,
            apply_tuckey_window=apply_tuckey_window,
            _token=cls._token,
        )

    # --- Properties ---
    @property
    def image(self) -> Dataset2d:
        return self._image

    @image.setter
    def image(self, value: Union[Dataset2d, NDArray]):
        if isinstance(value, np.ndarray):
            self._image = Dataset2d.from_array(value)
        elif isinstance(value, Dataset2d):
            self._image = value
        else:
            raise TypeError("Image must be a Dataset2d or a 2D numpy array")

    @property
    def q_vectors(self) -> List[Tuple[float, float]]:
        return self._q_vectors

    @q_vectors.setter
    def q_vectors(self, value: List[Tuple[float, float]]):
        if not isinstance(value, list):
            raise ValueError("q_vectors must be a list of (qx, qy) tuples")
        for q in value:
            if not (isinstance(q, (list, tuple)) and len(q) == 2):
                raise ValueError("Each q-vector must be a tuple or list of two values (qx, qy)")
        self._q_vectors = [(float(q[0]), float(q[1])) for q in value]

    @property
    def filter_radius(self) -> float:
        return self._filter_radius

    @filter_radius.setter
    def filter_radius(self, value: float):
        if value <= 0:
            raise ValueError("filter_radius must be positive")
        self._filter_radius = float(value)

    @property
    def filter_power(self) -> float:
        return self._filter_power

    @filter_power.setter
    def filter_power(self, value: float):
        if value <= 0:
            raise ValueError("filter_power must be positive")
        self._filter_power = float(value)

    @property
    def apply_tuckey_window(self) -> bool:
        return self._applytuckey_window

    @apply_tuckey_window.setter
    def apply_tuckey_window(self, value: bool):
        self._apply_tuckey_window = bool(value)

    def _make_fourier_coords(self, shape: Tuple[int, int]) -> Tuple[NDArray, NDArray]:
        """
        Generate Fourier space coordinates for the given shape.
        
        Parameters
        ----------
        shape : tuple of int
            Shape of the image in the spatial domain
            
        Returns
        -------
        tuple of NDArray
            (qx_grid, qy_grid) - Meshgrid of Fourier coordinates
        """
        # Generate Fourier coordinates for each dimension
        qx = np.fft.fftfreq(shape[0])
        qy = np.fft.fftfreq(shape[1])
        
        # Create meshgrid
        qy_grid, qx_grid = np.meshgrid(qy, qx)
        
        return qx_grid, qy_grid

    def _create_tukey_window(self, shape: Tuple[int, int], alpha: float = 0.5) -> NDArray:
        """
        Create a 2D Tukey window for the given shape.
        
        Parameters
        ----------
        shape : tuple of int
            Shape of the window
        alpha : float, default 0.5
            Parameter controlling the shape of the window
            
        Returns
        -------
        NDArray
            2D Tukey window
        """
        # Create 1D Tukey windows
        window_x = np.ones(shape[0])
        window_y = np.ones(shape[1])
        
        # Apply taper to edges
        nx = shape[0]
        ny = shape[1]
        
        # X direction
        taper_x = int(alpha * nx / 2)
        if taper_x > 0:
            window_x[:taper_x] = 0.5 * (1 + np.cos(np.pi * (np.arange(taper_x) / taper_x - 1)))
            window_x[-taper_x:] = 0.5 * (1 + np.cos(np.pi * (np.arange(taper_x) / taper_x)))
        
        # Y direction
        taper_y = int(alpha * ny / 2)
        if taper_y > 0:
            window_y[:taper_y] = 0.5 * (1 + np.cos(np.pi * (np.arange(taper_y) / taper_y - 1)))
            window_y[-taper_y:] = 0.5 * (1 + np.cos(np.pi * (np.arange(taper_y) / taper_y)))
        
        # Create 2D window
        return np.outer(window_x, window_y)

    def _create_filter_mask(self, q_vector: Tuple[float, float], qx_grid: NDArray, qy_grid: NDArray) -> NDArray:
        """
        Create a frequency domain filter mask for a specific q-vector.
        
        Parameters
        ----------
        q_vector : tuple of float
            (qx, qy) coordinates of the q-vector to filter around
        qx_grid : NDArray
            X-coordinates in Fourier space
        qy_grid : NDArray
            Y-coordinates in Fourier space
            
        Returns
        -------
        NDArray
            2D filter mask in frequency domain
        """
        # Calculate squared distance from q-vector
        qra2 = (qx_grid - q_vector[0])**2 + (qy_grid - q_vector[1])**2
        
        # Create Butterworth-like filter mask
        mask = 1.0 / (1.0 + (qra2 ** (self.filter_power / 2.0)) / (self.filter_radius ** self.filter_power))
        
        return mask

    def apply_filter(
        self,
        combine_filters: bool = False,
        show_images: bool = True,
    ):
        """
        Apply Fourier filtering around specified q-vectors to the input image.
        
        Parameters
        ----------
        combine_filters : bool, default False
            If True, combine all q-vector filters into a single output.
            If False, generate separate outputs for each q-vector.
        show_images : bool, default True
            Whether to display the filtered images after processing.
            
        Returns
        -------
        self : FourierFilter
            Returns self for method chaining.
        """
        if not self.q_vectors:
            raise ValueError("No q-vectors specified. Use q_vectors parameter to specify filtering locations.")
        
        # Get image shape
        base_shape = self.image.shape
        
        # Generate Fourier coordinates
        qx_grid, qy_grid = self._make_fourier_coords(base_shape)
        self.fourier_coords = (qx_grid, qy_grid)
        
        # Create filter masks for each q-vector
        self.filter_masks = {}
        for i, q in enumerate(self.q_vectors):
            self.filter_masks[f"q{i}"] = self._create_filter_mask(q, qx_grid, qy_grid)
        
        # Create Tukey window if needed
        if self.apply_tuckey_window:
            window = self._create_tukey_window(base_shape)
            img_data = self.image.array * window
        else:
            img_data = self.image.array
        
        # Apply FFT
        img_fft = fft2(img_data)
        
        # Initialize filtered images dictionary
        self.filtered_images = {}
        
        if combine_filters:
            # Initialize combined result
            combined_result = np.zeros(base_shape, dtype=complex)
            
            # Apply each filter and add to combined result
            for q_idx, q in enumerate(self.q_vectors):
                mask = self.filter_masks[f"q{q_idx}"]
                filtered_fft = img_fft * mask
                combined_result += filtered_fft
            
            # Inverse FFT for combined result
            filtered_img = np.abs(ifft2(combined_result))
            
            # Store as Dataset2d
            self.filtered_images["combined"] = Dataset2d.from_array(
                filtered_img,
                name="Combined Fourier filtered image",
                origin=self.image.origin,
                sampling=self.image.sampling,
                units=self.image.units,
            )
        else:
            # Process each q-vector separately
            for q_idx, q in enumerate(self.q_vectors):
                mask = self.filter_masks[f"q{q_idx}"]
                filtered_fft = img_fft * mask
                filtered_img = np.abs(ifft2(filtered_fft))
                
                # Store as Dataset2d
                self.filtered_images[f"q{q_idx}"] = Dataset2d.from_array(
                    filtered_img,
                    name=f"Fourier filtered image (q-vector {q_idx})",
                    origin=self.image.origin,
                    sampling=self.image.sampling,
                    units=self.image.units,
                )
        
        # Display results if requested
        if show_images:
            self.plot_filtered_images()
        
        return self

    def plot_filtered_images(self, q_index: Optional[int] = None, **kwargs):
        """
        Plot the filtered images using the show_2d visualization function.
        
        Parameters
        ----------
        q_index : int, optional
            Index of the q-vector to plot. If None and multiple q-vectors were used,
            all filtered images will be shown.
        **kwargs
            Additional arguments passed to show_2d
            
        Returns
        -------
        tuple
            (fig, ax) from matplotlib
        """
        if self.filtered_images is None:
            raise RuntimeError("No filtered images available. Call apply_filter() first.")
        
        title = kwargs.pop("title", "Fourier filtered images")
        
        if q_index is not None:
            q_label = f"q{q_index}"
            if q_label not in self.filtered_images:
                raise ValueError(f"No filtered images for q-vector index {q_index}")
            return show_2d(
                [self.filtered_images[q_label].array],
                title=f"{title} (q-vector {q_index})",
                **kwargs,
            )
        else:
            # Show all filtered images
            all_images = [dataset.array for dataset in self.filtered_images.values()]
            return show_2d(
                all_images,
                title=title,
                **kwargs,
            )

    def plot_original_image(self, **kwargs):
        """
        Plot the original image using the show_2d visualization function.
        
        Parameters
        ----------
        **kwargs
            Additional arguments passed to show_2d
            
        Returns
        -------
        tuple
            (fig, ax) from matplotlib
        """
        title = kwargs.pop("title", "Original image")
        return show_2d(
            [self.image.array],
            title=title,
            **kwargs,
        )

    def plot_frequency_domain(
        self, 
        log_scale: bool = True, 
        window: List[int] = [-60, 60],
        scale_circle: float = 1.0,
        circle_color: Union[str, Tuple] = 'red',
        **kwargs
    ):
        """
        Plot the frequency domain representation of the image and filter masks
        using the show_2d visualization function.

        Parameters
        ----------
        log_scale : bool, default True
            Whether to display the frequency magnitude in log scale
        window : List[int], default [-60, 60]
            Window limits for displaying the frequency domain
        scale_circle : float, default 1.0
            Scaling factor for the circles indicating filtered frequencies
        circle_color : str or tuple, default 'red'
            Color for the circles indicating filtered frequencies
        **kwargs
            Additional arguments passed to matplotlib

        Returns
        -------
        tuple
            (fig, ax) from matplotlib
        """
        if self.filter_masks is None:
            raise RuntimeError("No filter masks available. Call apply_filter() first.")

        # Create FFT of image for visualization
        if self.apply_tuckey_window:
            window_func = self._create_tukey_window(self.image.shape)
            img_data = self.image.array * window_func
        else:
            img_data = self.image.array

        img_fft = fft2(img_data)
        img_fft_shifted = fftshift(img_fft)

        # Prepare magnitude for display
        if log_scale:
            magnitude = np.log(np.abs(img_fft_shifted) + 1)
        else:
            magnitude = np.abs(img_fft_shifted)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))

        # Define the window limits
        y_mid = magnitude.shape[0] // 2
        x_mid = magnitude.shape[1] // 2

        # Slice the FFT data to the desired window
        magnitude_window = magnitude[
            y_mid + window[0]:y_mid + window[1], 
            x_mid + window[0]:x_mid + window[1]
        ]

        # Plot the sliced FFT data with the adjusted extent
        im_fft_plot = ax.imshow(
            magnitude_window, 
            extent=[window[0], window[1], window[0], window[1]], 
            vmin=kwargs.get('vmin', None),
            vmax=kwargs.get('vmax', None),
            cmap=kwargs.get('cmap', 'gray_r')
        )

        # Draw circles around each q-vector to indicate filtered frequencies
        for i, q_vector in enumerate(self.q_vectors):
            # Convert q-vector to shifted coordinates
            qx_shifted = q_vector[0] * magnitude.shape[0]
            qy_shifted = q_vector[1] * magnitude.shape[1]

            # Draw circle with radius equal to filter_radius
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = self.filter_radius * scale_circle * np.cos(theta) + qy_shifted
            circle_y = self.filter_radius * scale_circle * np.sin(theta) + qx_shifted

            # Use different colors for different q-vectors if multiple
            if isinstance(circle_color, list) and len(circle_color) > i:
                current_color = circle_color[i]
            else:
                current_color = circle_color

            ax.plot(circle_x, circle_y, color=current_color, linewidth=1)

            # Also plot circle for the conjugate q-vector (opposite side of FFT)
            circle_x_conj = self.filter_radius * scale_circle * np.cos(theta) - qy_shifted
            circle_y_conj = self.filter_radius * scale_circle * np.sin(theta) - qx_shifted
            ax.plot(circle_x_conj, circle_y_conj, color=current_color, linewidth=1)

        # Add title and adjust display
        ax.set_title(kwargs.get('title', 'Frequency Domain with Filtered Regions'))
        ax.axis('equal')

        if kwargs.get('axis_off', True):
            ax.axis('off')

        if kwargs.get('colorbar', False):
            fig.colorbar(im_fft_plot, ax=ax, extend='both')

        plt.tight_layout()
    
        return fig, ax
    ## Comprehensive plotting function;
    def plot_complete_results(
        self,
        figsize: Tuple[int, int] = (10, 10),
        window: List[int] = [-60, 60],
        scale_circle: float = 1.0,
        log_scale: bool = True,
        q_colors: List = None,
        save_path: str = None,
        **kwargs
    ):
        """
        Create a comprehensive visualization of the original image, 
        frequency domain, and filtered images.

        Parameters
        ----------
        figsize : tuple, default (10, 10)
            Figure size
        window : list, default [-60, 60]
            Window limits for displaying the frequency domain
        scale_circle : float, default 1.0
            Scaling factor for the circles indicating filtered frequencies
        log_scale : bool, default True
            Whether to display the frequency magnitude in log scale
        q_colors : list, default None
            List of colors for different q-vectors
        save_path : str, default None
            Path to save the figure
        **kwargs
            Additional arguments passed to matplotlib

        Returns
        -------
        tuple
            (fig, ax) from matplotlib
        """
        if self.filtered_images is None:
            raise RuntimeError("No filtered images available. Call apply_filter() first.")

        # Set default colors if not provided
        if q_colors is None:
            q_colors = ['red', (0.031, 0.188, 0.420)]  # Red and blue

        # Create figure with subplots
        fig, ax = plt.subplots(2, 2, figsize=figsize)

        # Get coordinate ranges
        if hasattr(self.image, 'origin') and hasattr(self.image, 'sampling'):
            y_range = [
                self.image.origin[1], 
                self.image.origin[1] + self.image.sampling[1] * self.image.shape[1]
            ]
            x_range = [
                self.image.origin[0], 
                self.image.origin[0] + self.image.sampling[0] * self.image.shape[0]
            ]
        else:
            y_range = [0, self.image.shape[1]]
            x_range = [0, self.image.shape[0]]

        # Plot original image
        im_val_plot = ax[0, 0].imshow(
            self.image.array, 
            cmap='gray',
            extent=[y_range[0], y_range[1], x_range[0], x_range[1]],
        )
        ax[0, 0].set_title('Original Image')
        ax[0, 0].axis('equal')
        ax[0, 0].axis('off')

        # Plot frequency domain
        # Create FFT of image for visualization
        if self.apply_tuckey_window:
            window_func = self._create_tukey_window(self.image.shape)
            img_data = self.image.array * window_func
        else:
            img_data = self.image.array

        img_fft = fft2(img_data)
        img_fft_shifted = fftshift(img_fft)

        # Prepare magnitude for display
        if log_scale:
            magnitude = np.log(np.abs(img_fft_shifted) + 1)
        else:
            magnitude = np.abs(img_fft_shifted)

        # Define the window limits
        y_mid = magnitude.shape[0] // 2
        x_mid = magnitude.shape[1] // 2

        # Slice the FFT data to the desired window
        magnitude_window = magnitude[
            y_mid + window[0]:y_mid + window[1], 
            x_mid + window[0]:x_mid + window[1]
        ]

        # Plot the sliced FFT data with the adjusted extent
        im_fft_plot = ax[0, 1].imshow(
            magnitude_window, 
            extent=[window[0], window[1], window[0], window[1]], 
            vmin=kwargs.get('vmin', 1e-10),
            vmax=kwargs.get('vmax', 1e4),
            cmap=kwargs.get('cmap', 'gray_r')
        )

        # Draw circles around each q-vector to indicate filtered frequencies
        for i, q_vector in enumerate(self.q_vectors):
            # Get color for this q-vector
            if i < len(q_colors):
                current_color = q_colors[i]
            else:
                current_color = 'red'

            # Draw circle with radius equal to filter_radius
            theta = np.linspace(0, 2*np.pi, 100)

            # Convert q-vector from normalized to pixel coordinates within the window
            qx_window = q_vector[0] * self.image.shape[0] / (window[1] - window[0])
            qy_window = q_vector[1] * self.image.shape[1] / (window[1] - window[0])

            circle_x = self.filter_radius * scale_circle * np.cos(theta) + qy_window
            circle_y = self.filter_radius * scale_circle * np.sin(theta) + qx_window

            ax[0, 1].plot(circle_x, circle_y, color=current_color, linewidth=1)

            # Also plot circle for the conjugate q-vector (opposite side of FFT)
            circle_x_conj = self.filter_radius * scale_circle * np.cos(theta) - qy_window
            circle_y_conj = self.filter_radius * scale_circle * np.sin(theta) - qx_window
            ax[0, 1].plot(circle_x_conj, circle_y_conj, color=current_color, linewidth=1)

        ax[0, 1].set_title('Frequency Domain')
        ax[0, 1].axis('equal')
        ax[0, 1].axis('off')

        # Plot filtered images
        if len(self.filtered_images) >= 2:
            # Plot first filtered image
            filtered_keys = list(self.filtered_images.keys())

            # First filtered image (bottom left)
            ax[1, 0].imshow(
                self.filtered_images[filtered_keys[0]].array,
                cmap='Reds',
                vmin=kwargs.get('vmin_filtered', 0.9),
                vmax=kwargs.get('vmax_filtered', 1.0),
                extent=[y_range[0], y_range[1], x_range[0], x_range[1]]
            )
            ax[1, 0].set_title(f'Filtered Image ({filtered_keys[0]})')
            ax[1, 0].axis('equal')
            ax[1, 0].axis('off')

            # Second filtered image (bottom right)
            ax[1, 1].imshow(
                self.filtered_images[filtered_keys[1]].array,
                cmap='Blues',
                vmin=kwargs.get('vmin_filtered', 0.9),
                vmax=kwargs.get('vmax_filtered', 1.0),
                extent=[y_range[0], y_range[1], x_range[0], x_range[1]]
            )
            ax[1, 1].set_title(f'Filtered Image ({filtered_keys[1]})')
            ax[1, 1].axis('equal')
            ax[1, 1].axis('off')
        else:
            # Only one filtered image, plot it in bottom left
            filtered_key = list(self.filtered_images.keys())[0]
            ax[1, 0].imshow(
                self.filtered_images[filtered_key].array,
                cmap='Reds',
                vmin=kwargs.get('vmin_filtered', 0.9),
                vmax=kwargs.get('vmax_filtered', 1.0),
                extent=[y_range[0], y_range[1], x_range[0], x_range[1]]
            )
            ax[1, 0].set_title(f'Filtered Image ({filtered_key})')
            ax[1, 0].axis('equal')
            ax[1, 0].axis('off')

            # Leave bottom right empty or add some text
            ax[1, 1].axis('off')
            ax[1, 1].text(
                0.5, 0.5, 
                "No second filtered image", 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax[1, 1].transAxes
            )

        fig.tight_layout()

        # Save figure if path provided
        if save_path:
            # Make backup of existing file if it exists
            if os.path.exists(save_path):
                backup_path = save_path.replace('.pdf', '_old.pdf')
                os.system(f'cp {save_path} {backup_path}')
            fig.savefig(save_path)

        return fig, ax

    def get_filtered_image(self, q_index: int = 0) -> Dataset2d:
        """
        Get a specific filtered image as a Dataset2d.
        
        Parameters
        ----------
        q_index : int, default 0
            Index of the q-vector filter to use
            
        Returns
        -------
        Dataset2d
            The filtered image
        """
        if self.filtered_images is None:
            raise RuntimeError("No filtered images available. Call apply_filter() first.")
        
        q_label = f"q{q_index}"
        if "combined" in self.filtered_images:
            # Combined filter case
            return self.filtered_images["combined"]
        elif q_label in self.filtered_images:
            # Specific q-vector filter
            return self.filtered_images[q_label]
        else:
            raise ValueError(f"No filtered image for q-vector index {q_index}")