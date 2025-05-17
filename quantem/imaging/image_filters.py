from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft2, ifft2, fftshift

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization import show_2d

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm


class FourierFilter(AutoSerialize):
    """
    FourierFitlering
    """

    _token = object()

    def __init__(
            self,
            images: Dataset2d,
            q_vectors: List[Tuple[float,float]],
            filter_radius: float = 6.0,
            filter_power: float = 2.0,
            apply_tukey_window: bool = True,
            _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use FourierFilter.from_data() or .from_file() to instantiate this class."
            )
        self._images = images
        self._q_vectors = q_vectors
        self._filter_radius = float(filter_radius)
        self._filter_power = float(filter_power)
        self._apply_tukey_window = bool(apply_tukey_window)

    @classmethod
    def from_data(
        cls,
        image: Dataset2d | NDArray,
        q_vectors: List[Tuple[float,float]] | None = None,
        filter_radius: float = 6.0,
        filter_power: float = 2.0,
        apply_tukey_window: bool = True,
    ):
        if isinstance(image, np.ndarray):
            image = Dataset2d.from_array(image)
        if q_vectors is None:
            q_vectors = []
        return cls(
            image,
            q_vectors,
            filter_radius,
            filter_power,
            apply_tukey_window,
            _token=cls._token,
        )
    ### Haven't wrote the from_class classmethod
    # --- Properties ----
    @property
    def image(self) -> Dataset2d:
        return self._images
    @image.setter
    def image(self, value: Union[Dataset2d, NDArray]):
        if isinstance(value, np.ndarray):
            self._image = Dataset2d.from_array(value)
        elif isinstance(value, Dataset2d):
            self._image = value
        else:
            raise TypeError("Image must be a Dataset2d or a 2D numpy array")
    
    @property
    def q_vectors(self) -> List[Tuple[float,float]]:
        return self._q_vectors
    @q_vectors.setter
    def q_vectors(self, value: List[Tuple[float,float]]):
        if not isinstance(value, list):
            raise valueError("q_vectors mst be a list of (qx,qy) tuples")
        for q in value:
            if not (isinstance(q,(list,tuple)) and len(q) == 2):
                raise ValueError("Each q-vector must be a tuple or list of two values (qx, qy)")
        self._q_vectors = [(float(q[0]), float(q[1])) for q in value]
    
    @property
    def filter_radius(self) -> float:
        return self._filter_radius
    @filter_radius.setter
    def filter_radius(self, value: float):
        if value <=0:
            raise ValueError("filter_radius must be positive")
        self._filter_radius = float(value)
    
    @property
    def filter_power(self) -> float:
        return self._filter_power
    @filter_power.setter
    def filter_power(self, value: float):
        if value <=0:
            raise ValueError("filter_power must be positive")
        self._filter_power = float(value)
    
    @property
    def apply_tukey_window(self) -> bool:
        return self._apply_tukey_window
    @apply_tukey_window.setter
    def apply_tukey_window(self, value: bool):
        self._apply_tukey_window = bool(value)
    
    def _create_tukey_window(self, windowSize:Tuple[int,int], windowFrac=None):
        x = np.linspace(1,windowSize[0], windowSize[0])
        windowOutput = np.minimum(1/(1 - windowFrac[0]) * (1 - np.abs((windowSize[0]+1)/2 - x) * 2 / windowSize[0]), 1)
        windowOutput = np.sin(windowOutput * (np.pi/2)) ** 2
        y = np.linspace(1, windowSize[1], windowSize[1])
        wy = np.minimum(1/(1 - windowFrac[1]) * (1 - np.abs((windowSize[1]+1)/2 - y) * 2 / windowSize[1]), 1)
        wy = np.sin(wy * (np.pi/2)) ** 2
        windowOutput = np.outer(windowOutput,wy)
        return windowOutput

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
            show_images: bool = True,
    ):
        """
        Apply Fourier Filtering around the specified q-vectors to the input image.
        Parameters
        ----------
        show_images: bool, default True
            Whether to isplay the filtered images after processing.
        
        Returns
        -------
        self: FouierFilter
            Returns self of the method chaining
        """

        if not self.q_vectors:
            raise ValueError("No q-vectors specified. Use q_vectors parameter to specify filtering locations.")
        
        # Check if the dataset has the physical pixel size;
        try:
            # I am assuming here that the self.image.sampling is returing sampling [x,y]
            pixel_size_x, pixel_size_y = self.image.sampling
        except AttributeError:
            # If a numpy array has been passed and the field is missing
            pixel_size_y = pixel_size_x = 1.0
        
        ny, nx = self.image.shape       ## Going to follow the convention of rows = y and columns = x

        qx = np.fft.fftfreq(nx,d=pixel_size_x)
        qy = np.fft.fftfreq(ny,d=pixel_size_y)
        qx_grid, qy_grid = np.meshgrid(qx,qy, indexing="xy") 

        #Create a window_size and window_frac for the tukey window. For now it is a place holder but I will make it a user input
        # Create tukey window if needed
        window_shape = (nx,ny)
        window_frac = (0,0)
        if self.apply_tukey_window:
            window =self._create_tukey_window(window_shape,window_frac)
            img_data = self.image.array* window
        else: 
            img_data = self.image.array
        
        # Apply FFT
        img_fft = fft2(img_data)

        # Create masks for each q-vector
        self.filter_masks ={}
        for i,q in enumerate(self.q_vectors):
            self.filter_masks[f"q{i}"] =self._create_filter_mask(q,qx_grid,qy_grid)

        # Initialize a dictionary of the fitlered images
        self.filtered_images = {}
        
        # Process each q-vector separately;
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
        


        
        if show_images:
            self.plot_filtered_image(img_data)
        
        return self

    ## Will get rid of this later.
    @staticmethod
    def _fft_with_axes(img: NDArray,
                       pixel_size_x: float,
                       pixel_size_y: float,
                       window: tuple[float, float] | None = None
                       ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Convenience:   return |FFT| (shifted) together with the q-axes that
        belong to the rows/columns of that array.

        Parameters
        ----------
        img              : real-space image
        pixel_size_x/y   : pixel sizes that turn pixel index -> q
        window           : (q_min, q_max)  – crop symmetric square window
                           around q = (0,0) in *frequency* units.
                           Pass None to keep the full FFT.

        Returns
        -------
        fft_mod          : |FFT|  (possibly cropped)
        qx_1d, qy_1d     : 1-D arrays that correspond to the axes of fft_mod
                           and can be used as ``extent`` in ``imshow``.
        """
        ny, nx = img.shape
        fft_mod = np.abs(fftshift(fft2(img)))

        qx = np.fft.fftshift(np.fft.fftfreq(nx, d=pixel_size_x))
        qy = np.fft.fftshift(np.fft.fftfreq(ny, d=pixel_size_y))

        if window is not None:
            q_min, q_max = window
            mask_x = (qx >= q_min) & (qx <= q_max)
            mask_y = (qy >= q_min) & (qy <= q_max)
            fft_mod = fft_mod[np.ix_(mask_y, mask_x)]
            qx = qx[mask_x]
            qy = qy[mask_y]

        return fft_mod, qx, qy
    
    def plot_results(self,
                 window: tuple[float, float] | None = (-60, 60),
                 cmap_real: str = "gray",
                 cmap_fft: str = "gray_r",
                 circle_color: str = "red",
                 circle_lw: float = 1.0,
                 scale_circle: float = 1.0,
                 save_path: str | None = None,
                 show: bool = True):
        """
        Visualise original image, Fourier space (with circle) and the filtered
        image(s).  For exactly one q-vector the layout is a simple 1×3 panel.
    
        Parameters
        ----------
        window        : (q_min, q_max) – square crop of the FFT shown.
        circle_color  : colour of the q-vector circle.
        scale_circle  : multiply self.filter_radius by this factor.
        save_path     : if not None -> `fig.savefig(save_path)`.
        show          : call plt.show() at the end.
        """
    
        # ----- basic checks -------------------------------------------------
        if not hasattr(self, "filtered_images"):
            raise RuntimeError("Call .apply_filter() before plotting results.")
        if len(self.q_vectors) == 0:
            raise RuntimeError("No q-vector present – nothing to plot.")
    
        img_real = self.image.array
        # pixel size may be absent
        try:
            px, py = self.image.sampling
        except AttributeError:
            px = py = 1.0
    
        # -------------------------------------------------------------------
        # obtain FFT slice + axes
        fft_mod, qx, qy = self._fft_with_axes(img_real, px, py, window)
        extent = [qx[0], qx[-1], qy[0], qy[-1]]
    
        # -------------------------------------------------------------------
        # decide on layout
        # -------------------------------------------------------------------
        single_q = len(self.q_vectors) == 1
        if single_q:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            ax_orig, ax_fft, ax_filt = axs
        else:
            # fallback to the previous generic layout
            n_q = len(self.q_vectors)
            ncols = 2
            nrows = 1 + (n_q + 1) // 2
            fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols,
                                                           5 * nrows))
            axs = np.asarray(axs).reshape(nrows, ncols)
            ax_orig = axs[0, 0]
            ax_fft  = axs[0, 1]
    
        # -------------------------------------------------------------------
        # original image
        # -------------------------------------------------------------------
        ax_orig.imshow(img_real, cmap=cmap_real, origin="lower")
        ax_orig.set_title("original image")
        ax_orig.axis("off")
    
        # -------------------------------------------------------------------
        # FFT with circle(s)
        # -------------------------------------------------------------------
        ax_fft.imshow(fft_mod, cmap=cmap_fft, origin="lower",
                      extent=extent,
                      norm=LogNorm(vmin=fft_mod.min() + 1e-10,
                                   vmax=fft_mod.max()))
        ax_fft.set_title("FFT (|F|)")
        ax_fft.axis("off")
    
        for q in self.q_vectors:
            circ = Circle((q[0], q[1]),
                          radius=scale_circle * self.filter_radius,
                          fill=False,
                          color=circle_color,
                          lw=circle_lw)
            ax_fft.add_patch(circ)
    
        # -------------------------------------------------------------------
        # filtered images
        # -------------------------------------------------------------------
        if single_q:
            # we know only one key exists: 'q0'
            ds = self.filtered_images["q0"]
            ax_filt.imshow(ds.array, cmap=cmap_real, origin="lower")
            ax_filt.set_title("filtered image")
            ax_filt.axis("off")
        else:
            for k, (key, ds) in enumerate(self.filtered_images.items()):
                row = 1 + (k // 2)
                col = k % 2
                ax = axs[row, col]
                ax.imshow(ds.array, cmap=cmap_real, origin="lower")
                ax.set_title(f"filtered – {key}")
                ax.axis("off")
            if len(self.q_vectors) % 2 == 1:          # blank panel, nice look
                axs[-1, -1].axis("off")
    
        # -------------------------------------------------------------------
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300)
        if show:
            plt.show()
        return fig, axs

    def plot_filtered_image(self, img_data,**kwargs):
        """
        

        Returns
        -------
        tuple
            (fig, ax) from matplotlib
        """
        title = kwargs.pop("title", "Original image")
        return show_2d(
            img_data,
            title=title,
            **kwargs,
        )     
        





        