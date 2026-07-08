"""
quark.supersampling
-------------------
Supersampling utilities for subpixel coordinate generation.

Provides:
- SpatialSupersampling: XYZ-based adaptive pixel widths (spherical geometry)
- ConstantSupersampling: Fixed meter-based spacing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


def _compute_subpixel_offset(i: int, j: int, supersampling: int) -> tuple[float, float]:
    """
    Compute fractional offset for subpixel (i, j).
    
    Parameters
    ----------
    i : int
        Subpixel index in x direction (0 to ss-1)
    j : int
        Subpixel index in y direction (0 to ss-1)
    supersampling : int
        Supersampling factor (1, 2, or 3)
    
    Returns
    -------
    offset_y, offset_x : float, float
        Fractional offsets in range [-0.5, +0.5]
    
    Examples
    --------
    ss=2: offsets = [-0.25, +0.25]
    ss=3: offsets = [-1/3, 0, +1/3]
    """
    ss = supersampling
    center = (ss - 1) / 2.0
    offset_x = (i - center) / ss
    offset_y = (j - center) / ss
    return offset_y, offset_x


def _compute_pixel_widths_xyz(
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """
    Compute per-pixel angular width using XYZ distance method.
    
    Converts lat/lon to unit sphere XYZ coordinates, computes mean
    chord distance to available neighbors (top, bottom, left, right),
    and converts back to angular width in degrees.
    
    More robust than direct angular interpolation as it accounts for
    spherical geometry and automatically handles latitude-dependent spacing.
    
    Parameters
    ----------
    lat : np.ndarray, shape (ny, nx)
        Latitude grid in degrees
    lon : np.ndarray, shape (ny, nx)
        Longitude grid in degrees
    
    Returns
    -------
    angular_width : np.ndarray, shape (ny, nx)
        Angular pixel width in degrees per pixel
    """
    # Convert to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # Convert to XYZ on unit sphere
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    ny, nx = lat.shape
    
    # Compute chord distances to all 4 neighbors
    # Top neighbor (y-1)
    dist_top = np.sqrt(
        (x[:-1, :] - x[1:, :])**2 +
        (y[:-1, :] - y[1:, :])**2 +
        (z[:-1, :] - z[1:, :])**2
    )  # shape (ny-1, nx)
    
    # Bottom neighbor (y+1)
    dist_bottom = np.sqrt(
        (x[1:, :] - x[:-1, :])**2 +
        (y[1:, :] - y[:-1, :])**2 +
        (z[1:, :] - z[:-1, :])**2
    )  # shape (ny-1, nx)
    
    # Left neighbor (x-1)
    dist_left = np.sqrt(
        (x[:, :-1] - x[:, 1:])**2 +
        (y[:, :-1] - y[:, 1:])**2 +
        (z[:, :-1] - z[:, 1:])**2
    )  # shape (ny, nx-1)
    
    # Right neighbor (x+1)
    dist_right = np.sqrt(
        (x[:, 1:] - x[:, :-1])**2 +
        (y[:, 1:] - y[:, :-1])**2 +
        (z[:, 1:] - z[:, :-1])**2
    )  # shape (ny, nx-1)
    
    # Compute mean chord distance directly
    sum_dist = np.zeros((ny, nx), dtype=np.float32)
    count_dist = np.zeros((ny, nx), dtype=np.uint8)
    
    # Top neighbor
    sum_dist[1:, :] += dist_top
    count_dist[1:, :] += 1
    
    # Bottom neighbor
    sum_dist[:-1, :] += dist_bottom
    count_dist[:-1, :] += 1
    
    # Left neighbor
    sum_dist[:, 1:] += dist_left
    count_dist[:, 1:] += 1
    
    # Right neighbor
    sum_dist[:, :-1] += dist_right
    count_dist[:, :-1] += 1
    
    # Mean chord distance per pixel
    mean_chord = sum_dist / count_dist  # (ny, nx)
    
    # Convert chord distance to angular distance
    # For unit sphere: angular_distance = 2 * arcsin(chord_distance / 2)
    angular_width_rad = 2 * np.arcsin(mean_chord / 2)
    angular_width_deg = np.rad2deg(angular_width_rad)
    
    return angular_width_deg


def _supersample_subpixel_coords_spatial(
    lat: np.ndarray,
    lon: np.ndarray,
    offset_y: float,
    offset_x: float,
    pixel_widths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    """
    Compute interpolated subpixel coordinates using XYZ-based adaptive method.
    
    Computes per-pixel angular width from 3D Cartesian neighbor distances (more
    robust than direct angular interpolation), then applies fractional offsets.
    Processes full grid (no edge exclusion) since all pixels have ≥2 neighbors.
    
    Parameters
    ----------
    lat : np.ndarray, shape (ny, nx)
        Latitude grid (2D)
    lon : np.ndarray, shape (ny, nx)
        Longitude grid (2D)
    offset_y : float
        Fractional offset in y direction
    offset_x : float
        Fractional offset in x direction
    pixel_widths : np.ndarray | None, shape (ny, nx)
        Pre-computed angular pixel widths in degrees.
        If None, will be computed (but prefer passing for efficiency).
    
    Returns
    -------
    lat_sub : np.ndarray
        Interpolated latitude coordinates
    lon_sub : np.ndarray
        Interpolated longitude coordinates
    values_slices : (slice, slice)
        Slices to apply to values arrays (full grid: [:, :])
    """
    # Compute or use provided pixel widths
    if pixel_widths is None:
        pixel_widths = _compute_pixel_widths_xyz(lat, lon)
    
    # Apply fractional offsets using adaptive pixel widths
    lat_sub = lat + offset_y * pixel_widths
    lon_sub = lon + offset_x * pixel_widths
    
    # Clamp latitude to valid range
    lat_sub = np.clip(lat_sub, -90.0, 90.0)
    
    # No edge exclusion - process full grid
    return lat_sub, lon_sub, (slice(None), slice(None))


def _supersample_subpixel_coords_constant(
    lat: np.ndarray,
    lon: np.ndarray,
    offset_y: float,
    offset_x: float,
    pixel_width_m: float,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    """
    Compute subpixel coordinates using constant pixel width.
    
    Uses fixed meter-based spacing instead of adaptive neighbor-based spacing.
    Allows processing entire grid (no edge exclusion).
    
    Parameters
    ----------
    lat : np.ndarray, shape (ny, nx)
        Latitude grid (2D)
    lon : np.ndarray, shape (ny, nx)
        Longitude grid (2D)
    offset_y : float
        Fractional offset in y direction
    offset_x : float
        Fractional offset in x direction
    pixel_width_m : float
        Pixel width in meters
    
    Returns
    -------
    lat_sub : np.ndarray
        Interpolated latitude coordinates (same shape as input)
    lon_sub : np.ndarray
        Interpolated longitude coordinates (same shape as input)
    values_slices : (slice, slice)
        Slices for values array (full grid: [:, :])
    """
    # Convert pixel width from meters to degrees
    # Latitude: 1 degree ≈ 111,320 meters (constant)
    delta_lat_deg = pixel_width_m / 111320.0
    
    # Longitude: Constant (no latitude dependency per user request)
    # Using same conversion as latitude (valid at equator)
    delta_lon_deg = pixel_width_m / 111320.0
    
    # Apply offsets
    lat_sub = lat + offset_y * delta_lat_deg
    lon_sub = lon + offset_x * delta_lon_deg
    
    # Clamp latitude to valid range
    lat_sub = np.clip(lat_sub, -90.0, 90.0)
    
    # Full grid - no slicing needed
    return lat_sub, lon_sub, (slice(None), slice(None))


def _interpolate_subpixel_coords(
    lat: np.ndarray,
    lon: np.ndarray,
    offset_y: float,
    offset_x: float,
    subpixel_mode: Literal["spatial", "constant"],
    pixel_width_m: float | None = None,
    pixel_widths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    """
    Compute interpolated subpixel coordinates (dispatcher function).
    
    Dispatches to either spatial or constant mode implementation.
    
    Parameters
    ----------
    lat : np.ndarray, shape (ny, nx)
        Latitude grid (2D)
    lon : np.ndarray, shape (ny, nx)
        Longitude grid (2D)
    offset_y : float
        Fractional offset in y direction
    offset_x : float
        Fractional offset in x direction
    subpixel_mode : {"spatial", "constant"}
        Subpixel coordinate computation mode
    pixel_width_m : float | None
        Pixel width in meters (required for constant mode)
    pixel_widths : np.ndarray | None
        Pre-computed pixel widths for spatial mode (computed once per dataset)
    
    Returns
    -------
    lat_sub : np.ndarray
        Interpolated latitude coordinates
    lon_sub : np.ndarray
        Interpolated longitude coordinates
    values_slices : (slice, slice)
        Slices to apply to values arrays to match lat_sub/lon_sub shape
    """
    if subpixel_mode == "constant":
        if pixel_width_m is None:
            raise ValueError("pixel_width_m required for constant mode")
        return _supersample_subpixel_coords_constant(
            lat, lon, offset_y, offset_x, pixel_width_m
        )
    else:  # spatial mode
        return _supersample_subpixel_coords_spatial(
            lat, lon, offset_y, offset_x, pixel_widths
        )


# ---------------------------------------------------------------------------
# Supersampler Classes (Strategy Pattern)
# ---------------------------------------------------------------------------

class _BaseSuperSampler(ABC):
    """
    Abstract base class for supersampling strategies.
    
    Parameters
    ----------
    factor : int
        Supersampling factor (must be >= 2). Creates a factor×factor subpixel grid.
        For example, factor=2 creates 4 subpixels, factor=3 creates 9 subpixels.
    project_center : bool
        Whether to project the center pixel (offset 0,0)
    """
    
    def __init__(self, factor: int, project_center: bool = True):
        if not isinstance(factor, int) or factor < 2:
            raise ValueError(f"factor must be an integer >= 2, got {factor}")
        self.factor = factor
        self.project_center = project_center
    
    def prepare(self, lat: np.ndarray, lon: np.ndarray) -> None:
        """
        Prepare for supersampling (compute cached data).
        
        Called once per dataset before iterating subpixels.
        
        Parameters
        ----------
        lat : np.ndarray, shape (ny, nx)
            Latitude grid
        lon : np.ndarray, shape (ny, nx)
            Longitude grid
        """
        pass  # Override in subclasses if needed
    
    @abstractmethod
    def compute_coords(
        self, lat: np.ndarray, lon: np.ndarray, i: int, j: int
    ) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
        """
        Compute subpixel coordinates for subpixel (i, j).
        
        Parameters
        ----------
        lat : np.ndarray, shape (ny, nx)
            Latitude grid
        lon : np.ndarray, shape (ny, nx)
            Longitude grid
        i : int
            Subpixel index in x direction (0 to factor-1)
        j : int
            Subpixel index in y direction (0 to factor-1)
        
        Returns
        -------
        lat_sub : np.ndarray
            Interpolated latitude coordinates
        lon_sub : np.ndarray
            Interpolated longitude coordinates
        values_slices : (slice, slice)
            Slices to apply to values arrays
        """
        pass


class SpatialSuperSampler(_BaseSuperSampler):
    """
    XYZ-based adaptive supersampling.
    
    Computes per-pixel angular widths from 3D Cartesian neighbor distances,
    accounting for spherical geometry and latitude-dependent spacing.
    Processes the full grid (no edge exclusion)
    
    Parameters
    ----------
    factor : int
        Supersampling factor (integer >= 2). Creates a factor×factor subpixel grid.
    project_center : bool
        Whether to project the center pixel (offset 0,0)
    
    Examples
    --------
    >>> supersampler = SpatialSupersampling(factor=2)  # 4 subpixels
    >>> supersampler = SpatialSupersampling(factor=5)  # 25 subpixels
    >>> supersampler.prepare(lat, lon)
    >>> lat_sub, lon_sub, slices = supersampler.compute_coords(lat, lon, 0, 0)
    """
    
    def __init__(self, factor: int, project_center: bool = True):
        super().__init__(factor, project_center)
        self._pixel_widths: np.ndarray | None = None
    
    def prepare(self, lat: np.ndarray, lon: np.ndarray) -> None:
        """Compute pixel widths once per dataset."""
        self._pixel_widths = _compute_pixel_widths_xyz(lat, lon)
    
    def compute_coords(
        self, lat: np.ndarray, lon: np.ndarray, i: int, j: int
    ) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
        """Compute adaptive subpixel coordinates."""
        offset_y, offset_x = _compute_subpixel_offset(i, j, self.factor)
        return _supersample_subpixel_coords_spatial(
            lat, lon, offset_y, offset_x, self._pixel_widths
        )


class ConstantSuperSampler(_BaseSuperSampler):
    """
    Constant-spacing supersampling.
    
    Uses fixed meter-based pixel width for uniform subpixel spacing.
    
    Parameters
    ----------
    factor : int
        Supersampling factor (integer >= 2). Creates a factor×factor subpixel grid.
    pixel_width : str
        Pixel width string (e.g., "1km", "500m")
    project_center : bool
        Whether to project the center pixel (offset 0,0)
    
    Examples
    --------
    >>> supersampler = ConstantSupersampling(factor=2, pixel_width="1km")  # 4 subpixels
    >>> supersampler = ConstantSupersampling(factor=10, pixel_width="100m")  # 100 subpixels
    >>> lat_sub, lon_sub, slices = supersampler.compute_coords(lat, lon, 0, 1)
    """
    
    def __init__(self, factor: int, pixel_width: str, project_center: bool = True):
        super().__init__(factor, project_center)
        self.pixel_width = pixel_width
        self.pixel_width_m = self._parse_pixel_width(pixel_width)
    
    @staticmethod
    def _parse_pixel_width(pixel_width: str) -> float:
        """
        Parse pixel width string to meters.
        
        Parameters
        ----------
        pixel_width : str
            Pixel width string (e.g., "1km", "500m", "0.3km")
        
        Returns
        -------
        float
            Pixel width in meters
        """
        pixel_width = pixel_width.strip().lower()
        
        if pixel_width.endswith("km"):
            try:
                value = float(pixel_width[:-2])
                return value * 1000.0
            except ValueError:
                raise ValueError(
                    f"Invalid pixel_width format: '{pixel_width}'. "
                    f"Expected format: '1km' or '500m'."
                )
        elif pixel_width.endswith("m"):
            try:
                value = float(pixel_width[:-1])
                return value
            except ValueError:
                raise ValueError(
                    f"Invalid pixel_width format: '{pixel_width}'. "
                    f"Expected format: '1km' or '500m'."
                )
        else:
            raise ValueError(
                f"Invalid pixel_width format: '{pixel_width}'. "
                f"Must end with 'km' or 'm' (e.g., '1km' or '500m')."
            )
    
    def compute_coords(
        self, lat: np.ndarray, lon: np.ndarray, i: int, j: int
    ) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
        """Compute constant-spacing subpixel coordinates."""
        offset_y, offset_x = _compute_subpixel_offset(i, j, self.factor)
        return _supersample_subpixel_coords_constant(
            lat, lon, offset_y, offset_x, self.pixel_width_m
        )
