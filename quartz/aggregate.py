"""
quartz.aggregate
----------------
Core aggregation engine: reproject and accumulate N-dimensional xarray Datasets
onto a target projection grid.

Iteration 1 (current):
  - In-memory mode only (vars_batch_size must be None)
  - No supersampling (supersampling must be 1)
  - naive sum_method only ("kahan" accepted by interface, raises NotImplementedError)
  - Full NDIMS support: variables may be 2-D, 3-D, 4-D …
    (lat/lon rasters are always 2-D; extra dims are preserved in output)
"""

from __future__ import annotations

import warnings
from typing import Callable, Literal

import numpy as np
import xarray as xr

from quartz import accumulate

from core import log

# ---------------------------------------------------------------------------
# Aggregator Class
# ---------------------------------------------------------------------------

class Aggregator:
    """
    Stateful aggregation pipeline.
    
    Encapsulates projection, configuration, variable metadata, and accumulators.
    Eliminates the need to pass dozens of arguments through helper functions.
    """
    
    def __init__(
        self,
        projection,
        datasets: list[xr.Dataset] | xr.Dataset,
        lat_name: str = "latitude",
        lon_name: str = "longitude",
        variables: list[str] | None = None,
        fail_on_schema_mismatch: bool = True,
        sum_method: Literal["simple", "kahan"] = "simple",
        skipna: bool = True,
        supersampling: int = 1,
        pixel_width: str | None = None,
        subpixel_mode: Literal["spatial", "constant"] | None = None,
        return_counts: bool = False,
        return_sums: bool = False,
        dtype=None,
    ):
        """
        Initialize aggregator with projection and configuration.
        
        Parameters
        ----------
        projection : ProjectionInterface
            Target grid projection
        datasets : list[xr.Dataset]
            Source datasets (for metadata extraction)
        lat_name, lon_name : str
            Geolocation variable names
        variables : list[str] | None
            Variables to aggregate (None = auto-detect)
        fail_on_schema_mismatch : bool
            Raise if variable sets differ across datasets
        sum_method : {"simple", "kahan"}
            Summation strategy (only 'simple' is currently implemented)
        skipna : bool
            Skip NaN/inf values
        supersampling : int
            Sub-pixel sampling factor (1, 2, or 3)
        pixel_width : str | None
            Pixel width for constant-spacing subpixel mode (e.g., "1km", "500m").
            If provided, uses constant spacing instead of spatial neighbor-based.
            Triggers 'constant' subpixel mode automatically.
        subpixel_mode : {"spatial", "constant"} | None
            Subpixel coordinate computation mode:
            - 'spatial': Use actual neighbor coordinates (adaptive, default)
            - 'constant': Use fixed pixel_width spacing (uniform)
            If None, auto-detected from pixel_width parameter.
        return_counts : bool
            Include count arrays in output
        return_sums : bool
            Include sum arrays in output
        dtype : np.dtype | None
            Override output dtype for all variables
        """
        
        if not isinstance(datasets, list):
            datasets = [datasets]
        
        # Validate inputs
        if not datasets:
            raise ValueError("datasets must be a non-empty list.")
        if supersampling not in (1, 2, 3):
            raise ValueError("supersampling must be 1, 2, or 3.")
        
        # Parse pixel_width and determine mode
        pixel_width_m = None
        if pixel_width is not None:
            pixel_width_m = self._parse_pixel_width(pixel_width)
        
        # Auto-detect or validate subpixel mode
        if subpixel_mode is None:
            # Auto-detect from pixel_width
            if pixel_width_m is not None:
                subpixel_mode = "constant"
            else:
                subpixel_mode = "spatial"
        else:
            # Explicit mode provided - validate consistency
            if pixel_width_m is not None and subpixel_mode == "spatial":
                raise ValueError(
                    "Conflicting parameters: pixel_width is provided but subpixel_mode='spatial'. "
                    "Either remove pixel_width to use spatial mode, or set subpixel_mode='constant'."
                )
            if pixel_width_m is None and subpixel_mode == "constant":
                raise ValueError(
                    "subpixel_mode='constant' requires pixel_width parameter (e.g., pixel_width='1km')."
                )
        
        self.projection = projection
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.skipna = skipna
        self.sum_method = sum_method
        self.supersampling = supersampling
        self.subpixel_mode = subpixel_mode
        self.pixel_width_m = pixel_width_m
        self.return_counts = return_counts
        self.return_sums = return_sums
        self.dtype = dtype
        self.datasets = datasets
        
        # Grid dimensions
        self.height = projection.height
        self.width = projection.width
        self.grid_size = self.height * self.width
        
        # Track number of datasets processed
        self.n_datasets = 0
        
        # Validate and extract metadata
        self._validate_datasets(datasets)
        self.reproject_dims = tuple(datasets[0][lat_name].dims)
        self.geo_vars = {lat_name, lon_name}
        
        # Collect target variables
        self.target_vars = _collect_variables(
            datasets, variables, self.geo_vars, fail_on_schema_mismatch
        )
        if not self.target_vars:
            raise ValueError(
                "No numeric variables found to aggregate. "
                "Check variable names and dataset contents."
            )
        
        # Extract variable metadata
        self.var_preserved_dims, self.var_preserved_shape, self.var_dtype = \
            _prepare_variable_metadata(datasets, self.target_vars, self.reproject_dims)
        
        # Filter variables that passed spatial check
        self.target_vars = [v for v in self.target_vars if v in self.var_preserved_shape]
        if not self.target_vars:
            raise ValueError(
                "None of the candidate variables contain the reprojection dimensions "
                f"{self.reproject_dims}."
            )
        
        # Allocate accumulators with per-variable index dtypes
        self.accumulators: dict[str, accumulate.BaseAccumulator] = {}
        self.var_index_dtype: dict[str, type] = {}
        
        for var_name in self.target_vars:
            pres_shape = self.var_preserved_shape[var_name]
            P = int(np.prod(pres_shape)) if pres_shape else 1
            flat_shape = (P * self.grid_size,)
            
            # Compute index dtype based on total array size (P * grid_size)
            # Maximum index value will be (P-1) * grid_size + (grid_size-1) = P * grid_size - 1
            total_size = P * self.grid_size
            if total_size < np.iinfo(np.uint32).max:
                self.var_index_dtype[var_name] = np.uint32
            else:
                self.var_index_dtype[var_name] = np.uint64
            
            # Create accumulator based on sum_method
            if sum_method == "simple":
                self.accumulators[var_name] = accumulate.SimpleAccumulator(
                    shape=flat_shape,
                    sum_dtype=np.float64,
                )
            elif sum_method == "kahan":
                self.accumulators[var_name] = accumulate.KahanAccumulator(
                    shape=flat_shape,
                    sum_dtype=np.float64,
                )
            else:
                raise ValueError(
                    f"Unknown sum_method '{sum_method}'. "
                    f"Supported methods: 'simple', 'kahan'."
                )
    
    def _validate_datasets(self, datasets: list[xr.Dataset]) -> None:
        """Validate geolocation for all datasets."""
        for ds in datasets:
            _validate_geolocation(ds, self.lat_name, self.lon_name)
    
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
        
        Examples
        --------
        >>> _parse_pixel_width("1km")
        1000.0
        >>> _parse_pixel_width("500m")
        500.0
        >>> _parse_pixel_width("0.5km")
        500.0
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
    
    def _compute_subpixel_offset(self, i: int, j: int) -> tuple[float, float]:
        """
        Compute fractional offset for subpixel (i, j).
        
        Parameters
        ----------
        i : int
            Subpixel index in x direction (0 to ss-1)
        j : int
            Subpixel index in y direction (0 to ss-1)
        
        Returns
        -------
        offset_y, offset_x : float, float
            Fractional offsets in range [-0.5, +0.5]
        
        Examples
        --------
        ss=2: offsets = [-0.25, +0.25]
        ss=3: offsets = [-1/3, 0, +1/3]
        """
        ss = self.supersampling
        center = (ss - 1) / 2.0
        offset_x = (i - center) / ss
        offset_y = (j - center) / ss
        return offset_y, offset_x
    
    def _compute_slices_for_offset(
        self, offset_y: float, offset_x: float
    ) -> tuple[tuple[slice, slice], tuple[slice, slice] | None]:
        """
        Compute array slices for center and neighbor based on offset direction.
        
        Parameters
        ----------
        offset_y : float
            Fractional offset in y direction
        offset_x : float
            Fractional offset in x direction
        
        Returns
        -------
        center_slices : (slice, slice)
            Slices (y_slice, x_slice) for center grid
        neighbor_slices : (slice, slice) | None
            Slices for neighbor grid, or None if offset is (0, 0)
        
        Examples
        --------
        offset_y=-0.25, offset_x=-0.25 (top-left):
            center: [1:, 1:], neighbor: [:-1, :-1]
        offset_y=0, offset_x=-0.25 (left):
            center: [:, 1:], neighbor: [:, :-1]
        offset_y=0, offset_x=0 (center):
            center: [:, :], neighbor: None
        """
        # Determine Y slicing
        if offset_y < 0:
            # Needs top neighbor (y-1) - exclude top edge
            y_slice_center = slice(1, None)
            y_slice_neighbor = slice(None, -1)
        elif offset_y > 0:
            # Needs bottom neighbor (y+1) - exclude bottom edge
            y_slice_center = slice(None, -1)
            y_slice_neighbor = slice(1, None)
        else:
            # No Y neighbor needed
            y_slice_center = slice(None)
            y_slice_neighbor = slice(None)
        
        # Determine X slicing
        if offset_x < 0:
            # Needs left neighbor (x-1) - exclude left edge
            x_slice_center = slice(1, None)
            x_slice_neighbor = slice(None, -1)
        elif offset_x > 0:
            # Needs right neighbor (x+1) - exclude right edge
            x_slice_center = slice(None, -1)
            x_slice_neighbor = slice(1, None)
        else:
            # No X neighbor needed
            x_slice_center = slice(None)
            x_slice_neighbor = slice(None)
        
        center_slices = (y_slice_center, x_slice_center)
        
        # If both offsets are 0, this is the center - no neighbor needed
        if offset_y == 0 and offset_x == 0:
            neighbor_slices = None
        else:
            neighbor_slices = (y_slice_neighbor, x_slice_neighbor)
        
        return center_slices, neighbor_slices
    
    def _interpolate_subpixel_coords(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        offset_y: float,
        offset_x: float,
    ) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
        """
        Compute interpolated subpixel coordinates (dispatcher method).
        
        Dispatches to either spatial or constant mode based on self.subpixel_mode.
        
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
        
        Returns
        -------
        lat_sub : np.ndarray
            Interpolated latitude coordinates
        lon_sub : np.ndarray
            Interpolated longitude coordinates
        values_slices : (slice, slice)
            Slices to apply to values arrays to match lat_sub/lon_sub shape
        """
        if self.subpixel_mode == "constant":
            return self._interpolate_subpixel_coords_constant(lat, lon, offset_y, offset_x)
        else:  # spatial mode
            return self._interpolate_subpixel_coords_spatial(lat, lon, offset_y, offset_x)
    
    def _interpolate_subpixel_coords_spatial(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        offset_y: float,
        offset_x: float,
    ) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
        """
        Compute interpolated subpixel coordinates using spatial neighbor-based method.
        
        Uses actual neighbor coordinates for adaptive spacing.
        Excludes edges where neighbors are unavailable.
        
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
        
        Returns
        -------
        lat_sub : np.ndarray
            Interpolated latitude coordinates
        lon_sub : np.ndarray
            Interpolated longitude coordinates
        values_slices : (slice, slice)
            Slices to apply to values arrays to match lat_sub/lon_sub shape
        """
        center_slices, neighbor_slices = self._compute_slices_for_offset(offset_y, offset_x)
        
        # Extract center grid
        lat_center = lat[center_slices]
        lon_center = lon[center_slices]
        
        # If this is the center (offset 0,0), return as-is
        if neighbor_slices is None:
            return lat_center, lon_center, center_slices
        
        # Extract neighbor grid
        lat_neighbor = lat[neighbor_slices]
        lon_neighbor = lon[neighbor_slices]
        
        # Compute deltas
        delta_lat = lat_neighbor - lat_center
        delta_lon = lon_neighbor - lon_center
        
        # Apply fractional interpolation
        # Moving from center toward neighbor by offset fraction
        lat_sub = lat_center + offset_y * delta_lat
        lon_sub = lon_center + offset_x * delta_lon
        
        # Clamp latitude to valid range
        lat_sub = np.clip(lat_sub, -90.0, 90.0)
        
        return lat_sub, lon_sub, center_slices
    
    def _interpolate_subpixel_coords_constant(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        offset_y: float,
        offset_x: float,
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
        
        Returns
        -------
        lat_sub : np.ndarray
            Interpolated latitude coordinates (same shape as input)
        lon_sub : np.ndarray
            Interpolated longitude coordinates (same shape as input)
        values_slices : (slice, slice)
            Slices for values array (full grid: [:, :])
        """
        # Type assertion for constant mode
        assert self.pixel_width_m is not None, "pixel_width_m must be set for constant mode"
        
        # Convert pixel width from meters to degrees
        # Latitude: 1 degree ≈ 111,320 meters (constant)
        delta_lat_deg = self.pixel_width_m / 111320.0
        
        # Longitude: Constant (no latitude dependency per user request)
        # Using same conversion as latitude (valid at equator)
        delta_lon_deg = self.pixel_width_m / 111320.0
        
        # Apply offsets
        lat_sub = lat + offset_y * delta_lat_deg
        lon_sub = lon + offset_x * delta_lon_deg
        
        # Clamp latitude to valid range
        lat_sub = np.clip(lat_sub, -90.0, 90.0)
        
        # Full grid - no slicing needed
        return lat_sub, lon_sub, (slice(None), slice(None))
    
    def process_dataset(self, ds: xr.Dataset) -> None:
        """
        Process one dataset: project and accumulate all variables.
        
        Parameters
        ----------
        ds : xr.Dataset
            Source dataset to process
        """
        # Extract lat/lon as 2D arrays (not flattened yet)
        lat = ds[self.lat_name].values.astype(np.float64)
        lon = ds[self.lon_name].values.astype(np.float64)
        
        # Always project center first (current behavior when ss=1)
        log.info(f"Projecting center coordinates...")
        self._project_batch(ds, lat, lon, values_slices=(slice(None), slice(None)))
        
        # If supersampling > 1, project additional subpixels
        if self.supersampling > 1:
            ss = self.supersampling
            for i in range(ss):
                for j in range(ss):
                    # Compute offset for this subpixel
                    offset_y, offset_x = self._compute_subpixel_offset(i, j)
                    
                    # Skip center (already projected)
                    if offset_y == 0 and offset_x == 0:
                        continue
                    
                    log.info(f"Projecting subpixel ({i},{j}) with offset ({offset_y:.3f}, {offset_x:.3f})...")
                    
                    # Compute interpolated subpixel coordinates
                    lat_sub, lon_sub, values_slices = self._interpolate_subpixel_coords(
                        lat, lon, offset_y, offset_x
                    )
                    
                    # Project this subpixel batch
                    self._project_batch(ds, lat_sub, lon_sub, values_slices)
        
        # Increment dataset counter
        self.n_datasets += 1
    
    def _project_batch(
        self,
        ds: xr.Dataset,
        lat: np.ndarray,
        lon: np.ndarray,
        values_slices: tuple[slice, slice],
    ) -> None:
        """
        Project a batch of lat/lon coordinates and accumulate all variables.
        
        Parameters
        ----------
        ds : xr.Dataset
            Source dataset
        lat : np.ndarray, shape (ny, nx) or subset
            Latitude coordinates (2D)
        lon : np.ndarray, shape (ny, nx) or subset
            Longitude coordinates (2D)
        values_slices : (slice, slice)
            Slices (y_slice, x_slice) to apply to values arrays to match lat/lon shape
        """
        # Flatten coordinates for projection
        lat_flat = lat.ravel()
        lon_flat = lon.ravel()
        
        # Project to target grid
        x_idx, y_idx = self.projection.project_to_indexes(lat_flat, lon_flat)
        spatial_valid = self.projection.is_valid_index(x_idx, y_idx)
        
        # Accumulate all variables
        for var_name in self.target_vars:
            if var_name not in ds:
                continue
            
            da = ds[var_name]
            
            # Validate consistent preserved shape (only on first batch)
            if self.n_datasets == 0:
                actual_pres_shape = tuple(
                    da.sizes[d] for d in da.dims if d not in self.reproject_dims
                )
                expected = self.var_preserved_shape[var_name]
                if actual_pres_shape != expected:
                    raise ValueError(
                        f"Variable '{var_name}' has inconsistent preserved dim shape "
                        f"across datasets: expected {expected}, got {actual_pres_shape}."
                    )
            
            # Get index dtype for this variable
            idx_dtype = self.var_index_dtype[var_name]
            
            # Compute flat target index (reused for this variable)
            target_flat_idx = (
                y_idx.astype(idx_dtype) * idx_dtype(self.width)
                + x_idx.astype(idx_dtype)
            )
            
            self._accumulate_variable(
                values=da.values,
                var_dims=da.dims,
                target_flat_idx=target_flat_idx,
                spatial_valid=spatial_valid,
                var_name=var_name,
                values_slices=values_slices,
            )
        
    
    def _accumulate_variable(
        self,
        values: np.ndarray,
        var_dims: tuple[str, ...],
        target_flat_idx: np.ndarray,
        spatial_valid: np.ndarray,
        var_name: str,
        values_slices: tuple[slice, slice],
    ) -> None:
        """
        Accumulate one variable using its stateful Accumulator instance.

        Transforms variable data to align spatial dims, builds flat indices,
        and delegates to the accumulator's add() method.

        Parameters
        ----------
        values : np.ndarray
            Variable data in its original dim order.
        var_dims : tuple[str, ...]
            Dim names corresponding to *values* axes.
        target_flat_idx : np.ndarray, shape (S,), dtype uint32 or uint64
            Flat target pixel index per source pixel: y_idx * width + x_idx.
        spatial_valid : np.ndarray, shape (S,), dtype bool
            True where source pixel projects inside the target grid.
        var_name : str
            Variable name (to lookup accumulator and metadata)
        values_slices : (slice, slice)
            Slices (y_slice, x_slice) to apply to spatial dimensions
        """
        # --- Reorder dims so spatial axes are last: (*preserved, ny_src, nx_src) ---
        spatial_axes = tuple(var_dims.index(d) for d in self.reproject_dims)
        preserved_axes = tuple(i for i in range(len(var_dims)) if i not in spatial_axes)
        perm = preserved_axes + spatial_axes
        # Avoid a copy when the order is already correct
        if perm != tuple(range(len(var_dims))):
            values = np.transpose(values, perm)
        
        # --- Apply spatial slices to match the lat/lon grid subset ---
        # Build full slicing tuple: all preserved dims + spatial slices
        full_slices = (slice(None),) * len(preserved_axes) + values_slices
        values = values[full_slices]

        # Flatten spatial dims and compute P (product of preserved dims)
        P = int(np.prod(values.shape[: len(preserved_axes)])) if preserved_axes else 1
        values_ps = np.asarray(values.reshape(P, -1), dtype=np.float64)  # (P, S)
        S = values_ps.shape[1]

        # Combined flat index: (p, s) -> p * grid_size + target_flat_idx[s]
        # Match dtype with target_flat_idx (uint32 or uint64)
        idx_dtype = target_flat_idx.dtype
        grid_size_typed = np.array(self.grid_size, dtype=idx_dtype).item()
        combined_idx = (
            np.arange(P, dtype=idx_dtype)[:, np.newaxis] * grid_size_typed
            + target_flat_idx[np.newaxis, :]
        )  # (P, S)

        # Build validity mask: OOB + (optionally) finite value check
        # Use a writeable copy to allow in-place &=
        valid = np.broadcast_to(spatial_valid[np.newaxis, :], (P, S)).copy()
        if self.skipna:
            valid &= np.isfinite(values_ps)

        flat_idx = combined_idx[valid]   # (N,)
        flat_vals = values_ps[valid]     # (N,)

        # Use pluggable accumulator for summation
        self.accumulators[var_name].add(flat_idx, flat_vals)
    
    def compute(self) -> xr.Dataset:
        """
        Process all datasets and return the aggregated result.
        
        Returns
        -------
        xr.Dataset
            Aggregated dataset with coordinates and attributes
        """
        # Process all datasets
        for ds in self.datasets:
            self.process_dataset(ds)
        
        # Finalize and return
        return self._finalize()
    
    def _finalize(self) -> xr.Dataset:
        """
        Build final xarray Dataset from accumulators.
        
        Returns
        -------
        xr.Dataset
            Aggregated dataset with coordinates and attributes
        """
        # Get output coordinates
        out_lat, out_lon = self.projection.get_coordinates()
        
        data_vars: dict[str, xr.DataArray] = {}
        
        # Build DataArrays for each variable
        for var_name in self.target_vars:
            pres_shape = self.var_preserved_shape[var_name]
            pres_dims = self.var_preserved_dims[var_name]
            out_shape = (*pres_shape, self.height, self.width)
            original_dtype = self.var_dtype[var_name]
            # Use override dtype if provided, otherwise use original
            output_dtype = self.dtype if self.dtype is not None else original_dtype
            out_dims = pres_dims + ("y", "x")
            
            # Finalize accumulator (only create views if needed)
            mean_grid, sum_grid, cnt_grid = self.accumulators[var_name].finalize(
                output_shape=out_shape,
                output_dtype=output_dtype,
                return_sums=self.return_sums,
                return_counts=self.return_counts,
            )
            
            data_vars[var_name] = xr.DataArray(mean_grid, dims=out_dims)
            
            if self.return_counts and cnt_grid is not None:
                data_vars[f"count_{var_name}"] = xr.DataArray(cnt_grid, dims=out_dims)
            if self.return_sums and sum_grid is not None:
                data_vars[f"sum_{var_name}"] = xr.DataArray(
                    sum_grid.astype(original_dtype), dims=out_dims
                )
        
        # Build coordinates
        coords: dict[str, xr.DataArray] = {
            "latitude": xr.DataArray(out_lat, dims=["y"]),
            "longitude": xr.DataArray(out_lon, dims=["x"]),
        }
        
        # Build global attributes
        area_attrs: dict[str, float] = {
            attr: getattr(self.projection, attr)
            for attr in ("north", "south", "east", "west")
            if hasattr(self.projection, attr)
        }
        
        global_attrs = {
            "projection_name": type(self.projection).__name__,
            "projection_width": self.projection.width,
            "projection_height": self.projection.height,
            "projection_area": str(area_attrs) if area_attrs else "N/A",
            "supersampling": self.supersampling,
            "mode": "in-memory",
            "sum_method": self.sum_method,
            "reproject_dims": str(self.reproject_dims),
            "n_datasets": self.n_datasets,
        }
        
        return xr.Dataset(data_vars, coords=coords, attrs=global_attrs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_geolocation(ds: xr.Dataset, lat_var: str, lon_var: str) -> None:
    """Raise ValueError if lat/lon are not 2-D or do not share dims/shape."""
    if lat_var not in ds:
        raise ValueError(
            f"Latitude variable '{lat_var}' not found in dataset. "
            f"Available variables: {sorted(ds.data_vars)}."
        )
    if lon_var not in ds:
        raise ValueError(
            f"Longitude variable '{lon_var}' not found in dataset. "
            f"Available variables: {sorted(ds.data_vars)}."
        )
    
    lat = ds[lat_var]
    lon = ds[lon_var]
    
    if lat.ndim != 2:
        raise ValueError(
            f"Latitude variable '{lat_var}' must be 2-D, got {lat.ndim}-D."
        )
    if lon.ndim != 2:
        raise ValueError(
            f"Longitude variable '{lon_var}' must be 2-D, got {lon.ndim}-D."
        )
    if lat.dims != lon.dims:
        raise ValueError(
            f"Latitude and longitude must have the same dimensions; "
            f"got lat.dims={lat.dims}, lon.dims={lon.dims}."
        )
    if lat.shape != lon.shape:
        raise ValueError(
            f"Latitude and longitude must have the same shape; "
            f"got lat={lat.shape}, lon={lon.shape}."
        )


def _collect_variables(
    datasets: list[xr.Dataset],
    requested: list[str] | None,
    geo_vars: set[str],
    fail_on_schema_mismatch: bool,
) -> list[str]:
    """
    Return the list of variable names to aggregate.

    * If *requested* is given: every name must exist in every dataset (hard error).
    * Otherwise: union of numeric data_vars across all datasets.
    * If *fail_on_schema_mismatch*: raise when variable sets differ across datasets.
    """
    all_var_sets: list[set[str]] = []
    for ds in datasets:
        numeric_vars = {
            v
            for v in ds.data_vars
            if v not in geo_vars and np.issubdtype(ds[v].dtype, np.number)
        }
        all_var_sets.append(numeric_vars)

    if requested is not None:
        for ds in datasets:
            for v in requested:
                if v not in ds:
                    raise ValueError(
                        f"Requested variable '{v}' not found in dataset. "
                        f"Available variables: {sorted(ds.data_vars)}."
                    )
        return list(requested)

    union_vars: set[str] = set().union(*all_var_sets) if all_var_sets else set()

    if fail_on_schema_mismatch and len(all_var_sets) > 1:
        ref = all_var_sets[0]
        for i, vs in enumerate(all_var_sets[1:], 1):
            if vs != ref:
                raise ValueError(
                    f"Dataset {i} has different numeric variables than dataset 0. "
                    f"Dataset 0: {sorted(ref)}, Dataset {i}: {sorted(vs)}. "
                    f"Pass fail_on_schema_mismatch=False to allow this."
                )

    return sorted(union_vars)


def _prepare_variable_metadata(
    datasets: list[xr.Dataset],
    target_vars: list[str],
    reproject_dims: tuple[str, ...],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[int, ...]],
    dict[str, np.dtype],
]:
    """
    Extract preserved dims, shapes, and dtypes for all target variables.
    
    Finds the first dataset containing each variable, validates it contains
    the spatial dims, and extracts metadata about non-spatial (preserved) dims.
    Variables missing spatial dims are filtered out with a warning.
    
    Parameters
    ----------
    datasets : list[xr.Dataset]
        Source datasets
    target_vars : list[str]
        Variable names to extract metadata for
    reproject_dims : tuple[str, ...]
        Spatial dimension names to reproject away
    
    Returns
    -------
    var_preserved_dims : dict[str, tuple[str, ...]]
        Preserved (non-spatial) dimension names per variable
    var_preserved_shape : dict[str, tuple[int, ...]]
        Preserved dimension sizes per variable
    var_dtype : dict[str, np.dtype]
        Data type per variable
    """
    var_preserved_dims: dict[str, tuple[str, ...]] = {}
    var_preserved_shape: dict[str, tuple[int, ...]] = {}
    var_dtype: dict[str, np.dtype] = {}

    for var_name in target_vars:
        for ds in datasets:
            if var_name not in ds:
                continue
            da = ds[var_name]
            
            # Verify variable contains the spatial dims
            missing = [d for d in reproject_dims if d not in da.dims]
            if missing:
                warnings.warn(
                    f"Variable '{var_name}' does not contain reproject_dims "
                    f"{missing}; skipping.",
                    stacklevel=2,
                )
                break
            
            # Extract preserved (non-spatial) dimensions
            pres_dims = tuple(d for d in da.dims if d not in reproject_dims)
            pres_shape = tuple(
                da.sizes[d] for d in da.dims if d not in reproject_dims
            )
            var_preserved_dims[var_name] = pres_dims
            var_preserved_shape[var_name] = pres_shape
            var_dtype[var_name] = da.dtype
            break

    return var_preserved_dims, var_preserved_shape, var_dtype
