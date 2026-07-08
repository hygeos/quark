"""
quark.aggregate
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

from quark import accumulate, supersampling

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
        datasets: list[xr.Dataset] | xr.Dataset,
        projection,
        lat_name: str = "latitude",
        lon_name: str = "longitude",
        variables: list[str] | None = None,
        fail_on_schema_mismatch: bool = True,
        sum_method: Literal["simple", "kahan"] = "simple",
        skipna: bool = True,
        supersampler: supersampling._BaseSuperSampler | None = None,
        return_counts: bool = False,
        return_sums: bool = False,
        dtype=None,
    ):
        """
        Initialize aggregator with projection and configuration.
        
        Parameters
        ----------
        datasets : list[xr.Dataset]
            Source datasets (for metadata extraction)
        projection : ProjectionInterface
            Target grid projection
        lat_name, lon_name : str
            Geolocation variable names
        variables : list[str] | None
            Variables to aggregate (None = auto-detect)
        fail_on_schema_mismatch : bool
            Raise if variable sets differ across datasets
        sum_method : {"simple", "kahan"}
            Summation strategy
        skipna : bool
            Skip NaN/inf values
        supersampler : BaseSupersampler | None
            Supersampling strategy (SpatialSupersampling, ConstantSupersampling).
            If None, no supersampling is applied.
        return_counts : bool
            Include count arrays in output
        return_sums : bool
            Include sum arrays in output
        dtype : np.dtype | None
            Override output dtype for all variables
        
        Examples
        --------
        >>> from quark.supersampling import SpatialSupersampling
        >>> supersampler = SpatialSupersampling(factor=2)
        >>> agg = Aggregator(projection, datasets, supersampler=supersampler)
        """
        
        if not isinstance(datasets, list):
            datasets = [datasets]
        
        # Validate inputs
        if not datasets:
            raise ValueError("datasets must be a non-empty list.")
        
        self.projection = projection
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.skipna = skipna
        self.sum_method = sum_method
        self.supersampler = supersampler
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
        
        # No supersampler: project once and done
        if self.supersampler is None:
            log.info("Projecting coordinates (no supersampling)...")
            self._project_batch(ds, lat, lon, values_slices=(slice(None), slice(None)))
        else:
            # Prepare supersampler (compute cached data like pixel widths)
            self.supersampler.prepare(lat, lon)
            
            # Iterate over all subpixels
            factor = self.supersampler.factor
            for i in range(factor):
                for j in range(factor):
                    # Check if we should project center
                    offset_y, offset_x = supersampling._compute_subpixel_offset(i, j, factor)
                    is_center = (offset_y == 0 and offset_x == 0)
                    
                    if is_center and not self.supersampler.project_center:
                        log.info(f"Skipping center subpixel ({i},{j})...")
                        continue
                    
                    log.info(f"Projecting subpixel ({i},{j}) of {factor}x{factor}...")
                    
                    # Compute subpixel coordinates
                    lat_sub, lon_sub, values_slices = self.supersampler.compute_coords(
                        lat, lon, i, j
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
        
        # Build coordinates (handle both 1D separable and 2D coupled projections)
        lat_dims = ["y", "x"] if out_lat.ndim == 2 else ["y"]
        lon_dims = ["y", "x"] if out_lon.ndim == 2 else ["x"]
        coords: dict[str, xr.DataArray] = {
            "latitude": xr.DataArray(out_lat, dims=lat_dims),
            "longitude": xr.DataArray(out_lon, dims=lon_dims),
        }
        
        # Build global attributes
        area_attrs: dict[str, float] = {
            attr: getattr(self.projection, attr)
            for attr in ("north", "south", "east", "west")
            if hasattr(self.projection, attr)
        }
        
        # Supersampling info
        if self.supersampler is not None:
            ss_info = {
                "supersampling_factor": self.supersampler.factor,
                "supersampling_type": type(self.supersampler).__name__,
                "supersampling_project_center": int(self.supersampler.project_center),
            }
            # Add pixel_width for ConstantSupersampling
            if isinstance(self.supersampler, supersampling.ConstantSuperSampler):
                ss_info["supersampling_pixel_width"] = self.supersampler.pixel_width
        else:
            ss_info = {"supersampling_factor": 1}
        
        global_attrs = {
            "projection_name": type(self.projection).__name__,
            "projection_width": self.projection.width,
            "projection_height": self.projection.height,
            "projection_area": str(area_attrs) if area_attrs else "N/A",
            **ss_info,
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
