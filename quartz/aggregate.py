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


def _accumulate_variable(
    values: np.ndarray,
    var_dims: tuple[str, ...],
    reproject_dims: tuple[str, ...],
    target_flat_idx: np.ndarray,
    spatial_valid: np.ndarray,
    sum_acc_flat: np.ndarray,
    cnt_acc_flat: np.ndarray,
    T: int,
    skipna: bool,
) -> None:
    """
    Accumulate one variable into flat sum/count accumulators (in-place).

    Strategy: single np.bincount call over all preserved-dim slices at once —
    no Python loop over P.

    Parameters
    ----------
    values : np.ndarray
        Variable data in its original dim order.
    var_dims : tuple[str, ...]
        Dim names corresponding to *values* axes.
    reproject_dims : tuple[str, ...]
        The two spatial dim names to reproject away (must be in *var_dims*).
    target_flat_idx : np.ndarray, shape (S,), dtype uint32 or uint64
        Flat target pixel index per source pixel: y_idx * width + x_idx.
    spatial_valid : np.ndarray, shape (S,), dtype bool
        True where source pixel projects inside the target grid.
    sum_acc_flat : np.ndarray, shape (P*T,), dtype float64
        Running weighted sum; modified in-place.
    cnt_acc_flat : np.ndarray, shape (P*T,), dtype uint32
        Running sample count; modified in-place.
    T : int
        Total target pixels = height * width.
    skipna : bool
        If True, NaN/inf values do not contribute to accumulators.
    """
    # --- Reorder dims so spatial axes are last: (*preserved, ny_src, nx_src) ---
    spatial_axes = tuple(var_dims.index(d) for d in reproject_dims)
    preserved_axes = tuple(i for i in range(len(var_dims)) if i not in spatial_axes)
    perm = preserved_axes + spatial_axes
    # Avoid a copy when the order is already correct
    if perm != tuple(range(len(var_dims))):
        values = np.transpose(values, perm)

    # Flatten spatial dims and compute P (product of preserved dims)
    P = int(np.prod(values.shape[: len(preserved_axes)])) if preserved_axes else 1
    values_ps = np.asarray(values.reshape(P, -1), dtype=np.float64)  # (P, S)
    S = values_ps.shape[1]

    # Combined flat index: (p, s) -> p * T + target_flat_idx[s]
    # Match dtype with target_flat_idx (uint32 or uint64)
    idx_dtype = target_flat_idx.dtype
    T_typed = np.array(T, dtype=idx_dtype).item()
    combined_idx = (
        np.arange(P, dtype=idx_dtype)[:, np.newaxis] * T_typed
        + target_flat_idx[np.newaxis, :]
    )  # (P, S)

    # Build validity mask: OOB + (optionally) finite value check
    # Use a writeable copy to allow in-place &=
    valid = np.broadcast_to(spatial_valid[np.newaxis, :], (P, S)).copy()
    if skipna:
        valid &= np.isfinite(values_ps)

    flat_idx = combined_idx[valid]   # (N,)
    flat_vals = values_ps[valid]     # (N,)

    n = len(sum_acc_flat)
    sum_acc_flat += np.bincount(flat_idx, weights=flat_vals, minlength=n)
    cnt_acc_flat += np.bincount(flat_idx, minlength=n).astype(np.uint32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate(
    datasets: list[xr.Dataset]|xr.Dataset,
    projection,
    *,
    variables: list[str] | None = None,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    fail_on_schema_mismatch: bool = False,
    supersampling: int = 1,
    mask: Callable | None = None,
    vars_batch_size: int | None = None,
    sum_method: Literal["naive", "kahan"] = "naive",
    skipna: bool = True,
    return_counts: bool = False,
    return_sums: bool = False,
) -> xr.Dataset:
    """
    Reproject and aggregate a list of xarray Datasets onto a target projection grid.

    Parameters
    ----------
    datasets : list[xr.Dataset]
        Source datasets to aggregate. Must each contain 2-D lat/lon variables.
    projection : ProjectionInterface
        Target grid projection (e.g. EquirectangularProjection).
    variables : list[str] | None
        Variables to aggregate. ``None`` → union of all numeric variables found
        across datasets.
    lat_name : str
        Name of the latitude variable (default ``"latitude"``).
    lon_name : str
        Name of the longitude variable (default ``"longitude"``).
    fail_on_schema_mismatch : bool
        Raise if numeric variable sets differ across datasets (only in union mode).
    supersampling : int
        Sub-pixel sampling factor. Must be 1 (>1 not yet implemented).
    mask : callable | None
        ``mask(ds) -> xr.DataArray`` — return a boolean DataArray with the same
        spatial dims as the source lat/lon grid. ``True`` = keep pixel.
    vars_batch_size : int | None
        ``None`` = in-memory mode. Streaming not yet implemented.
    sum_method : {"naive", "kahan"}
        Summation backend. ``"naive"`` is implemented; ``"kahan"`` is
        interface-ready and will be added in a later iteration.
    skipna : bool
        If ``True`` (default), NaN/inf values do not contribute to sums or counts.
    return_counts : bool
        Add ``count_<var>`` arrays to output.
    return_sums : bool
        Add ``sum_<var>`` arrays to output.

    Returns
    -------
    xr.Dataset
        Aggregated dataset. Each output variable has dims
        ``(*preserved_dims, y, x)`` with ``latitude`` / ``longitude`` helper
        coordinates attached. Output dtypes match input variable dtypes.
    """
    # ------------------------------------------------------------------
    # V1 guard rails
    # ------------------------------------------------------------------
    
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not datasets:
        raise ValueError("datasets must be a non-empty list.")
    if supersampling != 1:
        raise NotImplementedError("supersampling > 1 is not yet implemented.")
    if vars_batch_size is not None:
        raise NotImplementedError(
            "Streaming mode (vars_batch_size is not None) is not yet implemented."
        )
    if sum_method == "kahan":
        raise NotImplementedError("sum_method='kahan' is not yet implemented.")

    height, width = projection.height, projection.width
    T = height * width

    # Determine index dtype based on grid size
    # uint32 max = 4,294,967,295 (fits grids up to ~65k × 65k)
    # uint64 max = 18,446,744,073,709,551,615 (for larger grids)
    if T < np.iinfo(np.uint32).max:
        index_dtype = np.uint32
    else:
        index_dtype = np.uint64

    # ------------------------------------------------------------------
    # Validate geolocation variables and infer spatial dimensions
    # ------------------------------------------------------------------
    first_ds = datasets[0]
    
    for ds in datasets:
        _validate_geolocation(ds, lat_name, lon_name)

    # Infer spatial dims from lat/lon dimensions (must be identical)
    reproject_dims = tuple(first_ds[lat_name].dims)

    geo_vars: set[str] = {lat_name, lon_name}

    # ------------------------------------------------------------------
    # Collect target variables
    # ------------------------------------------------------------------
    target_vars = _collect_variables(
        datasets, variables, geo_vars, fail_on_schema_mismatch
    )

    if not target_vars:
        raise ValueError(
            "No numeric variables found to aggregate. "
            "Check variable names and dataset contents."
        )

    # ------------------------------------------------------------------
    # Determine preserved shape and dtype per variable (from first dataset that has it)
    # ------------------------------------------------------------------
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
            pres_dims = tuple(d for d in da.dims if d not in reproject_dims)
            pres_shape = tuple(
                da.sizes[d] for d in da.dims if d not in reproject_dims
            )
            var_preserved_dims[var_name] = pres_dims
            var_preserved_shape[var_name] = pres_shape
            var_dtype[var_name] = da.dtype
            break

    # Keep only variables that passed the spatial-dims check
    target_vars = [v for v in target_vars if v in var_preserved_shape]

    if not target_vars:
        raise ValueError(
            "None of the candidate variables contain the reprojection dimensions "
            f"{reproject_dims}."
        )

    # ------------------------------------------------------------------
    # Allocate flat accumulators: shape (P * T,) per variable
    # ------------------------------------------------------------------
    sum_accs: dict[str, np.ndarray] = {}
    cnt_accs: dict[str, np.ndarray] = {}

    for var_name in target_vars:
        pres_shape = var_preserved_shape[var_name]
        P = int(np.prod(pres_shape)) if pres_shape else 1
        sum_accs[var_name] = np.zeros(P * T, dtype=np.float64)
        cnt_accs[var_name] = np.zeros(P * T, dtype=np.uint32)

    # ------------------------------------------------------------------
    # Main accumulation loop
    # ------------------------------------------------------------------
    for ds in datasets:
        lat_flat = ds[lat_name].values.ravel().astype(np.float64)
        lon_flat = ds[lon_name].values.ravel().astype(np.float64)

        x_idx, y_idx = projection.project_to_indexes(lat_flat, lon_flat)
        spatial_valid = projection.is_valid_index(x_idx, y_idx)  # (S,) bool

        if mask is not None:
            user_mask_da = mask(ds)
            spatial_valid = spatial_valid & user_mask_da.values.ravel().astype(bool)

        # Flat target index — computed once, reused for all variables
        # Use uint32/uint64 based on grid size to minimize memory
        target_flat_idx = (
            y_idx.astype(index_dtype) * index_dtype(width)
            + x_idx.astype(index_dtype)
        )  # (S,)

        for var_name in target_vars:
            if var_name not in ds:
                continue

            da = ds[var_name]
            values = da.values

            # Validate consistent preserved shape across datasets
            actual_pres_shape = tuple(
                da.sizes[d] for d in da.dims if d not in reproject_dims
            )
            expected = var_preserved_shape[var_name]
            if actual_pres_shape != expected:
                raise ValueError(
                    f"Variable '{var_name}' has inconsistent preserved dim shape "
                    f"across datasets: expected {expected}, got {actual_pres_shape}."
                )

            P = int(np.prod(expected)) if expected else 1

            _accumulate_variable(
                values=values,
                var_dims=da.dims,
                reproject_dims=reproject_dims,
                target_flat_idx=target_flat_idx,
                spatial_valid=spatial_valid,
                sum_acc_flat=sum_accs[var_name],
                cnt_acc_flat=cnt_accs[var_name],
                T=T,
                skipna=skipna,
            )

    # ------------------------------------------------------------------
    # Finalize: mean = sum / count; NaN where count == 0
    # Output in original dtype (accumulated in float64 for precision)
    # ------------------------------------------------------------------
    out_lat, out_lon = projection.get_coordinates()  # 1-D: (height,), (width,)

    data_vars: dict[str, xr.DataArray] = {}

    for var_name in target_vars:
        pres_shape = var_preserved_shape[var_name]
        pres_dims = var_preserved_dims[var_name]
        out_shape = (*pres_shape, height, width)
        original_dtype = var_dtype[var_name]

        sum_grid = sum_accs[var_name].reshape(out_shape)
        cnt_grid = cnt_accs[var_name].reshape(out_shape)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_grid = np.where(
                cnt_grid > 0, sum_grid / cnt_grid, np.nan
            ).astype(original_dtype)

        out_dims = pres_dims + ("y", "x")
        data_vars[var_name] = xr.DataArray(mean_grid, dims=out_dims)

        if return_counts:
            data_vars[f"count_{var_name}"] = xr.DataArray(cnt_grid, dims=out_dims)
        if return_sums:
            data_vars[f"sum_{var_name}"] = xr.DataArray(
                sum_grid.astype(original_dtype), dims=out_dims
            )

    # ------------------------------------------------------------------
    # Build output coordinates and global attributes
    # ------------------------------------------------------------------
    coords: dict[str, xr.DataArray] = {
        "latitude": xr.DataArray(out_lat, dims=["y"]),
        "longitude": xr.DataArray(out_lon, dims=["x"]),
    }

    area_attrs: dict[str, float] = {
        attr: getattr(projection, attr)
        for attr in ("north", "south", "east", "west")
        if hasattr(projection, attr)
    }

    global_attrs = {
        "projection_name": type(projection).__name__,
        "projection_width": projection.width,
        "projection_height": projection.height,
        "projection_area": str(area_attrs) if area_attrs else "N/A",
        "supersampling": supersampling,
        "index_dtype": str(index_dtype),
        "mode": "in-memory",
        "sum_method": sum_method,
        "reproject_dims": str(reproject_dims),
        "n_datasets": len(datasets),
    }

    return xr.Dataset(data_vars, coords=coords, attrs=global_attrs)
