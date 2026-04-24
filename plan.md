
# QUARTZ
Quantized Array Reprojection Transform on Z-levels

## GOAL

Provide a simple fast python library to project geospatial xarray data to Level 3 grids, by accumulating / aggregating one or several datasets.

## STATUS

The design direction is solid, but the plan is not yet complete for implementation.
Missing parts are mainly about exact API behavior, output contract, invalid-data policy, and measurable definition of done.

## CONSTRAINTS

A: must be able to aggregate many volumetric datasets
B: must be fast
C: must be able to produce aggregated datasets without holes (to a certain degree)
D: must be able to ingest N DIMS datasets as long as XY latitudes longitudes rasters exists
E: must be able to aggregate large amount of datasets
F: must be able to mitigate floating point precision error (to a degree)
    -> cf error_comp.md (Kahan sums)

## AXIOMS

A: RAM must at least fit a single output raster + a intermediary counter raster 
B: RAM may NOT fit all output rasters (solution: batch process variables, store to disk between batches)



## INPUTS

datasets = list of xarray Datasets objects (one or more)
projection = Projection object instance (EquirectangularProjection or PolarProjection)
    Example: EquirectangularProjection(width=3600, height=1800, area={"north": 60, "west": -10, "south": 40, "east": 30})
    NOTE: equirectangular is not necessarily for the whole world - can be regional
supersampling = INT (how much to decompose source pixels; 2 = 1px -> 2x2sub px) [CONSTRAINTS.C]
dtype = numpy dtype for output grids (default: np.float64, can be float32 for memory efficiency)
mask = optional callable that takes a dataset and returns a boolean DataArray aligned to source geolocation grid
vars_batch_size = INT number of output variables to process simultaneously (default: 1 for streaming, None for in-memory)
reproject_dims = tuple of source spatial dims to reproject (default: infer from lat/lon dims)
preserve_dims = tuple of non-spatial dims to preserve in outputs (default: preserve all non-reprojected dims)
sum_method = "naive" or "kahan" (kahan is interface-ready, implementation planned later)

## OUTPUT CONTRACT

Output must be an xarray.Dataset with one DataArray per aggregated variable, plus optional diagnostics.

Required:
- Output variables contain mean values over projected samples.
- Coordinates are 1D (`y`, `x`) or (`latitude`, `longitude`) with explicit metadata.
- Missing cells (counter=0) are filled with `np.nan` (default policy).

Optional diagnostics (controlled by flags):
- `count_<var>`: number of samples accumulated per pixel.
- `sum_<var>`: raw sum (useful for audits).
- `kahan_comp_<var>`: compensation grid if Kahan is enabled and exported.

Output metadata:
- Projection metadata in attrs (`projection_name`, `area`, `width`, `height`).
- Processing metadata in attrs (`supersampling`, `dtype`, `mode`, `sum_method`, `reproject_dims`, `preserve_dims`).

## PUBLIC API (V1)

```python
def aggregate(
    datasets: list[xr.Dataset],
    projection: ProjectionInterface,
    *,
    variables: list[str] | None = None,
    fail_on_schema_mismatch: bool = False,
    reproject_dims: tuple[str, ...] | None = None,
    preserve_dims: tuple[str, ...] | None = None,
    supersampling: int = 1,
    dtype: np.dtype = np.float64,
    mask: callable | None = None,
    vars_batch_size: int | None = None,
    sum_method: Literal["naive", "kahan"] = "naive",
    skipna: bool = True,
    return_counts: bool = False,
    return_sums: bool = False,
) -> xr.Dataset:
    ...
```

Rules:
- If `variables` is None: aggregate union of numeric variables across datasets.
- If `fail_on_schema_mismatch=True`: raise if variable sets differ across datasets.
- If `variables` is explicitly provided: missing variable in any dataset raises an error.
- `reproject_dims` define which source dims are projected to target (`y`, `x`).
- If `reproject_dims=None`: infer from latitude/longitude DataArray dimensions.
- `preserve_dims=None` preserves all non-reprojected dims (time, z, channel, ...).
- `supersampling` must be >=1, otherwise raise ValueError.
- `vars_batch_size=None` means in-memory mode, otherwise streaming mode.
- `vars_batch_size<=0` raises ValueError.
- `mask` takes one dataset and must return a boolean DataArray aligned with source lat/lon raster.
- `sum_method` is API-stable now: `naive` implemented in V1, `kahan` plugs in later without API changes.

## DATA ASSUMPTIONS AND VALIDATION

Required per dataset:
- Latitude and longitude rasters must exist, be 2D, and have same shape.
- Aggregated variable data must be broadcast-compatible with lat/lon raster shape.

Validation behavior:
- Hard errors for missing geolocation fields or incompatible shapes.
- Warn-and-skip for variables that cannot be cast to numeric.
- Hard error if any explicitly requested variable is missing.

## NDIMS REPROJECTION MODEL

This library must support N-dimensional variables, not only 2D rasters.

Principle:
- Source geolocation (`latitude`, `longitude`) is always 2D.
- Reprojection is applied only across `reproject_dims` (the 2D source geolocation dimensions).
- All other variable dimensions are preserved in output.

Example:
- Input variable dims: `(time, z, y_src, x_src)`.
- `reproject_dims=("y_src", "x_src")`.
- Output variable dims: `(time, z, y, x)`.

Additional valid examples:
- 3D variable: `(y_src, x_src, band)` -> output `(y, x, band)`.
- 4D variable: `(time, y_src, x_src, band)` -> output `(time, y, x, band)`.

Implementation implications:
- Compute projection indexes once per dataset geolocation grid.
- Vectorize over preserved dims as independent slices for accumulation.
- Accumulator/counter shapes must be `(preserved_dims..., projection.height, projection.width)`.
- Streaming mode can batch by variable and, if needed, by preserved-dim slices for memory safety.

## INVALID VALUES POLICY

Default policy:
- NaN/inf input values do not contribute to sums or counters.
- Out-of-bounds projected indexes are always dropped.
- Final output cell is NaN if counter is 0.

This policy must be shared by both normal summation and Kahan summation paths.


## MODES

### In-Memory Mode (vars_batch_size=None)
- All output variables created/loaded in RAM simultaneously
- Single pass through all datasets
- FASTEST: no disk I/O between output variables
- PARALLELIZABLE: if ≥2 variables in memory, can parallelize accumulation
- REQUIRES: RAM ≥ (num_variables * grid_size * dtype_size * 2)  # *2 for counters
- THIS MODE IS OF HIGHER PRIORITY FOR IMPLEMENTATION

### Streaming Mode (vars_batch_size=1..N)
- Process vars_batch_size variables at a time
- Multiple passes through datasets (one per batch)
- Slower due to repeated dataset iteration
- MEMORY EFFICIENT: only hold subset of variables
- Use when RAM < all variables (need margin though for opening input dataset + computations)

#### RAM detection avoid
- for now assume the user is responsible for choosing the mode / batch size for its own hardware.

### Mode Selection Strategy
- Try in-memory first
- Fall back to streaming if OOM
- Batch size optimization: max(1, floor(available_RAM / (grid_size * dtype_size * 2)))

## PROJECTION INTERFACE

### Supported Projections
The aggregation engine supports multiple projection types through a common interface:
- **EquirectangularProjection** (PRIORITY 1 - implement first)
- **PolarProjection** (PRIORITY 2 - implement later)

### Projection Contract
All projection classes must implement this interface:

```python
class ProjectionInterface:
    """
    Required interface for projection classes.
    All projections must implement these methods/attributes.
    """
    
    # REQUIRED ATTRIBUTES
    width: int                    # Output grid width in pixels
    height: int                   # Output grid height in pixels
    uint_type: np.dtype          # Index dtype (uint16 or uint32)
    FILL_VALUE_OOB: int          # Sentinel value for out-of-bounds (max of uint_type)
    
    # REQUIRED METHODS
    def project_to_indexes(self, latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project lat/lon coordinates to pixel indexes.
        
        Args:
            latitude: Array of latitude values
            longitude: Array of longitude values
        
        Returns:
            (x_indexes, y_indexes): Pixel coordinates as uint arrays
                                    Out-of-bounds coords return FILL_VALUE_OOB
        """
        pass
    
    def is_valid_index(self, x_indexes: np.ndarray, y_indexes: np.ndarray) -> np.ndarray:
        """
        Check which indexes are valid (not out-of-bounds).
        
        Args:
            x_indexes: Array of x pixel coordinates
            y_indexes: Array of y pixel coordinates
        
        Returns:
            Boolean array: True where indexes are valid (not FILL_VALUE_OOB)
        """
        pass
```

### Using Projection Classes

```python
from quartz.projection.equirectangular import EquirectangularProjection
# from quartz.projection.polar import PolarProjection  # Future

# Create projection for target grid
projection = EquirectangularProjection(
    width=3600,
    height=1800,
    area={"north": 70, "west": -20, "south": 30, "east": 40}  # European region
)

# The aggregation engine uses:
# - projection.project_to_indexes(lat, lon) -> (x, y)
# - projection.is_valid_index(x, y) -> valid_mask
# - projection.FILL_VALUE_OOB to detect out-of-bounds
# - projection.width, projection.height for output grid shape
```

### Why This Interface?
- **Decoupling**: Aggregation logic is independent of projection math
- **Extensibility**: Easy to add new projections (Mercator, Lambert, etc.)
- **Testability**: Each projection can be tested independently
- **Consistency**: FILL_VALUE_OOB pattern works for all projections

### Implementation Priority
1. **EquirectangularProjection** - Already implemented (see `quartz/projection/equirectangular.py`)
   - Handles global and regional areas
   - uint16/uint32 automatic selection
   - FILL_VALUE_OOB for out-of-bounds
   
2. **PolarProjection** - Implement later
   - Stereographic or azimuthal equidistant
   - Same interface, different math
   - More complex handling of singularities

## IMPLEMENTATION NOTES

### FILL_VALUE Handling
- Projection returns FILL_VALUE (uint max) for out-of-bounds coordinates
- Must filter using projection.is_valid_index() before accumulation
- Only accumulate valid indexes to avoid writing to invalid memory locations

### Supersampling Deltas
- Lat/lon deltas are NOT constant - they vary per pixel
- Must compute deltas per-pixel or per-source-grid
- Cannot use simple broadcasting for all sub-pixels

Minimum V1 simplification:
- Assume regular source raster spacing and derive deltas from local neighboring pixels.
- Document that highly irregular swath geolocation supersampling is out of V1 scope.

### Coordinate Edge Cases
- Antimeridian wrapping (±180° crossing) is DANGEROUS on non-global area projections
- Verify projection handles partial-globe areas correctly
- Polar singularities may cause issues at high latitudes

### Parallelization Strategy (in-memory mode only)
- If ≥2 variables in memory: parallelize across variables
- Each thread/process accumulates to its own variable grid
- No race conditions since variables are independent
- Use multiprocessing.Pool or concurrent.futures

### Summation Backend Strategy
- Summation backend is selected by `sum_method`.
- V1 uses `naive` summation.
- `kahan` is intentionally part of the public API now and will be implemented in a later iteration.
- Both backends must share identical masking, validity filtering, and mean finalization behavior.

## FUNCTIONNING

```py
# In-memory mode (all output rasters fit in RAM)
# No supersampling

# Get output grid dimensions from projection
height, width = projection.height, projection.width

# Initialize output grids and counters for all variables
output_grids = {}
counters = {}
for var_name in all_variable_names:
    output_grids[var_name] = np.zeros((height, width), dtype=dtype)
    counters[var_name] = np.zeros((height, width), dtype=np.uint32)

for each dataset:
    # Optional mask BEFORE projection
    valid_input_mask = None
    if mask:
        valid_input_mask = mask(dataset)
    
    # Compute projection's indexes using projection object
    lat, lon = dataset.lat, dataset.lon
    x_idx, y_idx = projection.project_to_indexes(lat, lon)
    
    # Filter out-of-bounds coordinates using projection's method
    valid_mask = projection.is_valid_index(x_idx, y_idx)
    if valid_input_mask is not None:
        valid_mask = valid_mask & np.asarray(valid_input_mask)
    x_valid = x_idx[valid_mask]
    y_valid = y_idx[valid_mask]
    
    for each variable in dataset:
        values = dataset[variable].values
        values_valid = values[valid_mask]
        
        # Accumulate using fancy indexing with np.add.at (handles duplicates)
        np.add.at(output_grids[variable], (y_valid, x_valid), values_valid)
        np.add.at(counters[variable], (y_valid, x_valid), 1)

# Finalize: divide by counters to get mean
for var_name in output_grids:
    mask = counters[var_name] > 0
    output_grids[var_name][mask] /= counters[var_name][mask]
```

```py
# In-memory mode with supersampling
# Deltas vary per pixel!

# Get output grid dimensions from projection
height, width = projection.height, projection.width

# Initialize output grids and counters
output_grids = {}
counters = {}
for var_name in all_variable_names:
    output_grids[var_name] = np.zeros((height, width), dtype=dtype)
    counters[var_name] = np.zeros((height, width), dtype=np.uint32)

for each dataset:
    # Optional mask
    valid_input_mask = None
    if mask:
        valid_input_mask = mask(dataset)
    
    lat, lon = dataset.lat, dataset.lon
    
    # Compute per-pixel deltas (varies spatially!)
    lat_delta = compute_lat_delta(lat)  # e.g., diff between adjacent pixels
    lon_delta = compute_lon_delta(lon)
    
    # Generate supersampling offsets
    offsets = np.linspace(-0.5, 0.5, supersampling, endpoint=False) + 0.5/supersampling
    
    for each variable in dataset:
        values = dataset[variable].values
        
        # Supersample and accumulate
        for i_offset in offsets:
            for j_offset in offsets:
                # Compute sub-pixel coordinates
                sub_lat = lat + lat_delta * i_offset
                sub_lon = lon + lon_delta * j_offset
                
                # Project sub-pixels using projection object
                x_idx, y_idx = projection.project_to_indexes(sub_lat, sub_lon)
                
                # Filter valid indexes
                valid_mask = projection.is_valid_index(x_idx, y_idx)
                if valid_input_mask is not None:
                    valid_mask = valid_mask & np.asarray(valid_input_mask)
                x_valid = x_idx[valid_mask]
                y_valid = y_idx[valid_mask]
                values_valid = values[valid_mask]
                
                # Accumulate (each sub-pixel contributes)
                np.add.at(output_grids[variable], (y_valid, x_valid), values_valid)
                np.add.at(counters[variable], (y_valid, x_valid), 1)
        
# Finalize
for var_name in output_grids:
    mask = counters[var_name] > 0
    output_grids[var_name][mask] /= counters[var_name][mask]
```

```py
# Streaming mode (vars_batch_size = 1 or small N)
# Process variables in batches to save memory

# Get output grid dimensions from projection
height, width = projection.height, projection.width

all_vars = list(all_variable_names)
for batch_start in range(0, len(all_vars), vars_batch_size):
    batch_vars = all_vars[batch_start:batch_start + vars_batch_size]
    
    # Initialize grids for this batch only
    output_grids = {var: np.zeros((height, width), dtype=dtype) for var in batch_vars}
    counters = {var: np.zeros((height, width), dtype=np.uint32) for var in batch_vars}
    
    # Iterate through ALL datasets for this batch
    for each dataset:
        valid_input_mask = None
        if mask:
            valid_input_mask = mask(dataset)
        
        # Project coordinates using projection object
        lat, lon = dataset.lat, dataset.lon
        x_idx, y_idx = projection.project_to_indexes(lat, lon)
        valid_mask = projection.is_valid_index(x_idx, y_idx)
        if valid_input_mask is not None:
            valid_mask = valid_mask & np.asarray(valid_input_mask)
        x_valid = x_idx[valid_mask]
        y_valid = y_idx[valid_mask]
        
        # Accumulate only variables in current batch
        for var_name in batch_vars:
            if var_name in dataset:
                values_valid = dataset[var_name].values[valid_mask]
                np.add.at(output_grids[var_name], (y_valid, x_valid), values_valid)
                np.add.at(counters[var_name], (y_valid, x_valid), 1)
    
    # Finalize and save this batch to disk
    for var_name in batch_vars:
        mask = counters[var_name] > 0
        output_grids[var_name][mask] /= counters[var_name][mask]
        save_to_disk(var_name, output_grids[var_name])  # e.g., NetCDF, Zarr
    
    # Free memory before next batch
    del output_grids, counters
```

## TEST STRATEGY (DEFINITION OF DONE)

Unit tests:
- Projection contract tests (already started for equirectangular).
- Aggregation correctness on synthetic grids with known expected means.
- NDIMS aggregation tests: preserve non-spatial dims and reproject only source spatial dims.
- NaN/inf handling tests.
- Out-of-bounds filtering tests.
- Supersampling behavior tests (`supersampling=1`, `2`, `4`).
- Kahan vs naive sum comparison on precision-sensitive cases.

Integration tests:
- Multiple datasets with partial variable overlap.
- In-memory and streaming mode produce same result within tolerance.

Tolerance policy for parity tests:
- Do not hardcode tolerances inside assertions.
- Define module-level constants at the top of each test file, e.g. `PARITY_RTOL` and `PARITY_ATOL`.
- Default moderate values: `PARITY_RTOL = 1e-6`, `PARITY_ATOL = 1e-9`.

Performance tests:
- Baseline benchmark and regression guard (time and memory).
- Report throughput in MPix/s or samples/s.

Acceptance criteria:
- Numerical parity between in-memory and streaming modes within fixed tolerance.
- No crashes on datasets with holes/masked values.
- Deterministic outputs for deterministic inputs.

## ITERATION ROADMAP

Iteration 1 (MVP):
- Implement `aggregate()` with in-memory mode, no supersampling, with NDIMS support.
- Return output means + optional counts.
- Add core aggregation tests.

Iteration 2:
- Add streaming mode (`vars_batch_size`) and parity tests vs in-memory mode.
- Add NetCDF persistence hook for batch outputs.

Iteration 3:
- Add supersampling v1 for regular grids.
- Add benchmarks and optimization pass (vectorization and memory access).

Iteration 4:
- Implement `sum_method="kahan"` backend with same interface and outputs as naive mode.
- Add precision-focused regression tests using scenarios from `error_comp.md`.

Iteration 5:
- Add PolarProjection implementation behind same interface.
- Add projection-agnostic integration tests.

## KNOWN OPEN QUESTIONS

Resolved for V1:
- Expose output with (`x`,`y`) plus helper (`latitude`,`longitude`) coordinates.
- `variables=None` means union of numeric variables across datasets.
- Streaming first-class persistence format is NetCDF.
- Replace `filter_func` with `mask` callable: `mask(ds) -> boolean DataArray` aligned to source geolocation raster.
- Missing explicitly requested variables raise hard error.
- NDIMS variables are first-class: reproject only `reproject_dims` and preserve remaining dims.
- `sum_method` is interface-stable now; `kahan` is planned for later implementation.