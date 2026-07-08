# Quark

Quantized Array Reprojection with Kahan Summation

---

Quark is a reprojection and aggregation engine for xarray datasets with 2-D geolocation arrays.
It is designed to be simple for common workflows and extensible for advanced use cases. 

## Map Example

<p align="center">
  <img src="docs/img/daily_PAR_polar_example.png" alt="Daily PAR reprojected to a polar view" width="600">
  <br>
  <em>Figure 1 — SEN3 OLCI: Daily PAR (Photosynthetically Active Radiation) aggregated and reprojected to a polar view using QUARK.</em>
</p>

## Highlights

- N-dimensional datasets — 2D, 3D, 4D, and beyond
- Multi-dataset processing
- Supersampling for improved spatial coverage
- Multiple projections
- Kahan Summation for improved accuracy (requires numba)

## Table of Contents

- [Installation](#installation)
- [Quick Example](#quick-example)
- [More Examples](#more-examples)
  - [ND Variables](#nd-variables)
  - [Supersampling](#supersampling)
  - [Multi-dataset Accumulation](#multi-dataset-accumulation)
- [Supersampling Modes](#supersampling-modes)
- [Projections](#projections)
  - [Polar Projection Example](#polar-projection-example)
- [Input Model](#input-model)
- [Advanced: Kahan Summation](#advanced-kahan-summation)
- [Development](#development)
- [License](#license)

## Installation

```bash
pixi install
```

or

```bash
pip install -e ".[git]"
```

## Quick example

```python
import xarray as xr

from quark.aggregate import Aggregator
from quark.projection.equirectangular import EquiRectangular
from quark.utils import bbox_area

ds = xr.open_dataset("input.nc")

projection = EquiRectangular(
    width=2000,
    height=2000,
    area=bbox_area(ds, margin=0.05),
)

result = Aggregator(
    projection=projection,
    datasets=[ds],
    return_counts=True,
).compute()

result.to_netcdf("output.nc")
```

## More examples

### ND variables

```python
result = Aggregator(
    projection=projection,
    datasets=[ds],
    variables=["radiance"],
).compute()
```

### Supersampling

```python
from quark.supersampling import ConstantSuperSampler

result = Aggregator(
    projection=projection,
    datasets=[ds],
    supersampler=ConstantSuperSampler(factor=3, pixel_width="500m"),
    return_counts=True,
).compute()
```

### Multi-dataset accumulation

```python
result = Aggregator(
    projection=projection,
    datasets=[ds1, ds2, ds3],
    variables=["lst"],
    return_counts=True,
    return_sums=True,
).compute()
```

## Supersampling modes

Supersampling projects a `factor x factor` subpixel grid for each source pixel. This improves coverage when the source footprint is large relative to the target grid.

`SpatialSuperSampler` estimates local pixel width from neighboring pixels in the 2-D lat/lon array. It works well for structured rasters and well-behaved swaths.

`ConstantSuperSampler` uses a fixed width such as `"1km"` or `"500m"`. It is the safer choice when the source geolocation is irregular.

Important limitation: spatial supersampling assumes that array neighbors are also spatial neighbors. It should not be used for unstructured inputs, badly ordered swaths, or 2-D arrays whose neighborhood topology is not physically meaningful.

## Projections

Projection support is class-based. The repository currently includes:

- `EquiRectangular`
- `PolarNorth`
- `PolarSouth`

Additional projection classes can be added as long as they expose the projection methods expected by `Aggregator`.

### Polar projection example

```python
import xarray as xr

from quark.aggregate import Aggregator
from quark.projection.polar import PolarSouth

ds = xr.open_dataset("input.nc")

projection = PolarSouth(
    width=2000,
    height=2000,
    radius_deg=45.0,       # angular radius from pole
    rotation_deg=0.0,
)

result = Aggregator(
    projection=projection,
    datasets=[ds],
    variables=["PAR"],
    return_counts=True,
).compute()

result.to_netcdf("output_polar.nc")
```

## Input model

QUARK expects:

- a 2-D `latitude` variable
- a 2-D `longitude` variable
- matching shapes and dimensions for both
- data variables that include those spatial dimensions

This makes it a strong fit for swaths and geolocated rasters, but not for generic point clouds or arbitrary meshes.

## Advanced: Kahan Summation

For high-precision accumulation, QUARK supports Kahan Summation via the `sum_method="kahan"` option.

```python
result = Aggregator(
    projection=projection,
    datasets=[ds],
    sum_method="kahan",
    return_counts=True,
).compute()
```

**Note:** Kahan Summation requires `numba` to be installed in the environment.
**Note:** Kahan Summation is slower than naive summation.


## Development

```bash
pixi run pytest tests/
```

## License

MIT. See [LICENSE](LICENSE).