"""
Reproject OLCI daily PAR from equirectangular to azimuthal equidistant (polar) projection.

Source: Global equirectangular 8640×4320 (~0.02° resolution)
Target: Azimuthal equidistant with parametrized center point
"""

from pathlib import Path

import numpy as np
import xarray as xr

from quark.aggregate import Aggregator
from quark.projection.polar import AzimuthalEquidistant
from quark.supersampling import ConstantSuperSampler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

src = Path(
    "/mnt/ceph/data/PAR/L3/OLCI/valid/CAMS/"
    "OLCI-AB-20230615__NEQ_8640__L3-dailyPAR.nc"
)

# Output directory
out_dir = Path("/tmp/quark_polar_test")
out_dir.mkdir(parents=True, exist_ok=True)

# Projection center — change these to explore different views
CENTER_LAT = 90.0       # North Pole (try 0.0 for equatorial, 50.0 for Europe)
CENTER_LON = 0.0        # Prime meridian (try 10.0 for Europe)
RADIUS = 90.0           # Hemisphere (try 45.0 for zoomed view)

# Output resolution
OUTPUT_SIZE = 2000      # 2000×2000 pixels

# Aggregation settings
SUM_METHOD = "simple"   # or "kahan"
SSF_FACTOR = 2          # supersampling factor
SUBPX_MODE = "constant" # or "spatial"
PIXEL_WIDTH = "1km"     # pixel width for constant supersampling

# Variables to aggregate (None = all)
VARIABLES = ["daily_planar_PAR_(0+)"]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 60)
print("OLCI PAR → Azimuthal Equidistant Reprojection")
print("=" * 60)
print(f"Source: {src}")
print(f"Center: ({CENTER_LAT}°, {CENTER_LON}°)")
print(f"Radius: {RADIUS}°")
print(f"Output: {OUTPUT_SIZE}×{OUTPUT_SIZE} px")
print("=" * 60)

# Open source dataset
ds = xr.open_dataset(src)
print(f"\nSource dataset: {ds.sizes}")
print(f"Variables: {list(ds.data_vars)}")

# Create projection
projection = AzimuthalEquidistant(
    width=OUTPUT_SIZE,
    height=OUTPUT_SIZE,
    center_latitude=CENTER_LAT,
    center_longitude=CENTER_LON,
    radius=RADIUS,
)

print(f"\nProjection center pixel: ({projection._cx:.0f}, {projection._cy:.0f})")
print(f"Max pixel radius: {projection._max_pixel_radius:.0f}")

# Create supersampler
if SUBPX_MODE == "spatial":
    from quark.supersampling import SpatialSuperSampler
    supersampler = SpatialSuperSampler(factor=SSF_FACTOR, project_center=True)
else:
    supersampler = ConstantSuperSampler(
        factor=SSF_FACTOR, pixel_width=PIXEL_WIDTH, project_center=True
    )

# Build aggregator
agg = Aggregator(
    projection=projection,
    datasets=[ds],
    lat_name="lat",
    lon_name="lon",
    variables=VARIABLES,
    sum_method=SUM_METHOD,
    supersampler=supersampler,
    return_counts=True,
    return_sums=False,
    dtype=np.float32,
)

# Run aggregation
print("\nRunning aggregation...")
result = agg.compute()

# Save result
out_name = (
    f"PAR_20230615__az_eq__center_{CENTER_LAT:+.0f}_{CENTER_LON:+.0f}"
    f"__r{RADIUS:.0f}__{OUTPUT_SIZE}x{OUTPUT_SIZE}"
    f"__{SUM_METHOD}__ss{SSF_FACTOR}__{SUBPX_MODE}.nc"
)
dst = out_dir / out_name

print(f"\nResult dataset:\n{result}")
print(f"\nSaving to {dst}...")
result.to_netcdf(dst)
print(f"Saved → {dst}")

ds.close()
print("\nDone!")
