"""
Compare Simple vs Kahan summation on LST dataset with float32 and float64.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from quartz.aggregate import Aggregator
from quartz.projection.equirectangular import EquiRectangular

from core.monitor import Chrono
from core import log

src = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__input_ds.nc")
dst_dir = Path("/mnt/ceph/proj/USINE/LST/LST/")

ds = xr.open_dataset(src)

# Compute bounding box from geolocation arrays
lat = ds["latitude"].values
lon = ds["longitude"].values
valid = np.isfinite(lat) & np.isfinite(lon)

lat_min = float(lat[valid].min())
lat_max = float(lat[valid].max())
lon_min = float(lon[valid].min())
lon_max = float(lon[valid].max())

margin = 0.05
area = {
    "north": lat_max + margin,
    "south": lat_min - margin,
    "west": lon_min - margin,
    "east": lon_max + margin,
}

# Output resolution
width, height = 5000, 5000

print(f"Bounding box: {area}")
print(f"Output grid : {width} x {height} px")
print()

projection = EquiRectangular(width=width, height=height, area=area)

# Variable to test
variable = "lste_eco"  # Only process this one variable

# Test configurations
configs = [
    ("simple", np.float32),
    ("simple", np.float64),
    ("kahan", np.float32),
    ("kahan", np.float64),
]

results = {}

for sum_method, dtype in configs:
    label = f"{sum_method}_{dtype.__name__}"
    print(f"{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")
    
    with Chrono(f"Aggregation ({label})"):
        agg = Aggregator(
            projection=projection,
            datasets=[ds],
            variables=[variable],
            sum_method=sum_method,
            skipna=True,
            supersampling=1,
            return_counts=True,
            return_sums=True,
            dtype=dtype,
        )
        
        result = agg.compute()
        results[label] = result
    
    # Save result
    output_path = dst_dir / f"TESTING__acc__{label}.nc"
    log.info(f"Saving {label} to {output_path}...")
    result.to_netcdf(output_path)
    log.info(f"Saved → {output_path}")
    print()

# Compare results
print(f"{'='*60}")
print("COMPARISON ANALYSIS")
print(f"{'='*60}")

print(f"Variable: {variable}")
print()

# Compare simple float32 vs float64
diff_simple = results["simple_float64"][variable].values - results["simple_float32"][variable].values
valid_simple = np.isfinite(diff_simple)

print("Simple: float64 - float32")
print(f"  Mean abs diff : {np.abs(diff_simple[valid_simple]).mean():.6e}")
print(f"  Max abs diff  : {np.abs(diff_simple[valid_simple]).max():.6e}")
print(f"  RMS diff      : {np.sqrt((diff_simple[valid_simple]**2).mean()):.6e}")
print()

# Compare kahan float32 vs float64
diff_kahan = results["kahan_float64"][variable].values - results["kahan_float32"][variable].values
valid_kahan = np.isfinite(diff_kahan)

print("Kahan: float64 - float32")
print(f"  Mean abs diff : {np.abs(diff_kahan[valid_kahan]).mean():.6e}")
print(f"  Max abs diff  : {np.abs(diff_kahan[valid_kahan]).max():.6e}")
print(f"  RMS diff      : {np.sqrt((diff_kahan[valid_kahan]**2).mean()):.6e}")
print()

# Compare simple vs kahan (float32)
diff_method_f32 = results["kahan_float32"][variable].values - results["simple_float32"][variable].values
valid_method_f32 = np.isfinite(diff_method_f32)

print("float32: kahan - simple")
print(f"  Mean abs diff : {np.abs(diff_method_f32[valid_method_f32]).mean():.6e}")
print(f"  Max abs diff  : {np.abs(diff_method_f32[valid_method_f32]).max():.6e}")
print(f"  RMS diff      : {np.sqrt((diff_method_f32[valid_method_f32]**2).mean()):.6e}")
print()

# Compare simple vs kahan (float64)
diff_method_f64 = results["kahan_float64"][variable].values - results["simple_float64"][variable].values
valid_method_f64 = np.isfinite(diff_method_f64)

print("float64: kahan - simple")
print(f"  Mean abs diff : {np.abs(diff_method_f64[valid_method_f64]).mean():.6e}")
print(f"  Max abs diff  : {np.abs(diff_method_f64[valid_method_f64]).max():.6e}")
print(f"  RMS diff      : {np.sqrt((diff_method_f64[valid_method_f64]**2).mean()):.6e}")
print()

print("Done!")
