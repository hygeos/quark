"""Compare quartz vs geoutils regrid outputs"""
from pathlib import Path
import numpy as np
import xarray as xr

quartz_file = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__reproj_ds.nc")
geoutils_file = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__reproj_ds_geoutils.nc")

print("Loading datasets...")
ds_quartz = xr.open_dataset(quartz_file)
ds_geoutils = xr.open_dataset(geoutils_file)

print(f"\nQuartz variables: {list(ds_quartz.data_vars.keys())}")
print(f"Geoutils variables: {list(ds_geoutils.data_vars.keys())}")

# Find common variables (exclude count_* variables)
common_vars = [v for v in ds_quartz.data_vars if v in ds_geoutils.data_vars and not v.startswith('count_')]

print(f"\n{'Variable':<20} {'Shape':<15} {'RMSE':<12} {'Max Diff':<12} {'Valid Pixels'}")
print("=" * 80)

for var in common_vars:
    q = ds_quartz[var].values
    g = ds_geoutils[var].values
    
    # Handle potential dimension mismatch (quartz preserves dims, geoutils might reorder)
    if q.shape != g.shape:
        print(f"{var:<20} SHAPE MISMATCH: {q.shape} vs {g.shape}")
        continue
    
    # Compare only where both have valid data
    valid = np.isfinite(q) & np.isfinite(g)
    n_valid = np.sum(valid)
    
    if n_valid == 0:
        print(f"{var:<20} {str(q.shape):<15} NO VALID DATA")
        continue
    
    diff = q[valid] - g[valid]
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))
    
    print(f"{var:<20} {str(q.shape):<15} {rmse:<12.6f} {max_diff:<12.6f} {n_valid}")

print("\nComparison complete!")
