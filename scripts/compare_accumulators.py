#!/usr/bin/env python3
"""
Compare Kahan vs Simple accumulator results.

Computes differences between corresponding variables in two NetCDF files
and outputs a diff dataset.
"""

import xarray as xr
import numpy as np

# Input files
kahan_file = "/mnt/ceph/proj/USINE/LST/LST/TESTING__reproj_ds__acc_kahan__ss_2__constant__50m.nc"
simple_file = "/mnt/ceph/proj/USINE/LST/LST/TESTING__reproj_ds__acc_simple__ss_2__constant__50m.nc"

# Load datasets
print("Loading Kahan dataset...")
ds_kahan = xr.open_dataset(kahan_file)

print("Loading Simple dataset...")
ds_simple = xr.open_dataset(simple_file)

print(f"\nKahan dataset shape: {ds_kahan.dims}")
print(f"Simple dataset shape: {ds_simple.dims}")

# Create diff dataset - collect all diffs first
diff_vars = {}

# Compute differences for all non-count variables
print("\nComputing differences for variables:")
for var in ds_kahan.data_vars:
    if var.startswith("count_"):
        print(f"  Skipping {var} (count variable)")
        continue
    
    if var not in ds_simple:
        print(f"  Warning: {var} not in simple dataset, skipping")
        continue
    
    print(f"  {var}")
    diff = ds_kahan[var] - ds_simple[var]
    diff.attrs["long_name"] = f"Difference: Kahan - Simple ({var})"
    diff_vars[var] = diff
    
    # Print statistics
    abs_diff = np.abs(diff.values)
    valid_mask = ~np.isnan(abs_diff)
    if valid_mask.any():
        max_diff = np.nanmax(abs_diff)
        mean_diff = np.nanmean(abs_diff)
        print(f"    Max absolute diff: {max_diff:.6e}")
        print(f"    Mean absolute diff: {mean_diff:.6e}")
    else:
        print(f"    No valid differences (all NaN)")

# Create dataset from diff variables
diff_ds = xr.Dataset(diff_vars)

# Output file
output_file = "/mnt/ceph/proj/USINE/LST/LST/TESTING__diff_kahan_vs_simple__ss_2__constant__50m.nc"

print(f"\nSaving diff dataset to: {output_file}")
diff_ds.to_netcdf(output_file)

print("\nDone!")
print(f"\nDiff dataset contains {len(diff_ds.data_vars)} variables:")
for var in diff_ds.data_vars:
    print(f"  - {var}")
