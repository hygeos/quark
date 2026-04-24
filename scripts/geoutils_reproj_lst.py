from pathlib import Path

import numpy as np
import xarray as xr
from geoutils.regrid.proj import Proj_latlon, Proj_rect
from geoutils.regrid.regrid import regrid

from core.monitor import Chrono, RAM


src = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__input_ds.nc")
dst = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__reproj_ds_geoutils.nc")

ds = xr.open_dataset(src)

with Chrono("Reprojecting LST dataset with geoutils"):
# with RAM("Reprojecting LST dataset with geoutils"):

    # Compute bounding box from geolocation arrays
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    valid = np.isfinite(lat) & np.isfinite(lon)

    lat_min = float(lat[valid].min())
    lat_max = float(lat[valid].max())
    lon_min = float(lon[valid].min())
    lon_max = float(lon[valid].max())

    margin = 0.05
    lat_min_out = lat_min - margin
    lat_max_out = lat_max + margin
    lon_min_out = lon_min - margin
    lon_max_out = lon_max + margin

    # Output resolution
    width, height = 5000, 5000

    print(f"Bounding box: lat=[{lat_min_out:.2f}, {lat_max_out:.2f}], lon=[{lon_min_out:.2f}, {lon_max_out:.2f}]")
    print(f"Output grid : {width} x {height} px")

    # Create projections
    in_proj = Proj_latlon(lat, lon)
    out_proj = Proj_rect(shape=(height, width), 
                        lat_min=lat_min_out, lat_max=lat_max_out,
                        lon_min=lon_min_out, lon_max=lon_max_out)

    # Get output lat/lon coordinates
    out_lat = np.linspace(lat_max_out, lat_min_out, height)
    out_lon = np.linspace(lon_min_out, lon_max_out, width)

    # Process each variable
    data_vars = {}
    coords = {
        'latitude': ('y', out_lat),
        'longitude': ('x', out_lon),
    }

    # Get list of data variables (exclude lat/lon)
    var_names = [v for v in ds.data_vars if v not in ['latitude', 'longitude'] and np.issubdtype(ds[v].dtype, np.number)]

    print(f"\nProcessing {len(var_names)} variables...")

    for var_name in var_names:
        print(f"  {var_name} ({ds[var_name].dtype})...", end=' ', flush=True)
        
        da = ds[var_name]
        
        # Check dimensionality
        spatial_dims = ('scan_line', 'scan_pixel')  # Adjust based on your data
        if not all(d in da.dims for d in spatial_dims):
            # Try to infer spatial dims from lat/lon
            spatial_dims = tuple(ds['latitude'].dims)
            if not all(d in da.dims for d in spatial_dims):
                print(f"SKIP (missing spatial dims)")
                continue
        
        # Identify preserved dimensions (non-spatial)
        preserved_dims = [d for d in da.dims if d not in spatial_dims]
        
        if len(preserved_dims) > 0:
            # Handle 3D/4D data by looping over extra dimensions
            print(f"3D/4D ({preserved_dims})", end=' ', flush=True)
            
            # Get shapes
            preserved_shape = tuple(da.sizes[d] for d in preserved_dims)
            out_shape = preserved_shape + (height, width)
            
            # Initialize output arrays
            out_sum = np.zeros(out_shape, dtype=np.float64)
            out_counts = np.zeros(out_shape, dtype=np.uint32)
            
            # Create iterator for preserved dimensions
            import itertools
            n_slices = int(np.prod(preserved_shape))
            
            for idx in itertools.product(*[range(s) for s in preserved_shape]):
                # Extract 2D slice
                selector = {d: i for d, i in zip(preserved_dims, idx)}
                in_grid_2d = da.sel(selector).values.astype(np.float64)
                
                # Create output slice views
                out_sum_2d = out_sum[idx]
                out_counts_2d = out_counts[idx]
                
                # Create mask for valid input data
                in_mask = np.isfinite(in_grid_2d)
                
                # Regrid this slice
                regrid(in_grid_2d, out_sum_2d, out_counts_2d, 
                    'forward', in_proj, out_proj, 
                    in_mask=in_mask, verbose=False)
            
            # Compute mean
            with np.errstate(invalid='ignore', divide='ignore'):
                out_mean = np.where(out_counts > 0, out_sum / out_counts, np.nan).astype(da.dtype)
            
            # Create output DataArray with preserved dimensions
            out_dims = tuple(preserved_dims) + ('y', 'x')
            data_vars[var_name] = xr.DataArray(out_mean, dims=out_dims)
            data_vars[f'count_{var_name}'] = xr.DataArray(out_counts, dims=out_dims)
            
            print(f"OK ({n_slices} slices)")
        else:
            # Simple 2D case
            print("2D", end=' ', flush=True)
            
            # Initialize output arrays
            out_sum = np.zeros((height, width), dtype=np.float64)
            out_counts = np.zeros((height, width), dtype=np.uint32)
            
            # Get input data
            in_grid = da.values.astype(np.float64)
            
            # Create mask for valid input data
            in_mask = np.isfinite(in_grid)
            
            # Regrid
            regrid(in_grid, out_sum, out_counts, 
                'forward', in_proj, out_proj, 
                in_mask=in_mask, verbose=False)
            
            # Compute mean
            with np.errstate(invalid='ignore', divide='ignore'):
                out_mean = np.where(out_counts > 0, out_sum / out_counts, np.nan).astype(da.dtype)
            
            data_vars[var_name] = xr.DataArray(out_mean, dims=('y', 'x'))
            data_vars[f'count_{var_name}'] = xr.DataArray(out_counts, dims=('y', 'x'))
            
            print("OK")

    # Create output dataset
    result = xr.Dataset(data_vars, coords=coords)
    result.to_netcdf(dst)
    print(f"\nSaved → {dst}")
