from pathlib import Path

import numpy as np
import xarray as xr

from quartz.aggregate import Aggregator
from quartz.projection.equirectangular import EquiRectangular

from core.monitor import Chrono, RAM
from core import log

src = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__input_ds.nc")
dst = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__reproj_ds.nc")

ds = xr.open_dataset(src)

with Chrono("Reprojecting LST dataset"):
# with RAM("Reprojecting LST dataset"):

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

    # Output resolution in degrees (~0.01° ≈ 1 km at mid-latitudes); adjust as needed
    width, height = 5000, 5000

    print(f"Bounding box: {area}")
    print(f"Output grid : {width} x {height} px")

    projection = EquiRectangular(width=width, height=height, area=area)

    mode = "simple"
    # mode = "kahan"
    ssfactor = 1

    agg = Aggregator(
        projection=projection,
        datasets=[ds],
        # lat_name="latitude",
        # lon_name="longitude",
        variables=None,
        # fail_on_schema_mismatch=False,
        sum_method=mode,
        # skipna=True,
        supersampling=ssfactor,
        return_counts=True,
        return_sums=False,
        dtype=np.float32,
    )
    
    result = agg.compute()

    dst = dst.parent / (dst.stem + f"__acc_{mode}__ss_{ssfactor}.nc")

    log.info(f"Saving result to {dst}...")
    result.to_netcdf(dst)
    log.info(f"Saved → {dst}")