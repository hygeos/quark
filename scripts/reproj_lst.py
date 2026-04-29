from pathlib import Path

import numpy as np
import xarray as xr

from quartz.aggregate import Aggregator
from quartz.projection.equirectangular import EquiRectangular
from quartz.utils import bbox_area

from core.monitor import Chrono, RAM
from core import log

src = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__input_ds.nc")
dst = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__reproj_ds.nc")

ds = xr.open_dataset(src)

with Chrono("Reprojecting LST dataset"):
# with RAM("Reprojecting LST dataset"):

    # Compute bounding box from geolocation arrays
    area = bbox_area(ds, margin=0.05, lat_name="latitude", lon_name="longitude")

    # Output resolution in degrees (~0.01° ≈ 1 km at mid-latitudes); adjust as needed
    width, height = 5000, 5000

    print(f"Bounding box: {area}")
    print(f"Output grid : {width} x {height} px")

    projection = EquiRectangular(width=width, height=height, area=area)

    mode = "simple"
    # mode = "kahan"
    ssfactor = 2
    # subpxmode = "constant"
    subpxmode = "spatial"
    
    px_width = "50m"

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
        subpixel_mode=subpxmode,
        # pixel_width=px_width,
        return_counts=True,
        return_sums=False,
        dtype=np.float32,
    )
    
    result = agg.compute()

    dst = dst.parent / (dst.stem + f"__acc_{mode}__ss_{ssfactor}__{subpxmode}__{px_width}.nc")

    log.info(f"Saving result to {dst}...")
    result.to_netcdf(dst)
    log.info(f"Saved → {dst}")