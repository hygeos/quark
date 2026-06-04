from pathlib import Path
import site

import numpy as np
import xarray as xr

from quark.aggregate import Aggregator
from quark.projection.equirectangular import EquiRectangular
from quark.supersampling import SpatialSuperSampler, ConstantSuperSampler
from quark.utils import bbox_area, bbox_from_point, get_size_from_bbox

from core.monitor import Chrono, RAM
from core import log

# src = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__input_ds.nc")
# dst = Path("/mnt/ceph/proj/USINE/LST/LST/TESTING__reproj_ds.nc")

src = Path("/mnt/ceph/proj/USINE/LST/ECOSTRESS/ECOSTRESS__UNPROJ__Lille__2024-08-01__002.nc")
dst = Path("/mnt/ceph/proj/USINE/LST/ECOSTRESS/ECOSTRESS__REPROJ__Lille__2024-08-01__002.nc")

ds = xr.open_dataset(src)

class Site:
    """
    Describes a site of interest, including its name and location.
    """
    def __init__(self, name: str, lat: float, lon: float):
        self.name = name
        self.lat = lat
        self.lon = lon
        
with Chrono("Reprojecting LST dataset"):
# with RAM("Reprojecting LST dataset"):

    site = Site(name="Lille", lat=50.6292, lon=3.0573)

    # Compute bounding box from geolocation arrays
    # area = bbox_area(ds, margin=0.05, lat_name="latitude", lon_name="longitude")
    area = bbox_from_point(site.lat, site.lon, width=0.5, height=0.5)


    # Output resolution in degrees (~0.01° ≈ 1 km at mid-latitudes); adjust as needed
    width, height = get_size_from_bbox(area, resolution="70m")

    print(f"Bounding box: {area}")
    print(f"Output grid : {width} x {height} px")

    projection = EquiRectangular(width=width, height=height, area=area)

    mode = "simple"
    # mode = "kahan"
    ssfactor = 2
    # subpxmode = "constant"
    subpxmode = "constant"
    
    px_width = "70m"
    
    # Create supersampler based on mode
    if subpxmode == "spatial":
        supersampler = SpatialSuperSampler(factor=ssfactor, project_center=True)
    else:  # constant
        supersampler = ConstantSuperSampler(factor=ssfactor, pixel_width=px_width, project_center=True)

    agg = Aggregator(
        projection=projection,
        datasets=[ds],
        # lat_name="latitude",
        # lon_name="longitude",
        variables=None,
        # fail_on_schema_mismatch=False,
        sum_method=mode,
        # skipna=True,
        supersampler=supersampler,
        return_counts=True,
        return_sums=False,
        dtype=np.float32,
    )
    
    result = agg.compute()

    dst = dst.parent / (dst.stem + f"__acc_{mode}__ss_{ssfactor}__{subpxmode}__{px_width}.nc")

    log.info(f"Saving result to {dst}...")
    result.to_netcdf(dst)
    log.info(f"Saved → {dst}")