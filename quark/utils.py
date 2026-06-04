import re

import numpy as np
import xarray as xr

def bbox_area(
    ds: xr.Dataset, 
    margin = 0.05,
    lat_name="latitude",
    lon_name="longitude",
    ) -> dict[str, float]:
    """
    Compute the bounding area (north, south, east, west) from the latitude and longitude variables in the dataset.
    
    Args:
        ds (xr.Dataset): The input dataset containing 'latitude' and 'longitude'
        margin (float): Margin to add to the bounding box (in degrees)
        lat_name (str): Name of the latitude variable in the dataset
        lon_name (str): Name of the longitude variable in the dataset
    Returns:
        dict[str, float]: A dictionary with keys 'north', 'south', 'east', 'west' representing the bounding area with margin.
    """
    
    # Compute bounding box from geolocation arrays
    lat = ds[lat_name]
    lon = ds[lon_name]

    lat_min = float(lat.where(~lat.isnull()).min())
    lat_max = float(lat.where(~lat.isnull()).max())
    lon_min = float(lon.where(~lon.isnull()).min())
    lon_max = float(lon.where(~lon.isnull()).max())

    margin = margin
    
    area = {
        "north": lat_max + margin,
        "south": lat_min - margin,
        "west": lon_min - margin,
        "east": lon_max + margin,
    }
    
    return area


def bbox_from_point(
    lat: float,
    lon: float,
    width: float,
    height: float,
) -> dict[str, float]:
    """
    Compute a bounding box (north, south, east, west) from a central point (latitude, longitude) and width/height in degrees.
    
    Args:
        lat (float): Latitude of the central point
        lon (float): Longitude of the central point
        width (float): Width of the bounding box in degrees
        height (float): Height of the bounding box in degrees
    Returns:
        dict[str, float]: A dictionary with keys 'north', 'south', 'east', 'west' representing the bounding box.
    
    Note:
        Latitude values are clamped to [-90, 90]. Longitude values can exceed [-180, 180] and may need
        to be handled by the caller if wrapping around the antimeridian is required.
    """
    
    # Clamp latitude to valid range [-90, 90]
    north = min(90.0, lat + height / 2)
    south = max(-90.0, lat - height / 2)
    
    bbox = {
        "north": north,
        "south": south,
        "west": lon - width / 2,
        "east": lon + width / 2,
    }
    
    return bbox


_RESOLUTION_RE = re.compile(r"^([0-9]*\.?[0-9]+)\s*(km|m|cm)$", re.IGNORECASE)

def _parse_resolution(resolution: str) -> float:
    """Parse a resolution string (e.g. '50m', '1km', '100cm') to metres."""
    m = _RESOLUTION_RE.match(resolution.strip())
    if not m:
        raise ValueError(
            f"Invalid resolution '{resolution}'. "
            "Expected a number followed by 'km', 'm', or 'cm' (e.g. '50m', '1km', '250cm')."
        )
    value, unit = float(m.group(1)), m.group(2).lower()
    return value * {"km": 1_000.0, "m": 1.0, "cm": 0.01}[unit]


def get_size_from_bbox(bbox: dict, resolution: str = "1km") -> tuple[int, int]:
    """
    Compute the grid dimensions (width, height) required to cover a bounding box
    at a given spatial resolution.

    Latitude extent is converted to metres using the standard 111 320 m/° factor.
    Longitude extent is corrected for latitude by cos(centre_lat).

    Args:
        bbox (dict): Bounding box with keys 'north', 'south', 'east', 'west' (degrees).
        resolution (str): Desired pixel size with unit suffix. Supported units:
                          'm', 'km', 'cm'. Examples: '50m', '1km', '250m', '0.5km'.

    Returns:
        tuple[int, int]: (width, height) — number of pixels in x and y.
    """
    res_m = _parse_resolution(resolution)

    lat_span = bbox["north"] - bbox["south"]
    lon_span = bbox["east"]  - bbox["west"]

    M_PER_DEG = 111_320.0 # ~40 075 km circumference / 360 degrees

    width  = max(1, round(lon_span * M_PER_DEG / res_m))
    height = max(1, round(lat_span * M_PER_DEG  / res_m))

    return width, height