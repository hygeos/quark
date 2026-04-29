import xarray as xr
import numpy as np

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