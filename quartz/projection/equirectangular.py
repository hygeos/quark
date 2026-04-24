import numpy as np
from typing import Union, Optional

class EquirectangularProjection:
    def __init__(self, width: int, height: int, area: Union[dict, list, None] = None):
        """
        Initialize an equirectangular projection with custom area bounds.
        
        Args:
            width (int): The width of the equirectangular projection.
            height (int): The height of the equirectangular projection.
            area (dict|list, optional): A dictionary like {"north": 90, "west": -180, "south": -90, "east": 180} 
                                       specifying the area of the projection. Or list like [90, -180, -90, 180] 
                                       (north, west, south, east). Defaults to global coverage.
        """
        # Choose dtype based on dimensions (default to uint16 for efficiency)
        # Reserve max value as FILL_VALUE for out-of-bounds coordinates
        if max(width, height) < np.iinfo(np.uint16).max:  # uint16 max is 65535, reserve 65535 as FILL_VALUE
            self.uint_type = np.uint16
        elif max(width, height) < np.iinfo(np.uint32).max:  # uint32 max is 4294967295, reserve as FILL_VALUE
            self.uint_type = np.uint32
        else:
            raise ValueError(f"Dimensions too large: max(width={width}, height={height}) must be < 4294967295")
        
        self.FILL_VALUE_OOB = np.iinfo(self.uint_type).max
        
        self.width = width
        self.height = height
        
        # Default to global area if none specified
        if area is None:
            area = {"north": 90, "west": -180, "south": -90, "east": 180}
        
        if isinstance(area, dict):
            self.north = area["north"]
            self.west = area["west"]
            self.south = area["south"]
            self.east = area["east"]
        elif isinstance(area, list) and len(area) == 4:
            self.north, self.west, self.south, self.east = area
        else:
            raise ValueError("Area must be a dictionary like {'north': 90, 'west': -180, 'south': -90, 'east': 180} or a list like [90, -180, -90, 180].")
        
        # Validate area bounds
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError(f"Invalid latitude bounds: south={self.south}, north={self.north}. Must be in [-90, 90] with south < north.")
        if not (-180 <= self.west <= 180 and -180 <= self.east <= 180):
            raise ValueError(f"Invalid longitude bounds: west={self.west}, east={self.east}. Must be in [-180, 180].")
        
        # Calculate the span of the area
        self.lat_span = self.north - self.south
        self.lon_span = self.east - self.west
        if self.lon_span <= 0:
            self.lon_span += 360  # Handle wrapping around the antimeridian
    
    def project_to_indexes(self, latitude: Union[float, np.ndarray], longitude: Union[float, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Projects latitude and longitude coordinates to pixel indexes in the equirectangular projection.
        
        Args:
            latitude (float|np.ndarray): The latitude value(s) to project.
            longitude (float|np.ndarray): The longitude value(s) to project.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: The x and y pixel indexes in the equirectangular projection.
                                          Returns (FILL_VALUE, FILL_VALUE) for coordinates outside the area.
                                          Check with `is_valid_index()` or compare against `projection.FILL_VALUE`.
        """
        # Convert to numpy arrays
        lat = np.atleast_1d(np.asarray(latitude, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(longitude, dtype=np.float64))
        
        # Normalize longitude to handle wrapping
        lon_normalized = lon.copy()
        if self.west > self.east:  # Area crosses antimeridian
            lon_normalized = np.where(lon_normalized < 0, lon_normalized + 360, lon_normalized)
        
        # Check bounds (for non-clipped mode)
        out_of_bounds = (lat < self.south) | (lat > self.north) | (lon < self.west) | (lon > self.east)
        
        # Map latitude from [south, north] to [height-1, 0] (top to bottom in image coordinates)
        # North is at y=0, South is at y=height-1
        y_indexes = (self.north - lat) / self.lat_span * (self.height - 1)
        
        # Map longitude from [west, east] to [0, width-1]
        x_indexes = (lon_normalized - self.west) / self.lon_span * (self.width - 1)
        
        # Round to nearest integer and convert to uint
        x_indexes = np.round(x_indexes).astype(self.uint_type)
        y_indexes = np.round(y_indexes).astype(self.uint_type)
        
        # Mark out-of-bounds with FILL_VALUE (max value of uint type)
        x_indexes = np.where(out_of_bounds, self.FILL_VALUE_OOB, x_indexes)
        y_indexes = np.where(out_of_bounds, self.FILL_VALUE_OOB, y_indexes)
        
        return x_indexes, y_indexes
    
    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the latitude and longitude coordinates for each pixel in the equirectangular projection.
        Returns coordinates based on the defined area bounds.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: The latitude and longitude coordinates for each pixel 
                                          in the equirectangular projection.
        """
        # Create arrays of pixel indexes
        x_indexes = np.arange(self.width)
        y_indexes = np.arange(self.height)
        
        # Map from pixel coordinates to geographic coordinates
        # x: [0, width-1] -> [west, east]
        longitude = self.west + (x_indexes / (self.width - 1)) * self.lon_span
        
        # y: [0, height-1] -> [north, south] (top to bottom)
        latitude = self.north - (y_indexes / (self.height - 1)) * self.lat_span
        
        return latitude, longitude
    
    def is_in_bounds(self, latitude: Union[float, np.ndarray], longitude: Union[float, np.ndarray]) -> np.ndarray:
        """
        Check if coordinates are within the projection area bounds.
        
        Args:
            latitude (float|np.ndarray): The latitude value(s) to check.
            longitude (float|np.ndarray): The longitude value(s) to check.
        
        Returns:
            np.ndarray: Boolean array indicating which coordinates are in bounds.
        """
        lat = np.atleast_1d(np.asarray(latitude))
        lon = np.atleast_1d(np.asarray(longitude))
        
        in_bounds = (lat >= self.south) & (lat <= self.north) & \
                   (lon >= self.west) & (lon <= self.east)
        
        return in_bounds
    
    def is_valid_index(self, x_indexes: np.ndarray, y_indexes: np.ndarray) -> np.ndarray:
        """
        Check if pixel indexes are valid (not FILL_VALUE).
        
        Args:
            x_indexes (np.ndarray): The x pixel indexes to check.
            y_indexes (np.ndarray): The y pixel indexes to check.
        
        Returns:
            np.ndarray: Boolean array indicating which indexes are valid (not out-of-bounds).
        """
        return (x_indexes != self.FILL_VALUE_OOB) & (y_indexes != self.FILL_VALUE_OOB)