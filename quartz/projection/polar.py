import numpy as np
from typing import Union, Optional

from pyparsing import Literal


# TODO ! ============================================================
# NOTE: UNFINISHED, DO NOT USE YET
# NOTE: actually implement a specific projection
# like: https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
# this one keeps the area of pixels constant, and seems to be the most common for polar projections, but there are others.

class PolarProjection:
    def __init__(self, width: int, height: int, pole: Literal["north", "south"], center_longitude: float = 0.0):
        """
        Initialize a polar projection centered on the specified pole.
        
        Args:
            width (int): The width of the polar projection.
            height (int): The height of the polar projection.
            pole (str, optional): The pole to center the projection on ("north" or "south"). Defaults to "north".
            center_longitude (float, optional): The longitude to center the projection on. Defaults to 0 degrees.
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
        
        if pole not in ["north", "south"]:
            raise ValueError("Pole must be either 'north' or 'south'.")
        
        self.pole = pole
        self.center_longitude = center_longitude % 360  # Ensure center longitude is within [0, 360)