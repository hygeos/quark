import numpy as np
from typing import Union, Optional, Literal


class LambertAzimuthal:
    """
    Lambert azimuthal equidistant projection with a parametrized center point.

    This projection preserves distances and directions from the center point.
    It maps a hemisphere (or any angular radius) centered on an arbitrary
    latitude/longitude to a square image.

    See: https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection
    """

    def __init__(
        self,
        width: int,
        height: int,
        center_latitude: float = 90.0,
        center_longitude: float = 0.0,
        radius: Optional[float] = None,
        rotation: float = 0.0,
    ):
        """
        Initialize an azimuthal equidistant projection.

        Args:
            width (int): The width of the projection in pixels.
            height (int): The height of the projection in pixels.
            center_latitude (float, optional): Latitude of the projection center in degrees.
                Defaults to 90 (North Pole). Use 0 for an equatorial view, or any value
                in [-90, 90].
            center_longitude (float, optional): Longitude of the projection center in degrees.
                Defaults to 0. Any value in [-180, 180] is valid.
            radius (float, optional): Maximum angular distance (in degrees) from the center
                that will be visible in the projection. Defaults to 90 (hemisphere).
                Use smaller values for zoomed-in views (e.g., 45 for a quarter-sphere).
            rotation (float, optional): Rotation of the view in degrees clockwise from the
                default orientation. In the default orientation (rotation=0), the meridian
                through the center points "down" in the image (following the meridian
                southward from the center). Positive rotation spins the view clockwise
                around the center. Defaults to 0.
        """
        # Choose dtype based on dimensions
        if max(width, height) < np.iinfo(np.uint16).max:
            self.uint_type = np.uint16
        elif max(width, height) < np.iinfo(np.uint32).max:
            self.uint_type = np.uint32
        else:
            raise ValueError(
                f"Dimensions too large: max(width={width}, height={height}) must be < 4294967295"
            )

        self.FILL_VALUE_OOB = np.iinfo(self.uint_type).max

        self.width = width
        self.height = height

        # Validate and store center point
        if not (-90 <= center_latitude <= 90):
            raise ValueError(
                f"center_latitude must be in [-90, 90], got {center_latitude}"
            )
        if not (-180 <= center_longitude <= 180):
            raise ValueError(
                f"center_longitude must be in [-180, 180], got {center_longitude}"
            )

        self.center_latitude = center_latitude
        self.center_longitude = center_longitude

        # Validate and store radius
        if radius is None:
            radius = 90.0  # hemisphere by default
        if not (0 < radius <= 180):
            raise ValueError(f"radius must be in (0, 180], got {radius}")
        self.radius = radius

        # Store rotation
        self.rotation = rotation
        self._rotation_rad = np.radians(rotation)

        # Pre-compute trig values for the center point
        self._sin_lat0 = np.sin(np.radians(center_latitude))
        self._cos_lat0 = np.cos(np.radians(center_latitude))
        self._lon0_rad = np.radians(center_longitude)

        # Image center (where the projection center maps to)
        self._cx = (self.width - 1) / 2.0
        self._cy = (self.height - 1) / 2.0

        # Maximum pixel radius (distance from center to edge of image)
        # For a square image, this is the distance to the nearest edge
        self._max_pixel_radius = min(self._cx, self._cy)

        # Angular radius in radians
        self._radius_rad = np.radians(radius)

    def _angular_distance_and_azimuth(
        self, lat: np.ndarray, lon: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the angular distance θ and azimuth φ from the center point
        to each (lat, lon) coordinate.

        Uses the spherical law of cosines for the angular distance, and the
        proper formula for the azimuth from the center.

        Args:
            lat: Latitude array in degrees.
            lon: Longitude array in degrees.

        Returns:
            theta: Angular distance from center in radians.
            phi: Azimuth (direction) from center in radians, measured clockwise from north.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Angular distance using spherical law of cosines
        # θ = arccos(sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(lon - lon0))
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        dlon = lon_rad - self._lon0_rad

        cos_theta = (
            self._sin_lat0 * sin_lat
            + self._cos_lat0 * cos_lat * np.cos(dlon)
        )
        # Clamp to [-1, 1] to avoid numerical issues with arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # Azimuth φ from the center point
        # φ = atan2(
        #     sin(lon - lon0) * cos(lat),
        #     cos(lat0) * sin(lat) - sin(lat0) * cos(lat) * cos(lon - lon0)
        # )
        # Measured clockwise from north
        numerator = np.sin(dlon) * cos_lat
        denominator = (
            self._cos_lat0 * sin_lat
            - self._sin_lat0 * cos_lat * np.cos(dlon)
        )
        phi = np.arctan2(numerator, denominator)

        return theta, phi

    def project_to_indexes(
        self,
        latitude: Union[float, np.ndarray],
        longitude: Union[float, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project latitude and longitude coordinates to pixel indexes in the
        azimuthal equidistant projection.

        Args:
            latitude: Latitude value(s) in degrees.
            longitude: Longitude value(s) in degrees.

        Returns:
            tuple[np.ndarray, np.ndarray]: (x_indexes, y_indexes) — pixel coordinates.
                Returns (FILL_VALUE, FILL_VALUE) for coordinates outside the projection
                radius. Check validity with `is_valid_index()`.
        """
        lat = np.atleast_1d(np.asarray(latitude, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(longitude, dtype=np.float64))

        # Compute angular distance and azimuth from center
        theta, phi = self._angular_distance_and_azimuth(lat, lon)

        # Points beyond the projection radius are out of bounds
        # Use a small epsilon to handle numerical precision at exact boundaries
        out_of_bounds = (
            ~np.isfinite(lat)
            | ~np.isfinite(lon)
            | (theta > self._radius_rad + 1e-10)
        )

        # Apply rotation to azimuth (clockwise rotation of the view)
        phi_rotated = phi - self._rotation_rad

        # Scale angular distance to pixel distance
        # θ / radius_rad maps [0, radius] -> [0, 1]
        # Then scale to pixel coordinates
        scale = self._max_pixel_radius / self._radius_rad
        x_offset = scale * theta * np.sin(phi_rotated)
        y_offset = scale * theta * np.cos(phi_rotated)

        # In image coordinates, y increases downward, so we subtract the y offset
        x_indexes = self._cx + x_offset
        y_indexes = self._cy - y_offset

        # Round to nearest integer
        x_indexes = np.floor(x_indexes + 0.5).astype(self.uint_type)
        y_indexes = np.floor(y_indexes + 0.5).astype(self.uint_type)

        # Mark out-of-bounds with FILL_VALUE
        x_indexes = np.where(out_of_bounds, self.FILL_VALUE_OOB, x_indexes)
        y_indexes = np.where(out_of_bounds, self.FILL_VALUE_OOB, y_indexes)

        return x_indexes, y_indexes

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the latitude and longitude for each pixel in the projection.

        Returns a meshgrid of coordinates for the entire image.
        Pixels outside the circular projection area have NaN coordinates.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - latitude: 2D array of latitude values for each pixel.
                - longitude: 2D array of longitude values for each pixel.
        """
        # Create pixel coordinate grids
        x = np.arange(self.width, dtype=np.float64)
        y = np.arange(self.height, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)

        # Convert pixel offsets to angular distance and azimuth
        dx = xx - self._cx
        dy = self._cy - yy  # Flip y because image y goes down

        # Pixel distance from center
        pixel_dist = np.sqrt(dx**2 + dy**2)

        # Pixels outside the circular projection area
        valid = pixel_dist <= self._max_pixel_radius

        # Angular distance from center
        theta = pixel_dist * self._radius_rad / self._max_pixel_radius

        # Azimuth from center (clockwise from north), accounting for rotation
        phi = np.arctan2(dx, dy) + self._rotation_rad

        # Convert back to lat/lon using inverse formulas
        # sin(lat) = sin(lat0)*cos(θ) + cos(lat0)*sin(θ)*cos(φ)
        # lon = lon0 + atan2(sin(φ)*sin(θ)*cos(lat0), cos(θ) - sin(lat0)*sin(lat))
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        sin_lat = (
            self._sin_lat0 * cos_theta
            + self._cos_lat0 * sin_theta * cos_phi
        )
        # Clamp for numerical stability
        sin_lat = np.clip(sin_lat, -1.0, 1.0)
        latitude = np.degrees(np.arcsin(sin_lat))

        # For longitude, handle the pole case specially
        # At the poles, longitude is undefined (all directions are south/north)
        cos_lat = np.sqrt(1 - sin_lat**2)

        numerator = sin_phi * sin_theta
        denominator = cos_theta - self._sin_lat0 * sin_lat

        # Where cos_lat is near zero (near poles), longitude is arbitrary
        lon_offset = np.arctan2(numerator, denominator)
        longitude = np.degrees(self._lon0_rad + lon_offset)

        # Normalize longitude to [-180, 180]
        longitude = (longitude + 180) % 360 - 180

        # Set invalid pixels to NaN
        latitude = np.where(valid, latitude, np.nan)
        longitude = np.where(valid, longitude, np.nan)

        return latitude, longitude

    def is_in_bounds(
        self,
        latitude: Union[float, np.ndarray],
        longitude: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Check if coordinates are within the projection radius.

        Args:
            latitude: Latitude value(s) in degrees.
            longitude: Longitude value(s) in degrees.

        Returns:
            np.ndarray: Boolean array indicating which coordinates are within
                the projection radius.
        """
        lat = np.atleast_1d(np.asarray(latitude, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(longitude, dtype=np.float64))

        theta, _ = self._angular_distance_and_azimuth(lat, lon)

        in_bounds = (
            np.isfinite(lat)
            & np.isfinite(lon)
            & (theta <= self._radius_rad)
        )

        return in_bounds

    def is_valid_index(
        self, x_indexes: np.ndarray, y_indexes: np.ndarray
    ) -> np.ndarray:
        """
        Check if pixel indexes are valid (not FILL_VALUE).

        Args:
            x_indexes: The x pixel indexes to check.
            y_indexes: The y pixel indexes to check.

        Returns:
            np.ndarray: Boolean array indicating which indexes are valid.
        """
        return (
            (x_indexes != self.FILL_VALUE_OOB)
            & (y_indexes != self.FILL_VALUE_OOB)
        )