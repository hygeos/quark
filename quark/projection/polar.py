from quark.projection.lambertazimutal import LambertAzimutal


class PolarNorth(LambertAzimutal):
    """
    Lambert azimuthal equidistant projection centered on the North Pole.

    Convenience subclass of :class:`LambertAzimutal` with the center fixed at
    latitude 90° (North Pole) and longitude 0°.  Distances and directions from
    the pole are preserved, making this projection ideal for visualizing the
    Arctic region or any area within a given angular radius of the pole.

    The projection maps a circular cap around the pole onto a square image.
    Meridians radiate outward from the center, and parallels appear as
    concentric circles.

    See: [Azimuthal equidistant projection on Wikipedia](https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection)
    """

    def __init__(
        self,
        width: int,
        height: int,
        radius_deg: float = 90.0,
        rotation_deg: float = 0.0,
    ):
        """
        Initialize a North Pole-centered azimuthal equidistant projection.

        Args:
            width: The width of the projection in pixels.
            height: The height of the projection in pixels.
            radius_deg: Maximum angular distance (in degrees) from the North
                Pole that will be visible in the projection.  Defaults to 90°
                (full hemisphere).  Use smaller values for zoomed-in views
                (e.g., 45° for the Arctic Circle region).
            rotation_deg: Rotation of the view in degrees clockwise from the
                default orientation.  In the default orientation (rotation=0),
                the 0° meridian (Greenwich) points "down" in the image.
                Positive rotation spins the view clockwise around the pole.
                Defaults to 0.
        """
        super().__init__(
            width=width,
            height=height,
            center_latitude=90.0,
            center_longitude=0.0,
            radius=radius_deg,
            rotation=rotation_deg,
        )


class PolarSouth(LambertAzimutal):
    """
    Lambert azimuthal equidistant projection centered on the South Pole.

    Convenience subclass of :class:`LambertAzimutal` with the center fixed at
    latitude -90° (South Pole) and longitude 0°.  Distances and directions
    from the pole are preserved, making this projection ideal for visualizing
    Antarctica or any area within a given angular radius of the pole.

    The projection maps a circular cap around the pole onto a square image.
    Meridians radiate outward from the center, and parallels appear as
    concentric circles.

    See: [Azimuthal equidistant projection on Wikipedia](https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection)
    """

    def __init__(
        self,
        width: int,
        height: int,
        radius_deg: float = 90.0,
        rotation_deg: float = 0.0,
    ):
        """
        Initialize a South Pole-centered azimuthal equidistant projection.

        Args:
            width: The width of the projection in pixels.
            height: The height of the projection in pixels.
            radius_deg: Maximum angular distance (in degrees) from the South
                Pole that will be visible in the projection.  Defaults to 90°
                (full hemisphere).  Use smaller values for zoomed-in views
                (e.g., 45° for the Antarctic Circle region).
            rotation_deg: Rotation of the view in degrees clockwise from the
                default orientation.  In the default orientation (rotation=0),
                the 0° meridian (Greenwich) points "down" in the image.
                Positive rotation spins the view clockwise around the pole.
                Defaults to 0.
        """
        super().__init__(
            width=width,
            height=height,
            center_latitude=-90.0,
            center_longitude=0.0,
            radius=radius_deg,
            rotation=rotation_deg,
        )