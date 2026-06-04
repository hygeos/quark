from quark.projection.lambertazimutal import LambertAzimuthal



class PolarNorth(LambertAzimuthal):
    """
    Polar projection centered on the North Pole.
    """
    def __init__(self, width: int, height: int, radius_deg: float, rotation_deg: float = 0.0):
        super().__init__(
            width=width,
            height=height,
            center_latitude=90.0,
            center_longitude=0.0,
            radius=radius_deg,
            rotation=rotation_deg
        )


class PolarSouth(LambertAzimuthal):
    """
    Polar projection centered on the South Pole.
    """
    def __init__(self, width: int, height: int, radius_deg: float, rotation_deg: float = 0.0):
        super().__init__(
            width=width,
            height=height,
            center_latitude=-90.0,
            center_longitude=0.0,
            radius=radius_deg,
            rotation=rotation_deg
        )