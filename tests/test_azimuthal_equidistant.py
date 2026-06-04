import pytest
import numpy as np
from quark.projection.polar import AzimuthalEquidistant


class TestAzimuthalEquidistantInit:
    """Test initialization and validation of AzimuthalEquidistant."""

    def test_init_north_pole_default(self):
        """Test initialization with default North Pole center."""
        proj = AzimuthalEquidistant(500, 500)
        assert proj.width == 500
        assert proj.height == 500
        assert proj.center_latitude == 90.0
        assert proj.center_longitude == 0.0
        assert proj.radius == 90.0  # hemisphere
        assert proj.uint_type == np.uint16

    def test_init_south_pole(self):
        """Test initialization with South Pole center."""
        proj = AzimuthalEquidistant(500, 500, center_latitude=-90.0)
        assert proj.center_latitude == -90.0
        assert proj.radius == 90.0

    def test_init_equatorial(self):
        """Test initialization with equatorial center."""
        proj = AzimuthalEquidistant(
            500, 500, center_latitude=0.0, center_longitude=45.0
        )
        assert proj.center_latitude == 0.0
        assert proj.center_longitude == 45.0

    def test_init_custom_radius(self):
        """Test initialization with custom radius."""
        proj = AzimuthalEquidistant(500, 500, radius=45.0)
        assert proj.radius == 45.0

    def test_init_invalid_latitude(self):
        """Test that invalid center latitude raises ValueError."""
        with pytest.raises(ValueError, match="center_latitude"):
            AzimuthalEquidistant(100, 100, center_latitude=91.0)
        with pytest.raises(ValueError, match="center_latitude"):
            AzimuthalEquidistant(100, 100, center_latitude=-91.0)

    def test_init_invalid_longitude(self):
        """Test that invalid center longitude raises ValueError."""
        with pytest.raises(ValueError, match="center_longitude"):
            AzimuthalEquidistant(100, 100, center_longitude=181.0)
        with pytest.raises(ValueError, match="center_longitude"):
            AzimuthalEquidistant(100, 100, center_longitude=-181.0)

    def test_init_invalid_radius(self):
        """Test that invalid radius raises ValueError."""
        with pytest.raises(ValueError, match="radius"):
            AzimuthalEquidistant(100, 100, radius=0.0)
        with pytest.raises(ValueError, match="radius"):
            AzimuthalEquidistant(100, 100, radius=181.0)

    def test_init_uint32_selection(self):
        """Test that uint32 is selected for large dimensions."""
        proj = AzimuthalEquidistant(70000, 70000)
        assert proj.uint_type == np.uint32


class TestProjectToIndexesNorthPole:
    """Test project_to_indexes with North Pole center."""

    def test_center_maps_to_image_center(self):
        """Test that the center point maps to the image center."""
        proj = AzimuthalEquidistant(101, 101, center_latitude=90.0)
        x, y = proj.project_to_indexes(90.0, 0.0)
        assert x[0] == 50
        assert y[0] == 50

    def test_equator_points(self):
        """Test that equator points are at the edge of the projection."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, center_longitude=0.0)
        # Equator at lon=0 should be at the bottom of the image
        x, y = proj.project_to_indexes(0.0, 0.0)
        assert x[0] == 100  # center x
        assert y[0] == 200  # bottom edge

    def test_equator_90_east(self):
        """Test equator at 90E is at the right edge."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, center_longitude=0.0)
        x, y = proj.project_to_indexes(0.0, 90.0)
        assert x[0] == 200  # right edge
        assert y[0] == 100  # center y

    def test_equator_90_west(self):
        """Test equator at 90W is at the left edge."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, center_longitude=0.0)
        x, y = proj.project_to_indexes(0.0, -90.0)
        assert x[0] == 0  # left edge
        assert y[0] == 100  # center y

    def test_equator_180(self):
        """Test equator at 180 is at the top edge."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, center_longitude=0.0)
        x, y = proj.project_to_indexes(0.0, 180.0)
        assert x[0] == 100  # center x
        assert y[0] == 0  # top edge

    def test_antipodal_point_out_of_bounds(self):
        """Test that the antipodal point (South Pole) is out of bounds."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0)
        x, y = proj.project_to_indexes(-90.0, 0.0)
        assert x[0] == proj.FILL_VALUE_OOB
        assert y[0] == proj.FILL_VALUE_OOB

    def test_mid_latitude(self):
        """Test a mid-latitude point."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, center_longitude=0.0)
        # 45N at lon=0 should be halfway between center and edge
        x, y = proj.project_to_indexes(45.0, 0.0)
        assert x[0] == 100  # center x
        assert y[0] == 150  # halfway to bottom


class TestProjectToIndexesEquatorial:
    """Test project_to_indexes with equatorial center."""

    def test_center_maps_to_image_center(self):
        """Test that the center point maps to the image center."""
        proj = AzimuthalEquidistant(101, 101, center_latitude=0.0, center_longitude=0.0)
        x, y = proj.project_to_indexes(0.0, 0.0)
        assert x[0] == 50
        assert y[0] == 50

    def test_north_pole_at_top(self):
        """Test that North Pole is at the top edge."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=0.0, center_longitude=0.0)
        x, y = proj.project_to_indexes(90.0, 0.0)
        assert x[0] == 100  # center x
        assert y[0] == 0  # top edge

    def test_south_pole_at_bottom(self):
        """Test that South Pole is at the bottom edge."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=0.0, center_longitude=0.0)
        x, y = proj.project_to_indexes(-90.0, 0.0)
        assert x[0] == 100  # center x
        assert y[0] == 200  # bottom edge

    def test_antimeridian_out_of_bounds(self):
        """Test that the antimeridian is out of bounds for equatorial view."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=0.0, center_longitude=0.0)
        x, y = proj.project_to_indexes(0.0, 180.0)
        assert x[0] == proj.FILL_VALUE_OOB
        assert y[0] == proj.FILL_VALUE_OOB


class TestProjectToIndexesCustomRadius:
    """Test project_to_indexes with custom radius."""

    def test_points_beyond_radius_out_of_bounds(self):
        """Test that points beyond the radius are out of bounds."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, radius=45.0)
        # 45N is at angular distance 45 from North Pole — should be at edge
        x, y = proj.project_to_indexes(45.0, 0.0)
        assert x[0] == 100
        assert y[0] == 200

        # 44N is beyond the radius — should be out of bounds
        x, y = proj.project_to_indexes(44.0, 0.0)
        assert x[0] == proj.FILL_VALUE_OOB
        assert y[0] == proj.FILL_VALUE_OOB

    def test_points_within_radius_valid(self):
        """Test that points within the radius are valid."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, radius=45.0)
        x, y = proj.project_to_indexes(60.0, 0.0)
        assert x[0] != proj.FILL_VALUE_OOB
        assert y[0] != proj.FILL_VALUE_OOB


class TestGetCoordinates:
    """Test the get_coordinates method."""

    def test_returns_correct_shape(self):
        """Test that get_coordinates returns arrays of correct shape."""
        proj = AzimuthalEquidistant(100, 80)
        lat, lon = proj.get_coordinates()
        assert lat.shape == (80, 100)
        assert lon.shape == (80, 100)

    def test_center_pixel_coordinates(self):
        """Test that the center pixel has the correct coordinates."""
        proj = AzimuthalEquidistant(101, 101, center_latitude=90.0, center_longitude=0.0)
        lat, lon = proj.get_coordinates()
        assert np.isclose(lat[50, 50], 90.0, atol=1e-6)
        # Center should be finite (not NaN)
        assert np.isfinite(lat[50, 50])

    def test_edge_pixels_are_invalid(self):
        """Test that corner pixels have NaN coordinates."""
        proj = AzimuthalEquidistant(100, 100)
        lat, lon = proj.get_coordinates()
        assert np.isnan(lat[0, 0])  # top-left corner
        assert np.isnan(lat[0, -1])  # top-right corner
        assert np.isnan(lat[-1, 0])  # bottom-left corner
        assert np.isnan(lat[-1, -1])  # bottom-right corner

    def test_valid_pixels_are_finite(self):
        """Test that pixels within the projection circle have finite coordinates."""
        proj = AzimuthalEquidistant(100, 100)
        lat, lon = proj.get_coordinates()
        # Center pixels should be finite
        assert np.isfinite(lat[50, 50])
        assert np.isfinite(lon[50, 50])

    def test_invalid_pixels_are_nan(self):
        """Test that corner pixels (outside projection circle) have NaN coordinates."""
        proj = AzimuthalEquidistant(100, 100)
        lat, lon = proj.get_coordinates()
        assert np.isnan(lat[0, 0])
        assert np.isnan(lon[0, 0])


class TestIsInBounds:
    """Test the is_in_bounds method."""

    def test_center_in_bounds(self):
        """Test that the center point is in bounds."""
        proj = AzimuthalEquidistant(100, 100, center_latitude=90.0)
        assert proj.is_in_bounds(90.0, 0.0)[0]

    def test_equator_in_bounds_hemisphere(self):
        """Test that equator is in bounds for hemisphere view."""
        proj = AzimuthalEquidistant(100, 100, center_latitude=90.0, radius=90.0)
        assert proj.is_in_bounds(0.0, 0.0)[0]

    def test_south_pole_out_of_bounds_north_view(self):
        """Test that South Pole is out of bounds for North Pole view."""
        proj = AzimuthalEquidistant(100, 100, center_latitude=90.0)
        assert not proj.is_in_bounds(-90.0, 0.0)[0]

    def test_array_input(self):
        """Test that array inputs work correctly."""
        proj = AzimuthalEquidistant(100, 100, center_latitude=90.0)
        lats = np.array([90.0, 45.0, 0.0, -45.0])
        lons = np.array([0.0, 0.0, 0.0, 0.0])
        result = proj.is_in_bounds(lats, lons)
        assert result[0]  # 90N
        assert result[1]  # 45N
        assert result[2]  # 0 (equator — on the boundary)
        assert not result[3]  # 45S


class TestIsValidIndex:
    """Test the is_valid_index method."""

    def test_valid_indexes(self):
        """Test that valid indexes return True."""
        proj = AzimuthalEquidistant(100, 100)
        assert proj.is_valid_index(np.array([50]), np.array([50]))[0]

    def test_fill_value_indexes(self):
        """Test that FILL_VALUE indexes return False."""
        proj = AzimuthalEquidistant(100, 100)
        assert not proj.is_valid_index(
            np.array([proj.FILL_VALUE_OOB]), np.array([proj.FILL_VALUE_OOB])
        )[0]


class TestRoundTrip:
    """Test round-trip projection (coordinates -> indexes -> coordinates)."""

    def test_north_pole_round_trip(self):
        """Test that projecting known coordinates gives expected pixels."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, center_longitude=0.0)

        # Known points and their expected pixel positions
        test_cases = [
            # (lat, lon, expected_x, expected_y)
            (90.0, 0.0, 100, 100),    # Center (North Pole)
            (45.0, 0.0, 100, 150),    # 45N on prime meridian (below center)
            (45.0, 90.0, 150, 100),   # 45N at 90E (right of center)
            (45.0, -90.0, 50, 100),   # 45N at 90W (left of center)
            (0.0, 0.0, 100, 200),     # Equator at prime meridian (bottom edge)
            (0.0, 90.0, 200, 100),    # Equator at 90E (right edge)
            (0.0, -90.0, 0, 100),     # Equator at 90W (left edge)
        ]

        for lat, lon, exp_x, exp_y in test_cases:
            x, y = proj.project_to_indexes(lat, lon)
            assert int(x[0]) == exp_x, f"({lat}, {lon}): expected x={exp_x}, got {x[0]}"
            assert int(y[0]) == exp_y, f"({lat}, {lon}): expected y={exp_y}, got {y[0]}"

    def test_equatorial_round_trip(self):
        """Test round-trip for equatorial projection."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=0.0, center_longitude=0.0)

        lat_grid, lon_grid = proj.get_coordinates()

        # Check a few specific points
        test_points = [
            (30.0, 30.0),
            (-30.0, -30.0),
            (60.0, 0.0),
            (0.0, 60.0),
        ]

        for lat, lon in test_points:
            if not proj.is_in_bounds(lat, lon)[0]:
                continue

            x, y = proj.project_to_indexes(lat, lon)
            assert x[0] != proj.FILL_VALUE_OOB

            # Get coordinates at this pixel
            pixel_lat = lat_grid[y[0], x[0]]
            pixel_lon = lon_grid[y[0], x[0]]

            # Should be close to original (within a few degrees due to pixel discretization)
            assert abs(pixel_lat - lat) < 2.0
            # Longitude may wrap around
            lon_diff = abs(pixel_lon - lon)
            if lon_diff > 180:
                lon_diff = 360 - lon_diff
            assert lon_diff < 2.0


class TestDistancesPreserved:
    """Test that distances from center are preserved (key property of azimuthal equidistant)."""

    def test_distance_preservation_north_pole(self):
        """Test that angular distance from center equals pixel distance ratio."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, center_longitude=0.0)

        # Points at different latitudes on the same meridian
        lats = np.array([90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0])
        lons = np.zeros_like(lats)

        x, y = proj.project_to_indexes(lats, lons)

        # Pixel distances from center
        pixel_dists = np.sqrt((x - 100) ** 2 + (y - 100) ** 2)

        # Angular distances from North Pole (in degrees)
        angular_dists = 90.0 - lats

        # The ratio should be constant (linear relationship)
        # pixel_dist / angular_dist should be constant
        ratios = pixel_dists[angular_dists > 0] / angular_dists[angular_dists > 0]

        # All ratios should be approximately equal (within ~1% due to pixel discretization)
        assert np.allclose(ratios, ratios[0], rtol=0.02)

    def test_distance_preservation_equatorial(self):
        """Test distance preservation for equatorial projection."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=0.0, center_longitude=0.0)

        # Points along the equator
        lats = np.zeros(10)
        lons = np.linspace(0, 80, 10)

        x, y = proj.project_to_indexes(lats, lons)

        pixel_dists = np.sqrt((x - 100) ** 2 + (y - 100) ** 2)

        # Angular distances should be proportional to longitude difference on equator
        angular_dists = np.abs(lons)

        ratios = pixel_dists[angular_dists > 0] / angular_dists[angular_dists > 0]
        assert np.allclose(ratios, ratios[0], rtol=0.02)


class TestRotation:
    """Test the rotation parameter."""

    def test_rotation_zero_is_default(self):
        """Test that rotation=0 gives the same result as no rotation."""
        proj_default = AzimuthalEquidistant(201, 201, center_latitude=90.0)
        proj_rotated = AzimuthalEquidistant(201, 201, center_latitude=90.0, rotation=0.0)

        x1, y1 = proj_default.project_to_indexes(0.0, 90.0)
        x2, y2 = proj_rotated.project_to_indexes(0.0, 90.0)
        assert x1[0] == x2[0]
        assert y1[0] == y2[0]

    def test_rotation_90_degrees(self):
        """Test that rotation=90 shifts equator points by 90 degrees."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, rotation=90.0)

        # Without rotation, equator at lon=90 is at (200, 100) — right edge
        # With rotation=90, equator at lon=0 should now be at (200, 100)
        x, y = proj.project_to_indexes(0.0, 0.0)
        assert x[0] == 200  # right edge
        assert y[0] == 100  # center y

    def test_rotation_180_degrees(self):
        """Test that rotation=180 flips the view upside down."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, rotation=180.0)

        # Without rotation, equator at lon=0 is at (100, 200) — bottom edge
        # With rotation=180, equator at lon=0 should be at (100, 0) — top edge
        x, y = proj.project_to_indexes(0.0, 0.0)
        assert x[0] == 100  # center x
        assert y[0] == 0  # top edge

    def test_rotation_360_same_as_zero(self):
        """Test that rotation=360 is same as rotation=0."""
        proj_default = AzimuthalEquidistant(201, 201, center_latitude=90.0)
        proj_360 = AzimuthalEquidistant(201, 201, center_latitude=90.0, rotation=360.0)

        x1, y1 = proj_default.project_to_indexes(0.0, 45.0)
        x2, y2 = proj_360.project_to_indexes(0.0, 45.0)
        assert x1[0] == x2[0]
        assert y1[0] == y2[0]

    def test_rotation_preserves_distances(self):
        """Test that rotation doesn't change distances from center."""
        proj_default = AzimuthalEquidistant(201, 201, center_latitude=90.0)
        proj_rotated = AzimuthalEquidistant(201, 201, center_latitude=90.0, rotation=45.0)

        x1, y1 = proj_default.project_to_indexes(45.0, 0.0)
        x2, y2 = proj_rotated.project_to_indexes(45.0, 0.0)

        dist1 = np.sqrt((x1[0] - 100) ** 2 + (y1[0] - 100) ** 2)
        dist2 = np.sqrt((x2[0] - 100) ** 2 + (y2[0] - 100) ** 2)
        # Within 1 pixel due to rounding on diagonal
        assert abs(dist1 - dist2) <= 1

    def test_rotation_negative(self):
        """Test that negative rotation works (counter-clockwise)."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, rotation=-90.0)

        # rotation=-90: lon=0 moves to left edge, lon=180 moves to right edge
        x, y = proj.project_to_indexes(0.0, 0.0)
        assert x[0] == 0  # left edge
        assert y[0] == 100  # center y

    def test_rotation_equatorial_center(self):
        """Test rotation with equatorial center."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=0.0, center_longitude=0.0, rotation=90.0)

        # Without rotation, North Pole is at top (100, 0)
        # With rotation=90, North Pole should be at left (0, 100)
        x, y = proj.project_to_indexes(90.0, 0.0)
        assert x[0] == 0  # left edge
        assert y[0] == 100  # center y

    def test_rotation_round_trip(self):
        """Test that get_coordinates with rotation gives consistent results."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, rotation=45.0)

        lat_grid, lon_grid = proj.get_coordinates()

        # Center should still be at the pole
        assert np.isclose(lat_grid[100, 100], 90.0, atol=1e-6)

        # Corners should be NaN
        assert np.isnan(lat_grid[0, 0])


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_nan_coordinates(self):
        """Test that NaN coordinates are handled gracefully."""
        proj = AzimuthalEquidistant(100, 100)
        x, y = proj.project_to_indexes(np.nan, 0.0)
        assert x[0] == proj.FILL_VALUE_OOB
        assert y[0] == proj.FILL_VALUE_OOB

    def test_inf_coordinates(self):
        """Test that infinite coordinates are handled gracefully."""
        proj = AzimuthalEquidistant(100, 100)
        x, y = proj.project_to_indexes(np.inf, 0.0)
        assert x[0] == proj.FILL_VALUE_OOB
        assert y[0] == proj.FILL_VALUE_OOB

    def test_batch_projection(self):
        """Test projecting multiple points at once."""
        proj = AzimuthalEquidistant(100, 100, center_latitude=90.0)
        lats = np.array([90.0, 80.0, 70.0])
        lons = np.array([0.0, 0.0, 0.0])
        x, y = proj.project_to_indexes(lats, lons)
        assert len(x) == 3
        assert len(y) == 3

    def test_non_square_image(self):
        """Test with non-square image dimensions."""
        proj = AzimuthalEquidistant(300, 200, center_latitude=90.0)
        x, y = proj.project_to_indexes(90.0, 0.0)
        # Center: floor((299/2) + 0.5) = 150, floor((199/2) + 0.5) = 100
        assert x[0] == 150
        assert y[0] == 100

    def test_longitude_wrapping(self):
        """Test that longitude wrapping works correctly."""
        proj = AzimuthalEquidistant(201, 201, center_latitude=90.0, center_longitude=0.0)
        # 180 and -180 should give the same result
        x1, y1 = proj.project_to_indexes(0.0, 180.0)
        x2, y2 = proj.project_to_indexes(0.0, -180.0)
        assert x1[0] == x2[0]
        assert y1[0] == y2[0]
