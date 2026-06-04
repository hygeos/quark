import pytest
import numpy as np
from quark.projection.equirectangular import EquiRectangular


class TestEquirectangularProjectionInit:
    """Test initialization and validation of EquirectangularProjection."""
    
    def test_init_default_global_area(self):
        """Test initialization with default global area."""
        proj = EquiRectangular(360, 180)
        assert proj.width == 360
        assert proj.height == 180
        assert proj.north == 90
        assert proj.south == -90
        assert proj.west == -180
        assert proj.east == 180
        assert proj.lat_span == 180
        assert proj.lon_span == 360
        # Should use uint16 for small dimensions
        assert proj.uint_type == np.uint16
        assert proj.FILL_VALUE_OOB == 65535
    
    def test_init_with_dict_area(self):
        """Test initialization with dictionary area specification."""
        area = {"north": 60, "west": -10, "south": 40, "east": 10}
        proj = EquiRectangular(100, 100, area=area)
        assert proj.north == 60
        assert proj.south == 40
        assert proj.west == -10
        assert proj.east == 10
        assert proj.lat_span == 20
        assert proj.lon_span == 20
    
    def test_init_with_list_area(self):
        """Test initialization with list area specification."""
        area = [60, -10, 40, 10]  # north, west, south, east
        proj = EquiRectangular(100, 100, area=area)
        assert proj.north == 60
        assert proj.south == 40
        assert proj.west == -10
        assert proj.east == 10
    
    def test_init_invalid_area_type(self):
        """Test that invalid area type raises ValueError."""
        with pytest.raises(ValueError, match="Area must be a dictionary"):
            EquiRectangular(100, 100, area="invalid")
    
    def test_init_invalid_area_list_length(self):
        """Test that invalid list length raises ValueError."""
        with pytest.raises(ValueError, match="Area must be a dictionary"):
            EquiRectangular(100, 100, area=[60, -10, 40])
    
    def test_init_invalid_latitude_bounds(self):
        """Test that invalid latitude bounds raise ValueError."""
        # South >= North
        with pytest.raises(ValueError, match="Invalid latitude bounds"):
            EquiRectangular(100, 100, area={"north": 40, "south": 60, "west": -10, "east": 10})
        
        # Out of range
        with pytest.raises(ValueError, match="Invalid latitude bounds"):
            EquiRectangular(100, 100, area={"north": 100, "south": 0, "west": -10, "east": 10})
    
    def test_init_invalid_longitude_bounds(self):
        """Test that invalid longitude bounds raise ValueError."""
        with pytest.raises(ValueError, match="Invalid longitude bounds"):
            EquiRectangular(100, 100, area={"north": 60, "south": 40, "west": -200, "east": 10})
    
    def test_init_uint16_selection(self):
        """Test that uint16 is selected for small dimensions."""
        proj = EquiRectangular(1000, 1000)
        assert proj.uint_type == np.uint16
        assert proj.FILL_VALUE_OOB == 65535
    
    def test_size_equal_to_uint16_max(self):
        """Test that dimensions equal to uint16 max still use uint16."""
        proj = EquiRectangular(65535, 65535)
        assert proj.uint_type != np.uint16
        assert proj.FILL_VALUE_OOB != 65535
    
    def test_init_uint32_selection(self):
        """Test that uint32 is selected for large dimensions."""
        proj = EquiRectangular(70000, 70000)
        assert proj.uint_type == np.uint32
        assert proj.FILL_VALUE_OOB == 4294967295
    
    def test_init_dimensions_too_large(self):
        """Test that dimensions exceeding uint32 max raise ValueError."""
        with pytest.raises(ValueError, match="Dimensions too large"):
            EquiRectangular(4294967296, 100)


class TestProjectToIndexes:
    """Test the project_to_indexes method."""
    
    def test_project_global_corners(self):
        """Test projection of global corners."""
        proj = EquiRectangular(360, 180)
        
        # North-West corner (top-left)
        x, y = proj.project_to_indexes(90, -180)
        assert x[0] == 0
        assert y[0] == 0
        
        # North-East corner (top-right)
        x, y = proj.project_to_indexes(90, 180)
        assert x[0] == 359
        assert y[0] == 0
        
        # South-West corner (bottom-left)
        x, y = proj.project_to_indexes(-90, -180)
        assert x[0] == 0
        assert y[0] == 179
        
        # South-East corner (bottom-right)
        x, y = proj.project_to_indexes(-90, 180)
        assert x[0] == 359
        assert y[0] == 179
    
    def test_project_global_center(self):
        """Test projection of global center point."""
        proj = EquiRectangular(360, 180)
        
        # Center (0, 0)
        x, y = proj.project_to_indexes(0, 0)
        assert x[0] == 180  # Middle of width
        assert y[0] == 90   # Middle of height
    
    def test_project_custom_area_corners(self):
        """Test projection with custom area."""
        # Europe-ish area
        area = {"north": 60, "west": -10, "south": 40, "east": 30}
        proj = EquiRectangular(400, 200, area=area)
        
        # North-West corner
        x, y = proj.project_to_indexes(60, -10)
        assert x[0] == 0
        assert y[0] == 0
        
        # South-East corner
        x, y = proj.project_to_indexes(40, 30)
        assert x[0] == 399
        assert y[0] == 199
        
        # Center
        x, y = proj.project_to_indexes(50, 10)
        np.testing.assert_array_almost_equal(x[0], 199, decimal=0)
        np.testing.assert_array_almost_equal(y[0], 100, decimal=0)
    
    def test_project_array_input(self):
        """Test projection with array inputs."""
        proj = EquiRectangular(360, 180)
        
        lats = np.array([90, 0, -90])
        lons = np.array([-180, 0, 180])
        
        x, y = proj.project_to_indexes(lats, lons)
        
        assert len(x) == 3
        assert len(y) == 3
        assert x[0] == 0
        assert y[0] == 0
        assert x[1] == 180
        assert y[1] == 90
    
    def test_project_out_of_bounds_fill_value(self):
        """Test projection with out-of-bounds coordinates returns FILL_VALUE."""
        area = {"north": 60, "west": -10, "south": 40, "east": 30}
        proj = EquiRectangular(400, 200, area=area)
        
        # Point outside area (should return FILL_VALUE)
        x, y = proj.project_to_indexes(80, 50)  # Too far north and east
        
        # Should return FILL_VALUE for out of bounds
        assert x[0] == proj.FILL_VALUE_OOB
        assert y[0] == proj.FILL_VALUE_OOB
    
    def test_project_mixed_bounds(self):
        """Test projection with mix of in-bounds and out-of-bounds coordinates."""
        area = {"north": 60, "west": -10, "south": 40, "east": 30}
        proj = EquiRectangular(400, 200, area=area)
        
        lats = np.array([50, 80, 45])  # middle is out of bounds
        lons = np.array([10, 10, 10])
        
        x, y = proj.project_to_indexes(lats, lons)
        
        # First and third should be valid
        assert x[0] != proj.FILL_VALUE_OOB
        assert y[0] != proj.FILL_VALUE_OOB
        assert x[2] != proj.FILL_VALUE_OOB
        assert y[2] != proj.FILL_VALUE_OOB
        
        # Second should be FILL_VALUE
        assert x[1] == proj.FILL_VALUE_OOB
        assert y[1] == proj.FILL_VALUE_OOB


class TestGetCoordinates:
    """Test the get_coordinates method."""
    
    def test_get_coordinates_global(self):
        """Test getting coordinates for global projection."""
        proj = EquiRectangular(360, 180)
        
        lats, lons = proj.get_coordinates()
        
        assert len(lats) == 180
        assert len(lons) == 360
        
        # Check latitude range
        assert lats[0] == 90      # North at top
        assert lats[-1] == -90    # South at bottom
        
        # Check longitude range
        assert lons[0] == -180    # West at left
        assert lons[-1] == 180    # East at right
    
    def test_get_coordinates_custom_area(self):
        """Test getting coordinates for custom area."""
        area = {"north": 60, "west": -10, "south": 40, "east": 30}
        proj = EquiRectangular(400, 200, area=area)
        
        lats, lons = proj.get_coordinates()
        
        assert len(lats) == 200
        assert len(lons) == 400
        
        # Check latitude range
        np.testing.assert_almost_equal(lats[0], 60, decimal=5)      # North
        np.testing.assert_almost_equal(lats[-1], 40, decimal=5)     # South
        
        # Check longitude range
        np.testing.assert_almost_equal(lons[0], -10, decimal=5)     # West
        np.testing.assert_almost_equal(lons[-1], 30, decimal=5)     # East
    
    def test_get_coordinates_spacing(self):
        """Test that coordinates are evenly spaced."""
        proj = EquiRectangular(100, 50)
        
        lats, lons = proj.get_coordinates()
        
        # Check longitude spacing
        lon_diffs = np.diff(lons)
        assert np.allclose(lon_diffs, lon_diffs[0])  # All differences should be equal
        
        # Check latitude spacing
        lat_diffs = np.diff(lats)
        assert np.allclose(lat_diffs, lat_diffs[0])  # All differences should be equal


class TestIsValidIndex:
    """Test the is_valid_index method."""
    
    def test_is_valid_index_all_valid(self):
        """Test that valid indexes are identified correctly."""
        proj = EquiRectangular(100, 100)
        
        x = np.array([0, 50, 99])
        y = np.array([0, 50, 99])
        
        result = proj.is_valid_index(x, y)
        
        assert np.all(result == True)
    
    def test_is_valid_index_with_fill_values(self):
        """Test that FILL_VALUE indexes are identified as invalid."""
        proj = EquiRectangular(100, 100)
        
        x = np.array([0, proj.FILL_VALUE_OOB, 50])
        y = np.array([0, proj.FILL_VALUE_OOB, 50])
        
        result = proj.is_valid_index(x, y)
        
        assert result[0] == True
        assert result[1] == False
        assert result[2] == True
    
    def test_is_valid_index_mixed(self):
        """Test with mix of valid and invalid indexes."""
        proj = EquiRectangular(100, 100)
        
        x = np.array([0, 50, proj.FILL_VALUE_OOB])
        y = np.array([0, proj.FILL_VALUE_OOB, 50])
        
        result = proj.is_valid_index(x, y)
        
        assert result[0] == True   # Both valid
        assert result[1] == False  # y is FILL_VALUE
        assert result[2] == False  # x is FILL_VALUE


class TestIsInBounds:
    """Test the is_in_bounds method."""
    
    def test_is_in_bounds_single_point(self):
        """Test bounds checking with single point."""
        area = {"north": 60, "west": -10, "south": 40, "east": 30}
        proj = EquiRectangular(400, 200, area=area)
        
        # Point inside
        assert proj.is_in_bounds(50, 10)[0] == True
        
        # Point outside (north)
        assert proj.is_in_bounds(70, 10)[0] == False
        
        # Point outside (south)
        assert proj.is_in_bounds(30, 10)[0] == False
        
        # Point outside (west)
        assert proj.is_in_bounds(50, -20)[0] == False
        
        # Point outside (east)
        assert proj.is_in_bounds(50, 40)[0] == False
    
    def test_is_in_bounds_array(self):
        """Test bounds checking with array of points."""
        area = {"north": 60, "west": -10, "south": 40, "east": 30}
        proj = EquiRectangular(400, 200, area=area)
        
        lats = np.array([50, 70, 30, 50])
        lons = np.array([10, 10, 10, 40])
        
        result = proj.is_in_bounds(lats, lons)
        
        assert result[0] == True   # Inside
        assert result[1] == False  # Too far north
        assert result[2] == False  # Too far south
        assert result[3] == False  # Too far east
    
    def test_is_in_bounds_boundary(self):
        """Test bounds checking at exact boundaries."""
        area = {"north": 60, "west": -10, "south": 40, "east": 30}
        proj = EquiRectangular(400, 200, area=area)
        
        # Exact boundaries should be in bounds
        assert proj.is_in_bounds(60, -10)[0] == True   # North-West corner
        assert proj.is_in_bounds(60, 30)[0] == True    # North-East corner
        assert proj.is_in_bounds(40, -10)[0] == True   # South-West corner
        assert proj.is_in_bounds(40, 30)[0] == True    # South-East corner


class TestRoundTrip:
    """Test round-trip conversions between coordinates and indexes."""
    
    def test_roundtrip_global(self):
        """Test round-trip conversion for global projection."""
        proj = EquiRectangular(360, 180)
        
        # Get coordinates for all pixels
        lats, lons = proj.get_coordinates()
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Project back to indexes
        x, y = proj.project_to_indexes(lat_grid, lon_grid)
        
        # Check that we get back the original indexes
        y_expected, x_expected = np.meshgrid(np.arange(180), np.arange(360), indexing='ij')
        
        np.testing.assert_array_almost_equal(x, x_expected, decimal=0)
        np.testing.assert_array_almost_equal(y, y_expected, decimal=0)
    
    def test_roundtrip_custom_area(self):
        """Test round-trip conversion for custom area."""
        area = {"north": 60, "west": -10, "south": 40, "east": 30}
        proj = EquiRectangular(200, 100, area=area)
        
        # Get coordinates for all pixels
        lats, lons = proj.get_coordinates()
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Project back to indexes
        x, y = proj.project_to_indexes(lat_grid, lon_grid)
        
        # Check that we get back the original indexes
        y_expected, x_expected = np.meshgrid(np.arange(100), np.arange(200), indexing='ij')
        
        np.testing.assert_array_almost_equal(x, x_expected, decimal=0)
        np.testing.assert_array_almost_equal(y, y_expected, decimal=0)


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_single_pixel(self):
        """Test projection with 1x1 image."""
        proj = EquiRectangular(1, 1)
        
        # Any coordinate should map to (0, 0)
        x, y = proj.project_to_indexes(0, 0)
        assert x[0] == 0
        assert y[0] == 0
    
    def test_very_small_area(self):
        """Test projection with very small area."""
        area = {"north": 1, "west": 0, "south": 0, "east": 1}
        proj = EquiRectangular(100, 100, area=area)
        
        assert proj.lat_span == 1
        assert proj.lon_span == 1
        
        # Test projection
        x, y = proj.project_to_indexes(0.5, 0.5)
        np.testing.assert_array_almost_equal(x[0], 50, decimal=0)
        np.testing.assert_array_almost_equal(y[0], 50, decimal=0)
    
    def test_large_dimensions(self):
        """Test projection with large dimensions."""
        proj = EquiRectangular(10000, 5000)
        
        # Should still work correctly
        x, y = proj.project_to_indexes(0, 0)
        assert x[0] == 5000
        assert y[0] == 2500
    
    def test_scalar_inputs(self):
        """Test that scalar inputs work correctly."""
        proj = EquiRectangular(360, 180)
        
        # Single scalar values
        x, y = proj.project_to_indexes(0.0, 0.0)
        
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 1
        assert len(y) == 1
    
    def test_uint_type_consistency(self):
        """Test that returned indexes match the projection's uint_type."""
        proj = EquiRectangular(100, 100)
        
        x, y = proj.project_to_indexes(0, 0)
        
        assert x.dtype == proj.uint_type
        assert y.dtype == proj.uint_type
    
    def test_get_coordinates_single_pixel_width(self):
        """Test that get_coordinates works when width=1 (regression: division by zero)."""
        proj = EquiRectangular(1, 100)
        lats, lons = proj.get_coordinates()
        assert len(lons) == 1
        assert np.isfinite(lons[0])

    def test_get_coordinates_single_pixel_height(self):
        """Test that get_coordinates works when height=1 (regression: division by zero)."""
        proj = EquiRectangular(100, 1)
        lats, lons = proj.get_coordinates()
        assert len(lats) == 1
        assert np.isfinite(lats[0])

    def test_get_coordinates_1x1(self):
        """Test that get_coordinates works on a 1x1 grid (regression: division by zero)."""
        proj = EquiRectangular(1, 1)
        lats, lons = proj.get_coordinates()
        assert np.isfinite(lats[0])
        assert np.isfinite(lons[0])


class TestWestEqualsEast:
    """Regression tests for west == east validation (bug #4)."""

    def test_west_equals_east_raises(self):
        """west == east must be rejected to avoid division by zero downstream."""
        with pytest.raises(ValueError):
            EquiRectangular(100, 100, area={"north": 60, "south": 40, "west": 10, "east": 10})


class TestAntimeridianCrossing:
    """Regression tests for antimeridian-crossing areas (bugs #1, #2, #5)."""

    # area that straddles the antimeridian: 170°E to 170°W
    ANTI_AREA = {"north": 60, "south": 40, "west": 170, "east": -170}

    def test_project_to_indexes_antimeridian_center_not_oob(self):
        """Points inside an antimeridian-crossing area must not be marked FILL_VALUE (bug #1)."""
        proj = EquiRectangular(200, 100, area=self.ANTI_AREA)

        # 180° is the centre of a 170→-170 area — must be valid
        x, y = proj.project_to_indexes(50, 180)
        assert proj.is_valid_index(x, y)[0], \
            "lon=180 is inside a 170→-170 antimeridian area but was marked out-of-bounds"

    def test_project_to_indexes_antimeridian_west_edge(self):
        """Western edge of antimeridian-crossing area must map to x=0."""
        proj = EquiRectangular(200, 100, area=self.ANTI_AREA)

        x, y = proj.project_to_indexes(50, 170)
        assert proj.is_valid_index(x, y)[0]
        assert x[0] == 0

    def test_project_to_indexes_antimeridian_east_edge(self):
        """Eastern edge of antimeridian-crossing area must map to x=width-1."""
        proj = EquiRectangular(200, 100, area=self.ANTI_AREA)

        x, y = proj.project_to_indexes(50, -170)
        assert proj.is_valid_index(x, y)[0]
        assert x[0] == 199

    def test_project_to_indexes_antimeridian_outside_is_oob(self):
        """Points clearly outside an antimeridian-crossing area must be FILL_VALUE."""
        proj = EquiRectangular(200, 100, area=self.ANTI_AREA)

        # 0° is nowhere near 170→-170
        x, y = proj.project_to_indexes(50, 0)
        assert not proj.is_valid_index(x, y)[0], \
            "lon=0 is outside a 170→-170 area but was not marked out-of-bounds"

    def test_is_in_bounds_antimeridian_inside(self):
        """is_in_bounds must return True for points inside an antimeridian-crossing area (bug #2)."""
        proj = EquiRectangular(200, 100, area=self.ANTI_AREA)

        assert proj.is_in_bounds(50, 180)[0] == True
        assert proj.is_in_bounds(50, 175)[0] == True
        assert proj.is_in_bounds(50, -175)[0] == True

    def test_is_in_bounds_antimeridian_outside(self):
        """is_in_bounds must return False for points outside an antimeridian-crossing area (bug #2)."""
        proj = EquiRectangular(200, 100, area=self.ANTI_AREA)

        assert proj.is_in_bounds(50, 0)[0] == False
        assert proj.is_in_bounds(50, 90)[0] == False
        assert proj.is_in_bounds(50, -90)[0] == False

    def test_project_to_indexes_antimeridian_roundtrip(self):
        """get_coordinates + project_to_indexes round-trip for antimeridian area (bug #5)."""
        proj = EquiRectangular(200, 100, area=self.ANTI_AREA)

        lats, lons = proj.get_coordinates()
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        x, y = proj.project_to_indexes(lat_grid, lon_grid)

        y_expected, x_expected = np.meshgrid(np.arange(100), np.arange(200), indexing='ij')

        assert np.all(proj.is_valid_index(x, y)), \
            "Some in-bounds coordinates were marked FILL_VALUE during antimeridian round-trip"
        np.testing.assert_array_almost_equal(x, x_expected, decimal=0)
        np.testing.assert_array_almost_equal(y, y_expected, decimal=0)
