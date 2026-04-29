"""
Tests for quartz.aggregate.aggregate().

Tolerance constants (do not hardcode inside assertions):
"""

import numpy as np
import pytest
import xarray as xr

from quartz.aggregate import Aggregator
from quartz.supersampling import SpatialSuperSampler
from quartz.projection.equirectangular import EquiRectangular

# ---------------------------------------------------------------------------
# Tolerance constants for numeric parity checks
# ---------------------------------------------------------------------------
PARITY_RTOL = 1e-6
PARITY_ATOL = 1e-9


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def proj_5x3():
    """
    5-wide × 3-tall global equirectangular projection.
    Pixel centre coordinates:
      x: 0→lon=-180, 1→lon=-90, 2→lon=0, 3→lon=90, 4→lon=180
      y: 0→lat=90,   1→lat=0,   2→lat=-90
    """
    return EquiRectangular(width=5, height=3)


def make_ds_2d(lat, lon, value, var_name="temperature"):
    """
    Single-pixel 2-D dataset at a single (lat, lon) with a given scalar value.
    lat/lon dims: ('y_src', 'x_src').
    """
    lat_arr = np.array([[lat]], dtype=np.float64)
    lon_arr = np.array([[lon]], dtype=np.float64)
    val_arr = np.array([[value]], dtype=np.float64)
    return xr.Dataset(
        {
            "latitude": xr.DataArray(lat_arr, dims=["y_src", "x_src"]),
            "longitude": xr.DataArray(lon_arr, dims=["y_src", "x_src"]),
            var_name: xr.DataArray(val_arr, dims=["y_src", "x_src"]),
        }
    )


def make_ds_3d(lat, lon, band_values, var_name="reflectance"):
    """
    Single-pixel 3-D dataset at (lat, lon) with per-band values.
    Variable dims: ('y_src', 'x_src', 'band').
    """
    lat_arr = np.array([[lat]], dtype=np.float64)
    lon_arr = np.array([[lon]], dtype=np.float64)
    # shape (1, 1, n_bands)
    val_arr = np.array([[band_values]], dtype=np.float64)
    return xr.Dataset(
        {
            "latitude": xr.DataArray(lat_arr, dims=["y_src", "x_src"]),
            "longitude": xr.DataArray(lon_arr, dims=["y_src", "x_src"]),
            var_name: xr.DataArray(val_arr, dims=["y_src", "x_src", "band"]),
        }
    )


def make_ds_4d(lat, lon, data_4d, var_name="radiance"):
    """
    Single-pixel 4-D dataset.  data_4d shape: (n_time, n_band).
    Variable dims: ('time', 'y_src', 'x_src', 'band').
    """
    lat_arr = np.array([[lat]], dtype=np.float64)
    lon_arr = np.array([[lon]], dtype=np.float64)
    # shape (n_time, 1, 1, n_band)
    val_arr = data_4d[:, np.newaxis, np.newaxis, :]
    return xr.Dataset(
        {
            "latitude": xr.DataArray(lat_arr, dims=["y_src", "x_src"]),
            "longitude": xr.DataArray(lon_arr, dims=["y_src", "x_src"]),
            var_name: xr.DataArray(val_arr, dims=["time", "y_src", "x_src", "band"]),
        }
    )


def aggregate(datasets, projection, **kwargs):
    """
    Helper function to run aggregation using Aggregator class.
    Maintains backwards compatibility with test code.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]
    
    # Extract keyword arguments
    variables = kwargs.pop("variables", None)
    lat_name = kwargs.pop("lat_name", "latitude")
    lon_name = kwargs.pop("lon_name", "longitude")
    fail_on_schema_mismatch = kwargs.pop("fail_on_schema_mismatch", False)
    sum_method = kwargs.pop("sum_method", "simple")
    skipna = kwargs.pop("skipna", True)
    supersampling = kwargs.pop("supersampling", 1)
    return_counts = kwargs.pop("return_counts", False)
    return_sums = kwargs.pop("return_sums", False)
    dtype = kwargs.pop("dtype", None)
    
    # vars_batch_size is not supported, just pop it
    vars_batch_size = kwargs.pop("vars_batch_size", None)
    if vars_batch_size is not None:
        raise NotImplementedError(
            "Streaming mode (vars_batch_size is not None) is not yet implemented."
        )
    
    # Convert old supersampling integer to new supersampler object
    # Validate first (maintain backward compatibility with old test API)
    if not isinstance(supersampling, int) or supersampling < 1:
        raise ValueError(f"factor must be an integer >= 1, got {supersampling}")
    
    supersampler = None
    if supersampling > 1:
        supersampler = SpatialSuperSampler(factor=supersampling, project_center=True)
    
    # Create aggregator and compute
    agg = Aggregator(
        projection=projection,
        datasets=datasets,
        lat_name=lat_name,
        lon_name=lon_name,
        variables=variables,
        fail_on_schema_mismatch=fail_on_schema_mismatch,
        sum_method=sum_method,
        skipna=skipna,
        supersampler=supersampler,
        return_counts=return_counts,
        return_sums=return_sums,
        dtype=dtype,
    )
    
    return agg.compute()


# ---------------------------------------------------------------------------
# 2-D (no preserved dims)
# ---------------------------------------------------------------------------

class TestAggregate2D:

    def test_single_pixel_single_dataset(self):
        """Single source pixel maps to expected target pixel; value preserved."""
        proj = proj_5x3()
        ds = make_ds_2d(lat=0.0, lon=0.0, value=42.0)
        result = aggregate([ds], proj)

        assert "temperature" in result
        out = result["temperature"]
        assert out.dims == ("y", "x")
        assert out.shape == (3, 5)
        np.testing.assert_allclose(
            out.values[1, 2], 42.0, rtol=PARITY_RTOL, atol=PARITY_ATOL
        )

    def test_mean_from_two_datasets(self):
        """Two datasets at same pixel → mean of their values."""
        proj = proj_5x3()
        ds1 = make_ds_2d(lat=0.0, lon=0.0, value=10.0)
        ds2 = make_ds_2d(lat=0.0, lon=0.0, value=20.0)
        result = aggregate([ds1, ds2], proj)

        np.testing.assert_allclose(
            result["temperature"].values[1, 2], 15.0,
            rtol=PARITY_RTOL, atol=PARITY_ATOL,
        )

    def test_empty_cells_are_nan(self):
        """Pixels with no data must be NaN."""
        proj = proj_5x3()
        ds = make_ds_2d(lat=0.0, lon=0.0, value=1.0)
        result = aggregate([ds], proj)

        out = result["temperature"].values
        assert np.isnan(out[0, 0])   # top-left untouched
        assert np.isnan(out[2, 4])   # bottom-right untouched
        assert not np.isnan(out[1, 2])  # the projected pixel

    def test_output_shape_and_dims(self):
        """Output has correct shape, dims, and coordinate arrays."""
        proj = proj_5x3()
        ds = make_ds_2d(lat=0.0, lon=0.0, value=1.0)
        result = aggregate([ds], proj)

        assert result["temperature"].dims == ("y", "x")
        assert result["temperature"].shape == (proj.height, proj.width)
        assert "latitude" in result.coords
        assert "longitude" in result.coords
        assert result.coords["latitude"].shape == (proj.height,)
        assert result.coords["longitude"].shape == (proj.width,)

    def test_return_counts(self):
        """return_counts adds count_<var> with correct values."""
        proj = proj_5x3()
        ds1 = make_ds_2d(lat=0.0, lon=0.0, value=5.0)
        ds2 = make_ds_2d(lat=0.0, lon=0.0, value=5.0)
        result = aggregate([ds1, ds2], proj, return_counts=True)

        assert "count_temperature" in result
        cnt = result["count_temperature"].values
        assert cnt[1, 2] == 2     # two datasets hit this pixel
        assert cnt[0, 0] == 0     # no data

    def test_return_sums(self):
        """return_sums adds sum_<var> arrays."""
        proj = proj_5x3()
        ds = make_ds_2d(lat=0.0, lon=0.0, value=7.0)
        result = aggregate([ds], proj, return_sums=True)

        assert "sum_temperature" in result
        np.testing.assert_allclose(
            result["sum_temperature"].values[1, 2], 7.0,
            rtol=PARITY_RTOL, atol=PARITY_ATOL,
        )

    def test_out_of_bounds_ignored(self):
        """Points outside the projection area do not write to any pixel."""
        proj = EquiRectangular(
            width=10, height=5,
            area={"north": 10, "south": -10, "west": -10, "east": 10},
        )
        # source point far outside the area
        ds = make_ds_2d(lat=80.0, lon=150.0, value=999.0)
        result = aggregate([ds], proj)

        assert np.all(np.isnan(result["temperature"].values))

    def test_nan_values_skipped_by_default(self):
        """NaN source values do not increment sum or count (skipna=True)."""
        proj = proj_5x3()
        ds_nan = make_ds_2d(lat=0.0, lon=0.0, value=float("nan"))
        ds_ok  = make_ds_2d(lat=0.0, lon=0.0, value=42.0)

        result_nan_only = aggregate([ds_nan], proj)
        # pixel that received only NaN stays NaN
        assert np.isnan(result_nan_only["temperature"].values[1, 2])

        result_mixed = aggregate([ds_nan, ds_ok], proj)
        # NaN dataset ignored → only ds_ok contributes → mean = 42
        np.testing.assert_allclose(
            result_mixed["temperature"].values[1, 2], 42.0,
            rtol=PARITY_RTOL, atol=PARITY_ATOL,
        )

    def test_inf_values_skipped(self):
        """Inf source values are also skipped when skipna=True."""
        proj = proj_5x3()
        ds_inf = make_ds_2d(lat=0.0, lon=0.0, value=float("inf"))
        result = aggregate([ds_inf], proj)
        assert np.isnan(result["temperature"].values[1, 2])

    def test_skipna_false(self):
        """With skipna=False, NaN values contaminate the pixel output."""
        proj = proj_5x3()
        ds_nan = make_ds_2d(lat=0.0, lon=0.0, value=float("nan"))
        ds_ok  = make_ds_2d(lat=0.0, lon=0.0, value=10.0)
        result = aggregate([ds_nan, ds_ok], proj, skipna=False)
        # nan + 10 / 2 = nan
        assert np.isnan(result["temperature"].values[1, 2])

    def test_global_attrs(self):
        """Output Dataset carries expected metadata attributes."""
        proj = proj_5x3()
        ds = make_ds_2d(lat=0.0, lon=0.0, value=1.0)
        result = aggregate([ds], proj)

        assert result.attrs["projection_name"] == "EquiRectangular"
        assert result.attrs["projection_width"] == proj.width
        assert result.attrs["projection_height"] == proj.height
        assert result.attrs["n_datasets"] == 1
        assert result.attrs["sum_method"] == "simple"
        assert result.attrs["mode"] == "in-memory"

    def test_dtype_float32(self):
        """Output respects the requested dtype."""
        proj = proj_5x3()
        ds = make_ds_2d(lat=0.0, lon=0.0, value=1.0)
        result = aggregate([ds], proj, dtype=np.float32)
        assert result["temperature"].dtype == np.float32


# ---------------------------------------------------------------------------
# Multiple pixels / accumulation correctness
# ---------------------------------------------------------------------------

class TestAccumulationCorrectness:

    def test_multiple_pixels_independent(self):
        """Two datasets hitting different pixels do not bleed into each other."""
        proj = proj_5x3()
        ds1 = make_ds_2d(lat=90.0, lon=-180.0, value=100.0)  # -> y=0, x=0
        ds2 = make_ds_2d(lat=-90.0, lon=180.0, value=200.0)  # -> y=2, x=4
        result = aggregate([ds1, ds2], proj)

        out = result["temperature"].values
        np.testing.assert_allclose(out[0, 0], 100.0, rtol=PARITY_RTOL, atol=PARITY_ATOL)
        np.testing.assert_allclose(out[2, 4], 200.0, rtol=PARITY_RTOL, atol=PARITY_ATOL)
        assert np.isnan(out[1, 2])

    def test_many_datasets_same_pixel_mean(self):
        """Mean over N datasets is accurate regardless of N."""
        proj = proj_5x3()
        values = np.arange(1, 11, dtype=float)  # 1..10
        datasets = [make_ds_2d(0.0, 0.0, v) for v in values]
        result = aggregate(datasets, proj)

        expected_mean = values.mean()
        np.testing.assert_allclose(
            result["temperature"].values[1, 2], expected_mean,
            rtol=PARITY_RTOL, atol=PARITY_ATOL,
        )

    def test_variable_union_across_datasets(self):
        """Union of variables: both 'a' and 'b' appear in the output."""
        proj = proj_5x3()
        lat_arr = np.array([[0.0]])
        lon_arr = np.array([[0.0]])
        ds1 = xr.Dataset({
            "latitude": xr.DataArray(lat_arr, dims=["y_src", "x_src"]),
            "longitude": xr.DataArray(lon_arr, dims=["y_src", "x_src"]),
            "a":   xr.DataArray(np.array([[1.0]]), dims=["y_src", "x_src"]),
        })
        ds2 = xr.Dataset({
            "latitude": xr.DataArray(lat_arr, dims=["y_src", "x_src"]),
            "longitude": xr.DataArray(lon_arr, dims=["y_src", "x_src"]),
            "b":   xr.DataArray(np.array([[2.0]]), dims=["y_src", "x_src"]),
        })
        result = aggregate([ds1, ds2], proj)

        assert "a" in result
        assert "b" in result
        np.testing.assert_allclose(result["a"].values[1, 2], 1.0,
                                   rtol=PARITY_RTOL, atol=PARITY_ATOL)
        np.testing.assert_allclose(result["b"].values[1, 2], 2.0,
                                   rtol=PARITY_RTOL, atol=PARITY_ATOL)

    def test_explicit_variables_missing_raises(self):
        """Hard error when an explicitly requested variable is absent from any dataset."""
        proj = proj_5x3()
        ds = make_ds_2d(0.0, 0.0, 1.0, var_name="temperature")
        with pytest.raises(ValueError, match="not found"):
            aggregate([ds], proj, variables=["nonexistent"])

    def test_fail_on_schema_mismatch(self):
        """Raise when variable sets differ and fail_on_schema_mismatch=True."""
        proj = proj_5x3()
        lat = np.array([[0.0]])
        lon = np.array([[0.0]])
        ds1 = xr.Dataset({
            "latitude": xr.DataArray(lat, dims=["y_src", "x_src"]),
            "longitude": xr.DataArray(lon, dims=["y_src", "x_src"]),
            "a": xr.DataArray(np.array([[1.0]]), dims=["y_src", "x_src"]),
        })
        ds2 = xr.Dataset({
            "latitude": xr.DataArray(lat, dims=["y_src", "x_src"]),
            "longitude": xr.DataArray(lon, dims=["y_src", "x_src"]),
            "b": xr.DataArray(np.array([[1.0]]), dims=["y_src", "x_src"]),
        })
        with pytest.raises(ValueError, match="different"):
            aggregate([ds1, ds2], proj, fail_on_schema_mismatch=True)


# ---------------------------------------------------------------------------
# NDIMS: 3-D variables (y_src, x_src, band)
# ---------------------------------------------------------------------------

class TestAggregate3D:

    def test_3d_shape_and_dims(self):
        """3-D variable produces (*preserved, y, x) output."""
        proj = proj_5x3()
        ds = make_ds_3d(lat=0.0, lon=0.0, band_values=[1.0, 2.0, 3.0])
        result = aggregate([ds], proj)

        out = result["reflectance"]
        assert out.dims == ("band", "y", "x")
        assert out.shape == (3, 3, 5)

    def test_3d_values_correct(self):
        """Each band contains the correct reprojected mean."""
        proj = proj_5x3()
        band_vals = [10.0, 20.0, 30.0]
        ds = make_ds_3d(lat=0.0, lon=0.0, band_values=band_vals)
        result = aggregate([ds], proj)

        out = result["reflectance"]
        for b, expected in enumerate(band_vals):
            np.testing.assert_allclose(
                out.values[b, 1, 2], expected,
                rtol=PARITY_RTOL, atol=PARITY_ATOL,
            )

    def test_3d_empty_bands_are_nan(self):
        """Empty pixels in every band are NaN."""
        proj = proj_5x3()
        ds = make_ds_3d(lat=0.0, lon=0.0, band_values=[1.0, 2.0])
        result = aggregate([ds], proj)

        out = result["reflectance"].values
        for b in range(2):
            assert np.isnan(out[b, 0, 0])

    def test_3d_mean_across_two_datasets(self):
        """Mean per band is computed correctly from two datasets."""
        proj = proj_5x3()
        ds1 = make_ds_3d(lat=0.0, lon=0.0, band_values=[0.0, 10.0])
        ds2 = make_ds_3d(lat=0.0, lon=0.0, band_values=[20.0, 30.0])
        result = aggregate([ds1, ds2], proj)

        out = result["reflectance"].values
        np.testing.assert_allclose(out[0, 1, 2], 10.0, rtol=PARITY_RTOL, atol=PARITY_ATOL)
        np.testing.assert_allclose(out[1, 1, 2], 20.0, rtol=PARITY_RTOL, atol=PARITY_ATOL)

    def test_3d_nan_in_one_band_does_not_affect_other(self):
        """NaN in band 0 leaves band 1 unaffected."""
        proj = proj_5x3()
        ds = make_ds_3d(lat=0.0, lon=0.0, band_values=[float("nan"), 5.0])
        result = aggregate([ds], proj)

        out = result["reflectance"].values
        assert np.isnan(out[0, 1, 2])
        np.testing.assert_allclose(out[1, 1, 2], 5.0, rtol=PARITY_RTOL, atol=PARITY_ATOL)


# ---------------------------------------------------------------------------
# NDIMS: 4-D variables (time, y_src, x_src, band)
# ---------------------------------------------------------------------------

class TestAggregate4D:

    def test_4d_shape_and_dims(self):
        """4-D variable produces (time, band, y, x) output."""
        proj = proj_5x3()
        data = np.ones((4, 2), dtype=np.float64)  # 4 time steps, 2 bands
        ds = make_ds_4d(lat=0.0, lon=0.0, data_4d=data)
        result = aggregate([ds], proj)

        out = result["radiance"]
        assert out.dims == ("time", "band", "y", "x")
        assert out.shape == (4, 2, 3, 5)

    def test_4d_values_correct(self):
        """Each (time, band) slice has the correct reprojected value."""
        proj = proj_5x3()
        data = np.arange(1, 7, dtype=np.float64).reshape(3, 2)  # 3 times, 2 bands
        ds = make_ds_4d(lat=0.0, lon=0.0, data_4d=data)
        result = aggregate([ds], proj)

        out = result["radiance"].values
        for t in range(3):
            for b in range(2):
                np.testing.assert_allclose(
                    out[t, b, 1, 2], data[t, b],
                    rtol=PARITY_RTOL, atol=PARITY_ATOL,
                )


# ---------------------------------------------------------------------------
# Mask callable
# ---------------------------------------------------------------------------

class TestMaskCallable:
    """Mask parameter has been removed - these tests are skipped."""

    @pytest.mark.skip(reason="mask parameter removed from Aggregator")
    def test_mask_blocks_pixel(self):
        """Pixels where mask=False are not accumulated."""
        proj = proj_5x3()
        ds = make_ds_2d(lat=0.0, lon=0.0, value=99.0)

        # mask returns False for all pixels → nothing accumulates
        def all_false_mask(ds):
            lat = ds["latitude"]
            return xr.DataArray(
                np.zeros(lat.shape, dtype=bool), dims=lat.dims
            )

        result = aggregate([ds], proj, mask=all_false_mask)
        assert np.isnan(result["temperature"].values[1, 2])

    @pytest.mark.skip(reason="mask parameter removed from Aggregator")
    def test_mask_passes_pixel(self):
        """Pixels where mask=True are accumulated normally."""
        proj = proj_5x3()
        ds = make_ds_2d(lat=0.0, lon=0.0, value=42.0)

        def all_true_mask(ds):
            lat = ds["latitude"]
            return xr.DataArray(
                np.ones(lat.shape, dtype=bool), dims=lat.dims
            )

        result = aggregate([ds], proj, mask=all_true_mask)
        np.testing.assert_allclose(
            result["temperature"].values[1, 2], 42.0,
            rtol=PARITY_RTOL, atol=PARITY_ATOL,
        )


# ---------------------------------------------------------------------------
# Validation and error handling
# ---------------------------------------------------------------------------

class TestValidation:

    def test_empty_datasets_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            aggregate([], proj_5x3())

    def test_missing_geo_var_raises(self):
        ds = xr.Dataset({"x": xr.DataArray(np.array([[1.0]]), dims=["a", "b"])})
        with pytest.raises(ValueError, match="latitude"):
            aggregate([ds], proj_5x3())

    def test_lat_lon_not_2d_raises(self):
        ds = xr.Dataset({
            "latitude": xr.DataArray(np.array([0.0]), dims=["y_src"]),
            "longitude": xr.DataArray(np.array([0.0]), dims=["y_src"]),
            "v":   xr.DataArray(np.array([1.0]), dims=["y_src"]),
        })
        with pytest.raises(ValueError, match="2-D"):
            aggregate([ds], proj_5x3())

    def test_supersampling_invalid_raises(self):
        """Test that invalid supersampling values raise an error."""
        ds = make_ds_2d(0.0, 0.0, 1.0)
        # Test factor < 1
        with pytest.raises(ValueError, match="factor must be an integer >= 1"):
            aggregate([ds], proj_5x3(), supersampling=0)
        with pytest.raises(ValueError, match="factor must be an integer >= 1"):
            aggregate([ds], proj_5x3(), supersampling=-1)

    def test_vars_batch_size_raises(self):
        ds = make_ds_2d(0.0, 0.0, 1.0)
        with pytest.raises(NotImplementedError, match="Streaming"):
            aggregate([ds], proj_5x3(), vars_batch_size=1)

    def test_kahan_raises(self):
        """Kahan summation requires numba."""
        ds = make_ds_2d(0.0, 0.0, 42.0)
        try:
            import numba
            # If numba is available, should work
            result = aggregate([ds], proj_5x3(), sum_method="kahan")
            assert "temperature" in result
            np.testing.assert_allclose(
                result["temperature"].values[1, 2], 42.0,
                rtol=PARITY_RTOL, atol=PARITY_ATOL
            )
        except ImportError:
            # If numba not available, should raise ImportError
            with pytest.raises(ImportError, match="Numba is required"):
                aggregate([ds], proj_5x3(), sum_method="kahan")

    def test_unknown_sum_method_raises(self):
        ds = make_ds_2d(0.0, 0.0, 1.0)
        with pytest.raises(ValueError, match="Unknown sum_method"):
            aggregate([ds], proj_5x3(), sum_method="magic")
