"""
Simple test to verify supersampling implementation.
"""

import numpy as np
import xarray as xr
from quark.projection.equirectangular import EquiRectangular
from quark.aggregate import Aggregator
from quark.supersampling import SpatialSuperSampler, ConstantSuperSampler


def create_simple_test_dataset():
    """Create a simple test dataset with known values."""
    # Create a small 4x4 grid
    ny, nx = 4, 4
    
    # Simple lat/lon grid
    lat = np.linspace(10, 40, ny)
    lon = np.linspace(-10, 20, nx)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    # Simple values (constant for easy verification)
    values = np.ones((ny, nx)) * 42.0
    
    ds = xr.Dataset(
        {
            'latitude': (['y', 'x'], lat_grid),
            'longitude': (['y', 'x'], lon_grid),
            'temperature': (['y', 'x'], values),
        }
    )
    
    return ds


def test_supersampling_basic():
    """Test that supersampling runs without errors."""
    print("Creating test dataset...")
    ds = create_simple_test_dataset()
    
    # Create projection
    projection = EquiRectangular(
        width=10,
        height=10,
        area={"north": 50, "south": 0, "east": 30, "west": -20}
    )
    
    # Test ss=1 (baseline)
    print("\nTesting ss=1 (no supersampling)...")
    agg1 = Aggregator(
        projection=projection,
        datasets=[ds],
        variables=['temperature'],
        supersampler=None,  # No supersampling
        return_counts=True,
    )
    result1 = agg1.compute()
    print(f"  Result shape: {result1['temperature'].shape}")
    print(f"  Non-NaN pixels: {np.isfinite(result1['temperature'].values).sum()}")
    print(f"  Mean value: {np.nanmean(result1['temperature'].values):.3f}")
    
    # Test ss=2
    print("\nTesting ss=2...")
    agg2 = Aggregator(
        projection=projection,
        datasets=[ds],
        variables=['temperature'],
        supersampler=SpatialSuperSampler(factor=2),
        return_counts=True,
    )
    result2 = agg2.compute()
    print(f"  Result shape: {result2['temperature'].shape}")
    print(f"  Non-NaN pixels: {np.isfinite(result2['temperature'].values).sum()}")
    print(f"  Mean value: {np.nanmean(result2['temperature'].values):.3f}")
    print(f"  Max count: {np.nanmax(result2['count_temperature'].values)}")
    
    # Test ss=3
    print("\nTesting ss=3...")
    agg3 = Aggregator(
        projection=projection,
        datasets=[ds],
        variables=['temperature'],
        supersampler=SpatialSuperSampler(factor=3),
        return_counts=True,
    )
    result3 = agg3.compute()
    print(f"  Result shape: {result3['temperature'].shape}")
    print(f"  Non-NaN pixels: {np.isfinite(result3['temperature'].values).sum()}")
    print(f"  Mean value: {np.nanmean(result3['temperature'].values):.3f}")
    print(f"  Max count: {np.nanmax(result3['count_temperature'].values)}")
    
    # Verify that mean values are preserved (all should be ~42.0)
    print("\n=== Verification ===")
    print(f"All means should be ~42.0:")
    print(f"  ss=1: {np.nanmean(result1['temperature'].values):.6f}")
    print(f"  ss=2: {np.nanmean(result2['temperature'].values):.6f}")
    print(f"  ss=3: {np.nanmean(result3['temperature'].values):.6f}")
    
    # Verify that higher supersampling gives more coverage
    count1 = np.isfinite(result1['temperature'].values).sum()
    count2 = np.isfinite(result2['temperature'].values).sum()
    count3 = np.isfinite(result3['temperature'].values).sum()
    
    print(f"\nCoverage (non-NaN pixels):")
    print(f"  ss=1: {count1}")
    print(f"  ss=2: {count2} (should be >= ss=1)")
    print(f"  ss=3: {count3} (should be >= ss=2)")
    
    assert count2 >= count1, "ss=2 should have at least as much coverage as ss=1"
    assert count3 >= count2, "ss=3 should have at least as much coverage as ss=2"
    
    print("\n✅ All tests passed!")


def test_high_supersampling_factors():
    """Test that high supersampling factors work correctly."""
    print("\n" + "="*60)
    print("Testing High Supersampling Factors")
    print("="*60)
    
    ds = create_simple_test_dataset()
    
    projection = EquiRectangular(
        width=10,
        height=10,
        area={"north": 50, "south": 0, "east": 30, "west": -20}
    )
    
    # Test various high factors
    for factor in [4, 5, 7, 10]:
        print(f"\nTesting factor={factor} ({factor*factor} subpixels per source pixel)...")
        
        supersampler = SpatialSuperSampler(factor=factor, project_center=True)
        
        agg = Aggregator(
            projection=projection,
            datasets=[ds],
            variables=['temperature'],
            supersampler=supersampler,
            return_counts=True,
        )
        
        result = agg.compute()
        
        mean_val = np.nanmean(result['temperature'].values)
        max_count = np.nanmax(result['count_temperature'].values)
        non_nan = np.isfinite(result['temperature'].values).sum()
        
        print(f"  Non-NaN pixels: {non_nan}")
        print(f"  Mean value: {mean_val:.3f} (should be ~42.0)")
        print(f"  Max count: {max_count}")
        
        # Verify mean is preserved
        assert np.abs(mean_val - 42.0) < 0.1, f"Mean should be ~42.0, got {mean_val}"
        
        # Max count should be reasonable (up to factor*factor per source pixel)
        assert max_count <= factor * factor, f"Max count {max_count} exceeds {factor*factor}"
    
    print("\n✅ High factor tests passed!")


def test_grid_pattern_verification():
    """Verify that supersampling uses a proper grid pattern."""
    print("\n" + "="*60)
    print("Verifying Grid Pattern")
    print("="*60)
    
    # Test that subpixel offsets form a proper grid
    from quark.supersampling import compute_subpixel_offset
    
    for factor in [2, 3, 5]:
        print(f"\nFactor {factor}:")
        offsets = []
        for i in range(factor):
            for j in range(factor):
                offset_y, offset_x = compute_subpixel_offset(i, j, factor)
                offsets.append((offset_y, offset_x))
                
        # Check symmetry: offsets should be centered around (0, 0)
        mean_y = np.mean([o[0] for o in offsets])
        mean_x = np.mean([o[1] for o in offsets])
        
        print(f"  Generated {len(offsets)} offsets (expected {factor*factor})")
        print(f"  Mean offset: ({mean_y:.6f}, {mean_x:.6f}) (should be ~0)")
        print(f"  Offset range Y: [{min(o[0] for o in offsets):.3f}, {max(o[0] for o in offsets):.3f}]")
        print(f"  Offset range X: [{min(o[1] for o in offsets):.3f}, {max(o[1] for o in offsets):.3f}]")
        
        assert len(offsets) == factor * factor, "Should generate factor*factor offsets"
        assert abs(mean_y) < 1e-10, "Offsets should be centered at 0 in Y"
        assert abs(mean_x) < 1e-10, "Offsets should be centered at 0 in X"
        
        # Check that all offsets are within [-0.5, 0.5]
        for offset_y, offset_x in offsets:
            assert -0.5 <= offset_y <= 0.5, f"Y offset {offset_y} out of range"
            assert -0.5 <= offset_x <= 0.5, f"X offset {offset_x} out of range"
    
    print("\n✅ Grid pattern verified!")


def test_constant_mode():
    """Test constant pixel width mode."""
    print("\n" + "="*60)
    print("Testing Constant Pixel Width Mode")
    print("="*60)
    
    ds = create_simple_test_dataset()
    
    projection = EquiRectangular(
        width=10,
        height=10,
        area={"north": 50, "south": 0, "east": 30, "west": -20}
    )
    
    # Test with pixel_width parameter
    print("\nTesting with pixel_width='1km'...")
    agg = Aggregator(
        projection=projection,
        datasets=[ds],
        variables=['temperature'],
        supersampler=ConstantSuperSampler(factor=2, pixel_width="1km"),
        return_counts=True,
    )
    result = agg.compute()
    
    print(f"  Result shape: {result['temperature'].shape}")
    print(f"  Non-NaN pixels: {np.isfinite(result['temperature'].values).sum()}")
    print(f"  Mean value: {np.nanmean(result['temperature'].values):.3f}")
    print(f"  Max count: {np.nanmax(result['count_temperature'].values)}")
    
    # Verify mean is preserved
    assert np.abs(np.nanmean(result['temperature'].values) - 42.0) < 0.1
    
    # Test different units
    print("\nTesting with pixel_width='500m'...")
    agg2 = Aggregator(
        projection=projection,
        datasets=[ds],
        variables=['temperature'],
        supersampler=ConstantSuperSampler(factor=2, pixel_width="500m"),
        return_counts=True,
    )
    result2 = agg2.compute()
    print(f"  Mean value: {np.nanmean(result2['temperature'].values):.3f}")
    
    # Test project_center=False (skip center pixel)
    print("\nTesting with project_center=False...")
    agg3 = Aggregator(
        projection=projection,
        datasets=[ds],
        variables=['temperature'],
        supersampler=ConstantSuperSampler(factor=2, pixel_width="1km", project_center=False),
        return_counts=True,
    )
    result3 = agg3.compute()
    print(f"  Mean value: {np.nanmean(result3['temperature'].values):.3f}")
    
    # Test that factor < 2 raises error
    print("\nTesting error on invalid factor...")
    try:
        ConstantSuperSampler(factor=1, pixel_width="1km")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Caught expected error: {str(e)[:80]}...")
    
    # Test that high factors work
    print("\nTesting high factor (10)...")
    agg_high = Aggregator(
        projection=projection,
        datasets=[ds],
        variables=['temperature'],
        supersampler=ConstantSuperSampler(factor=10, pixel_width="1km"),
        return_counts=True,
    )
    result_high = agg_high.compute()
    print(f"  Mean value: {np.nanmean(result_high['temperature'].values):.3f}")
    print(f"  Max count: {np.nanmax(result_high['count_temperature'].values)}")
    
    print("\n✅ Constant mode tests passed!")


if __name__ == "__main__":
    test_supersampling_basic()
    test_high_supersampling_factors()
    test_grid_pattern_verification()
    test_constant_mode()
