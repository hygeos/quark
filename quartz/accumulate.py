"""
quartz._summation
-----------------
Stateful accumulator classes for aggregation with pluggable summation strategies.

Each Accumulator instance manages:
- Internal sum/count state arrays
- Accumulation of values via add() method
- Finalization (computing mean, reshaping, storing results as attributes)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Base Accumulator
# ---------------------------------------------------------------------------

class BaseAccumulator(ABC):
    """
    Base class for accumulation strategies.
    
    Handles array allocation with appropriate dtypes and provides
    the finalization interface. Subclasses implement accumulation logic.
    """
    
    def __init__(
        self,
        shape: tuple[int, ...],
        sum_dtype: type = np.float64,
    ):
        """
        Initialize accumulator arrays.
        
        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of flat accumulator arrays (typically (P * grid_size,))
        sum_dtype : type
            Data type for sum accumulation (default: np.float64)
        """
        # Infer minimal count dtype based on array size
        total_size = int(np.prod(shape))
        count_dtype = np.uint32 if total_size < np.iinfo(np.uint32).max else np.uint64
        
        self.sum_acc = np.zeros(shape, dtype=sum_dtype)
        self.cnt_acc = np.zeros(shape, dtype=count_dtype)
        self.sum_dtype = sum_dtype
        self.count_dtype = count_dtype
        
        # Result grids (populated after finalize())
        self.mean_grid = None
        self.sum_grid = None
        self.cnt_grid = None
    
    @abstractmethod
    def add(self, indices: np.ndarray, values: np.ndarray) -> None:
        """
        Accumulate values at specified indices.
        
        Parameters
        ----------
        indices : np.ndarray, shape (N,)
            Flat indices where to accumulate
        values : np.ndarray, shape (N,)
            Values to add
        """
        ...
    
    def finalize(
        self,
        output_shape: tuple[int, ...],
        output_dtype: np.dtype,
        return_sums: bool = True,
        return_counts: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Compute final mean and reshape to output dimensions.
        
        Stores results as attributes (mean_grid, sum_grid, cnt_grid) and
        returns them as a tuple for backwards compatibility.
        
        Parameters
        ----------
        output_shape : tuple[int, ...]
            Final output shape (e.g., (*preserved_dims, height, width))
        output_dtype : np.dtype
            Data type for the mean result (original variable dtype)
        return_sums : bool
            If True, create and return sum_grid view (default: True)
        return_counts : bool
            If True, create and return cnt_grid view (default: True)
        
        Returns
        -------
        mean_grid : np.ndarray
            Mean values (sum / count), NaN where count == 0
        sum_grid : np.ndarray | None
            Sum values (view of reshaped sum_acc), or None if not requested
        cnt_grid : np.ndarray | None
            Count values (view of reshaped cnt_acc), or None if not requested
        """
        # Create views only if requested (memory optimization)
        if return_sums:
            self.sum_grid = self.sum_acc.reshape(output_shape)
        else:
            self.sum_grid = None
        
        if return_counts:
            self.cnt_grid = self.cnt_acc.reshape(output_shape)
        else:
            self.cnt_grid = None
        
        # Compute mean = sum / count (NaN where count == 0)
        # Use reshaped views directly to avoid extra allocations
        sum_view = self.sum_acc.reshape(output_shape)
        cnt_view = self.cnt_acc.reshape(output_shape)
        
        with np.errstate(invalid="ignore", divide="ignore"):
            self.mean_grid = np.where(
                cnt_view > 0,
                sum_view / cnt_view,
                np.nan
            ).astype(output_dtype)
        
        return self.mean_grid, self.sum_grid, self.cnt_grid


# ---------------------------------------------------------------------------
# Concrete Accumulator Implementations
# ---------------------------------------------------------------------------

class SimpleAccumulator(BaseAccumulator):
    """
    Simple accumulator using np.bincount.
    
    Fast and straightforward but subject to floating-point error accumulation.
    Suitable for most use cases where numerical precision is not critical.
    """
    
    def add(self, indices: np.ndarray, values: np.ndarray) -> None:
        """Accumulate using direct bincount summation."""
        n = len(self.sum_acc)
        self.sum_acc += np.bincount(indices, weights=values, minlength=n)
        self.cnt_acc += np.bincount(indices, minlength=n).astype(self.count_dtype)


# ---------------------------------------------------------------------------
# Numba JIT-compiled Kahan summation kernel
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @njit(parallel=True)
    def _kahan_add_sorted(unique_indices, split_points, sorted_values, sum_acc, cmp_acc):
        """
        JIT-compiled Kahan summation for sorted indices.
        
        Processes pre-sorted values grouped by index, applying Kahan
        compensation algorithm to minimize floating-point error.
        Parallelized over unique bins.
        """
        for i in prange(len(unique_indices)):
            idx = unique_indices[i]
            start = split_points[i]
            end = split_points[i + 1]
            
            # Load current state for this bin
            s = sum_acc[idx]
            c = cmp_acc[idx]
            
            # Apply Kahan summation to all values for this bin
            for j in range(start, end):
                y = sorted_values[j] - c
                t = s + y
                c = (t - s) - y
                s = t
            
            # Store updated state
            sum_acc[idx] = s
            cmp_acc[idx] = c


class KahanAccumulator(BaseAccumulator):
    """
    Kahan (compensated) accumulator for reduced floating-point error.
    
    Tracks error compensation terms to improve numerical accuracy using
    JIT-compiled numba code for performance. Significantly reduces
    floating-point rounding errors compared to SimpleAccumulator.
    
    Requires numba to be installed. Will raise ImportError if not available.
    """
    
    def __init__(
        self,
        shape: tuple[int, ...],
        sum_dtype: type = np.float64,
    ):
        if not HAS_NUMBA:
            raise ImportError(
                "Numba is required for Kahan summation but is not installed. "
            )
        
        super().__init__(shape, sum_dtype)
        
        # Compensation array (tracks accumulated rounding errors)
        self.cmp_acc = np.zeros(shape, dtype=sum_dtype)
    
    def add(self, indices: np.ndarray, values: np.ndarray) -> None:
        """
        Accumulate using Kahan compensated summation.
        
        Sorts values by index and applies JIT-compiled Kahan algorithm
        for maximum performance while maintaining precision.
        """
        # Sort by index to group values going to same bin
        sort_order = np.argsort(indices)
        sorted_idx = indices[sort_order]
        sorted_val = values[sort_order]
        
        # Find boundaries where index changes
        unique_idx, split_points = np.unique(sorted_idx, return_index=True)
        split_points = np.append(split_points, len(sorted_idx))
        
        # Apply JIT-compiled Kahan summation
        _kahan_add_sorted(unique_idx, split_points, sorted_val, self.sum_acc, self.cmp_acc)
        
        # Count accumulation (vectorized, no compensation needed)
        n = len(self.sum_acc)
        self.cnt_acc += np.bincount(indices, minlength=n).astype(self.count_dtype)

