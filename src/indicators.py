# -*- coding: utf-8 -*-
"""
Technical indicators matching MT5 / Pine behaviour exactly.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# -- ATR ---------------------------------------------------------------------

def calc_wilder_atr(highs: np.ndarray, lows: np.ndarray,
                    closes: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's smoothed ATR (identical to MT5 iATR / Pine ta.atr).

    Initialisation:   atr[period-1] = SMA of TR over the first 'period' bars.
    Subsequent bars:  atr[i] = atr[i-1] * (period-1)/period  +  tr[i] / period
    Returns NaN for the first period-1 positions.
    """
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]),
        )

    atr = np.full(n, np.nan)
    if n < period:
        return atr

    atr[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha

    return atr


# -- Pivot high / low (vectorised) --------------------------------------------

def compute_pivot_highs(highs: np.ndarray, L: int, R: int) -> np.ndarray:
    """
    Vectorised port of MQL5 GetPivotHigh(baseShift, L, R).

    For every detection bar i, checks whether the bar at i-R is a pivot high:
        highs[i-R] >= all other highs in [i-L-R .. i]

    result[i] = highs[i-R]  if pivot, else NaN.

    MQL5 semantics: returns EMPTY_VALUE if any bar in the window is STRICTLY
    greater than the pivot -> ties are allowed (pivot can equal neighbours).
    """
    n = len(highs)
    window_size = L + R + 1
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    # windows[j] covers bars [j .. j + window_size - 1]
    windows   = sliding_window_view(highs, window_size)   # (n-ws+1, ws)
    pivot_pos = L                                          # pivot is L bars from left

    pivot_vals = windows[:, pivot_pos]

    mask = np.ones(window_size, dtype=bool)
    mask[pivot_pos] = False
    max_others = windows[:, mask].max(axis=1)

    # Valid where pivot >= all others (equiv. to MQL5 "no other bar > pivot")
    is_pivot = pivot_vals >= max_others

    # Detection bar i = j + window_size - 1  ->  result index = j + L + R
    result[window_size - 1:] = np.where(is_pivot, pivot_vals, np.nan)
    return result


def compute_pivot_lows(lows: np.ndarray, L: int, R: int) -> np.ndarray:
    """Vectorised port of MQL5 GetPivotLow. result[i] = lows[i-R] if pivot, else NaN."""
    n = len(lows)
    window_size = L + R + 1
    result = np.full(n, np.nan)
    if n < window_size:
        return result

    windows   = sliding_window_view(lows, window_size)
    pivot_pos = L

    pivot_vals = windows[:, pivot_pos]
    mask = np.ones(window_size, dtype=bool)
    mask[pivot_pos] = False
    min_others = windows[:, mask].min(axis=1)

    is_pivot = pivot_vals <= min_others
    result[window_size - 1:] = np.where(is_pivot, pivot_vals, np.nan)
    return result
