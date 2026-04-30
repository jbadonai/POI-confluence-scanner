# -*- coding: utf-8 -*-
"""
Overbought/Oversold Indicator — Python port of ceyhun's Pine Script v4.

Pine Script source
------------------
    ys1 = (high + low + close * 2) / 4
    rk3 = ema(ys1, n)
    rk4 = stdev(ys1, n)                    <- population std, ddof=0
    rk5 = (ys1 - rk3) * 100 / rk4
    rk6 = ema(rk5, n)
    up   = ema(rk6, n)
    down = ema(up,  n)
    Buy  = crossover(up,  down)            <- up crosses above down
    Sell = crossunder(up, down)            <- up crosses below  down

Pine Script behaviour notes
---------------------------
- ema() in Pine skips NaN inputs and seeds on the first non-NaN value.
- stdev() returns NaN until `period` bars of non-NaN src are available.
- rk5 is NaN wherever rk4 == 0 (impossible to normalise). Pine treats
  NaN inputs to ema() as transparent — it skips them and seeds once it
  has enough valid values. We replicate this by substituting NaN → 0.0
  in rk5 before the next EMA pass (same effective behaviour because
  rk5=0 means ys1 == rk3, i.e. price is exactly on its mean).

Public API
----------
obos_signals(highs, lows, closes, n) -> dict
    up, down : np.ndarray series
    buy, sell: bool arrays (crossover / crossunder bars)

last_signal(highs, lows, closes, n, lookback) -> (bar_offset, direction)
    bar_offset : bars ago (0 = last bar)
    direction  : +1 buy, -1 sell, 0 = none found
"""

from typing import Tuple
import numpy as np


# ---------------------------------------------------------------------------
# EMA helper — seeds on the first non-NaN value, matches Pine ema()
# ---------------------------------------------------------------------------

def _ema_series(src: np.ndarray, period: int) -> np.ndarray:
    n   = len(src)
    out = np.full(n, np.nan)
    k   = 2.0 / (period + 1)

    # Find first index where we have `period` consecutive non-NaN values
    valid = ~np.isnan(src)
    seed_idx = -1
    count = 0
    for i in range(n):
        if valid[i]:
            count += 1
            if count >= period:
                seed_idx = i
                break
        else:
            count = 0

    if seed_idx < 0:
        return out

    out[seed_idx] = np.mean(src[seed_idx - period + 1 : seed_idx + 1])
    for i in range(seed_idx + 1, n):
        if np.isnan(src[i]):
            out[i] = out[i - 1]   # carry forward (Pine skips NaN)
        else:
            out[i] = src[i] * k + out[i - 1] * (1.0 - k)
    return out


# ---------------------------------------------------------------------------
# Rolling population stdev — matches Pine ta.stdev() (ddof=0)
# ---------------------------------------------------------------------------

def _stdev_series(src: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(src), np.nan)
    for i in range(period - 1, len(src)):
        window = src[i - period + 1 : i + 1]
        if not np.any(np.isnan(window)):
            out[i] = float(np.std(window, ddof=0))
    return out


# ---------------------------------------------------------------------------
# Main indicator
# ---------------------------------------------------------------------------

def obos_signals(highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, n: int = 5) -> dict:
    """
    Compute the full OB/OS indicator series for all bars.
    Returns dict with: up, down, buy, sell, rk5, rk6
    """
    # ys1 = weighted close  (H + L + C*2) / 4
    ys1 = (highs + lows + closes * 2.0) / 4.0

    rk3 = _ema_series(ys1, n)

    rk4 = _stdev_series(ys1, n)

    # rk5 = normalised deviation. Where rk4==0 or NaN, substitute 0.0
    # (Pine skips NaN in ema, equivalent to treating deviation as 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        rk5_raw = np.where(rk4 > 0, (ys1 - rk3) * 100.0 / rk4, np.nan)
    rk5 = np.where(np.isnan(rk5_raw), 0.0, rk5_raw)

    rk6  = _ema_series(rk5, n)
    up   = _ema_series(rk6, n)
    down = _ema_series(up,  n)

    # crossover(up, down)  : up[i-1] <= down[i-1] AND up[i] > down[i]
    # crossunder(up, down) : up[i-1] >= down[i-1] AND up[i] < down[i]
    total  = len(closes)
    buy    = np.zeros(total, dtype=bool)
    sell   = np.zeros(total, dtype=bool)

    for i in range(1, total):
        u0, d0 = up[i - 1], down[i - 1]
        u1, d1 = up[i],     down[i]
        if np.isnan(u0) or np.isnan(d0) or np.isnan(u1) or np.isnan(d1):
            continue
        # crossover(up, down) — up crosses above down
        if u0 <= d0 and u1 > d1:
            # Valid BUY only when BOTH lines are in negative territory (below 0).
            # Crossovers near or above the 0 line are noise (yellow on the chart).
            if u1 < 0 and d1 < 0:
                buy[i] = True
        # crossunder(up, down) — up crosses below down
        elif u0 >= d0 and u1 < d1:
            # Valid SELL only when BOTH lines are in positive territory (above 0).
            # Crossunders near or below the 0 line are noise (yellow on the chart).
            if u1 > 0 and d1 > 0:
                sell[i] = True

    return {"up": up, "down": down, "buy": buy, "sell": sell,
            "rk5": rk5, "rk6": rk6}


def last_signal(highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, n: int,
                lookback: int) -> Tuple[int, int]:
    """
    Most recent valid OB/OS crossover within the last `lookback` confirmed bars.

    Returns (bar_offset, direction):
        bar_offset : bars ago from last bar (0 = current bar)
        direction  : +1 buy, -1 sell, 0 = none found

    Validity re-check at the REJECTION bar (last bar = index n-1):
    ---------------------------------------------------------------
    When the crossover fired BEFORE the rejection (offset > 0), we must
    verify that the OB/OS lines are still on the correct side of zero AT
    the rejection bar — not just at the original crossover bar.

    Example: SELL crossunder fires above 0 (valid), but by the time the
    rejection candle appears 3 bars later, up/down have crossed below 0.
    That means the signal context has changed — trade is invalidated.

    Re-check rules (applied at the last bar regardless of offset):
        BUY  (direction +1): up[-1] < 0  AND  down[-1] < 0
        SELL (direction -1): up[-1] > 0  AND  down[-1] > 0
    """
    sig      = obos_signals(highs, lows, closes, n)
    last_idx = len(closes) - 1

    # Current (rejection bar) up/down values
    up_now   = sig["up"][last_idx]
    down_now = sig["down"][last_idx]

    for offset in range(lookback + 1):
        i = last_idx - offset
        if i < 0:
            break

        if sig["buy"][i]:
            # Original crossover was valid (already guaranteed by obos_signals).
            # Re-check: are lines still below zero AT the rejection bar?
            if np.isnan(up_now) or np.isnan(down_now):
                return lookback + 1, 0
            if up_now < 0 and down_now < 0:
                return offset, +1
            else:
                # Lines have drifted — crossover context no longer valid
                return lookback + 1, 0

        if sig["sell"][i]:
            # Re-check: are lines still above zero AT the rejection bar?
            if np.isnan(up_now) or np.isnan(down_now):
                return lookback + 1, 0
            if up_now > 0 and down_now > 0:
                return offset, -1
            else:
                return lookback + 1, 0

    return lookback + 1, 0
