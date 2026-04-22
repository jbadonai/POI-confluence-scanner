# -*- coding: utf-8 -*-
"""
HTF (Higher Time Frame) Trend Filter — hardened against false flips.

Trend detection upgrades applied
---------------------------------
1. Dual EMA  (fast + slow)  — structure confirmation, not just price
2. EMA slope               — must be rising/falling, flat = neutral
3. Consecutive closes       — N closes same side before committing
4. ATR buffer zone          — ignores tiny EMA crosses (anti-whipsaw)
5. Trend lock               — once in a trend, require strong opposite to flip
6. Low-volatility filter    — neutral when market is not moving enough

Trend states
------------
+1  Bullish   ALL of: EMA20>EMA50, close>EMA+buffer, slope rising, N closes above
-1  Bearish   ALL of: EMA20<EMA50, close<EMA-buffer, slope falling, N closes below
 0  Neutral   Conditions not fully met (flat, choppy, buffer zone)

Trend lock
----------
Once +1 or -1 is established it persists until the OPPOSITE full condition
is met for `trend_lock_candles` consecutive bars.  Single-bar reversals are
ignored — the bot has memory.

Policy
------
STRICT  (strict=true)   counter-trend → skipped, Telegram failure if enabled
LENIENT (strict=false)  counter-trend → allowed, TP reduced to counter_trend_rr
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HTFResult:
    trend:          int    # +1 / -1 / 0
    with_trend:     bool
    counter_trend:  bool
    htf_close:      float
    fast_ema:       float
    slow_ema:       float
    ema_slope:      float  # positive = rising, negative = falling
    atr_value:      float
    htf_timeframe:  str
    skipped:        bool
    tp_rr_override: float
    reason:         str    # human-readable explanation of trend decision


@dataclass
class _TrendState:
    """Persistent per-symbol state across cycles (trend lock)."""
    locked_trend:      int   = 0   # the currently locked trend
    opposite_count:    int   = 0   # consecutive bars showing opposite signal
    confirmed_bullish: int   = 0   # consecutive closes above EMA (pre-lock counter)
    confirmed_bearish: int   = 0   # consecutive closes below EMA


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _ema_series(closes: np.ndarray, period: int) -> np.ndarray:
    """
    Full EMA series using exponential smoothing seeded with SMA.
    Returns array same length as closes (NaN for first period-1 values).
    """
    n   = len(closes)
    out = np.full(n, np.nan)
    if n < period:
        return out
    k        = 2.0 / (period + 1)
    out[period - 1] = closes[:period].mean()
    for i in range(period, n):
        out[i] = closes[i] * k + out[i - 1] * (1 - k)
    return out


def _atr(highs: np.ndarray, lows: np.ndarray,
         closes: np.ndarray, period: int) -> float:
    """Wilder ATR — returns the last valid value."""
    n  = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i]  - closes[i - 1]),
                    abs(lows[i]   - closes[i - 1]))
    if n < period:
        return float(tr.mean())
    val = float(tr[:period].mean())
    alpha = 1.0 / period
    for i in range(period, n):
        val = val * (1 - alpha) + tr[i] * alpha
    return val


# ---------------------------------------------------------------------------
# Binance HTF data fetch
# ---------------------------------------------------------------------------

def _fetch_htf(symbol: str, htf_tf: str, limit: int,
               base_url: str) -> Optional[Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]]:
    """
    Fetch HTF OHLC from Binance.
    Returns (opens, highs, lows, closes) of CONFIRMED bars (last bar excluded).
    Returns None on failure.
    """
    try:
        r = requests.get(
            f"{base_url}/api/v3/klines",
            params={"symbol": symbol, "interval": htf_tf, "limit": limit + 1},
            timeout=10,
        )
        if r.status_code != 200:
            logger.warning(f"[HTF {symbol}] Binance HTTP {r.status_code}")
            return None
        data = r.json()
        if not data or len(data) < 3:
            return None
        rows   = data[:-1]   # exclude forming candle
        opens  = np.array([float(row[1]) for row in rows])
        highs  = np.array([float(row[2]) for row in rows])
        lows   = np.array([float(row[3]) for row in rows])
        closes = np.array([float(row[4]) for row in rows])
        return opens, highs, lows, closes
    except Exception as e:
        logger.warning(f"[HTF {symbol}] fetch error: {e}")
        return None


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------

class HTFFilter:
    """
    Per-scanner-instance HTF filter.

    reset_cache()  — call at start of each cycle (clears price data cache)
    check()        — evaluate a signal; returns HTFResult
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self._enabled        = cfg.get("htf_enabled", False)
        self._strict         = cfg.get("htf_strict", True)
        self._htf_tf         = cfg.get("htf_timeframe", "1h")
        self._fast_period    = int(cfg.get("htf_fast_ema", 20))
        self._slow_period    = int(cfg.get("htf_slow_ema", 50))
        self._confirm        = int(cfg.get("htf_confirm_candles", 2))
        self._lock_candles   = int(cfg.get("htf_trend_lock_candles", 3))
        self._atr_mult       = float(cfg.get("htf_atr_buffer_mult", 0.1))
        self._min_atr_pct    = float(cfg.get("htf_min_atr_pct", 0.002))  # 0.2% min volatility
        self._ct_rr          = float(cfg.get("htf_counter_trend_rr", 0.5))
        self._base_url       = cfg.get("binance_base_url", "https://api1.binance.com")

        # Bars needed: slow EMA needs most, plus room for confirmation + lock
        self._bars_needed = self._slow_period + self._lock_candles + 10

        # Per-symbol persistent trend state (survives cycle resets)
        self._states: Dict[str, _TrendState] = {}

        # Per-cycle price data cache (reset each cycle)
        self._cache: Dict[str, Optional[HTFResult]] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset_cache(self):
        """Call at the start of each scan cycle to force fresh HTF data."""
        self._cache.clear()

    def check(self, symbol: str, signal_direction: int,
              signal_rr: float) -> HTFResult:
        """Evaluate a signal against the HTF trend."""
        if not self._enabled:
            return self._passthrough(signal_rr)

        if symbol not in self._cache:
            self._cache[symbol] = self._compute(symbol)

        result = self._cache[symbol]
        if result is None:
            # Data unavailable — don't block
            return self._passthrough(signal_rr,
                                     reason="HTF data unavailable — filter bypassed")

        return self._apply_policy(result, signal_direction, signal_rr)

    # -------------------------------------------------------------------------
    # Trend computation
    # -------------------------------------------------------------------------

    def _compute(self, symbol: str) -> Optional[HTFResult]:
        """Fetch HTF data and compute trend for symbol."""
        ohlc = _fetch_htf(symbol, self._htf_tf, self._bars_needed, self._base_url)
        if ohlc is None:
            return None

        opens, highs, lows, closes = ohlc
        n = len(closes)
        if n < self._slow_period + 2:
            logger.warning(f"[HTF {symbol}] too few bars ({n}), need {self._slow_period+2}")
            return None

        # ── Indicators ────────────────────────────────────────────────────────
        fast_ema = _ema_series(closes, self._fast_period)
        slow_ema = _ema_series(closes, self._slow_period)
        atr_val  = _atr(highs, lows, closes, 14)

        last_fast  = float(fast_ema[-1])
        last_slow  = float(slow_ema[-1])
        prev_fast  = float(fast_ema[-2])
        last_close = float(closes[-1])
        buffer     = atr_val * self._atr_mult

        # EMA slope: positive = rising, negative = falling
        slope = last_fast - prev_fast

        # Low-volatility guard: if ATR < min_atr_pct of price, market is flat
        atr_pct = atr_val / last_close if last_close > 0 else 0
        low_vol = atr_pct < self._min_atr_pct

        # ── Per-symbol trend lock state ────────────────────────────────────────
        state = self._states.setdefault(symbol, _TrendState())

        # Count consecutive closes above / below fast EMA
        above_ema = last_close > (last_fast + buffer)
        below_ema = last_close < (last_fast - buffer)

        if above_ema:
            state.confirmed_bullish += 1
            state.confirmed_bearish  = 0
        elif below_ema:
            state.confirmed_bearish += 1
            state.confirmed_bullish  = 0
        else:
            # Inside buffer zone — neither counter accumulates
            state.confirmed_bullish = max(0, state.confirmed_bullish - 1)
            state.confirmed_bearish = max(0, state.confirmed_bearish - 1)

        # ── Raw trend signal (all conditions required) ─────────────────────────
        bull_conditions = (
            last_fast > last_slow          # dual EMA structure
            and above_ema                  # close above EMA + buffer
            and slope > 0                  # EMA rising
            and state.confirmed_bullish >= self._confirm  # N consecutive
            and not low_vol                # market is moving
        )
        bear_conditions = (
            last_fast < last_slow
            and below_ema
            and slope < 0
            and state.confirmed_bearish >= self._confirm
            and not low_vol
        )

        raw_trend = +1 if bull_conditions else -1 if bear_conditions else 0

        # ── Trend lock ─────────────────────────────────────────────────────────
        if state.locked_trend == 0:
            # No established trend — adopt raw signal immediately
            if raw_trend != 0:
                state.locked_trend   = raw_trend
                state.opposite_count = 0
        elif raw_trend == -state.locked_trend:
            # Opposite signal building — increment counter
            state.opposite_count += 1
            if state.opposite_count >= self._lock_candles:
                # Strong enough flip — adopt the new trend
                state.locked_trend   = raw_trend
                state.opposite_count = 0
            # else: keep existing locked trend (ignore weak reversal)
        else:
            # Same direction or neutral — reset opposite counter
            state.opposite_count = 0
            if raw_trend != 0:
                state.locked_trend = raw_trend

        trend = state.locked_trend

        # ── Build explanation ──────────────────────────────────────────────────
        trend_str = "BULLISH" if trend > 0 else "BEARISH" if trend < 0 else "NEUTRAL"
        reason_parts = [
            f"EMA{self._fast_period}={last_fast:.5f}",
            f"EMA{self._slow_period}={last_slow:.5f}",
            f"slope={'rising' if slope > 0 else 'falling' if slope < 0 else 'flat'}",
            f"buf={buffer:.5f}",
            f"consec_bull={state.confirmed_bullish}",
            f"consec_bear={state.confirmed_bearish}",
            f"lock={state.locked_trend}",
            f"low_vol={low_vol}",
        ]
        reason = f"{self._htf_tf.upper()} {trend_str} — {' | '.join(reason_parts)}"
        logger.debug(f"[HTF {symbol}] {reason}")

        return HTFResult(
            trend=trend,
            with_trend=False,       # filled by _apply_policy
            counter_trend=False,
            htf_close=last_close,
            fast_ema=last_fast,
            slow_ema=last_slow,
            ema_slope=slope,
            atr_value=atr_val,
            htf_timeframe=self._htf_tf,
            skipped=False,
            tp_rr_override=signal_rr,
            reason=reason,
        )

    # -------------------------------------------------------------------------
    # Policy application
    # -------------------------------------------------------------------------

    def _apply_policy(self, base: HTFResult,
                      signal_direction: int,
                      signal_rr: float) -> HTFResult:
        """Determine with_trend / counter_trend and apply STRICT / LENIENT."""
        trend = base.trend

        if trend == 0:
            # Neutral / flat market — pass through unchanged
            return self._clone(base, with_trend=True, counter_trend=False,
                               skipped=False, tp_rr_override=signal_rr)

        with_trend    = (signal_direction == trend)
        counter_trend = not with_trend

        if with_trend:
            return self._clone(base, with_trend=True, counter_trend=False,
                               skipped=False, tp_rr_override=signal_rr)

        # Counter-trend signal
        if self._strict:
            return self._clone(base, with_trend=False, counter_trend=True,
                               skipped=True, tp_rr_override=signal_rr)
        else:
            return self._clone(base, with_trend=False, counter_trend=True,
                               skipped=False, tp_rr_override=self._ct_rr)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _passthrough(self, signal_rr: float, reason: str = "HTF filter disabled") -> HTFResult:
        return HTFResult(
            trend=0, with_trend=True, counter_trend=False,
            htf_close=0.0, fast_ema=0.0, slow_ema=0.0,
            ema_slope=0.0, atr_value=0.0,
            htf_timeframe=self._htf_tf,
            skipped=False, tp_rr_override=signal_rr,
            reason=reason,
        )

    @staticmethod
    def _clone(base: HTFResult, **overrides) -> HTFResult:
        return HTFResult(
            trend          = overrides.get("trend",          base.trend),
            with_trend     = overrides.get("with_trend",     base.with_trend),
            counter_trend  = overrides.get("counter_trend",  base.counter_trend),
            htf_close      = base.htf_close,
            fast_ema       = base.fast_ema,
            slow_ema       = base.slow_ema,
            ema_slope      = base.ema_slope,
            atr_value      = base.atr_value,
            htf_timeframe  = base.htf_timeframe,
            skipped        = overrides.get("skipped",        base.skipped),
            tp_rr_override = overrides.get("tp_rr_override", base.tp_rr_override),
            reason         = base.reason,
        )
