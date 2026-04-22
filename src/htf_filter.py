# -*- coding: utf-8 -*-
"""
HTF Trend Filter — Ichimoku Cloud implementation.

Trend detection matches 1:1 the "ADVANCED TREND DETECTOR" section of the
Pine Script (Entry Model - V2 - Simplified | Jbadonai).

Pine Script source (key section)
---------------------------------
    tenkan_expr  = math.avg(ta.highest(9),  ta.lowest(9))
    kijun_expr   = math.avg(ta.highest(26), ta.lowest(26))
    senkouA_expr = math.avg(tenkan_expr, kijun_expr)[26]
    senkouB_expr = math.avg(ta.highest(52), ta.lowest(52))[26]

    cloud_top    = math.max(senkouA, senkouB)
    cloud_bottom = math.min(senkouA, senkouB)

    trend = close > cloud_top    ? "Uptrend"
          : close < cloud_bottom ? "Downtrend"
          :                        "Ranging"

Python translation
------------------
At the last confirmed bar (index n-1), the cloud values are derived from
data 26 bars ago (the [26] displacement in Pine Script):

    senkou_a[n-1] = (tenkan[n-27] + kijun[n-27]) / 2
    senkou_b[n-1] = (highest_52[n-27] + lowest_52[n-27]) / 2

Minimum bars needed: senkou_b_period(52) + displacement(26) = 78.

Ranging market
--------------
When close is inside the cloud (Ranging), the trend is 0.
By default skip_ranging = true, which treats 0 as a SKIP signal
(same effect as STRICT mode for counter-trend, but reason says "Ranging").

Policy
------
STRICT  (strict=true)   counter-trend  -> skipped
LENIENT (strict=false)  counter-trend  -> allowed, TP = counter_trend_rr
BOTH modes: Ranging -> skipped when skip_ranging=true,
                    -> pass-through (neutral) when skip_ranging=false
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HTFResult:
    trend:          int      # +1 Uptrend / -1 Downtrend / 0 Ranging/neutral
    trend_label:    str      # "Uptrend" / "Downtrend" / "Ranging" / "Disabled"
    with_trend:     bool
    counter_trend:  bool
    ranging:        bool
    htf_close:      float
    cloud_top:      float
    cloud_bottom:   float
    senkou_a:       float
    senkou_b:       float
    htf_timeframe:  str
    skipped:        bool
    skip_reason:    str      # why it was skipped (empty if not skipped)
    tp_rr_override: float


# ---------------------------------------------------------------------------
# Ichimoku helpers — exact Pine Script equivalents
# ---------------------------------------------------------------------------

def _ichimoku(highs: np.ndarray, lows: np.ndarray,
              tenkan_period: int, kijun_period: int,
              senkou_b_period: int, displacement: int
              ) -> Tuple[float, float, float, float]:
    """
    Compute Ichimoku cloud values for the LAST confirmed bar.

    Pine Script equivalents
    -----------------------
    tenkan_expr  = math.avg(ta.highest(tenkan_period), ta.lowest(tenkan_period))
    kijun_expr   = math.avg(ta.highest(kijun_period),  ta.lowest(kijun_period))
    senkouA_expr = math.avg(tenkan_expr, kijun_expr)[displacement]
    senkouB_expr = math.avg(ta.highest(senkou_b_period),
                            ta.lowest(senkou_b_period))[displacement]

    The [displacement] in Pine means "look back displacement bars", so:
      senkouA at bar i = avg(tenkan[i-d], kijun[i-d])
      senkouB at bar i = avg(highest_sb[i-d], lowest_sb[i-d])

    Returns (tenkan, kijun, senkou_a, senkou_b) for the last bar.
    """
    n = len(highs)
    d = displacement

    # Current tenkan / kijun (for display info only — not used in cloud)
    t_hi = np.max(highs[n - tenkan_period:])
    t_lo = np.min(lows[n  - tenkan_period:])
    tenkan = (t_hi + t_lo) / 2.0

    k_hi = np.max(highs[n - kijun_period:])
    k_lo = np.min(lows[n  - kijun_period:])
    kijun = (k_hi + k_lo) / 2.0

    # Cloud values derived from d bars ago
    # tenkan d bars ago
    t_hi_d = np.max(highs[n - d - tenkan_period: n - d])
    t_lo_d = np.min(lows[n  - d - tenkan_period: n - d])
    tenkan_d = (t_hi_d + t_lo_d) / 2.0

    # kijun d bars ago
    k_hi_d = np.max(highs[n - d - kijun_period: n - d])
    k_lo_d = np.min(lows[n  - d - kijun_period: n - d])
    kijun_d = (k_hi_d + k_lo_d) / 2.0

    senkou_a = (tenkan_d + kijun_d) / 2.0

    # senkou B: highest/lowest of senkou_b_period, d bars ago
    sb_hi = np.max(highs[n - d - senkou_b_period: n - d])
    sb_lo = np.min(lows[n  - d - senkou_b_period: n - d])
    senkou_b = (sb_hi + sb_lo) / 2.0

    return tenkan, kijun, senkou_a, senkou_b


# ---------------------------------------------------------------------------
# Binance HTF data fetch
# ---------------------------------------------------------------------------

def _fetch_htf(symbol: str, htf_tf: str, limit: int,
               base_url: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fetch HTF OHLC from Binance.
    Returns (highs, lows, closes) of CONFIRMED bars (last forming bar excluded).
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
        highs  = np.array([float(row[2]) for row in rows])
        lows   = np.array([float(row[3]) for row in rows])
        closes = np.array([float(row[4]) for row in rows])
        return highs, lows, closes
    except Exception as e:
        logger.warning(f"[HTF {symbol}] fetch error: {e}")
        return None


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------

class HTFFilter:
    """
    Per-scanner-instance Ichimoku-based HTF filter.

    reset_cache() — call at start of each cycle
    check()       — evaluate a signal, returns HTFResult
    """

    def __init__(self, cfg: dict):
        self._enabled          = cfg.get("htf_enabled",          False)
        self._strict           = cfg.get("htf_strict",           True)
        self._htf_tf           = cfg.get("htf_timeframe",        "1h")
        self._tenkan           = int(cfg.get("htf_tenkan",        9))
        self._kijun            = int(cfg.get("htf_kijun",         26))
        self._senkou_b         = int(cfg.get("htf_senkou_b",      52))
        self._displacement     = int(cfg.get("htf_displacement",  26))
        self._skip_ranging     = cfg.get("htf_skip_ranging",      True)
        self._ct_rr            = float(cfg.get("htf_counter_trend_rr", 0.5))
        self._base_url         = cfg.get("binance_base_url",
                                         "https://api1.binance.com")

        # Bars needed: senkou_b_period + displacement + safety buffer
        self._bars_needed = self._senkou_b + self._displacement + 10

        # Per-symbol cycle cache (cleared each cycle)
        self._cache: Dict[str, HTFResult] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset_cache(self):
        """Clear per-cycle data cache — call at the start of each scan cycle."""
        self._cache.clear()

    def check(self, symbol: str, signal_direction: int,
              signal_rr: float) -> HTFResult:
        """
        Evaluate a signal against the HTF Ichimoku trend.
        Returns an HTFResult — never raises.
        """
        if not self._enabled:
            return self._passthrough(signal_rr)

        if symbol not in self._cache:
            self._cache[symbol] = self._compute(symbol)

        result = self._cache[symbol]
        if result is None:
            return self._passthrough(signal_rr,
                                     reason="HTF data unavailable — filter bypassed")

        return self._apply_policy(result, signal_direction, signal_rr)

    # -------------------------------------------------------------------------
    # Ichimoku computation
    # -------------------------------------------------------------------------

    def _compute(self, symbol: str) -> Optional[HTFResult]:
        """Fetch HTF bars and compute Ichimoku cloud trend."""
        ohlc = _fetch_htf(symbol, self._htf_tf,
                          self._bars_needed, self._base_url)
        if ohlc is None:
            return None

        highs, lows, closes = ohlc
        n = len(closes)
        min_needed = self._senkou_b + self._displacement + self._kijun
        if n < min_needed:
            logger.warning(f"[HTF {symbol}] only {n} bars, need {min_needed}")
            return None

        # Compute Ichimoku values
        tenkan, kijun, senkou_a, senkou_b = _ichimoku(
            highs, lows,
            self._tenkan, self._kijun,
            self._senkou_b, self._displacement,
        )

        last_close  = float(closes[-1])
        cloud_top    = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Pine Script 1:1 trend logic
        if last_close > cloud_top:
            trend = +1
            label = "Uptrend"
        elif last_close < cloud_bottom:
            trend = -1
            label = "Downtrend"
        else:
            trend = 0
            label = "Ranging"

        logger.debug(
            f"[HTF {symbol}] {self._htf_tf.upper()} "
            f"close={last_close:.5f}  "
            f"cloud_top={cloud_top:.5f}  "
            f"cloud_bottom={cloud_bottom:.5f}  "
            f"senkou_a={senkou_a:.5f}  "
            f"senkou_b={senkou_b:.5f}  "
            f"trend={label}"
        )

        return HTFResult(
            trend=trend, trend_label=label,
            with_trend=False, counter_trend=False, ranging=(trend == 0),
            htf_close=last_close,
            cloud_top=cloud_top, cloud_bottom=cloud_bottom,
            senkou_a=senkou_a, senkou_b=senkou_b,
            htf_timeframe=self._htf_tf,
            skipped=False, skip_reason="",
            tp_rr_override=signal_rr,
        )

    # -------------------------------------------------------------------------
    # Policy
    # -------------------------------------------------------------------------

    def _apply_policy(self, base: HTFResult,
                      signal_direction: int,
                      signal_rr: float) -> HTFResult:
        """Apply STRICT / LENIENT / ranging policy to a computed HTFResult."""
        trend = base.trend

        # Ranging market
        if trend == 0:
            if self._skip_ranging:
                return self._clone(base,
                                   skipped=True,
                                   skip_reason=(
                                       f"Ranging market — price inside "
                                       f"Ichimoku cloud "
                                       f"(top={base.cloud_top:.5f}  "
                                       f"bottom={base.cloud_bottom:.5f})"
                                   ),
                                   tp_rr_override=signal_rr)
            else:
                # Treat ranging as neutral — pass through unchanged
                return self._clone(base,
                                   with_trend=True, counter_trend=False,
                                   skipped=False, skip_reason="",
                                   tp_rr_override=signal_rr)

        # Trending market
        with_trend    = (signal_direction == trend)
        counter_trend = not with_trend
        d_str         = "LONG" if signal_direction > 0 else "SHORT"
        trend_str     = base.trend_label

        if with_trend:
            return self._clone(base,
                               with_trend=True, counter_trend=False,
                               skipped=False, skip_reason="",
                               tp_rr_override=signal_rr)

        # Counter-trend
        if self._strict:
            reason = (
                f"Counter-trend: signal is {d_str} but "
                f"HTF {self._htf_tf.upper()} is {trend_str} "
                f"(close={base.htf_close:.5f}  "
                f"cloud_top={base.cloud_top:.5f}  "
                f"cloud_bottom={base.cloud_bottom:.5f})"
            )
            return self._clone(base,
                               with_trend=False, counter_trend=True,
                               skipped=True, skip_reason=reason,
                               tp_rr_override=signal_rr)
        else:
            # LENIENT: allow with reduced TP
            return self._clone(base,
                               with_trend=False, counter_trend=True,
                               skipped=False, skip_reason="",
                               tp_rr_override=self._ct_rr)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _passthrough(self, signal_rr: float,
                     reason: str = "HTF filter disabled") -> HTFResult:
        return HTFResult(
            trend=0, trend_label="Disabled",
            with_trend=True, counter_trend=False, ranging=False,
            htf_close=0.0, cloud_top=0.0, cloud_bottom=0.0,
            senkou_a=0.0, senkou_b=0.0,
            htf_timeframe=self._htf_tf,
            skipped=False, skip_reason=reason,
            tp_rr_override=signal_rr,
        )

    @staticmethod
    def _clone(base: HTFResult, **kw) -> HTFResult:
        return HTFResult(
            trend          = kw.get("trend",          base.trend),
            trend_label    = kw.get("trend_label",    base.trend_label),
            with_trend     = kw.get("with_trend",     base.with_trend),
            counter_trend  = kw.get("counter_trend",  base.counter_trend),
            ranging        = kw.get("ranging",        base.ranging),
            htf_close      = base.htf_close,
            cloud_top      = base.cloud_top,
            cloud_bottom   = base.cloud_bottom,
            senkou_a       = base.senkou_a,
            senkou_b       = base.senkou_b,
            htf_timeframe  = base.htf_timeframe,
            skipped        = kw.get("skipped",        base.skipped),
            skip_reason    = kw.get("skip_reason",    base.skip_reason),
            tp_rr_override = kw.get("tp_rr_override", base.tp_rr_override),
        )
