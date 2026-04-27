# -*- coding: utf-8 -*-
"""
HTF Trend Filter — Ichimoku Cloud matching Pine Script ATD exactly.

Data source fix
---------------
The original version fetched Binance spot data for the HTF calculation.
TradingView (and Bybit traders) see Bybit PERPETUAL prices, which can differ
from Binance spot by 0.1-2% on altcoins — enough to flip the cloud comparison.

This version fetches HTF klines from BYBIT's public API (same data as TradingView
BYBIT:SYMBOL.P charts), with Binance as fallback if Bybit fetch fails.

Pine Script source (ADVANCED TREND DETECTOR section)
-----------------------------------------------------
    tenkan_expr  = math.avg(ta.highest(9),  ta.lowest(9))
    kijun_expr   = math.avg(ta.highest(26), ta.lowest(26))
    senkouA_expr = math.avg(tenkan_expr, kijun_expr)[26]
    senkouB_expr = math.avg(ta.highest(52), ta.lowest(52))[26]
    close_expr   = close

    [senkouA, senkouB, close_val] = request.security(ticker, tf, [...])
    cloud_top    = math.max(senkouA, senkouB)
    cloud_bottom = math.min(senkouA, senkouB)
    trend = close_val > cloud_top    ? "Uptrend"
          : close_val < cloud_bottom ? "Downtrend"
                                     : "Ranging"

Python translation
------------------
request.security with barmerge.lookahead_off returns values from the last
CONFIRMED HTF bar. The [26] displacement means 26 HTF bars ago.

At last confirmed bar T (index n-1, 0-based):
  tenkan(T-26)  = avg(max(highs[n-35 : n-26]), min(lows[n-35 : n-26]))
  kijun(T-26)   = avg(max(highs[n-52 : n-26]), min(lows[n-52 : n-26]))
  senkouA       = avg(tenkan(T-26), kijun(T-26))
  senkouB(T-26) = avg(max(highs[n-78 : n-26]), min(lows[n-78 : n-26]))
  compare: closes[n-1] vs max/min(senkouA, senkouB)

Minimum bars needed: 52 + 26 + 1 = 79. We request 90 for safety.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Bybit interval mapping from our timeframe format strings
_BYBIT_INTERVAL = {
    '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
    '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
    '1d': 'D', '1w': 'W',
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HTFResult:
    trend:          int      # +1 Uptrend / -1 Downtrend / 0 Ranging
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
    skip_reason:    str
    tp_rr_override: float


# ---------------------------------------------------------------------------
# Ichimoku computation — exact Pine Script match
# ---------------------------------------------------------------------------

def _ichimoku_at_last_bar(highs: np.ndarray, lows: np.ndarray,
                           tenkan_p: int, kijun_p: int,
                           senkou_b_p: int, displacement: int
                           ) -> Tuple[float, float, float, float]:
    """
    Compute Ichimoku cloud values at the last confirmed bar.

    Pine Script mapping (T = last confirmed bar = index n-1, D = displacement = 26):
      tenkan(T-D)  = avg(max(highs[T-D .. T-D-tenkan_p+1]),
                         min(lows [T-D .. T-D-tenkan_p+1]))
                   = avg(max(highs[n-D-tenkan_p : n-D]),
                         min(lows [n-D-tenkan_p : n-D]))

      kijun(T-D)   = avg(max(highs[n-D-kijun_p : n-D]),
                         min(lows [n-D-kijun_p : n-D]))

      senkouA      = avg(tenkan(T-D), kijun(T-D))

      senkouB(T-D) = avg(max(highs[n-D-senkou_b_p : n-D]),
                         min(lows [n-D-senkou_b_p : n-D]))

    All slices end at index n-D (exclusive), which is bar T-D+1 in 0-based,
    so the last bar INCLUDED is n-D-1 = T-D. This matches Pine Script's
    ta.highest(N)[D] = max of bars [T-D, T-D-1, ..., T-D-N+1].
    """
    n = len(highs)
    D = displacement

    def hi(start, end): return float(np.max(highs[start:end]))
    def lo(start, end): return float(np.min(lows [start:end]))

    # tenkan at T-D
    t_hi = hi(n - D - tenkan_p,   n - D)
    t_lo = lo(n - D - tenkan_p,   n - D)
    tenkan = (t_hi + t_lo) / 2.0

    # kijun at T-D
    k_hi = hi(n - D - kijun_p,    n - D)
    k_lo = lo(n - D - kijun_p,    n - D)
    kijun = (k_hi + k_lo) / 2.0

    senkou_a = (tenkan + kijun) / 2.0

    # senkouB at T-D
    sb_hi = hi(n - D - senkou_b_p, n - D)
    sb_lo = lo(n - D - senkou_b_p, n - D)
    senkou_b = (sb_hi + sb_lo) / 2.0

    return tenkan, kijun, senkou_a, senkou_b


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _fetch_bybit(symbol: str, htf_tf: str, limit: int,
                 bybit_base: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fetch HTF klines from Bybit public API (same data as TradingView BYBIT charts).
    Bybit returns data NEWEST FIRST — we reverse to oldest-first.
    Excludes the forming (most recent) bar.
    """
    interval = _BYBIT_INTERVAL.get(htf_tf)
    if not interval:
        logger.warning(f"[HTF] Unknown timeframe '{htf_tf}' for Bybit interval mapping")
        return None
    try:
        r = requests.get(
            f"{bybit_base}/v5/market/kline",
            params={"category": "linear", "symbol": symbol,
                    "interval": interval, "limit": limit + 1},
            timeout=10,
        )
        if r.status_code != 200:
            logger.debug(f"[HTF {symbol}] Bybit HTTP {r.status_code}")
            return None
        data = r.json()
        rows = data.get("result", {}).get("list", [])
        if not rows or len(rows) < 3:
            return None
        # Bybit: newest first → reverse to oldest first, then exclude last (forming)
        rows = list(reversed(rows))[:-1]
        highs  = np.array([float(row[2]) for row in rows])
        lows   = np.array([float(row[3]) for row in rows])
        closes = np.array([float(row[4]) for row in rows])
        return highs, lows, closes
    except Exception as e:
        logger.debug(f"[HTF {symbol}] Bybit fetch error: {e}")
        return None


def _fetch_binance(symbol: str, htf_tf: str, limit: int,
                   binance_base: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fallback: fetch HTF klines from Binance."""
    try:
        r = requests.get(
            f"{binance_base}/api/v3/klines",
            params={"symbol": symbol, "interval": htf_tf, "limit": limit + 1},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or len(data) < 3:
            return None
        rows   = data[:-1]   # exclude forming candle (Binance: oldest first)
        highs  = np.array([float(row[2]) for row in rows])
        lows   = np.array([float(row[3]) for row in rows])
        closes = np.array([float(row[4]) for row in rows])
        return highs, lows, closes
    except Exception as e:
        logger.debug(f"[HTF {symbol}] Binance fallback error: {e}")
        return None


def _fetch_htf(symbol: str, htf_tf: str, limit: int,
               bybit_base: str, binance_base: str
               ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fetch HTF data from Bybit first (matches TradingView BYBIT charts),
    fall back to Binance if Bybit fails.
    """
    result = _fetch_bybit(symbol, htf_tf, limit, bybit_base)
    if result is not None:
        return result
    logger.debug(f"[HTF {symbol}] Bybit fetch failed — trying Binance fallback")
    return _fetch_binance(symbol, htf_tf, limit, binance_base)


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------

class HTFFilter:
    """
    Ichimoku-based HTF trend filter.
    Data fetched from Bybit (matches TradingView charts).

    reset_cache() — call at start of each scan cycle.
    check()       — evaluate a signal, returns HTFResult.
    """

    def __init__(self, cfg: dict):
        self._enabled      = cfg.get("htf_enabled",          False)
        self._strict       = cfg.get("htf_strict",           True)
        self._htf_tf       = cfg.get("htf_timeframe",        "1h")
        self._tenkan       = int(cfg.get("htf_tenkan",        9))
        self._kijun        = int(cfg.get("htf_kijun",         26))
        self._senkou_b     = int(cfg.get("htf_senkou_b",      52))
        self._displacement = int(cfg.get("htf_displacement",  26))
        self._skip_ranging = cfg.get("htf_skip_ranging",      True)
        self._ct_rr        = float(cfg.get("htf_counter_trend_rr", 0.5))
        self._bybit_base   = cfg.get("bybit_base_url",        "https://api.bybit.com")
        self._binance_base = cfg.get("binance_base_url",      "https://api1.binance.com")

        # bars: senkouB(52) + displacement(26) + kijun(26) + buffer = 110
        self._bars_needed  = self._senkou_b + self._displacement + self._kijun + 20

        self._cache: Dict[str, HTFResult] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset_cache(self):
        self._cache.clear()

    def check(self, symbol: str, signal_direction: int,
              signal_rr: float) -> HTFResult:
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
        ohlc = _fetch_htf(symbol, self._htf_tf, self._bars_needed,
                          self._bybit_base, self._binance_base)
        if ohlc is None:
            return None

        highs, lows, closes = ohlc
        n = len(closes)
        min_needed = self._senkou_b + self._displacement + self._kijun
        if n < min_needed:
            logger.warning(f"[HTF {symbol}] only {n} bars, need {min_needed}")
            return None

        tenkan, kijun, senkou_a, senkou_b = _ichimoku_at_last_bar(
            highs, lows,
            self._tenkan, self._kijun,
            self._senkou_b, self._displacement,
        )

        last_close   = float(closes[-1])
        cloud_top    = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # 1:1 Pine Script ATD logic
        if   last_close > cloud_top:    trend, label = +1, "Uptrend"
        elif last_close < cloud_bottom: trend, label = -1, "Downtrend"
        else:                           trend, label =  0, "Ranging"

        logger.debug(
            f"[HTF {symbol}] {self._htf_tf.upper()} "
            f"close={last_close:.5f}  "
            f"cloud_top={cloud_top:.5f}  "
            f"cloud_bottom={cloud_bottom:.5f}  "
            f"-> {label}"
        )

        return HTFResult(
            trend=trend, trend_label=label,
            with_trend=False, counter_trend=False, ranging=(trend == 0),
            htf_close=last_close,
            cloud_top=cloud_top, cloud_bottom=cloud_bottom,
            senkou_a=senkou_a, senkou_b=senkou_b,
            htf_timeframe=self._htf_tf,
            skipped=False, skip_reason="",
            tp_rr_override=0.0,   # filled by _apply_policy with actual signal_rr
        )

    # -------------------------------------------------------------------------
    # Policy
    # -------------------------------------------------------------------------

    def _apply_policy(self, base: HTFResult,
                      signal_direction: int,
                      signal_rr: float) -> HTFResult:
        trend = base.trend

        if trend == 0:
            if self._skip_ranging:
                return self._clone(base,
                    skipped=True,
                    skip_reason=(
                        f"Ranging — price inside Ichimoku cloud "
                        f"(top={base.cloud_top:.5f} "
                        f"bottom={base.cloud_bottom:.5f})"
                    ),
                    tp_rr_override=signal_rr)
            return self._clone(base,
                with_trend=True, counter_trend=False,
                skipped=False, skip_reason="",
                tp_rr_override=signal_rr)

        with_trend = (signal_direction == trend)

        if with_trend:
            return self._clone(base,
                with_trend=True, counter_trend=False,
                skipped=False, skip_reason="",
                tp_rr_override=signal_rr)

        d_str = "LONG" if signal_direction > 0 else "SHORT"
        if self._strict:
            reason = (
                f"Counter-trend: signal is {d_str} but "
                f"HTF {self._htf_tf.upper()} is {base.trend_label} "
                f"(close={base.htf_close:.5f}  "
                f"cloud_top={base.cloud_top:.5f}  "
                f"cloud_bottom={base.cloud_bottom:.5f})"
            )
            return self._clone(base,
                with_trend=False, counter_trend=True,
                skipped=True, skip_reason=reason,
                tp_rr_override=signal_rr)
        else:
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
