# -*- coding: utf-8 -*-
"""
1:1 port of POI_Confluence_Zones_EA.mq5 detection logic.

Index convention (throughout this file)
---------------------------------------
  df / numpy arrays are indexed  0 = OLDEST  ->  N-1 = NEWEST.
  'ci' is the confirmed bar being processed (equiv. to MQL5 baseShift=1 on
  the live bar, or a higher shift during the historical replay).

  MQL5  iHigh(_Symbol, _, baseShift + k)  ==  Python  highs[ci - k]
  MQL5  GetHighest(baseShift, n)           ==  Python  highs[ci-n+1 : ci+1].max()
"""

import numpy as np
from typing import List, Optional, Tuple

import pandas as pd

from .models   import ZoneData, OBState, SignalInfo
from .obos     import last_signal as _obos_last_signal
from .indicators import calc_wilder_atr, compute_pivot_highs, compute_pivot_lows


# -- Bitmask helpers ----------------------------------------------------------

def has_ob(src: int) -> bool: return bool(src & 1)
def has_oc(src: int) -> bool: return bool(src & 2)
def has_ls(src: int) -> bool: return bool(src & 4)


def src_or(a: int, b: int) -> int:
    """Merge two source bitmasks (AddBit equivalent)."""
    r = a
    if has_ob(b) and not has_ob(r): r |= 1
    if has_oc(b) and not has_oc(r): r |= 2
    if has_ls(b) and not has_ls(r): r |= 4
    return r


def src_text(src: int) -> str:
    parts = []
    if has_ob(src): parts.append("OB")
    if has_oc(src): parts.append("Odd")
    if has_ls(src): parts.append("LS")
    return "+".join(parts) if parts else "?"


# -- Zone list ----------------------------------------------------------------

class ZoneList:
    MAX_ZONES = 500

    def __init__(self):
        self._zones: List[ZoneData] = []
        self._id_ctr = 0

    def reset(self):
        self._zones.clear()
        self._id_ctr = 0

    @property
    def zones(self) -> List[ZoneData]:
        return self._zones

    def _new_id(self) -> int:
        self._id_ctr += 1
        return self._id_ctr

    def _try_merge(self, top: float, bot: float, direction: int,
                   src: int, tol: float) -> bool:
        """Return True and merge if an overlapping same-direction zone exists."""
        for z in self._zones:
            if z.direction != direction:
                continue
            # Price ranges intersect within tolerance
            if top >= z.bottom - tol and bot <= z.top + tol:
                z.top    = max(z.top, top)
                z.bottom = min(z.bottom, bot)
                z.src    = src_or(z.src, src)
                return True
        return False

    def add(self, top: float, bot: float, direction: int,
            src: int, tol: float, created_bar: int) -> Optional[ZoneData]:
        """Add zone or merge into existing. Returns new ZoneData or None if merged."""
        if top <= bot or len(self._zones) >= self.MAX_ZONES:
            return None
        if self._try_merge(top, bot, direction, src, tol):
            return None
        z = ZoneData(top=top, bottom=bot, direction=direction,
                     src=src, hit=False,
                     created_bar=created_bar, zone_id=self._new_id())
        self._zones.append(z)
        return z

    def trim_ob(self, direction: int, max_blocks: int):
        """Drop oldest OB zones of given direction when count exceeds max_blocks."""
        while True:
            ob_idxs = [i for i, z in enumerate(self._zones)
                       if has_ob(z.src) and z.direction == direction]
            if len(ob_idxs) < max_blocks:
                break
            self._zones.pop(ob_idxs[0])   # oldest first (lowest index)

    def check_mitigation(self, ci: int, closes: np.ndarray,
                         highs: np.ndarray, lows: np.ndarray, mode: str):
        """Remove zones whose mitigation condition is met at bar ci."""
        c, h, l = closes[ci], highs[ci], lows[ci]
        to_del = []

        for i, z in enumerate(self._zones):
            top, bot, d = z.top, z.bottom, z.direction
            remove = False

            if mode == 'CLOSE_BEYOND':
                if d > 0 and c < bot:  remove = True
                if d < 0 and c > top:  remove = True

            elif mode == 'WICK_BEYOND':
                if d > 0 and l < bot:  remove = True
                if d < 0 and h > top:  remove = True

            else:   # BOTH_M
                # First touch marks the zone as 'hit'
                if not z.hit:
                    if d > 0 and l <= top and h >= bot: z.hit = True
                    if d < 0 and h >= bot and l <= top: z.hit = True
                # Close beyond removes
                if d > 0 and c < bot:  remove = True
                if d < 0 and c > top:  remove = True
                # After hit, closing back inside removes
                if z.hit and d > 0 and c > top: remove = True
                if z.hit and d < 0 and c < bot: remove = True

            if remove:
                to_del.append(i)

        for i in reversed(to_del):
            self._zones.pop(i)


# -- Detection functions ------------------------------------------------------

def _find_ob_box(opens: np.ndarray, closes: np.ndarray,
                 ci: int, lookback: int, bull: bool) -> Tuple[Optional[float], Optional[float]]:
    """
    Exact port of the MQL5 OB-box search loop.

    Searches bars ci-1 down to ci-(lookback-1) for:
      bull=True  -> bar with lowest  min(O,C)  -> that candle is the OB
      bull=False -> bar with highest max(O,C)  -> that candle is the OB

    MQL5 tie-breaking: initialise with ci-1, update only on strict improvement
    -> keeps the newest bar when values are equal.
    """
    if lookback < 1 or ci < 1:
        return None, None

    o0, c0 = opens[ci - 1], closes[ci - 1]

    if bull:
        box_btm = min(o0, c0)
        box_top = max(o0, c0)
        for k in range(1, lookback):          # k=1 re-checks ci-1 (same as init, no-op)
            idx = ci - k
            if idx < 0: break
            cand = min(opens[idx], closes[idx])
            if cand < box_btm:
                box_btm = cand
                box_top = max(opens[idx], closes[idx])
    else:
        box_top = max(o0, c0)
        box_btm = min(o0, c0)
        for k in range(1, lookback):
            idx = ci - k
            if idx < 0: break
            cand = max(opens[idx], closes[idx])
            if cand > box_top:
                box_top = cand
                box_btm = min(opens[idx], closes[idx])

    return box_top, box_btm


def _detect_ob(zl: ZoneList, ob: OBState,
               opens: np.ndarray, closes: np.ndarray,
               highs: np.ndarray, lows: np.ndarray,
               ci: int, a10: float, a14: float, cfg: dict):
    """
    Port of ProcessOBDetection().

    Swing detection:
      GetHighest(baseShift, swingLen) = highs[ci-swingLen+1 : ci+1].max()
      iHigh(baseShift + swingLen)     = highs[ci - swingLen]

    Type 0 = swing top (pivot high), type 1 = swing bottom (pivot low).
    On a transition -> record the swing bar.
    When close breaks the swing level -> find the OB candle and create a zone.
    """
    sw = cfg['ob_swing_len']
    if ci < sw:
        return

    swing_bar = ci - sw
    start     = max(0, ci - sw + 1)

    upper = highs[start : ci + 1].max()
    lower = lows [start : ci + 1].min()
    bar_h = highs[swing_bar]
    bar_l = lows [swing_bar]

    prev_type = ob.swing_type
    if   bar_h > upper: new_type = 0
    elif bar_l < lower: new_type = 1
    else:               new_type = prev_type

    ob.swing_type = new_type
    tol = a14 * cfg['overlap_tol']

    # Record new swing top
    if new_type == 0 and prev_type != 0:
        ob.top_bar     = swing_bar
        ob.top_y       = bar_h
        ob.top_crossed = False

    # Record new swing bottom
    if new_type == 1 and prev_type != 1:
        ob.btm_bar     = swing_bar
        ob.btm_y       = bar_l
        ob.btm_crossed = False

    c = closes[ci]

    # -- Bullish OB: close breaks above swing high ---------------------------
    if ob.top_y > 0 and c > ob.top_y and not ob.top_crossed:
        ob.top_crossed = True
        lookback = ci - ob.top_bar
        if lookback >= 1:
            box_top, box_btm = _find_ob_box(opens, closes, ci, lookback, bull=True)
            if box_top is not None:
                ob_size = abs(box_top - box_btm)
                if 0 < ob_size <= a10 * cfg['ob_max_atr_mult']:
                    zl.trim_ob(+1, cfg['ob_max_blocks'])
                    zl.add(box_top, box_btm, +1, 1, tol, ci - 1)

    # -- Bearish OB: close breaks below swing low ----------------------------
    if ob.btm_y > 0 and c < ob.btm_y and not ob.btm_crossed:
        ob.btm_crossed = True
        lookback = ci - ob.btm_bar
        if lookback >= 1:
            box_top, box_btm = _find_ob_box(opens, closes, ci, lookback, bull=False)
            if box_top is not None:
                ob_size = abs(box_top - box_btm)
                if 0 < ob_size <= a10 * cfg['ob_max_atr_mult']:
                    zl.trim_ob(-1, cfg['ob_max_blocks'])
                    zl.add(box_top, box_btm, -1, 1, tol, ci - 1)


def _detect_oc(zl: ZoneList,
               opens: np.ndarray, closes: np.ndarray,
               highs: np.ndarray, lows: np.ndarray,
               ci: int, a14: float, cfg: dict):
    """
    Port of ProcessOCDetection().

    Bullish demand: bearish odd candle at [ci-1] in a bullish series [ci-2..ci-1-minSeries]
                    confirmed by bullish [ci].
    Bearish supply: bullish odd candle at [ci-1] in a bearish series, confirmed bearish [ci].
    Zone = high/low of the odd candle.
    """
    ms = cfg['oc_min_series']
    if ci < ms + 2:
        return

    tol = a14 * cfg['overlap_tol']

    def is_bull(i: int) -> bool: return closes[i] > opens[i]
    def is_bear(i: int) -> bool: return closes[i] < opens[i]

    def check_series(offset: int, length: int, bullish: bool) -> bool:
        # MQL5 CheckSeries(baseShift+offset, length, bullish)
        # checks bars ci-offset, ci-offset-1, ..., ci-offset-(length-1)
        for k in range(length):
            idx = ci - offset - k
            if idx < 0:          return False
            if bullish  and not is_bull(idx): return False
            if not bullish and not is_bear(idx): return False
        return True

    # Bearish odd candle at [ci-1] -> bullish demand zone
    if is_bear(ci - 1) and is_bull(ci) and check_series(2, ms, True):
        zl.add(highs[ci - 1], lows[ci - 1], +1, 2, tol, ci - 1)

    # Bullish odd candle at [ci-1] -> bearish supply zone
    if is_bull(ci - 1) and is_bear(ci) and check_series(2, ms, False):
        zl.add(highs[ci - 1], lows[ci - 1], -1, 2, tol, ci - 1)


def _detect_ls(zl: ZoneList,
               opens: np.ndarray, closes: np.ndarray,
               highs: np.ndarray, lows: np.ndarray,
               pivot_highs: np.ndarray, pivot_lows: np.ndarray,
               ci: int, a14: float, cfg: dict):
    """
    Port of ProcessLSDetection().

    pivot_highs[ci] is non-NaN when a pivot high exists at bar ci - pivot_len.
    Pivot high -> bearish supply; Pivot low -> bullish demand.
    Zone extent depends on ls_area (WICK_EXTREMITY or FULL_RANGE).
    """
    pl = cfg['ls_pivot_len']
    if ci < 2 * pl:
        return

    tol      = a14 * cfg['overlap_tol']
    piv_idx  = ci - pl          # the actual pivot bar in the DataFrame

    if not np.isnan(pivot_highs[ci]):
        piv_h = highs[piv_idx]
        ph_bot = (max(closes[piv_idx], opens[piv_idx])
                  if cfg['ls_area'] == 'WICK_EXTREMITY'
                  else lows[piv_idx])
        if piv_h > ph_bot:
            zl.add(piv_h, ph_bot, -1, 4, tol, piv_idx)

    if not np.isnan(pivot_lows[ci]):
        piv_l = lows[piv_idx]
        pl_top = (min(closes[piv_idx], opens[piv_idx])
                  if cfg['ls_area'] == 'WICK_EXTREMITY'
                  else highs[piv_idx])
        if pl_top > piv_l:
            zl.add(pl_top, piv_l, +1, 4, tol, piv_idx)


def _detect_rejection(zl: ZoneList,
                      opens: np.ndarray, closes: np.ndarray,
                      highs: np.ndarray, lows: np.ndarray,
                      ci: int, a14: float, cfg: dict) -> int:
    """
    Port of ProcessRejectionDetection().
    Returns +1 (bull rejection), -1 (bear rejection), 0 (none).

    Rejection criteria:
      body  >= ATR14 * min_body_atr
      bull: lower_wick >= body * ratio   (long lower wick)
      bear: upper_wick >= body * ratio   (long upper wick)

    Zone interaction (WICK_IN_CLOSE_OUT default):
      bull: l <= zone_top  AND  c > zone_top
      bear: h >= zone_bot  AND  c < zone_bot
    """
    c, o, h, l = closes[ci], opens[ci], highs[ci], lows[ci]

    body       = abs(c - o)
    body_top   = max(o, c)
    body_bot   = min(o, c)
    upper_wick = h - body_top
    lower_wick = body_bot - l
    min_body   = a14 * cfg['rej_min_body_atr']

    body_ok       = body >= min_body
    bull_wick_ok  = body_ok and lower_wick >= body * cfg['rej_ratio']
    bear_wick_ok  = body_ok and upper_wick >= body * cfg['rej_ratio']

    is_bull_rej = is_bear_rej = False
    mode = cfg['rej_zone_mode']

    for z in zl.zones:
        zt, zb, zd = z.top, z.bottom, z.direction

        if bull_wick_ok and zd > 0:
            if   mode == 'WICK_IN_CLOSE_OUT' and l <= zt and c > zt: is_bull_rej = True
            elif mode == 'ANY_TOUCH'          and l <= zt and h >= zb: is_bull_rej = True
            elif mode == 'CLOSE_INSIDE'       and zb <= c <= zt:       is_bull_rej = True

        if bear_wick_ok and zd < 0:
            if   mode == 'WICK_IN_CLOSE_OUT' and h >= zb and c < zb: is_bear_rej = True
            elif mode == 'ANY_TOUCH'          and h >= zb and l <= zt: is_bear_rej = True
            elif mode == 'CLOSE_INSIDE'       and zb <= c <= zt:       is_bear_rej = True

    if is_bull_rej: return +1
    if is_bear_rej: return -1
    return 0


def _build_signal(symbol: str, df: pd.DataFrame,
                  ci: int, atr_sl: float, is_bull: bool,
                  zl: ZoneList, cfg: dict) -> SignalInfo:
    """Calculate trade levels and package into SignalInfo."""
    c = float(df['close'].iloc[ci])
    h = float(df['high'].iloc[ci])
    l = float(df['low'].iloc[ci])
    bar_time = int(df['open_time'].iloc[ci])

    en = c

    # SL is placed beyond the candle wick (natural structure) plus an ATR buffer.
    # This is more robust than pure ATR from entry: it respects the actual
    # price structure of the rejection candle and avoids dangerously tight stops
    # on low-volatility bars where ATR alone would place SL inside the spread.
    #
    #   Bull: SL = candle low  - ATR * sl_atr_mult
    #   Bear: SL = candle high + ATR * sl_atr_mult
    #
    # risk = |entry - SL|  (always entry-to-SL, used for TP and position sizing)
    sl   = (l - atr_sl * cfg['sl_atr_mult']) if is_bull else (h + atr_sl * cfg['sl_atr_mult'])
    risk = abs(en - sl)
    tp   = (en + risk * cfg['tp_rr']) if is_bull else (en - risk * cfg['tp_rr'])

    # Safety: if SL is somehow on the wrong side of entry (data anomaly), skip
    if is_bull and sl >= en:
        sl = en - atr_sl
        risk = abs(en - sl)
        tp = en + risk * cfg['tp_rr']
    elif not is_bull and sl <= en:
        sl = en + atr_sl
        risk = abs(en - sl)
        tp = en - risk * cfg['tp_rr']

    # Convert SL/TP distances to integer pip counts.
    # Multiplier is based on entry price magnitude so it works correctly
    # across all asset classes without manual pip_size configuration.
    #   BTC/ETH (>=1000)    : mult=1     -> 300 pips = $300
    #   SOL/BNB/LINK (>=10) : mult=100   -> 130 pips = $1.30
    #   XRP/ADA/etc (>=0.5) : mult=10000 -> 22 pips  = $0.0022
    #   DOGE/etc (<0.5)     : mult=1000  -> 3 pips   = $0.003
    if en >= 1000:       _pip_mult = 1
    elif en >= 10:       _pip_mult = 100
    elif en >= 0.5:      _pip_mult = 10000
    elif en >= 0.01:     _pip_mult = 1000
    else:                _pip_mult = 100000
    # Safety: if result rounds to 0 (very tight SL), scale up once more
    _sl_p = int(round(risk * _pip_mult))
    if _sl_p == 0 and risk > 0:
        _pip_mult *= 10
    sl_pips = int(round(risk             * _pip_mult))
    tp_pips = int(round(risk * cfg['tp_rr'] * _pip_mult))

    # Lot size estimate for Telegram alert display.
    # Actual qty is determined at order placement: order_value / entry_price.
    # order_value = manual_order_value (GUI/alert mode) or floor(max_lev/2) (auto mode).
    lot_size = None
    lot_info = "N/A"
    order_value = float(cfg.get('bybit_manual_order_value', 10.0))
    if en > 0 and order_value > 0:
        raw_qty  = order_value / en
        lot_size = round(raw_qty, 4)
        lot_info = f"~{lot_size} @ {order_value:.0f} USDT order value"

    # Determine which zone types triggered the rejection
    opens  = df['open'].values
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    mode   = cfg['rej_zone_mode']
    zone_src = 0

    for z in zl.zones:
        zt, zb, zd = z.top, z.bottom, z.direction
        triggered = False
        if is_bull and zd > 0:
            if   mode == 'WICK_IN_CLOSE_OUT' and lows[ci]  <= zt and closes[ci] > zt:  triggered = True
            elif mode == 'ANY_TOUCH'          and lows[ci]  <= zt and highs[ci] >= zb:  triggered = True
            elif mode == 'CLOSE_INSIDE'       and zb <= closes[ci] <= zt:               triggered = True
        elif not is_bull and zd < 0:
            if   mode == 'WICK_IN_CLOSE_OUT' and highs[ci] >= zb and closes[ci] < zb:  triggered = True
            elif mode == 'ANY_TOUCH'          and highs[ci] >= zb and lows[ci]  <= zt:  triggered = True
            elif mode == 'CLOSE_INSIDE'       and zb <= closes[ci] <= zt:               triggered = True
        if triggered:
            zone_src = src_or(zone_src, z.src)

    # Collect zone coordinates for triggered zones
    zone_coords = []
    _label_map = {1: "OB", 2: "OC", 4: "LS", 3: "OB+OC", 5: "OB+LS",
                  6: "OC+LS", 7: "OB+OC+LS"}
    for z in zl.zones:
        zt, zb, zd = z.top, z.bottom, z.direction
        triggered = False
        if is_bull and zd > 0:
            if   mode == 'WICK_IN_CLOSE_OUT' and lows[ci]  <= zt and closes[ci] > zt:  triggered = True
            elif mode == 'ANY_TOUCH'          and lows[ci]  <= zt and highs[ci] >= zb:  triggered = True
            elif mode == 'CLOSE_INSIDE'       and zb <= closes[ci] <= zt:               triggered = True
        elif not is_bull and zd < 0:
            if   mode == 'WICK_IN_CLOSE_OUT' and highs[ci] >= zb and closes[ci] < zb:  triggered = True
            elif mode == 'ANY_TOUCH'          and highs[ci] >= zb and lows[ci]  <= zt:  triggered = True
            elif mode == 'CLOSE_INSIDE'       and zb <= closes[ci] <= zt:               triggered = True
        if triggered:
            src_bits = z.src
            lbl = _label_map.get(src_bits, src_text(src_bits))
            zone_coords.append((lbl, round(zt, 8), round(zb, 8)))

    return SignalInfo(
        symbol=symbol,
        direction=+1 if is_bull else -1,
        zone_src=zone_src,
        entry=en, sl=sl, tp=tp,
        sl_pips=sl_pips, tp_pips=tp_pips,
        risk=risk, lot_size=lot_size, lot_info=lot_info,
        rr=cfg['tp_rr'],
        bar_time=bar_time,
        active_zones=len(zl.zones),
        zone_coords=zone_coords if zone_coords else None,
        obos_bars_ago=None,
    )


# -- Main entry point ---------------------------------------------------------

def scan_pair(symbol: str, df: pd.DataFrame, cfg: dict) -> Optional[SignalInfo]:
    """
    Full stateless scan for one pair.

    Fetches scan_bars of history, replays bar-by-bar (historical scan) to
    rebuild zone state, then checks the last confirmed bar for a rejection
    signal.  The forming candle (df.iloc[-1]) is never processed.

    This is called in a thread-pool so must be thread-safe (no shared state).
    """
    n = len(df)
    if n < 60:
        return None

    opens  = df['open'].values.astype(float)
    closes = df['close'].values.astype(float)
    highs  = df['high'].values.astype(float)
    lows   = df['low'].values.astype(float)

    # -- Pre-compute indicators -----------------------------------------------
    atr14  = calc_wilder_atr(highs, lows, closes, 14)
    atr10  = calc_wilder_atr(highs, lows, closes, 10)
    atr_sl = calc_wilder_atr(highs, lows, closes, cfg['sl_atr_len'])

    pl = cfg['ls_pivot_len']
    if cfg['ls_enabled']:
        pivot_highs = compute_pivot_highs(highs, pl, pl)
        pivot_lows  = compute_pivot_lows (lows,  pl, pl)
    else:
        pivot_highs = pivot_lows = np.full(n, np.nan)

    # -- State ----------------------------------------------------------------
    zl = ZoneList()
    ob = OBState()

    # Minimum bar index before detection can begin
    min_ci = max(
        cfg['ob_swing_len'] * 2 + 2,
        pl * 2 + 2,
        cfg['oc_min_series'] + 4,
        20,
    )

    # -- Historical replay  (oldest confirmed bar -> second-to-last row) -------
    # df.iloc[-1] = forming candle (excluded); df.iloc[-2] = live/confirmed bar
    for ci in range(min_ci, n - 1):
        a14 = atr14[ci];  a10 = atr10[ci]
        if np.isnan(a14) or np.isnan(a10) or a14 <= 0 or a10 <= 0:
            continue

        zl.check_mitigation(ci, closes, highs, lows, cfg['mitigation'])

        if cfg['ob_enabled']:
            _detect_ob(zl, ob, opens, closes, highs, lows, ci, a10, a14, cfg)
        if cfg['oc_enabled']:
            _detect_oc(zl, opens, closes, highs, lows, ci, a14, cfg)
        if cfg['ls_enabled']:
            _detect_ls(zl, opens, closes, highs, lows,
                       pivot_highs, pivot_lows, ci, a14, cfg)

    # -- Live bar check (df.iloc[-2]) -----------------------------------------
    ci = n - 2
    if ci < min_ci:
        return None

    a14 = atr14[ci];  a10 = atr10[ci];  asl = atr_sl[ci]
    if np.isnan(a14) or np.isnan(a10) or np.isnan(asl) or a14 <= 0:
        return None

    # Mitigation + zone detection on the live bar (same as historical)
    zl.check_mitigation(ci, closes, highs, lows, cfg['mitigation'])
    if cfg['ob_enabled']:
        _detect_ob(zl, ob, opens, closes, highs, lows, ci, a10, a14, cfg)
    if cfg['oc_enabled']:
        _detect_oc(zl, opens, closes, highs, lows, ci, a14, cfg)
    if cfg['ls_enabled']:
        _detect_ls(zl, opens, closes, highs, lows,
                   pivot_highs, pivot_lows, ci, a14, cfg)

    if not cfg['rej_enabled'] or not zl.zones:
        return None

    rej = _detect_rejection(zl, opens, closes, highs, lows, ci, a14, cfg)
    if rej == 0:
        return None

    sig = _build_signal(symbol, df, ci, asl, rej > 0, zl, cfg)
    if sig is None:
        return None

    # Confluence filter: skip signals that don't meet the minimum zone-type count.
    # zone_src is a bitmask: OB=1, OC=2, LS=4.
    # min_confluence is automatically capped to the number of ENABLED zone types
    # so that disabling e.g. Odd Candle never makes min_confluence=3 impossible.
    if cfg.get('require_confluence', True):
        enabled_count = sum([
            bool(cfg.get('ob_enabled', True)),
            bool(cfg.get('oc_enabled', True)),
            bool(cfg.get('ls_enabled', True)),
        ])
        min_conf  = min(int(cfg.get('min_confluence', 2)), enabled_count)
        zones_hit = bin(sig.zone_src).count('1')
        if zones_hit < min_conf:
            return None

    # ── OB/OS Confirmation Filter ──────────────────────────────────────────
    # Requires an OB/OS crossover in the same direction within obos_window bars.
    # Both sequences are valid:
    #   Rejection first -> crossover fires within N bars
    #   Crossover first -> rejection fires within N bars (current bar IS rejection)
    # Signal skipped when no matching crossover found within window.
    if cfg.get('obos_enabled', False):
        obos_n   = int(cfg.get('obos_length', 5))
        obos_win = int(cfg.get('obos_window', 5))
        h_sl = highs [:ci + 1]
        l_sl = lows  [:ci + 1]
        c_sl = closes[:ci + 1]
        offset, cross_dir = _obos_last_signal(h_sl, l_sl, c_sl, obos_n, obos_win)
        if cross_dir == 0 or cross_dir != sig.direction:
            return None
        sig.obos_bars_ago = offset

    return sig
