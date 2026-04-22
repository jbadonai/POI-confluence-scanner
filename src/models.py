# -*- coding: utf-8 -*-
"""
Pure data containers shared across modules.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ZoneData:
    top:         float
    bottom:      float
    direction:   int    # +1 = bullish demand, -1 = bearish supply
    src:         int    # bitmask: OB=1, OC=2, LS=4
    hit:         bool = False
    created_bar: int  = 0   # iloc index of bar that created the zone
    zone_id:     int  = 0


@dataclass
class OBState:
    """Persistent state for Order Block swing detection (equiv. to MQL5 'var' globals)."""
    swing_type:  int   = 0      # 0 = top, 1 = bottom  (starts 0, matches MQL5 global default)
    top_bar:     int   = -1     # iloc index of recorded swing-top bar
    top_y:       float = 0.0
    top_crossed: bool  = False
    btm_bar:     int   = -1
    btm_y:       float = 0.0
    btm_crossed: bool  = False


@dataclass
class SignalInfo:
    symbol:       str
    direction:    int            # +1 = long, -1 = short
    zone_src:     int            # bitmask of zone types that triggered
    entry:        float
    sl:           float
    tp:           float
    sl_pips:      int
    tp_pips:      int
    risk:         float          # price distance entry->SL
    lot_size:     Optional[float]
    lot_info:     str
    rr:           float
    bar_time:     int            # open_time of signal candle (ms)
    active_zones: int
    # HTF filter fields (defaults keep backward compatibility)
    counter_trend:  bool  = False   # True when signal opposes HTF trend
    htf_trend:      int   = 0       # +1 bull / -1 bear / 0 neutral / disabled
    htf_timeframe:  str   = ""      # e.g. "1h"
