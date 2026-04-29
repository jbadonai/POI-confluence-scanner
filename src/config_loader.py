# -*- coding: utf-8 -*-
"""
Loads and validates config.ini into a single flat dict consumed by all modules.
"""
import configparser
import os
from typing import Any, Dict

_TF_TO_SECONDS = {
    '1m': 60, '3m': 180, '5m': 300, '15m': 900,
    '30m': 1800, '1h': 3600, '4h': 14400, '1d': 86400,
}

_TRIGGER_MAP = {
    'LAST_PRICE': 'LastPrice',
    'MARK_PRICE': 'MarkPrice',
    'INDEX_PRICE': 'IndexPrice',
}


def load_config(path: str = 'config.ini') -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    cp = configparser.ConfigParser()
    cp.read(path, encoding='utf-8')

    def s(sec, key, fallback=''):    return cp.get(sec, key, fallback=fallback).strip()
    def b(sec, key, fallback=False): return cp.getboolean(sec, key, fallback=fallback)
    def f(sec, key, fallback=0.0):  return cp.getfloat(sec, key, fallback=fallback)
    def i(sec, key, fallback=0):    return cp.getint(sec, key, fallback=fallback)

    symbols_raw = s('PAIRS', 'symbols')
    symbols = [sym.strip().upper() for sym in symbols_raw.split(',') if sym.strip()]

    timeframe     = s('BINANCE', 'timeframe', '5m')
    auto_interval = _TF_TO_SECONDS.get(timeframe, 300)

    # Bybit trigger normalisation
    sl_trig_raw = s('BYBIT', 'sl_trigger', 'LAST_PRICE').upper()
    tp_trig_raw = s('BYBIT', 'tp_trigger', 'LAST_PRICE').upper()
    sl_trigger  = _TRIGGER_MAP.get(sl_trig_raw, 'LastPrice')
    tp_trigger  = _TRIGGER_MAP.get(tp_trig_raw, 'LastPrice')

    cfg: Dict[str, Any] = {
        # -- Telegram ---------------------------------------------------------
        'telegram_token':   s('TELEGRAM', 'bot_token'),
        'telegram_chat_id': s('TELEGRAM', 'chat_id'),

        # -- Binance -----------------------------------------------------------
        'binance_base_url': s('BINANCE', 'base_url', 'https://api1.binance.com'),
        'timeframe':        timeframe,
        'scan_bars':        i('BINANCE', 'scan_bars', 500),

        # -- Pairs -------------------------------------------------------------
        'symbols': symbols,

        # -- Bybit -------------------------------------------------------------
        'bybit_api_key':            s('BYBIT', 'api_key'),
        'bybit_api_secret':         s('BYBIT', 'api_secret'),
        'bybit_testnet':            b('BYBIT', 'testnet', False),
        'bybit_base_url':           s('BYBIT', 'base_url', 'https://api.bybit.com'),
        'bybit_use_manual_order_value': b('BYBIT', 'use_manual_order_value', False),
        'bybit_manual_order_value':     f('BYBIT', 'manual_order_value', 10.0),
        'bybit_order_value_multiplier': f('BYBIT', 'order_value_multiplier', 0.5),
        'bybit_auto_trade':         b('BYBIT', 'auto_trade', True),
        'bybit_max_concurrent_trades': i('BYBIT', 'max_concurrent_trades', 5),
        'bybit_notify_failures':     b('BYBIT', 'notify_failures', True),
        'bybit_order_type':         s('BYBIT', 'order_type', 'MARKET').upper(),
        'bybit_sl_trigger':         sl_trigger,
        'bybit_tp_trigger':         tp_trigger,

        # -- General -----------------------------------------------------------
        'mitigation': s('GENERAL', 'mitigation', 'CLOSE_BEYOND').upper(),

        # -- Order Block -------------------------------------------------------
        'ob_enabled':      b('ORDER_BLOCK', 'enabled', True),
        'ob_swing_len':    i('ORDER_BLOCK', 'swing_len', 10),
        'ob_max_atr_mult': f('ORDER_BLOCK', 'max_atr_mult', 3.5),
        'ob_max_blocks':   i('ORDER_BLOCK', 'max_blocks', 5),

        # -- Odd Candle --------------------------------------------------------
        'oc_enabled':    b('ODD_CANDLE', 'enabled', True),
        'oc_min_series': i('ODD_CANDLE', 'min_series', 2),

        # -- Liquidity Swing ---------------------------------------------------
        'ls_enabled':   b('LIQUIDITY_SWING', 'enabled', True),
        'ls_pivot_len': i('LIQUIDITY_SWING', 'pivot_len', 14),
        'ls_area':      s('LIQUIDITY_SWING', 'area', 'WICK_EXTREMITY').upper(),

        # -- Zone --------------------------------------------------------------
        'overlap_tol': f('ZONE', 'overlap_tol', 0.1),

        # -- Rejection ---------------------------------------------------------
        'rej_enabled':      b('REJECTION', 'enabled', True),
        'rej_ratio':        f('REJECTION', 'ratio', 1.5),
        'rej_min_body_atr': f('REJECTION', 'min_body_atr', 0.05),
        'rej_zone_mode':    s('REJECTION', 'zone_mode', 'WICK_IN_CLOSE_OUT').upper(),
        'require_confluence': b('REJECTION', 'require_confluence', True),
        'min_confluence':     i('REJECTION', 'min_confluence', 2),  # 2 or 3

        # -- Trade levels ------------------------------------------------------
        'sl_atr_mult': f('TRADE', 'sl_atr_mult', 0.5),
        'sl_atr_len':  i('TRADE', 'sl_atr_len', 14),
        'tp_rr':       f('TRADE', 'tp_rr', 1.0),
        'pip_size':    f('TRADE', 'pip_size', 1.0),

        # -- Legacy position sizing (alert-only mode) --------------------------

        # -- OB/OS confirmation filter ----------------------------------------
        'obos_enabled': b('OBOS_FILTER', 'enabled', False),
        'obos_length':  i('OBOS_FILTER', 'length',  5),
        'obos_window':  i('OBOS_FILTER', 'window',  5),

        # -- HTF trend filter (Ichimoku Cloud) ---------------------------------
        'htf_enabled':          b('HTF_FILTER', 'enabled',          False),
        'htf_strict':           b('HTF_FILTER', 'strict',           True),
        'htf_timeframe':        s('HTF_FILTER', 'timeframe',        '1h'),
        'htf_tenkan':           i('HTF_FILTER', 'tenkan_period',    9),
        'htf_kijun':            i('HTF_FILTER', 'kijun_period',     26),
        'htf_senkou_b':         i('HTF_FILTER', 'senkou_b_period',  52),
        'htf_displacement':     i('HTF_FILTER', 'displacement',     26),
        'htf_skip_ranging':     b('HTF_FILTER', 'skip_ranging',     True),
        'htf_counter_trend_rr': f('HTF_FILTER', 'counter_trend_rr', 0.5),

        # -- Scanner -----------------------------------------------------------
        'interval_seconds':        auto_interval,   # always = timeframe in seconds (e.g. 5m -> 300)
        'max_concurrent_requests': i('SCANNER', 'max_concurrent_requests', 50),
        'max_worker_threads':      i('SCANNER', 'max_worker_threads', 16),
    }

    # -- Validation -----------------------------------------------------------
    errors = []

    if not cfg['telegram_token'] or 'YOUR_' in cfg['telegram_token']:
        errors.append("TELEGRAM bot_token must be set in config.ini")
    if not cfg['telegram_chat_id'] or 'YOUR_' in cfg['telegram_chat_id']:
        errors.append("TELEGRAM chat_id must be set in config.ini")
    if not cfg['symbols']:
        errors.append("[PAIRS] symbols must contain at least one pair")
    if cfg['mitigation'] not in {'CLOSE_BEYOND', 'WICK_BEYOND', 'BOTH_M'}:
        errors.append(f"Invalid mitigation: {cfg['mitigation']}")
    if cfg['ls_area'] not in {'WICK_EXTREMITY', 'FULL_RANGE'}:
        errors.append(f"Invalid ls_area: {cfg['ls_area']}")
    if cfg['rej_zone_mode'] not in {'WICK_IN_CLOSE_OUT', 'ANY_TOUCH', 'CLOSE_INSIDE'}:
        errors.append(f"Invalid rej_zone_mode: {cfg['rej_zone_mode']}")
    if cfg['scan_bars'] < 100:
        errors.append("scan_bars should be at least 100")
    if cfg['bybit_auto_trade']:
        if not cfg['bybit_api_key'] or 'YOUR_' in cfg['bybit_api_key']:
            errors.append("BYBIT api_key must be set when auto_trade = true")
        if not cfg['bybit_api_secret'] or 'YOUR_' in cfg['bybit_api_secret']:
            errors.append("BYBIT api_secret must be set when auto_trade = true")
    if errors:
        raise ValueError("Config errors:\n  " + "\n  ".join(errors))

    return cfg
