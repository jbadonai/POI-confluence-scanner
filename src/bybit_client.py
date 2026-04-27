# -*- coding: utf-8 -*-
"""
Bybit V5 REST API client for POI scanner auto-trading.

Design
------
- Hedge Mode: positionIdx 1=long, 2=short (set once in Bybit account settings)
- Market orders by default; Limit optional via config
- SL/TP attached to the opening order (slTriggerBy/tpTriggerBy = LastPrice)
- Leverage queried from Bybit per symbol and cached for the process lifetime
- Qty floored to Bybit qtyStep — never rounds up (avoids over-risking)
- No pybit dependency; uses stdlib + requests only
"""

import hashlib
import hmac
import json
import logging
import math
import time
from threading import Lock
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Bybit retCode that means "leverage already set to this value" — not a real error
_ERR_LEVERAGE_ALREADY_SET = {110043}

# Public endpoints tried in order for connection probing and time sync
_TIME_ENDPOINTS = [
    ("/v5/market/time",  lambda d: int(d.get("result", {}).get("timeSecond", 0)) * 1000),
    ("/v5/public/time",  lambda d: int(d.get("result", {}).get("timeNano",   0)) // 1_000_000),
]
_PROBE_ENDPOINTS = [
    "/v5/market/time",
    "/v5/market/tickers?category=linear&symbol=BTCUSDT",
    "/v5/public/time",
]


class BybitClient:
    """Thread-safe Bybit V5 REST client. All methods are synchronous."""

    def __init__(self, cfg: dict):
        self._key    = cfg.get("bybit_api_key",    "").strip()
        self._secret = cfg.get("bybit_api_secret", "").strip()
        testnet      = cfg.get("bybit_testnet", False)

        self._base         = ("https://api-testnet.bybit.com" if testnet
                              else "https://api.bybit.com")
        self._recv_window  = "5000"
        self._order_lock   = Lock()
        self._time_offset  = 0

        # Per-process caches
        self._leverage_cache:   Dict[str, int]  = {}
        self._instrument_cache: Dict[str, dict] = {}

        if self._key and self._secret:
            self._sync_time()
        else:
            logger.warning("Bybit credentials not configured.")

    # -------------------------------------------------------------------------
    # Time sync
    # -------------------------------------------------------------------------

    def _sync_time(self):
        for path, extractor in _TIME_ENDPOINTS:
            try:
                r = requests.get(self._base + path, timeout=5)
                if r.status_code != 200:
                    continue
                ts = extractor(r.json())
                if ts > 0:
                    self._time_offset = ts - int(time.time() * 1000)
                    logger.debug(f"Bybit time synced via {path} offset={self._time_offset}ms")
                    return
            except Exception:
                continue
        logger.warning("Bybit time sync failed — using local clock (trades will still work)")

    def _now_ms(self) -> str:
        return str(int(time.time() * 1000) + self._time_offset)

    # -------------------------------------------------------------------------
    # Auth / HTTP
    # -------------------------------------------------------------------------

    def _sign(self, ts: str, payload: str) -> str:
        pre = ts + self._key + self._recv_window + payload
        return hmac.new(self._secret.encode(), pre.encode(), hashlib.sha256).hexdigest()

    def _auth_headers(self, ts: str, sig: str) -> dict:
        return {
            "Content-Type":       "application/json",
            "X-BAPI-API-KEY":     self._key,
            "X-BAPI-TIMESTAMP":   ts,
            "X-BAPI-RECV-WINDOW": self._recv_window,
            "X-BAPI-SIGN":        sig,
        }

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body, separators=(",", ":"))
        ts  = self._now_ms()
        sig = self._sign(ts, body_str)
        r   = requests.post(
            self._base + path,
            headers=self._auth_headers(ts, sig),
            data=body_str,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        # Surface Bybit-level errors as exceptions so callers don't have to check retCode
        code = data.get("retCode", -1)
        if code != 0:
            raise RuntimeError(f"Bybit API error [{code}]: {data.get('retMsg', '?')}")
        return data

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        params = params or {}
        qs     = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        ts     = self._now_ms()
        sig    = self._sign(ts, qs)
        url    = self._base + path + (f"?{qs}" if qs else "")
        r      = requests.get(url, headers=self._auth_headers(ts, sig), timeout=10)
        r.raise_for_status()
        data = r.json()
        code = data.get("retCode", -1)
        if code != 0:
            raise RuntimeError(f"Bybit API error [{code}]: {data.get('retMsg', '?')}")
        return data

    def _get_public(self, path: str, params: Optional[dict] = None) -> dict:
        params = params or {}
        qs     = "&".join(f"{k}={v}" for k, v in params.items())
        url    = self._base + path + (f"?{qs}" if qs else "")
        r      = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()

    # -------------------------------------------------------------------------
    # Instrument info (cached)
    # -------------------------------------------------------------------------

    def _get_instrument(self, symbol: str) -> dict:
        if symbol not in self._instrument_cache:
            data  = self._get_public("/v5/market/instruments-info",
                                     {"category": "linear", "symbol": symbol})
            items = data.get("result", {}).get("list", [])
            if not items:
                raise RuntimeError(f"No instrument info returned for {symbol}")
            self._instrument_cache[symbol] = items[0]
        return self._instrument_cache[symbol]

    def get_max_leverage(self, symbol: str) -> int:
        info = self._get_instrument(symbol)
        return int(float(info.get("leverageFilter", {}).get("maxLeverage", 50)))

    def _qty_precision(self, symbol: str) -> Tuple[float, float, float]:
        """Returns (qty_step, min_order_qty, tick_size)."""
        info = self._get_instrument(symbol)
        lsf  = info.get("lotSizeFilter", {})
        pf   = info.get("priceFilter",   {})
        return (
            float(lsf.get("qtyStep",     "0.001")),
            float(lsf.get("minOrderQty", "0.001")),
            float(pf.get("tickSize",     "0.01")),
        )

    def round_qty(self, qty: float, symbol: str) -> float:
        """Floor qty to Bybit's qtyStep. Returns 0.0 if below minOrderQty."""
        step, min_qty, _ = self._qty_precision(symbol)
        floored = math.floor(qty / step) * step
        floored = round(floored, 8)
        return floored if floored >= min_qty else 0.0

    def round_price(self, price: float, symbol: str) -> str:
        """Round price to Bybit's tickSize and return as string for API."""
        _, _, tick = self._qty_precision(symbol)
        rounded = round(round(price / tick) * tick, 8)
        # Format with enough decimal places to avoid scientific notation
        decimals = max(0, -int(math.floor(math.log10(tick)))) if tick < 1 else 0
        return f"{rounded:.{decimals}f}"

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    def get_balance(self) -> float:
        """
        Available USDT in the Unified Trading Account.
        Tries both UNIFIED and CONTRACT account types for compatibility.
        Returns 0.0 if balance cannot be determined.
        """
        for account_type in ("UNIFIED", "CONTRACT"):
            try:
                data  = self._get("/v5/account/wallet-balance",
                                  {"accountType": account_type})
                lists = data.get("result", {}).get("list", [])
                if not lists:
                    continue
                coins = lists[0].get("coin", [])
                for c in coins:
                    if c.get("coin") == "USDT":
                        raw = c.get("availableToWithdraw") or c.get("walletBalance") or "0"
                        val = float(raw) if str(raw).strip() not in ("", "None", "null") else 0.0
                        if val > 0:
                            logger.debug(f"Balance ({account_type}): {val:.2f} USDT")
                            return val
            except Exception as e:
                logger.debug(f"get_balance ({account_type}) failed: {e}")
                continue
        logger.warning("get_balance: could not read USDT balance from any account type")
        return 0.0

    def get_open_position_count(self) -> int:
        """Count of all open linear positions (any size > 0)."""
        data  = self._get("/v5/position/list",
                          {"category": "linear", "settleCoin": "USDT"})
        items = data.get("result", {}).get("list", [])
        return sum(1 for p in items if float(p.get("size", 0)) > 0)

    def has_open_position(self, symbol: str, side: str) -> bool:
        """
        Check if a position is already open for this symbol and side.

        side: "Buy" (long) or "Sell" (short) — Bybit Hedge Mode convention.
        In Hedge Mode, positionIdx 1 = long, positionIdx 2 = short.
        Returns True if an open position (size > 0) exists for that side.
        """
        try:
            data  = self._get("/v5/position/list",
                              {"category": "linear", "symbol": symbol})
            items = data.get("result", {}).get("list", [])
            target_idx = 1 if side == "Buy" else 2
            for p in items:
                if (int(p.get("positionIdx", 0)) == target_idx
                        and float(p.get("size", 0)) > 0):
                    logger.debug(f"[{symbol}] Open {side} position found "
                                 f"size={p.get('size')} avgPrice={p.get('avgPrice')}")
                    return True
            return False
        except Exception as e:
            logger.warning(f"[{symbol}] has_open_position check failed: {e} — assuming no position")
            return False

    # -------------------------------------------------------------------------
    # Leverage — always use maximum, gracefully handles already-set errors
    # -------------------------------------------------------------------------

    def set_max_leverage(self, symbol: str, max_leverage: int) -> bool:
        """
        Set buy+sell leverage to max_leverage for symbol in Hedge Mode.

        Gracefully handles:
        - 110043: leverage already set to this value (treated as success)
        - Any other Bybit error: logged, returns False (trade blocked)

        Uses a process-lifetime cache so each symbol is set only once per run.
        """
        cached = self._leverage_cache.get(symbol)
        if cached == max_leverage:
            logger.debug(f"[{symbol}] Leverage already confirmed at {max_leverage}x (cached)")
            return True
        try:
            self._post("/v5/position/set-leverage", {
                "category":     "linear",
                "symbol":       symbol,
                "buyLeverage":  str(max_leverage),
                "sellLeverage": str(max_leverage),
            })
            self._leverage_cache[symbol] = max_leverage
            logger.info(f"[{symbol}] Max leverage set to {max_leverage}x")
            return True
        except RuntimeError as e:
            err_str = str(e)
            if "110043" in err_str:
                # Already at this leverage — not an error, cache and continue
                self._leverage_cache[symbol] = max_leverage
                logger.info(f"[{symbol}] Max leverage already at {max_leverage}x")
                return True
            if "110044" in err_str:
                # Leverage not modified (same value) — also fine
                self._leverage_cache[symbol] = max_leverage
                logger.info(f"[{symbol}] Leverage unchanged at {max_leverage}x")
                return True
            # Genuine error — log and block the trade
            logger.error(f"[{symbol}] set-leverage failed: {e}")
            return False
        except Exception as e:
            logger.error(f"[{symbol}] set-leverage unexpected error: {e}")
            return False

    # -------------------------------------------------------------------------
    # Order value sizing
    # -------------------------------------------------------------------------

    def get_order_value(self, symbol: str) -> Tuple[int, float]:
        """
        Determine max leverage and order value for a symbol.

        Order value (USDT notional) = floor(max_leverage / 2)
        e.g. max_leverage=100 -> order_value=50 USDT
             max_leverage=75  -> order_value=37 USDT (floor)

        Returns (max_leverage, order_value_usdt).
        """
        max_lev     = self.get_max_leverage(symbol)
        order_value = math.floor(max_lev / 2.0)
        logger.info(f"[{symbol}] max_leverage={max_lev}x  "
                    f"order_value={order_value} USDT")
        return max_lev, float(order_value)

    # -------------------------------------------------------------------------
    # Order placement
    # -------------------------------------------------------------------------

    def place_order(self,
                    symbol:        str,
                    side:          str,
                    qty:           float,
                    sl_price:      float,
                    tp_price:      float,
                    order_link_id: str,
                    cfg:           dict,
                    entry_price:   float = 0.0) -> dict:
        """
        Place a market or limit order with SL and TP in Hedge Mode.
        entry_price is required for LIMIT orders (the limit price to post).
        Raises RuntimeError on any Bybit API error.
        """
        order_type   = cfg.get("bybit_order_type", "MARKET").upper()
        sl_trigger   = cfg.get("bybit_sl_trigger", "LastPrice")
        tp_trigger   = cfg.get("bybit_tp_trigger", "LastPrice")
        position_idx = 1 if side == "Buy" else 2
        is_limit     = order_type == "LIMIT"

        body = {
            "category":    "linear",
            "symbol":      symbol,
            "side":        side,
            "orderType":   "Limit" if is_limit else "Market",
            "qty":         str(qty),
            "timeInForce": "GTC",
            "positionIdx": position_idx,
            "orderLinkId": order_link_id,
            "stopLoss":    self.round_price(sl_price, symbol),
            "takeProfit":  self.round_price(tp_price, symbol),
            "slTriggerBy": sl_trigger,
            "tpTriggerBy": tp_trigger,
        }

        # Limit orders require a price field — use entry_price if provided,
        # otherwise fall back to sl_price (better than omitting it entirely)
        if is_limit:
            limit_price = entry_price if entry_price > 0 else sl_price
            body["price"] = self.round_price(limit_price, symbol)
            logger.info(f"[{symbol}] Limit order at price={body['price']}")

        logger.debug(f"place_order body: {json.dumps(body)}")

        with self._order_lock:
            resp = self._post("/v5/order/create", body)

        return resp.get("result", {})

    # -------------------------------------------------------------------------
    # Connection test
    # -------------------------------------------------------------------------

    def test_connection(self) -> bool:
        """
        Probe several public Bybit endpoints.
        Returns True as soon as one responds with a valid payload.
        Does NOT require API credentials.
        """
        for path in _PROBE_ENDPOINTS:
            try:
                r = requests.get(self._base + path, timeout=8)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("retCode") == 0 or "result" in data:
                        logger.info(f"Bybit reachable via {path}")
                        return True
                logger.debug(f"Bybit probe {path} -> HTTP {r.status_code}")
            except Exception as e:
                logger.debug(f"Bybit probe {path} failed: {e}")
        logger.error(
            "Bybit unreachable on all probed endpoints. "
            "Check network. If Bybit is geo-blocked, a VPN is needed."
        )
        return False
