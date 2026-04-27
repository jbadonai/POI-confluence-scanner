# -*- coding: utf-8 -*-
"""
TradeExecutor: bridges a SignalInfo to a live Bybit order.

Simplified execution pipeline
------------------------------
1.  Persistent dedup      - SQLite: skip if (symbol, bar_time, direction) already traded
2.  Symbol validation     - skip if symbol not on Bybit linear (e.g. ETHUSDC)
3.  Concurrent guard      - skip if open positions >= max_concurrent_trades
4.  Same-side guard       - skip if this side already open on Bybit for this symbol
5.  Within-cycle guard    - skip if same (symbol, side) already placed this cycle
6.  Max leverage          - get from Bybit, set it, handle errors gracefully
7.  Order value           - floor(max_leverage / 2) USDT
8.  Quantity              - order_value / entry_price, floored to qtyStep
9.  DB pre-reservation    - insert PLACING record before API call
10. Place order           - market or limit, SL+TP attached
11. DB update             - PLACED with bybitId, or FAILED with reason

No position sizing config — order value is always max_leverage // 2 USDT.
SL and TP are retained from the signal (calculated by detector.py).
"""

import logging
import threading
from typing import Optional, Tuple

from .bybit_client import BybitClient
from .models       import SignalInfo
from .trade_store  import TradeStore

logger = logging.getLogger(__name__)


class TradeExecutor:

    def __init__(self, cfg: dict, store: TradeStore):
        self.cfg   = cfg
        self.store = store
        self.bybit = BybitClient(cfg)
        self._invalid_symbols: set = set()

        # Within-cycle duplicate guard
        self._cycle_placed:      set = set()
        self._cycle_placed_lock       = threading.Lock()

        # Cycle balance tracker (kept for margin-check compatibility)
        self._cycle_balance:     Optional[float] = None
        self._cycle_balance_lock                  = threading.Lock()

    # -------------------------------------------------------------------------
    # Cycle reset (called by scanner at start of each cycle)
    # -------------------------------------------------------------------------

    def reset_cycle_balance(self):
        """Reset per-cycle state. Called once per scan cycle."""
        try:
            bal = self.bybit.get_balance()
            with self._cycle_balance_lock:
                self._cycle_balance = bal
            with self._cycle_placed_lock:
                self._cycle_placed.clear()
            logger.info(f"Cycle balance: {bal:.2f} USDT available")
        except Exception as e:
            logger.warning(f"Balance fetch failed: {e} — margin check disabled this cycle")
            with self._cycle_balance_lock:
                self._cycle_balance = None
            with self._cycle_placed_lock:
                self._cycle_placed.clear()

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def execute(self, sig: SignalInfo) -> Optional[dict]:
        """
        Execute a signal on Bybit.

        Returns dict(order_id, leverage, qty, order_value, failure=None) on success.
        Returns dict(order_id=None, failure=reason) on failure (caller notifies).
        Returns None on silent skip (dedup, position guard, cycle guard).
        """
        sym   = sig.symbol
        d_str = "LONG" if sig.direction > 0 else "SHORT"
        tag   = f"[{sym} {d_str}]"

        try:
            # Step 1: persistent dedup
            if self.store.already_traded(sym, sig.bar_time, sig.direction):
                logger.info(f"{tag} Skip — already traded bar {sig.bar_time}")
                return None

            # Step 2: symbol validation
            if sym in self._invalid_symbols:
                logger.info(f"{tag} Skip — {sym} not supported on Bybit linear")
                return None
            try:
                self.bybit.get_max_leverage(sym)
            except Exception as e:
                self._invalid_symbols.add(sym)
                reason = f"{sym} not on Bybit linear: {e}"
                logger.warning(f"{tag} {reason}")
                self._record(sig, "SKIPPED", note=reason)
                return None

            # Step 3: concurrent position guard
            max_pos    = self.cfg.get("bybit_max_concurrent_trades", 5)
            open_count = self.bybit.get_open_position_count()
            if open_count >= max_pos:
                reason = f"Max positions ({open_count}/{max_pos})"
                logger.info(f"{tag} Skip — {reason}")
                self._record(sig, "SKIPPED", note=reason)
                return None

            # Step 4: same-side Bybit position guard
            bybit_side = "Buy" if sig.direction > 0 else "Sell"
            if self.bybit.has_open_position(sym, bybit_side):
                reason = f"Already have an open {bybit_side} on {sym}"
                logger.info(f"{tag} Skip — {reason}")
                self._record(sig, "SKIPPED", note=reason)
                return None

            # Step 5: within-cycle duplicate guard
            cycle_key = (sym, bybit_side)
            with self._cycle_placed_lock:
                if cycle_key in self._cycle_placed:
                    logger.info(f"{tag} Skip — already placed {bybit_side} this cycle")
                    return None

            # Step 6: get max leverage and set it on Bybit
            max_leverage = self.bybit.get_max_leverage(sym)
            ok = self.bybit.set_max_leverage(sym, max_leverage)
            if not ok:
                reason = f"Failed to set max leverage ({max_leverage}x)"
                logger.error(f"{tag} {reason}")
                self._record(sig, "FAILED", note=reason)
                return {"order_id": None, "failure": reason}

            # Step 7: order value = floor(max_leverage / 2) USDT
            _, order_value = self.bybit.get_order_value(sym)
            logger.info(f"{tag} order_value={order_value:.0f} USDT  "
                        f"leverage={max_leverage}x")

            # Optional margin check against cycle balance
            margin_required = order_value / max_leverage
            with self._cycle_balance_lock:
                cycle_bal = self._cycle_balance
            if cycle_bal is not None and margin_required > cycle_bal:
                reason = (f"Insufficient balance — need {margin_required:.2f} USDT margin "
                          f"but only {cycle_bal:.2f} USDT available "
                          f"(order_value={order_value:.0f} USDT @ {max_leverage}x)")
                logger.warning(f"{tag} {reason}")
                self._record(sig, "FAILED", note=reason)
                return {"order_id": None, "failure": reason}

            # Step 8: quantity = order_value / entry_price
            raw_qty = order_value / sig.entry
            qty     = self.bybit.round_qty(raw_qty, sym)
            if qty <= 0:
                reason = (f"Qty rounded to 0 "
                          f"(order_value={order_value:.0f} entry={sig.entry})")
                logger.warning(f"{tag} {reason}")
                self._record(sig, "FAILED", note=reason)
                return {"order_id": None, "failure": reason}

            # Step 9: pre-reserve DB slot
            link_id  = self._make_link_id(sym, sig.bar_time, sig.direction)
            reserved = self.store.record(
                symbol=sym, direction=sig.direction, bar_time=sig.bar_time,
                zone_src=sig.zone_src, entry=sig.entry, sl=sig.sl, tp=sig.tp,
                qty=qty, notional=order_value,
                risk_usd=order_value,
                order_link_id=link_id, status="PLACING",
            )
            if not reserved:
                logger.info(f"{tag} Skip — DB slot already taken")
                return None

            # Step 10: place order
            logger.info(f"{tag} Placing {bybit_side} {qty} contracts  "
                        f"SL={sig.sl:.5f}  TP={sig.tp:.5f}  linkId={link_id}")
            result   = self.bybit.place_order(
                symbol=sym, side=bybit_side, qty=qty,
                sl_price=sig.sl, tp_price=sig.tp,
                order_link_id=link_id, cfg=self.cfg,
                entry_price=sig.entry,
            )

            # Step 11: update DB and cycle state
            bybit_id = result.get("orderId", "")
            self.store.update_status(link_id, "PLACED", note=f"bybitId={bybit_id}")
            logger.info(f"{tag} Order PLACED — bybitId={bybit_id}")

            with self._cycle_placed_lock:
                self._cycle_placed.add(cycle_key)
            with self._cycle_balance_lock:
                if self._cycle_balance is not None:
                    self._cycle_balance = max(0.0,
                                              self._cycle_balance - margin_required)

            return {
                "order_id":    bybit_id,
                "leverage":    max_leverage,
                "qty":         qty,
                "order_value": order_value,
                "failure":     None,
            }

        except Exception as e:
            reason = str(e)
            logger.error(f"{tag} Execution error: {reason}", exc_info=True)
            try:
                link_id = self._make_link_id(sym, sig.bar_time, sig.direction)
                self.store.update_status(link_id, "FAILED", note=reason)
            except Exception:
                self._record(sig, "FAILED", note=reason)
            return {"order_id": None, "failure": reason}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_link_id(symbol: str, bar_time: int, direction: int) -> str:
        d   = "L" if direction > 0 else "S"
        ts  = str(bar_time)[-10:]
        return f"POI_{symbol}_{ts}_{d}"[:36]

    def _record(self, sig: SignalInfo, status: str, note: str = ""):
        try:
            link_id = self._make_link_id(sig.symbol, sig.bar_time, sig.direction)
            inserted = self.store.record(
                symbol=sig.symbol, direction=sig.direction, bar_time=sig.bar_time,
                zone_src=sig.zone_src, entry=sig.entry, sl=sig.sl, tp=sig.tp,
                qty=0, notional=0, risk_usd=0,
                order_link_id=link_id, status=status, note=note,
            )
            if not inserted:
                self.store.update_status(link_id, status, note=note)
        except Exception as e:
            logger.debug(f"_record error (non-fatal): {e}")
