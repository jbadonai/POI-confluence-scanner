# -*- coding: utf-8 -*-
"""
TradeExecutor: bridges a SignalInfo to a live Bybit order.

Execution steps
---------------
1.  Persistent dedup      - SQLite: skip if (symbol, bar_time, direction) already traded
2.  Symbol validation     - skip if symbol not supported on Bybit linear (e.g. ETHUSDC)
3.  Concurrent guard      - skip if open positions >= max_concurrent_trades
4.  Leverage              - resolve max or fixed, set on Bybit (cached)
5.  Notional sizing       - FIXED_RISK or PCT_BALANCE
6.  Sanity bounds         - reject if notional < min or > max
7.  Balance check         - reject if required margin > available USDT (prevents 110007)
8.  Qty rounding          - floor to Bybit qtyStep, reject if below minOrderQty
9.  DB pre-reservation    - insert PLACING record before API call (race guard)
10. Place order           - market with SL+TP attached
11. DB update             - PLACED with bybitId, or FAILED with reason

Returns
-------
On success : dict(order_id, leverage, qty, notional, failure=None)
On skip    : None
On failure : dict(order_id=None, failure=reason_str)  — so caller can Telegram-notify
"""

import logging
from typing import Optional, Tuple

from .bybit_client import BybitClient
from .models       import SignalInfo
from .trade_store  import TradeStore

logger = logging.getLogger(__name__)

# Bybit linear perpetuals only support USDT-settled pairs.
# USDC pairs (BTCUSDC, ETHUSDC, etc.) are not listed as linear perps.
# We detect these by trying instrument info and catching the error.


class TradeExecutor:

    def __init__(self, cfg: dict, store: TradeStore):
        self.cfg   = cfg
        self.store = store
        self.bybit = BybitClient(cfg)
        # Cache of symbols confirmed invalid on Bybit (avoids repeated API calls)
        self._invalid_symbols: set = set()
        # Tracks (symbol, side) pairs placed THIS cycle to prevent duplicate
        # same-side orders within a single scan cycle (before Bybit position
        # list updates). Cleared by reset_cycle_balance() each cycle.
        self._cycle_placed: set = set()
        self._cycle_placed_lock = __import__('threading').Lock()
        # Running balance tracker: reduced by margin of each order placed this cycle.
        # Reset at the start of each cycle via reset_cycle_balance().
        # Prevents multiple signals in the same cycle all passing the balance check
        # before any of them actually place — which causes Bybit error 110007.
        self._cycle_balance: Optional[float] = None
        self._cycle_balance_lock = __import__('threading').Lock()

    # -------------------------------------------------------------------------
    # Cycle balance management
    # -------------------------------------------------------------------------

    def reset_cycle_balance(self):
        """
        Called once at the start of each scan cycle (from scanner).
        Fetches fresh balance from Bybit and stores it as the cycle budget.
        All execute() calls in this cycle deduct from this shared balance.

        Sets _cycle_balance to None only on genuine API failure (disables check).
        A balance of 0.0 is valid and will correctly block all orders.
        """
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
        Try to execute a signal on Bybit.

        Returns:
            dict(order_id, leverage, qty, notional, failure=None)  -- success
            dict(order_id=None, failure=reason_str)                -- failed (notify)
            None                                                    -- silently skipped
        """
        sym   = sig.symbol
        d_str = "LONG" if sig.direction > 0 else "SHORT"
        tag   = f"[{sym} {d_str}]"

        try:
            # Step 1: persistent dedup
            if self.store.already_traded(sym, sig.bar_time, sig.direction):
                logger.info(f"{tag} Skip — already traded bar {sig.bar_time}")
                return None

            # Step 2: symbol validation (skip USDC pairs and others not on Bybit linear)
            if sym in self._invalid_symbols:
                logger.info(f"{tag} Skip — {sym} not supported on Bybit linear")
                return None
            try:
                self.bybit.get_max_leverage(sym)   # uses instrument cache
            except Exception as e:
                self._invalid_symbols.add(sym)
                reason = f"{sym} not available on Bybit linear perps: {e}"
                logger.warning(f"{tag} {reason}")
                # Record once but don't Telegram-notify — it's a config issue, not a trade failure
                self._record(sig, "SKIPPED", note=reason)
                return None

            # Step 3: same-side position guard
            # In Hedge Mode: max 1 long AND 1 short per symbol at any time.
            # Block if this side already has an open position on Bybit.
            bybit_side = "Buy" if sig.direction > 0 else "Sell"
            if self.bybit.has_open_position(sym, bybit_side):
                reason = f"Already have an open {bybit_side} position on {sym}"
                logger.info(f"{tag} Skip — {reason}")
                self._record(sig, "SKIPPED", note=reason)
                return None

            # Step 3b: within-cycle duplicate guard
            # Bybit position list may not update instantly after placement,
            # so track what we placed this cycle ourselves.
            cycle_key = (sym, bybit_side)
            with self._cycle_placed_lock:
                if cycle_key in self._cycle_placed:
                    logger.info(f"{tag} Skip — already placed {bybit_side} this cycle")
                    return None

            # Step 4: concurrent position guard (total across all pairs)
            max_pos    = self.cfg.get("bybit_max_concurrent_trades", 5)
            open_count = self.bybit.get_open_position_count()
            if open_count >= max_pos:
                reason = f"Max positions reached ({open_count}/{max_pos})"
                logger.info(f"{tag} Skip — {reason}")
                self._record(sig, "SKIPPED", note=reason)
                return None

            # Step 5: leverage
            leverage, ok = self._resolve_leverage(sym)
            if not ok:
                reason = "Leverage setup failed"
                logger.error(f"{tag} {reason}")
                self._record(sig, "FAILED", note=reason)
                return {"order_id": None, "failure": reason}

            # Step 5: notional sizing
            sl_pct = abs(sig.entry - sig.sl) / sig.entry
            if sl_pct <= 1e-9:
                reason = f"SL distance too small (sl_pct={sl_pct:.8f})"
                logger.error(f"{tag} {reason}")
                self._record(sig, "FAILED", note=reason)
                return {"order_id": None, "failure": reason}

            notional = self.bybit.calc_notional(self.cfg, sl_pct)
            margin   = notional / leverage
            logger.info(f"{tag} Sizing: sl_pct={sl_pct:.4%}  "
                        f"notional={notional:.2f} USDT  "
                        f"margin={margin:.2f} USDT @ {leverage}x")

            # Step 6: sanity bounds
            min_n = self.cfg.get("bybit_min_notional",  5.0)
            max_n = self.cfg.get("bybit_max_notional",  500_000.0)
            if notional < min_n:
                reason = f"Notional {notional:.2f} USDT < minimum {min_n}"
                logger.warning(f"{tag} {reason}")
                self._record(sig, "FAILED", note=reason)
                return {"order_id": None, "failure": reason}
            if notional > max_n:
                reason = f"Notional {notional:.2f} USDT > maximum {max_n}"
                logger.warning(f"{tag} {reason}")
                self._record(sig, "FAILED", note=reason)
                return {"order_id": None, "failure": reason}

            # Step 7: balance check — prevent 110007 "not enough available balance"
            # Uses the cycle budget (set once per cycle) rather than a live API call,
            # so multiple signals in the same cycle each see the correctly reduced balance.
            with self._cycle_balance_lock:
                cycle_bal = self._cycle_balance

            if cycle_bal is not None:
                logger.info(f"{tag} Cycle balance: {cycle_bal:.2f} USDT  "
                            f"Required margin: {margin:.2f} USDT")
                if margin > cycle_bal:
                    reason = (f"Insufficient balance — need {margin:.2f} USDT margin "
                              f"but only {cycle_bal:.2f} USDT available")
                    logger.warning(f"{tag} {reason}")
                    self._record(sig, "FAILED", note=reason)
                    return {"order_id": None, "failure": reason}
            else:
                logger.warning(f"{tag} Balance unknown — proceeding without check")

            # Step 8: quantity
            raw_qty = notional / sig.entry
            qty     = self.bybit.round_qty(raw_qty, sym)
            if qty <= 0:
                reason = (f"Qty rounded to 0 "
                          f"(raw={raw_qty:.6f}  notional={notional:.2f}  "
                          f"entry={sig.entry})")
                logger.warning(f"{tag} {reason}")
                self._record(sig, "FAILED", note=reason)
                return {"order_id": None, "failure": reason}

            # Step 9: pre-reserve DB slot (prevents duplicate order on race)
            link_id  = self._make_link_id(sym, sig.bar_time, sig.direction)
            reserved = self.store.record(
                symbol=sym, direction=sig.direction, bar_time=sig.bar_time,
                zone_src=sig.zone_src, entry=sig.entry, sl=sig.sl, tp=sig.tp,
                qty=qty, notional=notional,
                risk_usd=self.cfg.get("bybit_risk_usd", 50.0),
                order_link_id=link_id, status="PLACING",
            )
            if not reserved:
                logger.info(f"{tag} Skip — DB slot already taken")
                return None

            # Step 10: place order
            side = "Buy" if sig.direction > 0 else "Sell"
            logger.info(f"{tag} Placing {side} {qty} contracts  "
                        f"SL={sig.sl:.5f}  TP={sig.tp:.5f}  linkId={link_id}")

            result   = self.bybit.place_order(
                symbol=sym, side=side, qty=qty,
                sl_price=sig.sl, tp_price=sig.tp,
                order_link_id=link_id, cfg=self.cfg,
                entry_price=sig.entry,
            )

            # Step 11: update DB on success
            bybit_id = result.get("orderId", "")
            self.store.update_status(link_id, "PLACED", note=f"bybitId={bybit_id}")
            logger.info(f"{tag} Order PLACED — bybitId={bybit_id}")

            # Mark this (symbol, side) as placed for the rest of this cycle
            with self._cycle_placed_lock:
                self._cycle_placed.add(cycle_key)

            # Deduct margin from cycle budget so the next signal in this cycle
            # sees the reduced available balance and won't over-commit.
            with self._cycle_balance_lock:
                if self._cycle_balance is not None:
                    self._cycle_balance = max(0.0, self._cycle_balance - margin)
                    logger.info(f"{tag} Cycle balance after order: {self._cycle_balance:.2f} USDT")

            return {
                "order_id": bybit_id,
                "leverage": leverage,
                "qty":      qty,
                "notional": notional,
                "failure":  None,
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

    def _resolve_leverage(self, symbol: str) -> Tuple[int, bool]:
        use_max = self.cfg.get("bybit_use_max_leverage", True)
        if use_max:
            try:
                leverage = self.bybit.get_max_leverage(symbol)
            except Exception as e:
                logger.warning(f"[{symbol}] get_max_leverage failed: {e} — using fixed")
                leverage = self.cfg.get("bybit_fixed_leverage", 10)
        else:
            leverage = self.cfg.get("bybit_fixed_leverage", 10)

        ok = self.bybit.ensure_leverage(symbol, leverage)
        return leverage, ok

    @staticmethod
    def _make_link_id(symbol: str, bar_time: int, direction: int) -> str:
        """Deterministic 36-char order link ID — survives restarts."""
        d   = "L" if direction > 0 else "S"
        ts  = str(bar_time)[-10:]
        return f"POI_{symbol}_{ts}_{d}"[:36]

    def _record(self, sig: SignalInfo, status: str, note: str = ""):
        """Insert or update a trade record safely (never raises)."""
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
