# -*- coding: utf-8 -*-
"""
Main scanning loop.

Per-cycle flow
--------------
1. Fetch klines for all pairs (aiohttp, concurrent)
2. Scan each pair in ThreadPoolExecutor (CPU-bound, stateless)
3. For each signal:
   - auto_trade=true  -> execute on Bybit, Telegram on success OR failure
   - auto_trade=false -> Telegram alert only
4. Sleep until next candle boundary + 1s buffer
"""
import asyncio
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Optional

from .binance_data       import fetch_all_pairs
from .detector           import scan_pair
from .htf_filter         import HTFFilter
from .models             import SignalInfo
from .telegram_notifier  import TelegramNotifier
from .trade_store        import TradeStore
from .trade_executor     import TradeExecutor

logger = logging.getLogger(__name__)


class Scanner:
    def __init__(self, cfg: dict, signal_queue: Optional[queue.Queue] = None):
        self.cfg          = cfg
        self.notifier     = TelegramNotifier(cfg["telegram_token"], cfg["telegram_chat_id"], cfg["timeframe"])
        self._executor    = ThreadPoolExecutor(max_workers=cfg["max_worker_threads"])
        self._signal_queue = signal_queue   # set when GUI is active (auto_trade=False)

        # In-memory dedup: (symbol, bar_time, direction) -> True
        self._seen: Dict[tuple, bool] = {}

        self._auto_trade = cfg.get("bybit_auto_trade", False)
        self._store:  Optional[TradeStore]    = None
        self._trader: Optional[TradeExecutor] = None

        if self._auto_trade:
            self._store  = TradeStore()
            self._trader = TradeExecutor(cfg, self._store)
            logger.info("Auto-trade: ENABLED (Bybit)")
        else:
            logger.info("Auto-trade: DISABLED (alert-only + GUI)")

        # HTF trend filter (shared across all pairs, cache reset each cycle)
        self._htf = HTFFilter(cfg)
        htf_on  = cfg.get("htf_enabled", False)
        htf_mode = "STRICT" if cfg.get("htf_strict", True) else "LENIENT"
        if htf_on:
            logger.info(f"HTF filter: ENABLED  "
                        f"TF={cfg.get('htf_timeframe','1h')}  "
                        f"EMA({cfg.get('htf_ema_period',20)})  "
                        f"mode={htf_mode}")
        else:
            logger.info("HTF filter: DISABLED")

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------

    async def run(self):
        logger.info(
            f"Scanner running | {len(self.cfg['symbols'])} pairs | "
            f"TF={self.cfg['timeframe']} | scan_bars={self.cfg['scan_bars']}"
        )
        try:
            await self.notifier.send_startup(
                self.cfg["symbols"],
                self.cfg["timeframe"],
                auto_trade=self._auto_trade,
                testnet=self.cfg.get("bybit_testnet", False),
            )
        except Exception as e:
            logger.warning(f"Startup notification failed: {e}")

        if self._auto_trade and self._trader:
            loop = asyncio.get_event_loop()
            ok   = await loop.run_in_executor(
                self._executor, self._trader.bybit.test_connection
            )
            if ok:
                logger.info("Bybit connection OK")
            else:
                logger.error("Bybit connection FAILED — check credentials and network")

        while True:
            try:
                await self._cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                try:
                    await self.notifier.send_error(str(e))
                except Exception:
                    pass
            await self._sleep_to_next_candle()

    # -------------------------------------------------------------------------
    # Scan cycle
    # -------------------------------------------------------------------------

    async def _cycle(self):
        t0 = time.perf_counter()
        logger.info("-- Scan cycle started --")

        loop = asyncio.get_event_loop()

        # Reset cycle balance so each signal in this cycle sees the correct
        # remaining margin budget (prevents 110007 on multi-signal cycles).
        if self._auto_trade and self._trader:
            await loop.run_in_executor(self._executor, self._trader.reset_cycle_balance)

        # Reset HTF cache so each cycle fetches fresh HTF data per symbol
        self._htf.reset_cache()

        pair_data = await fetch_all_pairs(
            symbols    = self.cfg["symbols"],
            interval   = self.cfg["timeframe"],
            limit      = self.cfg["scan_bars"],
            base_url   = self.cfg["binance_base_url"],
            max_concur = self.cfg["max_concurrent_requests"],
        )
        logger.info(f"Fetched {len(pair_data)}/{len(self.cfg['symbols'])} pairs "
                    f"in {time.perf_counter()-t0:.1f}s")
        futures = {
            sym: loop.run_in_executor(self._executor, scan_pair, sym, df, self.cfg)
            for sym, df in pair_data.items()
        }

        placed = failed = alerted = 0

        for sym, fut in futures.items():
            try:
                sig: Optional[SignalInfo] = await fut
                if sig is None:
                    continue

                key = (sym, sig.bar_time, sig.direction)
                if self._seen.get(key):
                    continue
                self._seen[key] = True

                d_str = "LONG" if sig.direction > 0 else "SHORT"
                logger.info(f"  * Signal: {sym} {d_str} "
                            f"entry={sig.entry:.5f} "
                            f"sl={sig.sl:.5f} tp={sig.tp:.5f} "
                            f"rr=1:{sig.rr:.1f}")

                # ── HTF Trend Filter ────────────────────────────────────────
                htf = self._htf.check(sym, sig.direction, sig.rr)

                if htf.skipped:
                    # STRICT mode: counter-trend signal blocked
                    trend_str = "BULL" if htf.trend > 0 else "BEAR"
                    reason = (f"Counter-trend: signal is {d_str} but "
                              f"HTF {htf.htf_timeframe} trend is {trend_str} "
                              f"(close={htf.htf_close:.5f} "
                              f"EMA={htf.ema_value:.5f})")
                    logger.info(f"  [HTF] {sym} SKIPPED — {reason}")
                    if self.cfg.get("bybit_notify_failures", True):
                        await self._notify_failure(sig, reason)
                    failed += 1
                    continue

                if htf.counter_trend:
                    # LENIENT mode: adjust TP for counter-trend
                    new_tp = (sig.entry + sig.risk * htf.tp_rr_override
                              if sig.direction > 0
                              else sig.entry - sig.risk * htf.tp_rr_override)
                    sig.tp            = new_tp
                    sig.rr            = htf.tp_rr_override
                    sig.counter_trend = True
                    sig.htf_trend     = htf.trend
                    sig.htf_timeframe = htf.htf_timeframe
                    trend_str = "BULL" if htf.trend > 0 else "BEAR"
                    logger.info(f"  [HTF] {sym} counter-trend — "
                                f"TP adjusted to RR={htf.tp_rr_override} "
                                f"(HTF {htf.htf_timeframe} is {trend_str})")
                else:
                    sig.htf_trend     = htf.trend
                    sig.htf_timeframe = htf.htf_timeframe
                # ── End HTF Filter ──────────────────────────────────────────

                if self._auto_trade and self._trader:
                    result = await loop.run_in_executor(
                        self._executor, self._trader.execute, sig
                    )
                    if result is None:
                        pass  # silently skipped (dedup / position guard)
                    elif result.get("failure") is None:
                        # Success
                        placed += 1
                        await self._notify_success(sig, result)
                    else:
                        # Failed — notify if enabled
                        failed += 1
                        await self._notify_failure(sig, result["failure"])
                else:
                    # Send Telegram alert
                    await self.notifier.send_signal(sig)
                    alerted += 1
                    # Push to GUI queue if active
                    if self._signal_queue is not None:
                        try:
                            self._signal_queue.put_nowait(sig)
                        except queue.Full:
                            pass

            except Exception as e:
                logger.error(f"  Error processing {sym}: {e}", exc_info=True)

        elapsed = time.perf_counter() - t0
        if self._auto_trade:
            logger.info(f"-- Cycle done: {placed} placed / {failed} failed "
                        f"in {elapsed:.1f}s --")
        else:
            logger.info(f"-- Cycle done: {alerted} alerts sent in {elapsed:.1f}s --")

    # -------------------------------------------------------------------------
    # Notification helpers
    # -------------------------------------------------------------------------

    async def _notify_success(self, sig: SignalInfo, result: dict):
        try:
            await self.notifier.send_trade_executed(
                sig,
                order_id = result["order_id"],
                lot_size = result["qty"],
                notional = result["notional"],
                leverage = result["leverage"],
            )
        except Exception as e:
            logger.warning(f"Success notification failed: {e}")

    async def _notify_failure(self, sig: SignalInfo, reason: str):
        if not self.cfg.get("bybit_notify_failures", True):
            return
        try:
            await self.notifier.send_trade_failed(sig, reason)
        except Exception as e:
            logger.warning(f"Failure notification failed: {e}")

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    async def _sleep_to_next_candle(self):
        interval  = self.cfg["interval_seconds"]
        now       = time.time()
        next_open = (int(now / interval) + 1) * interval
        sleep_for = next_open - now + 1.0

        next_dt = datetime.fromtimestamp(next_open, tz=timezone.utc)
        logger.info(f"Next scan: {next_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC "
                    f"(in {sleep_for:.0f}s)")
        await asyncio.sleep(sleep_for)
