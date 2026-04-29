# -*- coding: utf-8 -*-
"""
Sends formatted signal and trade execution alerts to Telegram.
Uses HTML parse mode.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from .models   import SignalInfo
from .detector import src_text

logger = logging.getLogger(__name__)

# TradingView interval parameter mapped from Binance timeframe strings
_TV_INTERVAL = {
    '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
    '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
    '1d': 'D', '1w': 'W',
}


def _tv_url(symbol: str, timeframe: str) -> str:
    """
    Full-screen TradingView chart URL for a Bybit perpetual.
    Opens directly to the correct symbol and timeframe.

    e.g. XRPUSDT on 5m ->
    https://www.tradingview.com/chart/?symbol=BYBIT:XRPUSDT.P&interval=5
    """
    interval = _TV_INTERVAL.get(timeframe, '5')
    return (
        f"https://www.tradingview.com/chart/"
        f"?symbol=BYBIT:{symbol}.P&interval={interval}"
    )


class TelegramNotifier:
    _API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str, chat_id: str, timeframe: str = '5m'):
        self._url       = self._API.format(token=token)
        self._chat_id   = chat_id
        self._timeframe = timeframe   # stored so chart links use the right interval

    # -------------------------------------------------------------------------
    # Price formatting
    # -------------------------------------------------------------------------

    @staticmethod
    def _fp(p: float) -> str:
        if   p >= 10_000: return f"{p:,.1f}"
        elif p >= 1:      return f"{p:,.4f}"
        else:             return f"{p:.6f}"

    # -------------------------------------------------------------------------
    # Message builders
    # -------------------------------------------------------------------------

    def _build_signal_msg(self, sig: SignalInfo,
                          order_id:  Optional[str]   = None,
                          lot_size:  Optional[float] = None,
                          notional:  Optional[float] = None,
                          leverage:  Optional[int]   = None,
                          add_chart: bool            = False) -> str:
        is_bull   = sig.direction > 0
        emoji     = "🟢" if is_bull else "🔴"
        direction = "LONG" if is_bull else "SHORT"
        zone_type = src_text(sig.zone_src) or "—"
        fp        = self._fp

        bar_dt  = datetime.fromtimestamp(sig.bar_time / 1000, tz=timezone.utc)
        bar_str = bar_dt.strftime("%Y-%m-%d %H:%M UTC")

        sl_dist = abs(sig.entry - sig.sl)
        tp_dist = abs(sig.entry - sig.tp)
        sl_side = f"-{fp(sl_dist)} ({sig.sl_pips}pts)" if is_bull else f"+{fp(sl_dist)} ({sig.sl_pips}pts)"
        tp_side = f"+{fp(tp_dist)} ({sig.tp_pips}pts)" if is_bull else f"-{fp(tp_dist)} ({sig.tp_pips}pts)"

        # Counter-trend warning banner (LENIENT mode)
        ct        = getattr(sig, "counter_trend", False)
        htf_tf    = getattr(sig, "htf_timeframe", "")
        htf_trend = getattr(sig, "htf_trend", 0)
        ct_line   = ""
        if ct and htf_tf:
            htf_dir = "Uptrend" if htf_trend > 0 else "Downtrend"
            ct_line = (
                f"\n⚠️ <b>COUNTER-TREND</b> — "
                f"HTF {htf_tf.upper()} is {htf_dir} "
                f"(reduced RR 1:{sig.rr:.1f})"
            )

        # Header varies: alert vs executed trade
        if order_id:
            header    = f"{emoji} <b>TRADE EXECUTED — {sig.symbol}</b>"
            exec_line = f"\n🆔 Bybit ID : <code>{order_id}</code>"
            size_line = ""
            if lot_size and notional and leverage:
                size_line = (
                    f"\n📦 Qty         : <code>{lot_size}</code> contracts"
                    f"\n💵 Order Value : <code>{fp(notional)} USDT</code>"
                    f"\n⚙️  Leverage    : <code>{leverage}x</code>"
                )
            chart_line = ""   # no chart link on executed trades (already in)
        else:
            header    = f"{emoji} <b>POI Signal — {sig.symbol}</b>"
            exec_line = ""
            size_line = f"\n📦 Lot size : {sig.lot_info}"
            # Chart link only for alert-only mode signals
            if add_chart:
                url        = _tv_url(sig.symbol, self._timeframe)
                chart_line = f"\n\n📈 <a href=\"{url}\">Open {sig.symbol} chart on TradingView</a>"
            else:
                chart_line = ""

        # OB/OS confirmation line
        obos_ago = getattr(sig, "obos_bars_ago", None)
        obos_line = ""
        if obos_ago is not None:
            bars_str = "current bar" if obos_ago == 0 else f"{obos_ago} bar{'s' if obos_ago > 1 else ''} ago"
            d_word   = "Buy crossover" if sig.direction > 0 else "Sell crossunder"
            obos_line = f"\n📊 <b>OB/OS</b> : {d_word} confirmed ({bars_str})"

        # POI zone coordinates
        coords = getattr(sig, "zone_coords", None)
        coords_line = ""
        if coords:
            coords_lines = []
            for lbl, top, bot in coords:
                coords_lines.append(
                    f"  <code>{lbl:<6}</code> "
                    f"<code>{fp(top)}</code> — <code>{fp(bot)}</code>"
                )
            coords_line = "\n\n📍 <b>POI Zones:</b>\n" + "\n".join(coords_lines)

        return (
            f"{header}\n"
            f"{ct_line}\n"
            f"\n"
            f"📊 Direction : <b>{direction}</b>\n"
            f"🏷  Zone Type : <b>{zone_type}</b>\n"
            f"\n"
            f"💰 Entry  : <code>{fp(sig.entry)}</code>\n"
            f"🛑 SL     : <code>{fp(sig.sl)}</code>  ({sl_side})\n"
            f"🎯 TP     : <code>{fp(sig.tp)}</code>  ({tp_side})\n"
            f"📐 RR     : <code>1:{sig.rr:.1f}</code>"
            f"{size_line}"
            f"{exec_line}\n"
            f"\n"
            f"⏰ Candle : {bar_str}\n"
            f"🔢 Active zones : {sig.active_zones}"
            f"{obos_line}"
            f"{coords_line}"
            f"{chart_line}"
        )

    # -------------------------------------------------------------------------
    # HTTP send
    # -------------------------------------------------------------------------

    async def _send(self, text: str):
        payload = {
            "chat_id":               self._chat_id,
            "text":                  text,
            "parse_mode":            "HTML",
            "disable_web_page_preview": True,   # don't unfurl the TV chart link
        }
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self._url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Telegram {resp.status}: {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def send_signal(self, sig: SignalInfo):
        """Alert-only signal (auto_trade=false). Includes TradingView chart link."""
        await self._send(self._build_signal_msg(sig, add_chart=True))

    async def send_trade_executed(self, sig: SignalInfo,
                                  order_id: str,
                                  lot_size: float,
                                  order_value: float,
                                  leverage: int):
        """Confirmation message after a Bybit order is placed."""
        await self._send(self._build_signal_msg(
            sig, order_id=order_id,
            lot_size=lot_size, notional=order_value, leverage=leverage,
        ))

    async def send_trade_failed(self, sig: SignalInfo, reason: str):
        is_bull   = sig.direction > 0
        emoji     = "🟢" if is_bull else "🔴"
        d_str     = "LONG" if is_bull else "SHORT"
        zone_type = src_text(sig.zone_src) or "?"
        fp        = self._fp
        sl_dist   = abs(sig.entry - sig.sl)
        tp_dist   = abs(sig.entry - sig.tp)
        sl_side   = f"-{fp(sl_dist)}" if is_bull else f"+{fp(sl_dist)}"
        tp_side   = f"+{fp(tp_dist)}" if is_bull else f"-{fp(tp_dist)}"
        text = (
            f"⚠️ <b>Order FAILED — {sig.symbol}</b>\n"
            f"\n"
            f"{emoji} Direction : <b>{d_str}</b>\n"
            f"🏷  Zone     : <b>{zone_type}</b>\n"
            f"\n"
            f"💰 Entry : <code>{fp(sig.entry)}</code>\n"
            f"🛑 SL    : <code>{fp(sig.sl)}</code>  ({sl_side})\n"
            f"🎯 TP    : <code>{fp(sig.tp)}</code>  ({tp_side})\n"
            f"📐 RR    : <code>1:{sig.rr:.1f}</code>\n"
            f"\n"
            f"❌ <b>Reason:</b> <code>{reason}</code>"
        )
        await self._send(text)

    async def send_startup(self, symbols, timeframe: str,
                           auto_trade: bool = False, testnet: bool = False):
        mode = "🤖 AUTO-TRADE" if auto_trade else "👁 ALERTS ONLY"
        net  = " [TESTNET]" if testnet else ""
        text = (
            f"🚀 <b>POI Confluence Scanner — Started{net}</b>\n"
            f"\n"
            f"📈 Pairs      : {len(symbols)}\n"
            f"⏱  Timeframe  : {timeframe}\n"
            f"🔍 Detecting  : OB · Odd Candle · Liquidity Swing\n"
            f"⚡ Mode       : {mode}\n"
            f"✅ Signals on confirmed candle close only."
        )
        await self._send(text)

    async def send_error(self, msg: str):
        await self._send(f"⚠️ <b>Scanner error</b>\n<code>{msg}</code>")
