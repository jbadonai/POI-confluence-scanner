# -*- coding: utf-8 -*-
"""
Binance public REST API — klines fetcher.
No API key required (public market data endpoint).

Automatically cycles through Binance's four mirror hosts when the primary
endpoint is slow or geo-blocked (common outside the US/EU).
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# Binance provides four interchangeable mirrors — tried in order per symbol.
_BINANCE_HOSTS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

_KLINES_COLS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'num_trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore',
]


async def _fetch_one(
    session:   aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol:    str,
    interval:  str,
    limit:     int,
    base_url:  str,          # preferred host (from config); mirrors tried first
) -> Tuple[str, Optional[list]]:
    """
    Fetch klines for one symbol.

    Tries the configured host first, then the four Binance mirrors in order,
    giving each host one attempt before moving on.  Total attempts = 5.
    """
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}

    # Build host list: configured host first, then mirrors (skip duplicates)
    hosts = [base_url] + [h for h in _BINANCE_HOSTS if h != base_url]

    async with semaphore:
        for host in hosts:
            url = f"{host}/api/v3/klines"
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        if host != base_url:
                            logger.info(f"[{symbol}] Succeeded via mirror: {host}")
                        return symbol, await resp.json()
                    if resp.status == 429:
                        wait = int(resp.headers.get('Retry-After', 30))
                        logger.warning(f"[{symbol}] Rate-limited on {host}. Waiting {wait}s.")
                        await asyncio.sleep(wait)
                        continue       # retry same host after back-off
                    logger.warning(f"[{symbol}] HTTP {resp.status} from {host}")
                    # Non-retryable HTTP error — try next mirror
                    continue

            except asyncio.TimeoutError:
                logger.warning(f"[{symbol}] Timeout on {host} — trying next mirror.")
            except aiohttp.ClientConnectorError as e:
                logger.warning(f"[{symbol}] Connection error on {host}: {e} — trying next mirror.")
            except aiohttp.ClientError as e:
                logger.warning(f"[{symbol}] Network error on {host}: {e} — trying next mirror.")
            except Exception as e:
                logger.error(f"[{symbol}] Unexpected error on {host}: {e}")

    logger.error(f"[{symbol}] All Binance mirrors exhausted — no data.")
    return symbol, None


def _to_df(raw: list) -> Optional[pd.DataFrame]:
    """Convert Binance klines JSON list to a typed OHLCV DataFrame."""
    if not raw:
        return None
    df = pd.DataFrame(raw, columns=_KLINES_COLS)
    for col in ('open', 'high', 'low', 'close', 'volume'):
        df[col] = df[col].astype(float)
    df['open_time'] = df['open_time'].astype(int)
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)


async def fetch_all_pairs(
    symbols:    List[str],
    interval:   str,
    limit:      int,
    base_url:   str,
    max_concur: int,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch klines for all symbols concurrently.

    Returns {symbol: DataFrame}.  The last row of each DataFrame is the
    *forming* (incomplete) candle — scan_pair() always excludes it.
    """
    semaphore = asyncio.Semaphore(max_concur)
    connector = aiohttp.TCPConnector(limit=max_concur, limit_per_host=10)

    # Per-request: 30 s to connect, 90 s total (generous for slow links).
    # The session timeout is disabled (None) so it does not cut across retries.
    req_timeout = aiohttp.ClientTimeout(connect=30, sock_read=60, total=None)

    async with aiohttp.ClientSession(connector=connector, timeout=req_timeout) as session:
        tasks   = [_fetch_one(session, semaphore, sym, interval, limit, base_url)
                   for sym in symbols]
        results = await asyncio.gather(*tasks)

    out: Dict[str, pd.DataFrame] = {}
    for sym, raw in results:
        if raw is None:
            continue
        df = _to_df(raw)
        if df is None or len(df) < 60:
            logger.warning(f"[{sym}] Skipped — only {len(raw) if raw else 0} bars returned.")
            continue
        out[sym] = df

    return out
