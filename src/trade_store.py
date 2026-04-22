# -*- coding: utf-8 -*-
"""
Persistent trade store — SQLite-backed, survives restarts.

Every executed trade is written here immediately after the Bybit order is
placed.  On startup the scanner reloads this state so it never re-fires a
signal for a bar it already traded.

Schema
------
trades
  id            INTEGER  PK autoincrement
  symbol        TEXT     trading pair
  direction     INTEGER  +1 = long, -1 = short
  bar_time      INTEGER  candle open_time (ms) — primary dedup key
  zone_src      INTEGER  bitmask (OB=1, OC=2, LS=4)
  entry         REAL
  sl            REAL
  tp            REAL
  qty           REAL     executed quantity (contracts)
  notional      REAL     position value in USDT
  risk_usd      REAL     intended USD risk
  bybit_order_id TEXT
  order_link_id TEXT
  status        TEXT     PLACED | FAILED | SKIPPED
  created_at    INTEGER  unix ms when record was inserted
  note          TEXT     free-form (error msg, skip reason, etc.)
"""
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    direction       INTEGER NOT NULL,
    bar_time        INTEGER NOT NULL,
    zone_src        INTEGER NOT NULL DEFAULT 0,
    entry           REAL    NOT NULL DEFAULT 0,
    sl              REAL    NOT NULL DEFAULT 0,
    tp              REAL    NOT NULL DEFAULT 0,
    qty             REAL    NOT NULL DEFAULT 0,
    notional        REAL    NOT NULL DEFAULT 0,
    risk_usd        REAL    NOT NULL DEFAULT 0,
    bybit_order_id  TEXT    NOT NULL DEFAULT '',
    order_link_id   TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'PLACED',
    created_at      INTEGER NOT NULL,
    note            TEXT    NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_trades_signal
    ON trades (symbol, bar_time, direction);
CREATE INDEX IF NOT EXISTS ix_trades_symbol
    ON trades (symbol, created_at DESC);
"""


class TradeStore:
    """Thread-safe SQLite wrapper.  One connection per instance; uses WAL mode."""

    def __init__(self, db_path: str = "trades.db"):
        self._path = str(Path(db_path).resolve())
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_DDL)
        self._conn.commit()
        logger.info(f"TradeStore opened: {self._path}")

    # -- Write ----------------------------------------------------------------

    def record(self,
               symbol: str, direction: int, bar_time: int,
               zone_src: int, entry: float, sl: float, tp: float,
               qty: float, notional: float, risk_usd: float,
               bybit_order_id: str = "", order_link_id: str = "",
               status: str = "PLACED", note: str = "") -> bool:
        """
        Insert a trade record.  Returns True on success, False if the
        (symbol, bar_time, direction) triplet already exists (duplicate).
        """
        try:
            self._conn.execute(
                """
                INSERT INTO trades
                    (symbol, direction, bar_time, zone_src, entry, sl, tp,
                     qty, notional, risk_usd, bybit_order_id, order_link_id,
                     status, created_at, note)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (symbol, direction, bar_time, zone_src, entry, sl, tp,
                 qty, notional, risk_usd, bybit_order_id, order_link_id,
                 status, int(time.time() * 1000), note)
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Unique constraint — already traded this signal
            return False
        except Exception as e:
            logger.error(f"TradeStore.record error: {e}")
            return False

    def update_status(self, order_link_id: str, status: str, note: str = ""):
        try:
            self._conn.execute(
                "UPDATE trades SET status=?, note=? WHERE order_link_id=?",
                (status, note, order_link_id)
            )
            self._conn.commit()
        except Exception as e:
            logger.error(f"TradeStore.update_status error: {e}")

    # -- Read -----------------------------------------------------------------

    def already_traded(self, symbol: str, bar_time: int, direction: int) -> bool:
        """True if this exact signal (pair + candle + direction) was already sent."""
        row = self._conn.execute(
            "SELECT 1 FROM trades WHERE symbol=? AND bar_time=? AND direction=? LIMIT 1",
            (symbol, bar_time, direction)
        ).fetchone()
        return row is not None

    def recent(self, limit: int = 50) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def open_count(self) -> int:
        """Returns count of PLACED trades created in last 24 h (proxy for open positions)."""
        cutoff = int(time.time() * 1000) - 86_400_000
        row = self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status='PLACED' AND created_at >= ?",
            (cutoff,)
        ).fetchone()
        return row[0] if row else 0

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
