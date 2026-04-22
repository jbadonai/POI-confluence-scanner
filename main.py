# -*- coding: utf-8 -*-
import os, sys

# -- Path fix -----------------------------------------------------------------
if hasattr(sys, '_MEIPASS'):
    _INTERNAL = sys._MEIPASS
    _EXE_DIR  = os.path.dirname(sys.executable)
else:
    _INTERNAL = os.path.dirname(os.path.abspath(__file__))
    _EXE_DIR  = _INTERNAL

if _INTERNAL not in sys.path: sys.path.insert(0, _INTERNAL)
if _EXE_DIR  not in sys.path: sys.path.insert(0, _EXE_DIR)
os.chdir(_EXE_DIR)
_CRASH_LOG = os.path.join(_EXE_DIR, "startup_error.log")
# -----------------------------------------------------------------------------

def _write_crash(msg: str):
    try:
        with open(_CRASH_LOG, "w", encoding="utf-8") as f:
            f.write(msg + "\n")
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    0,
                    f"POI Scanner crashed.\n\nCheck: {_CRASH_LOG}\n\n{msg[:400]}",
                    "POI Scanner — Error", 0x10)
            except Exception:
                pass
    except Exception:
        pass

if sys.platform == "win32":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import asyncio
    import logging
    import queue
    import threading
    import traceback
    from src.config_loader import load_config
    from src.scanner       import Scanner
except Exception:
    _write_crash(f"IMPORT ERROR\n\n{traceback.format_exc() if 'traceback' in dir() else ''}")
    sys.exit(1)


def setup_logging():
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("poi_scanner.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Auto-trade ON: original asyncio-only path
# ---------------------------------------------------------------------------

async def run_scanner(cfg):
    scanner = Scanner(cfg)
    await scanner.run()


# ---------------------------------------------------------------------------
# Auto-trade OFF: scanner in background thread + GUI on main thread
# ---------------------------------------------------------------------------

def _run_scanner_in_thread(cfg, sig_queue, loop):
    """Run the asyncio scanner loop in a daemon thread."""
    asyncio.set_event_loop(loop)
    scanner = Scanner(cfg, signal_queue=sig_queue)
    loop.run_until_complete(scanner.run())


def run_with_gui(cfg):
    from src.trade_store        import TradeStore
    from src.telegram_notifier  import TelegramNotifier
    from src.trade_gui          import TradeGUI

    sig_queue = queue.Queue(maxsize=500)
    loop      = asyncio.new_event_loop()

    # Start scanner thread
    t = threading.Thread(
        target=_run_scanner_in_thread,
        args=(cfg, sig_queue, loop),
        daemon=True,
    )
    t.start()

    # Notifier and store shared with GUI
    notifier = TelegramNotifier(
        cfg["telegram_token"], cfg["telegram_chat_id"], cfg["timeframe"]
    )
    store = TradeStore()

    # Run GUI on main thread (blocking until window closed)
    gui = TradeGUI(cfg, sig_queue, loop, notifier, store)
    gui.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    log = logging.getLogger("main")

    cfg_path = "config.ini"
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            cfg_path = sys.argv[idx + 1]

    try:
        cfg = load_config(cfg_path)
    except (FileNotFoundError, ValueError) as e:
        log.error(f"Config error: {e}")
        _write_crash(f"CONFIG ERROR\n\n{e}\n\nExpected: {os.path.join(_EXE_DIR, 'config.ini')}")
        sys.exit(1)

    log.info(f"Config loaded from '{cfg_path}'")
    log.info(f"Pairs     : {', '.join(cfg['symbols'])}")
    log.info(f"TF        : {cfg['timeframe']} ({cfg['interval_seconds']}s interval)")
    log.info(f"scan_bars : {cfg['scan_bars']}")
    log.info(f"Zones     : OB={cfg['ob_enabled']}  OC={cfg['oc_enabled']}  LS={cfg['ls_enabled']}")
    log.info(f"RR        : 1:{cfg['tp_rr']}   SL ATR*{cfg['sl_atr_mult']}")

    auto_trade = cfg.get("bybit_auto_trade", False)

    if auto_trade:
        log.info("Mode: AUTO-TRADE (Bybit)")
        try:
            asyncio.run(run_scanner(cfg))
        except KeyboardInterrupt:
            print("\nScanner stopped.")
    else:
        log.info("Mode: ALERTS ONLY + Manual Trade GUI")
        try:
            run_with_gui(cfg)
        except KeyboardInterrupt:
            print("\nScanner stopped.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        msg = f"RUNTIME ERROR\n\n{traceback.format_exc()}"
        _write_crash(msg)
        print("\n" + msg)
        print(f"\nError saved to: {_CRASH_LOG}")
        print("Window closes in 30s...")
        import time; time.sleep(30)
        sys.exit(1)
