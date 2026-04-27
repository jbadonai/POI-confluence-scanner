# -*- coding: utf-8 -*-
"""
Manual Trade Pad — tkinter GUI for alert-only mode.

Architecture
------------
- Runs on the main thread (tkinter requirement).
- Scanner runs asyncio in a background daemon thread.
- Signals arrive via a thread-safe queue.Queue polled every 500ms.
- Live price fetched from Binance in a separate daemon thread every 5s.
- Bybit calls (set leverage, place order) run in a ThreadPoolExecutor
  so the GUI never freezes.
- Telegram notification sent via asyncio.run_coroutine_threadsafe()
  using the scanner's event loop reference.
"""

import asyncio
import logging
import math
import queue
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from tkinter import messagebox, ttk
from typing import Dict, List, Optional

import requests
import webbrowser

from .bybit_client       import BybitClient
from .models             import SignalInfo
from .telegram_notifier  import TelegramNotifier
from .trade_store        import TradeStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette — dark high-contrast theme
# ---------------------------------------------------------------------------
C = {
    "bg":         "#0d1117",
    "panel":      "#161b22",
    "card":       "#1c2128",
    "card_sel":   "#1f3a5f",
    "border":     "#30363d",
    "text":       "#e6edf3",
    "dim":        "#8b949e",
    "green":      "#3fb950",
    "red":        "#f85149",
    "yellow":     "#e3b341",
    "blue":       "#58a6ff",
    "cyan":       "#39d0d8",
    "btn_place":  "#1a7f37",
    "btn_place_h":"#2ea043",
    "btn_del":    "#8b1c1a",
    "btn_del_h":  "#da3633",
    "btn_sec":    "#21262d",
    "btn_sec_h":  "#30363d",
    "entry_bg":   "#21262d",
    "entry_sel":  "#1f6feb",
}

FONT_TITLE  = ("Segoe UI", 11, "bold")
FONT_LABEL  = ("Segoe UI", 9)
FONT_BOLD   = ("Segoe UI", 9, "bold")
FONT_MONO   = ("Consolas", 9)
FONT_MONO_L = ("Consolas", 11, "bold")
FONT_SMALL  = ("Segoe UI", 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _btn(parent, text, command, bg, hover_bg, fg="#ffffff",
         font=FONT_BOLD, padx=16, pady=6, **kw):
    """Factory for flat styled buttons with hover effect."""
    b = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=hover_bg, activeforeground=fg,
        font=font, relief="flat", cursor="hand2",
        padx=padx, pady=pady, bd=0, **kw
    )
    b.bind("<Enter>", lambda e: b.config(bg=hover_bg))
    b.bind("<Leave>", lambda e: b.config(bg=bg))
    return b


def _label(parent, text, fg=None, font=None, **kw):
    return tk.Label(
        parent, text=text,
        bg=C["panel"], fg=fg or C["text"],
        font=font or FONT_LABEL, **kw
    )


def _sep(parent):
    return tk.Frame(parent, bg=C["border"], height=1)


def _entry(parent, textvariable=None, width=14, **kw):
    e = tk.Entry(
        parent, textvariable=textvariable,
        bg=C["entry_bg"], fg=C["text"],
        insertbackground=C["text"],
        relief="flat", bd=4,
        font=FONT_MONO, width=width, **kw
    )
    e.bind("<FocusIn>",  lambda ev: e.config(bg=C["entry_sel"]))
    e.bind("<FocusOut>", lambda ev: e.config(bg=C["entry_bg"]))
    return e


# ---------------------------------------------------------------------------
# Signal item stored in the GUI list
# ---------------------------------------------------------------------------

@dataclass
class GUISignal:
    sig:       SignalInfo
    checked:   bool = False          # selected for deletion
    timestamp: float = field(default_factory=time.time)

    @property
    def key(self):
        return (self.sig.symbol, self.sig.bar_time, self.sig.direction)

    @property
    def direction_str(self):
        return "LONG" if self.sig.direction > 0 else "SHORT"

    @property
    def age_str(self):
        mins = int((time.time() - self.timestamp) / 60)
        if mins < 60: return f"{mins}m ago"
        return f"{mins//60}h {mins%60}m ago"


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------

class TradeGUI:
    """
    Manual trade pad. Call .run() from the main thread.
    Signals are delivered via signal_queue from the scanner thread.
    """

    def __init__(self, cfg: dict,
                 signal_queue:  queue.Queue,
                 scanner_loop:  asyncio.AbstractEventLoop,
                 notifier:      TelegramNotifier,
                 store:         Optional[TradeStore] = None):

        self.cfg          = cfg
        self.signal_queue = signal_queue
        self.loop         = scanner_loop
        self.notifier     = notifier
        self.store        = store or TradeStore()
        self._bybit       = BybitClient(cfg)
        self._pool        = ThreadPoolExecutor(max_workers=4)

        # Signal list: key -> GUISignal
        self._signals:        Dict[tuple, GUISignal] = {}
        self._selected_key:   Optional[tuple]        = None
        self._live_price_var: Optional[tk.StringVar] = None
        self._live_stop       = threading.Event()

        # Field-edit tracking — prevents programmatic refreshes from
        # overwriting changes the user has made to the parameter fields.
        self._user_edited       = False   # True after any manual field edit
        self._suppress_populate = False   # True during programmatic selection restore
        self._populating        = False   # True while _populate_fields is running

        self._build_ui()

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("POI Scanner — Manual Trade Pad")
        self.root.configure(bg=C["bg"])
        self.root.resizable(True, True)
        self.root.minsize(860, 560)

        # ── Always-on-top ────────────────────────────────────────────────────
        self._topmost = tk.BooleanVar(value=True)
        self.root.attributes("-topmost", True)

        # ── Title bar ────────────────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg=C["panel"], pady=6)
        title_bar.pack(fill="x", padx=0, pady=(0, 1))

        tk.Label(title_bar, text="  POI SCANNER",
                 bg=C["panel"], fg=C["blue"],
                 font=("Segoe UI", 12, "bold")).pack(side="left")
        tk.Label(title_bar, text=" — Manual Trade Pad",
                 bg=C["panel"], fg=C["dim"],
                 font=("Segoe UI", 10)).pack(side="left")

        top_chk = tk.Checkbutton(
            title_bar, text="Always on Top",
            variable=self._topmost,
            command=self._toggle_topmost,
            bg=C["panel"], fg=C["dim"],
            selectcolor=C["card"],
            activebackground=C["panel"],
            activeforeground=C["text"],
            font=FONT_SMALL, cursor="hand2",
        )
        top_chk.pack(side="right", padx=12)

        # ── Main paned layout ────────────────────────────────────────────────
        paned = tk.PanedWindow(
            self.root, orient="horizontal",
            bg=C["border"], sashwidth=4,
            sashrelief="flat", bd=0,
        )
        paned.pack(fill="both", expand=True, padx=1, pady=1)

        left  = tk.Frame(paned, bg=C["panel"], width=260)
        right = tk.Frame(paned, bg=C["panel"])
        paned.add(left,  minsize=220)
        paned.add(right, minsize=520)

        self._build_signal_list(left)
        self._build_trade_panel(right)

        # ── Status bar ───────────────────────────────────────────────────────
        status_bar = tk.Frame(self.root, bg=C["card"], height=22)
        status_bar.pack(fill="x", side="bottom")
        self._status_var = tk.StringVar(value="Ready — waiting for signals")
        tk.Label(status_bar, textvariable=self._status_var,
                 bg=C["card"], fg=C["dim"],
                 font=FONT_SMALL, anchor="w").pack(side="left", padx=8)

    # ── Left panel: signal list ───────────────────────────────────────────────

    def _build_signal_list(self, parent):
        # Header row
        hdr = tk.Frame(parent, bg=C["panel"])
        hdr.pack(fill="x", padx=8, pady=(8, 4))
        self._sig_count_var = tk.StringVar(value="SIGNALS  (0)")
        tk.Label(hdr, textvariable=self._sig_count_var,
                 bg=C["panel"], fg=C["blue"],
                 font=FONT_BOLD).pack(side="left")

        # Treeview (signal list)
        tv_frame = tk.Frame(parent, bg=C["panel"])
        tv_frame.pack(fill="both", expand=True, padx=6, pady=2)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Signals.Treeview",
                        background=C["card"],
                        foreground=C["text"],
                        fieldbackground=C["card"],
                        bordercolor=C["border"],
                        lightcolor=C["border"],
                        darkcolor=C["border"],
                        rowheight=28,
                        font=FONT_MONO)
        style.configure("Signals.Treeview.Heading",
                        background=C["panel"],
                        foreground=C["dim"],
                        bordercolor=C["border"],
                        font=FONT_SMALL)
        style.map("Signals.Treeview",
                  background=[("selected", C["card_sel"])],
                  foreground=[("selected", C["text"])])

        cols = ("chk", "pair", "dir", "entry", "age")
        self._tv = ttk.Treeview(
            tv_frame, columns=cols, show="headings",
            style="Signals.Treeview", selectmode="browse",
        )
        self._tv.heading("chk",   text="")
        self._tv.heading("pair",  text="Pair")
        self._tv.heading("dir",   text="Dir")
        self._tv.heading("entry", text="Entry")
        self._tv.heading("age",   text="Age")
        self._tv.column("chk",   width=24,  stretch=False, anchor="center")
        self._tv.column("pair",  width=82,  stretch=False, anchor="w")
        self._tv.column("dir",   width=44,  stretch=False, anchor="center")
        self._tv.column("entry", width=70,  stretch=False, anchor="e")
        self._tv.column("age",   width=52,  stretch=True,  anchor="e")

        self._tv.tag_configure("long",  foreground=C["green"])
        self._tv.tag_configure("short", foreground=C["red"])

        sb = ttk.Scrollbar(tv_frame, orient="vertical", command=self._tv.yview)
        self._tv.configure(yscrollcommand=sb.set)
        self._tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._tv.bind("<<TreeviewSelect>>", self._on_signal_select)
        self._tv.bind("<Button-1>",         self._on_tv_click)

        # Bottom buttons
        btn_row = tk.Frame(parent, bg=C["panel"])
        btn_row.pack(fill="x", padx=6, pady=(4, 8))

        _btn(btn_row, "Select All", self._select_all,
             C["btn_sec"], C["btn_sec_h"], C["dim"],
             font=FONT_SMALL, padx=8, pady=4).pack(side="left", padx=2)
        _btn(btn_row, "Clear", self._clear_checks,
             C["btn_sec"], C["btn_sec_h"], C["dim"],
             font=FONT_SMALL, padx=8, pady=4).pack(side="left", padx=2)

        self._del_btn = _btn(btn_row, "Delete Checked",
                             self._delete_checked,
                             C["btn_del"], C["btn_del_h"],
                             font=FONT_SMALL, padx=8, pady=4)
        self._del_btn.pack(side="right", padx=2)

    # ── Right panel: trade parameters ────────────────────────────────────────

    def _build_trade_panel(self, parent):
        # Signal header
        hdr = tk.Frame(parent, bg=C["panel"])
        hdr.pack(fill="x", padx=12, pady=(10, 4))

        self._pair_var = tk.StringVar(value="— No signal selected —")
        self._dir_var  = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self._pair_var,
                 bg=C["panel"], fg=C["text"],
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        self._dir_lbl = tk.Label(hdr, textvariable=self._dir_var,
                                  bg=C["panel"], fg=C["dim"],
                                  font=("Segoe UI", 11, "bold"))
        self._dir_lbl.pack(side="left", padx=(10, 0))

        # TradingView chart link — shown when a pair is selected
        self._tv_url: str = ""
        self._chart_link = tk.Label(
            hdr, text="",
            bg=C["panel"], fg=C["blue"],
            font=("Segoe UI", 9, "underline"),
            cursor="hand2",
        )
        self._chart_link.pack(side="right", padx=(0, 4))
        self._chart_link.bind("<Button-1>", self._open_chart)

        _sep(parent).pack(fill="x", padx=12, pady=4)

        # Scrollable parameters area
        canvas  = tk.Canvas(parent, bg=C["panel"], highlightthickness=0)
        v_scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)
        v_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=C["panel"])
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_resize(e):
            canvas.itemconfig(inner_id, width=e.width)
        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_resize)
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        self._build_param_fields(inner)

    def _build_param_fields(self, parent):
        PAD = {"padx": 14, "pady": 3}

        def row(lbl_text, var, extra_var=None, extra_lbl=""):
            f = tk.Frame(parent, bg=C["panel"])
            f.pack(fill="x", **PAD)
            tk.Label(f, text=f"{lbl_text:<10}", bg=C["panel"],
                     fg=C["dim"], font=FONT_LABEL, width=10, anchor="w").pack(side="left")
            e = _entry(f, textvariable=var, width=16)
            e.pack(side="left", padx=(4, 0))
            if extra_var is not None:
                tk.Label(f, textvariable=extra_var, bg=C["panel"],
                         fg=C["yellow"], font=FONT_SMALL).pack(side="left", padx=(8, 0))
            return e

        # ── Price fields ─────────────────────────────────────────────────────
        _label(parent, "  TRADE PARAMETERS", fg=C["blue"],
               font=FONT_BOLD).pack(anchor="w", padx=14, pady=(8, 2))
        _sep(parent).pack(fill="x", padx=14, pady=2)

        self._entry_var = tk.StringVar()
        self._sl_var    = tk.StringVar()
        self._tp_var    = tk.StringVar()
        self._order_value_var = tk.StringVar()
        self._qty_var         = tk.StringVar()
        self._risk_var  = tk.StringVar(value="")
        self._rr_var    = tk.StringVar(value="")

        row("Entry",    self._entry_var)
        row("Stop Loss",self._sl_var,  self._risk_var, "risk")
        row("Take Profit",self._tp_var,self._rr_var,   "rr")

        # ── Order Value section ──────────────────────────────────────────────
        _label(parent, "  Order Value Mode", fg=C["blue"], font=FONT_BOLD).pack(
               anchor="w", padx=14, pady=(6, 2))

        self._ov_mode = tk.StringVar(value="manual")
        ov_mode_row = tk.Frame(parent, bg=C["panel"])
        ov_mode_row.pack(fill="x", padx=18, pady=2)
        for val, lbl in [("manual",   "Manual (USDT)"),
                          ("max_lev",  "Max Leverage"),
                          ("half_lev", "½ Max Leverage")]:
            tk.Radiobutton(
                ov_mode_row, text=lbl, variable=self._ov_mode, value=val,
                command=self._on_ov_mode_change,
                bg=C["panel"], fg=C["text"], selectcolor=C["card"],
                activebackground=C["panel"], activeforeground=C["text"],
                font=FONT_LABEL, cursor="hand2",
            ).pack(side="left", padx=(0, 14))

        f_ov = tk.Frame(parent, bg=C["panel"])
        f_ov.pack(fill="x", **PAD)
        tk.Label(f_ov, text=f"{'Order Val':<10}", bg=C["panel"],
                 fg=C["dim"], font=FONT_LABEL, width=10, anchor="w").pack(side="left")
        self._ov_entry = _entry(f_ov, textvariable=self._order_value_var, width=16)
        self._ov_entry.pack(side="left", padx=(4, 0))
        tk.Label(f_ov, text="USDT", bg=C["panel"],
                 fg=C["dim"], font=FONT_SMALL).pack(side="left", padx=(4, 0))
        # Fetch button (visible for max_lev / half_lev modes)
        self._ov_fetch_btn = _btn(
            f_ov, "Fetch", self._fetch_order_value,
            C["btn_sec"], C["btn_sec_h"], C["dim"],
            font=FONT_SMALL, padx=6, pady=2,
        )
        self._ov_fetch_btn.pack(side="left", padx=(6, 0))
        self._ov_fetch_btn.pack_forget()   # hidden by default (manual mode)
        self._ov_info = tk.Label(
            f_ov, text="", bg=C["panel"], fg=C["yellow"], font=FONT_SMALL)
        self._ov_info.pack(side="left", padx=(6, 0))

        f_qty = tk.Frame(parent, bg=C["panel"])
        f_qty.pack(fill="x", **PAD)
        tk.Label(f_qty, text=f"{'Qty':<10}", bg=C["panel"],
                 fg=C["dim"], font=FONT_LABEL, width=10, anchor="w").pack(side="left")
        _entry(f_qty, textvariable=self._qty_var, width=16).pack(side="left", padx=(4, 0))
        _btn(f_qty, "Auto", self._recalc_qty,
             C["btn_sec"], C["btn_sec_h"], C["dim"],
             font=FONT_SMALL, padx=6, pady=2).pack(side="left", padx=(6, 0))

        # Auto-recalc qty when entry or SL changes — only schedule if the user
        # is actually typing (not during _populate_fields which sets _populating=True).
        # Without this guard the delayed callback fires after _populating resets,
        # triggering _mark_edited and wrongly setting _user_edited=True on first load.
        def _schedule_recalc(*_):
            if not self._populating:
                self.root.after(100, self._recalc_qty)
        for v in (self._entry_var, self._sl_var):
            v.trace_add("write", _schedule_recalc)

        # Mark fields as user-edited whenever any editable var changes
        # (only fires when NOT inside _populate_fields)
        def _mark_edited(*_):
            if not self._populating:
                self._user_edited = True
                self._reset_btn.config(state="normal")   # enable Reset button
        for v in (self._entry_var, self._sl_var, self._tp_var,
                  self._order_value_var, self._qty_var):
            v.trace_add("write", _mark_edited)

        _sep(parent).pack(fill="x", padx=14, pady=6)

        # ── Order type ───────────────────────────────────────────────────────
        _label(parent, "  Order Type", fg=C["blue"], font=FONT_BOLD).pack(
               anchor="w", padx=14, pady=(4, 2))
        self._order_type = tk.StringVar(value="MARKET")
        ot_row = tk.Frame(parent, bg=C["panel"])
        ot_row.pack(fill="x", padx=18, pady=2)
        for val, lbl in [("MARKET", "Market"), ("LIMIT", "Limit")]:
            tk.Radiobutton(ot_row, text=lbl, variable=self._order_type, value=val,
                           bg=C["panel"], fg=C["text"], selectcolor=C["card"],
                           activebackground=C["panel"], activeforeground=C["text"],
                           font=FONT_LABEL, cursor="hand2").pack(side="left", padx=(0, 18))

        _sep(parent).pack(fill="x", padx=14, pady=6)

        # ── Trigger type ─────────────────────────────────────────────────────
        _label(parent, "  SL/TP Trigger", fg=C["blue"], font=FONT_BOLD).pack(
               anchor="w", padx=14, pady=(4, 2))
        self._trigger_type = tk.StringVar(value="LastPrice")
        trig_row = tk.Frame(parent, bg=C["panel"])
        trig_row.pack(fill="x", padx=18, pady=2)
        for val, lbl in [("LastPrice", "Last Price"), ("MarkPrice", "Mark Price")]:
            tk.Radiobutton(trig_row, text=lbl, variable=self._trigger_type, value=val,
                           bg=C["panel"], fg=C["text"], selectcolor=C["card"],
                           activebackground=C["panel"], activeforeground=C["text"],
                           font=FONT_LABEL, cursor="hand2").pack(side="left", padx=(0, 18))

        _sep(parent).pack(fill="x", padx=14, pady=6)

        # ── Execution mode ───────────────────────────────────────────────────
        _label(parent, "  Execution Mode", fg=C["blue"], font=FONT_BOLD).pack(
               anchor="w", padx=14, pady=(4, 2))
        self._exec_mode = tk.StringVar(value="direct")
        ex_row = tk.Frame(parent, bg=C["panel"])
        ex_row.pack(fill="x", padx=18, pady=2)
        for val, lbl in [("direct", "Direct (simplified)"),
                          ("full",   "Full (all safety checks)")]:
            tk.Radiobutton(ex_row, text=lbl, variable=self._exec_mode, value=val,
                           bg=C["panel"], fg=C["text"], selectcolor=C["card"],
                           activebackground=C["panel"], activeforeground=C["text"],
                           font=FONT_LABEL, cursor="hand2").pack(side="left", padx=(0, 18))

        _sep(parent).pack(fill="x", padx=14, pady=6)

        # ── Live price ───────────────────────────────────────────────────────
        lp_row = tk.Frame(parent, bg=C["panel"])
        lp_row.pack(fill="x", padx=14, pady=4)
        self._show_live = tk.BooleanVar(value=False)
        tk.Checkbutton(lp_row, text="Show Live Price",
                       variable=self._show_live,
                       command=self._toggle_live_price,
                       bg=C["panel"], fg=C["dim"],
                       selectcolor=C["card"],
                       activebackground=C["panel"], activeforeground=C["text"],
                       font=FONT_LABEL, cursor="hand2").pack(side="left")
        self._live_price_var = tk.StringVar(value="")
        self._live_lbl = tk.Label(lp_row, textvariable=self._live_price_var,
                                   bg=C["panel"], fg=C["cyan"],
                                   font=FONT_MONO_L)
        # Hidden by default; shown when checkbox ticked

        _sep(parent).pack(fill="x", padx=14, pady=6)

        # ── Place Trade + Reset buttons ──────────────────────────────────────
        btn_row = tk.Frame(parent, bg=C["panel"])
        btn_row.pack(fill="x", padx=14, pady=(4, 12))

        self._place_btn = _btn(btn_row, "  PLACE TRADE  ",
                               self._place_trade,
                               C["btn_place"], C["btn_place_h"],
                               font=("Segoe UI", 11, "bold"),
                               padx=28, pady=10)
        self._place_btn.pack(side="left")

        # Reset — restores original signal values, disabled until user edits
        self._reset_btn = _btn(btn_row, "Reset",
                               self._reset_to_signal,
                               C["btn_sec"], C["btn_sec_h"], C["yellow"],
                               font=FONT_SMALL, padx=10, pady=6)
        self._reset_btn.pack(side="left", padx=(8, 0))
        self._reset_btn.config(state="disabled")

        self._trade_status_var = tk.StringVar(value="")
        self._trade_status_lbl = tk.Label(
            btn_row, textvariable=self._trade_status_var,
            bg=C["panel"], fg=C["dim"], font=FONT_LABEL, wraplength=240, justify="left")
        self._trade_status_lbl.pack(side="left", padx=(12, 0))

    # -------------------------------------------------------------------------
    # Signal list management
    # -------------------------------------------------------------------------

    def _refresh_tv(self):
        """Redraw the Treeview from _signals dict."""
        # Remember selection
        sel_iid = self._tv.selection()

        self._tv.delete(*self._tv.get_children())
        for key, gs in sorted(self._signals.items(),
                               key=lambda x: -x[1].timestamp):
            chk  = "☑" if gs.checked else "☐"
            tag  = "long" if gs.sig.direction > 0 else "short"
            d    = "LONG" if gs.sig.direction > 0 else "SHORT"
            fp   = self._fmt_price(gs.sig.entry)
            iid  = "|".join(str(k) for k in key)
            self._tv.insert("", "end", iid=iid,
                            values=(chk, gs.sig.symbol, d, fp, gs.age_str),
                            tags=(tag,))

        # Restore selection without triggering field repopulation.
        # We set _suppress_populate=True, call selection_set (which fires
        # <<TreeviewSelect>> synchronously), then clear the flag.
        if sel_iid:
            try:
                self._suppress_populate = True
                self._tv.selection_set(sel_iid)
            except Exception:
                pass
            finally:
                self._suppress_populate = False

        n = len(self._signals)
        chk_n = sum(1 for g in self._signals.values() if g.checked)
        s = f"SIGNALS  ({n})"
        if chk_n:
            s += f"  [{chk_n} checked]"
        self._sig_count_var.set(s)

    def _on_tv_click(self, event):
        """Toggle checkbox if first column clicked; otherwise select."""
        region = self._tv.identify("region", event.x, event.y)
        col    = self._tv.identify("column", event.x, event.y)
        iid    = self._tv.identify("item", event.x, event.y)
        if not iid:
            return
        key = self._iid_to_key(iid)
        if key not in self._signals:
            return
        if col == "#1":   # checkbox column
            self._signals[key].checked = not self._signals[key].checked
            self._refresh_tv()

    def _on_signal_select(self, event):
        sel = self._tv.selection()
        if not sel:
            return
        key = self._iid_to_key(sel[0])
        if key not in self._signals:
            return

        # Programmatic selection restore (e.g. from _refresh_tv) — don't touch fields
        if self._suppress_populate:
            return

        different_signal = (key != self._selected_key)

        # Clicking the SAME signal while user has edited — preserve their changes
        if not different_signal and self._user_edited:
            return

        # Clicking a DIFFERENT signal — always repopulate regardless of edit state,
        # because the user is intentionally switching to a new signal.
        # force=True bypasses the _user_edited guard inside _populate_fields.
        self._selected_key = key
        self._populate_fields(self._signals[key].sig,
                              force=different_signal)

    def _iid_to_key(self, iid: str) -> tuple:
        parts = iid.split("|")
        return (parts[0], int(parts[1]), int(parts[2]))

    def _select_all(self):
        for gs in self._signals.values():
            gs.checked = True
        self._refresh_tv()

    def _clear_checks(self):
        for gs in self._signals.values():
            gs.checked = False
        self._refresh_tv()

    def _delete_checked(self):
        to_del = [k for k, g in self._signals.items() if g.checked]
        if not to_del:
            self._set_status("No signals checked for deletion.", C["yellow"])
            return
        for k in to_del:
            del self._signals[k]
        if self._selected_key in [k for k in to_del]:
            self._selected_key = None
            self._clear_fields()
        self._refresh_tv()
        self._set_status(f"Deleted {len(to_del)} signal(s).", C["dim"])

    # -------------------------------------------------------------------------
    # Field population & calculation
    # -------------------------------------------------------------------------

    def _populate_fields(self, sig: SignalInfo, force: bool = False):
        """
        Populate all trade parameter fields from a signal.
        Skipped if the user has already edited the fields (unless force=True).
        force=True is used by the Reset button.
        """
        if self._user_edited and not force:
            return

        self._populating = True   # suppress _mark_edited traces while we set values
        try:
            self._pair_var.set(sig.symbol)
            is_bull = sig.direction > 0
            self._dir_var.set("  LONG" if is_bull else "  SHORT")
            self._dir_lbl.config(fg=C["green"] if is_bull else C["red"])

            self._entry_var.set(self._fmt_price(sig.entry))
            self._sl_var.set(self._fmt_price(sig.sl))
            self._tp_var.set(self._fmt_price(sig.tp))
            self._order_value_var.set(
                str(self.cfg.get("bybit_manual_order_value", 10.0)))
            self._recalc_qty()

            # Update TradingView chart link
            from .telegram_notifier import _tv_url
            self._tv_url = _tv_url(sig.symbol, self.cfg.get("timeframe", "5m"))
            self._chart_link.config(text=f"Open {sig.symbol} on TradingView ->")

            # Clear edit flag and disable Reset button (fields now match signal)
            self._user_edited = False
            self._reset_btn.config(state="disabled")

            self._set_status(f"Selected {sig.symbol}  — review and place.", C["dim"])

            # Restart live price if shown
            if self._show_live.get():
                self._start_live_price(sig.symbol)
        finally:
            self._populating = False

    def _clear_fields(self):
        self._pair_var.set("— No signal selected —")
        self._dir_var.set("")
        self._dir_lbl.config(fg=C["dim"])
        self._tv_url = ""
        self._chart_link.config(text="")
        self._user_edited = False
        self._reset_btn.config(state="disabled")
        self._populating = True
        try:
            for v in (self._entry_var, self._sl_var, self._tp_var,
                      self._order_value_var, self._qty_var,
                      self._risk_var, self._rr_var):
                v.set("")
        finally:
            self._populating = False

    def _on_ov_mode_change(self):
        """Toggle entry editability and Fetch button based on OV mode."""
        mode = self._ov_mode.get()
        if mode == "manual":
            self._ov_entry.config(state="normal")
            self._ov_fetch_btn.pack_forget()
            self._ov_info.config(text="")
        else:
            self._ov_entry.config(state="readonly")
            self._ov_fetch_btn.pack(side="left", padx=(6, 0))
            self._ov_info.config(text="click Fetch", fg=C["dim"])

    def _fetch_order_value(self):
        """Fetch max leverage from Bybit and compute order value per selected mode."""
        if not self._selected_key or self._selected_key not in self._signals:
            self._set_status("Select a signal first.", C["yellow"])
            return
        sym  = self._signals[self._selected_key].sig.symbol
        mode = self._ov_mode.get()
        self._ov_fetch_btn.config(state="disabled", text="...")
        self._ov_info.config(text="fetching...", fg=C["dim"])

        def _do():
            try:
                max_lev = self._bybit.get_max_leverage(sym)
                return max_lev, None
            except Exception as e:
                return None, str(e)

        def _done(future):
            max_lev, err = future.result()
            self.root.after(0, lambda: self._on_ov_fetched(sym, max_lev, err, mode))

        self._pool.submit(_do).add_done_callback(_done)

    def _on_ov_fetched(self, sym: str, max_lev, err, mode: str):
        self._ov_fetch_btn.config(state="normal", text="Fetch")
        if err:
            self._ov_info.config(text="error", fg=C["red"])
            self._set_status(f"Leverage fetch failed: {err}", C["red"])
            return
        if mode == "max_lev":
            order_value = float(max_lev)
            label = f"max={max_lev}x → {order_value:.0f} USDT"
        else:   # half_lev
            import math as _math
            order_value = float(_math.floor(max_lev / 2))
            label = f"max={max_lev}x → ½={order_value:.0f} USDT"
        self._populating = True
        try:
            self._order_value_var.set(str(order_value))
        finally:
            self._populating = False
        self._ov_info.config(text=label, fg=C["yellow"])
        self._recalc_qty()
        self._set_status(f"{sym} order value set to {order_value:.0f} USDT ({mode})", C["green"])

    def _reset_to_signal(self):
        """Force-repopulate all fields from the original signal values."""
        if not self._selected_key or self._selected_key not in self._signals:
            self._set_status("No signal selected.", C["yellow"])
            return
        sig = self._signals[self._selected_key].sig
        self._populate_fields(sig, force=True)
        self._set_status(f"Fields reset to original signal values for {sig.symbol}.", C["dim"])

    def _open_chart(self, event=None):
        """Open the TradingView full-screen chart for the selected pair."""
        if self._tv_url:
            try:
                webbrowser.open(self._tv_url)
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")

    def _recalc_qty(self):
        """
        Auto-recalculate Qty from order_value / entry_price.
        Qty = order_value_usdt / entry_price (rounded to 3 dp as estimate).
        """
        try:
            entry       = float(self._entry_var.get())
            order_value = float(self._order_value_var.get() or
                                self.cfg.get("bybit_manual_order_value", 10.0))
            if entry <= 0 or order_value <= 0:
                return
            raw_qty = order_value / entry
            qty     = round(raw_qty, 3)
            self._qty_var.set(str(qty))

            # Margin info (for display only — actual margin set by Bybit's max leverage)
            sl_str = self._sl_var.get()
            if sl_str:
                sl    = float(sl_str)
                sl_pct = abs(entry - sl) / entry if entry > 0 else 0
                self._risk_var.set(
                    f"  order_val={order_value:.0f} USDT  "
                    f"sl~{sl_pct:.3%}")

            # TP RR
            try:
                tp   = float(self._tp_var.get())
                sl   = float(self._sl_var.get())
                tp_d = abs(entry - tp)
                sl_d = abs(entry - sl)
                rr   = tp_d / sl_d if sl_d > 0 else 0
                self._rr_var.set(f"  RR 1:{rr:.2f}")
            except ValueError:
                pass
        except (ValueError, ZeroDivisionError):
            pass

    @staticmethod
    def _fmt_price(p: float) -> str:
        if   p >= 10_000: return f"{p:.2f}"
        elif p >= 1:      return f"{p:.4f}"
        else:             return f"{p:.6f}"

    # -------------------------------------------------------------------------
    # Trade placement
    # -------------------------------------------------------------------------

    def _place_trade(self):
        if not self._selected_key or self._selected_key not in self._signals:
            self._set_status("Select a signal first.", C["yellow"])
            return

        gs  = self._signals[self._selected_key]
        sig = gs.sig

        # Read current field values
        try:
            entry       = float(self._entry_var.get())
            sl_price    = float(self._sl_var.get())
            tp_price    = float(self._tp_var.get())
            order_value = float(self._order_value_var.get())
            qty         = float(self._qty_var.get())
        except ValueError as e:
            self._set_status(f"Invalid value: {e}", C["red"])
            return

        if qty <= 0:
            self._set_status("Qty must be > 0", C["red"])
            return

        mode        = self._exec_mode.get()
        order_type  = self._order_type.get()
        trigger     = self._trigger_type.get()
        side        = "Buy" if sig.direction > 0 else "Sell"

        self._place_btn.config(state="disabled")
        self._set_status("Placing order...", C["yellow"])

        # Build a modified cfg for this specific order
        order_cfg = dict(self.cfg)
        order_cfg["bybit_order_type"]  = order_type
        order_cfg["bybit_sl_trigger"]  = trigger
        order_cfg["bybit_tp_trigger"]  = trigger

        def _do_place():
            try:
                # Step 1: Get max leverage and set it (always max for all trades)
                max_leverage = self._bybit.get_max_leverage(sig.symbol)
                self._bybit.set_max_leverage(sig.symbol, max_leverage)

                # Step 2: Round qty to Bybit precision
                rounded_qty = self._bybit.round_qty(qty, sig.symbol)
                if rounded_qty <= 0:
                    return None, "Qty rounded to 0 — check Bybit minOrderQty"

                # Step 3: Place
                if mode == "direct":
                    # Direct: no dedup / position / balance checks
                    link_id = f"MGUI_{sig.symbol}_{int(time.time())}"[:36]
                    result  = self._bybit.place_order(
                        symbol=sig.symbol, side=side, qty=rounded_qty,
                        sl_price=sl_price, tp_price=tp_price,
                        order_link_id=link_id, cfg=order_cfg,
                        entry_price=entry,
                    )
                    bybit_id = result.get("orderId", "")
                else:
                    # Full path via TradeExecutor (creates a new SignalInfo
                    # from modified values so all checks use user's prices)
                    from .trade_executor import TradeExecutor
                    executor = TradeExecutor(order_cfg, self.store)
                    from .models import SignalInfo as SI
                    modified_sig = SI(
                        symbol=sig.symbol, direction=sig.direction,
                        zone_src=sig.zone_src,
                        entry=entry, sl=sl_price, tp=tp_price,
                        sl_pips=sig.sl_pips, tp_pips=sig.tp_pips,
                        risk=abs(entry - sl_price),
                        lot_size=rounded_qty, lot_info=str(rounded_qty),
                        rr=sig.rr, bar_time=sig.bar_time,
                        active_zones=sig.active_zones,
                    )
                    res = executor.execute(modified_sig)
                    if res is None:
                        return None, "Skipped (dedup / guard)"
                    if res.get("failure"):
                        return None, res["failure"]
                    bybit_id = res["order_id"]

                # Record in trades.db for direct mode
                if mode == "direct":
                    self.store.record(
                        symbol=sig.symbol, direction=sig.direction,
                        bar_time=sig.bar_time, zone_src=sig.zone_src,
                        entry=entry, sl=sl_price, tp=tp_price,
                        qty=rounded_qty, notional=order_value,
                        risk_usd=order_value,
                        bybit_order_id=bybit_id,
                        order_link_id=f"MGUI_{sig.symbol}_{sig.bar_time}_{side[0]}",
                        status="PLACED",
                        note=f"manual|{mode}",
                    )
                return bybit_id, None

            except Exception as e:
                return None, str(e)

        def _on_done(future):
            bybit_id, err = future.result()
            self.root.after(0, lambda: self._after_place(
                sig, bybit_id, err, entry, sl_price, tp_price,
                qty, order_value, order_type, trigger
            ))

        fut = self._pool.submit(_do_place)
        fut.add_done_callback(_on_done)

    def _after_place(self, sig, bybit_id, err,
                     entry, sl, tp, qty, order_value, order_type, trigger):
        self._place_btn.config(state="normal")
        if err:
            self._set_status(f"FAILED: {err}", C["red"])
            logger.error(f"Manual trade failed: {err}")
        else:
            d   = "LONG" if sig.direction > 0 else "SHORT"
            msg = f"PLACED  {sig.symbol} {d}  {qty} contracts  ID: {bybit_id}"
            self._set_status(msg, C["green"])
            logger.info(f"Manual trade placed: {msg}")

            # Send Telegram notification
            try:
                from .models import SignalInfo as SI
                notif_sig = SI(
                    symbol=sig.symbol, direction=sig.direction,
                    zone_src=sig.zone_src,
                    entry=entry, sl=sl, tp=tp,
                    sl_pips=sig.sl_pips, tp_pips=sig.tp_pips,
                    risk=abs(entry - sl), lot_size=qty,
                    lot_info=str(qty), rr=sig.rr,
                    bar_time=sig.bar_time,
                    active_zones=sig.active_zones,
                )
                asyncio.run_coroutine_threadsafe(
                    self.notifier.send_trade_executed(
                        notif_sig, order_id=bybit_id,
                        lot_size=qty, order_value=order_value,
                        leverage=0,
                    ),
                    self.loop,
                )
            except Exception as e:
                logger.warning(f"Telegram notification after manual trade failed: {e}")

    # -------------------------------------------------------------------------
    # Live price
    # -------------------------------------------------------------------------

    def _toggle_live_price(self):
        if self._show_live.get():
            sym = self._pair_var.get()
            if sym and "—" not in sym:
                self._live_lbl.pack(side="left", padx=(12, 0))
                self._start_live_price(sym)
        else:
            self._live_stop.set()
            self._live_price_var.set("")
            self._live_lbl.pack_forget()

    def _start_live_price(self, symbol: str):
        self._live_stop.clear()
        t = threading.Thread(
            target=self._live_price_loop,
            args=(symbol,), daemon=True
        )
        t.start()

    def _live_price_loop(self, symbol: str):
        base = self.cfg.get("binance_base_url", "https://api1.binance.com")
        while not self._live_stop.is_set():
            try:
                r = requests.get(
                    f"{base}/api/v3/ticker/price",
                    params={"symbol": symbol}, timeout=5
                )
                if r.status_code == 200:
                    price = float(r.json().get("price", 0))
                    formatted = self._fmt_price(price)
                    self.root.after(0, lambda p=formatted: self._live_price_var.set(p))
            except Exception:
                pass
            for _ in range(50):   # sleep 5s in 0.1s increments for fast stop
                if self._live_stop.is_set():
                    return
                time.sleep(0.1)

    # -------------------------------------------------------------------------
    # Always on top
    # -------------------------------------------------------------------------

    def _toggle_topmost(self):
        self.root.attributes("-topmost", self._topmost.get())

    # -------------------------------------------------------------------------
    # Status helpers
    # -------------------------------------------------------------------------

    def _set_status(self, msg: str, color: str = None):
        self._trade_status_var.set(msg)
        if color:
            self._trade_status_lbl.config(fg=color)
        self._status_var.set(msg)

    # -------------------------------------------------------------------------
    # Queue polling — receives signals from the scanner thread
    # -------------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                sig: SignalInfo = self.signal_queue.get_nowait()
                key = (sig.symbol, sig.bar_time, sig.direction)
                if key not in self._signals:
                    self._signals[key] = GUISignal(sig=sig)
                    self._refresh_tv()
                    d = "LONG" if sig.direction > 0 else "SHORT"
                    self._set_status(
                        f"New signal: {sig.symbol} {d}  entry={self._fmt_price(sig.entry)}",
                        C["blue"]
                    )
        except queue.Empty:
            pass
        # Refresh ages every 30 s
        if int(time.time()) % 30 == 0:
            self._refresh_tv()
        self.root.after(500, self._poll_queue)

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------

    def run(self):
        self.root.after(500, self._poll_queue)
        try:
            self.root.mainloop()
        finally:
            self._live_stop.set()
            self._pool.shutdown(wait=False)
