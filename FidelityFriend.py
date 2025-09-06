import os
import io
import re
import time
import math
import platform
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib

from zoneinfo import ZoneInfo
from collections import deque
import threading

# --- Optional: filesystem watch (falls back to polling if not installed) ---
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    _WATCHDOG_OK = True
except Exception:
    _WATCHDOG_OK = False
    Observer = None
    FileSystemEventHandler = object

DEFAULT_CSV_PATH = r"C:\Users\an\Desktop\Trading\Temp\temp.csv"

PRICE_RE = re.compile(r"\$(\d+(?:\.\d+)?)")
DT_RE = re.compile(r"(\d{1,2}:\d{2}:\d{2})\s*(AM|PM)\s+(\d{2}/\d{2}/\d{4})", re.I)


matplotlib.use(
    "Agg"
)  # headless backend to avoid TkAgg blinks; we'll embed with canvas widget
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

REFRESH_INTERVAL_MS = 1000  # auto-refresh cadence
DARK_BG = "#0f1115"
PANEL_BG = "#151821"
CARD_BG = "#1d2230"
TEXT_PRIMARY = "#ffffff"
TEXT_MUTED = "#a9b0c3"
ACCENT = "#4cc9f0"
GREEN = "#00d97e"
RED = "#ff5470"
GRID = "#2a3041"


class TradingCompanionApp:
    def __init__(self):

        # --- Mini Mode state ---
        self.mini_mode = False
        self.mini_win = None
        self._mini_last_pos = None  # remember last geometry (e.g., "+1200+30")
        self.MINI_SYMBOL_ROWS = None  # show all rows in mini mode
        self.MINI_ROW_HEIGHT = 20  # matches Treeview rowheight style
        self.MINI_MAX_SYMBOL_ROWS = 200  # cap if you ever want one (or set to None)
        self.MINI_BASE_HEIGHT = 72  # ribbon + padding (adjust if you changed fonts)
        self.MINI_HEADER_HEIGHT = 24  # Treeview header height (approx)
        self.MINI_VERTICAL_PADDING = 16  # extra top/bottom padding under table
        # holds references to mini-mode stat value labels
        self._mini_blocks = {}
        self._mini_needs_data_repaint = False  # set True when new stats arrive
        self._mini_last_n_rows = None

        # --- Tk root & theme ---
        self.root = tk.Tk()
        self.root.title("Fidelity Day Trading Companion")
        self.root.geometry("900x540")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.minsize(700, 440)
        self.root.configure(bg=DARK_BG)
        try:
            self.root.iconbitmap("")  # harmless; keep empty
        except:
            pass

        # ttk theme
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            ".", background=DARK_BG, foreground=TEXT_PRIMARY, fieldbackground=CARD_BG
        )
        style.configure("TFrame", background=DARK_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("Muted.TLabel", foreground=TEXT_MUTED, background=DARK_BG)
        style.configure("CardMuted.TLabel", foreground=TEXT_MUTED, background=CARD_BG)
        style.configure("Card.TLabel", background=CARD_BG, foreground=TEXT_PRIMARY)
        style.configure("Green.TLabel", foreground=GREEN, background=CARD_BG)
        style.configure("Red.TLabel", foreground=RED, background=CARD_BG)
        style.configure("Accent.TLabel", foreground=ACCENT, background=CARD_BG)
        style.configure(
            "Header.TLabel", font=("Segoe UI", 14, "bold"), background=DARK_BG
        )
        style.configure("Big.TLabel", font=("Segoe UI", 28, "bold"), background=DARK_BG)
        style.configure(
            "StatValue.TLabel", font=("Segoe UI", 18, "bold"), background=CARD_BG
        )
        style.configure(
            "StatLabel.TLabel",
            font=("Segoe UI", 10),
            foreground=TEXT_MUTED,
            background=CARD_BG,
        )
        style.configure(
            "TButton", background=PANEL_BG, foreground=TEXT_PRIMARY, padding=6
        )
        style.map("TButton", background=[("active", "#222739")])

        style.configure(
            "Treeview",
            background="#1e2230",
            fieldbackground="#1e2230",
            foreground=TEXT_PRIMARY,
        )
        style.map(
            "Treeview",
            background=[("selected", "#2e3650")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Treeview.Heading",
            background="#2a3041",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
        )

        style.configure("CardGreen.TFrame", background="#152820")  # faint green tint
        style.configure("CardRed.TFrame", background="#281515")  # faint red tint

        # Styles used by mini mode table (defined once here)
        style.configure(
            "Mini.Treeview",
            background=CARD_BG,
            fieldbackground=CARD_BG,
            foreground=TEXT_PRIMARY,
            rowheight=self.MINI_ROW_HEIGHT,  # keep row height in sync
            borderwidth=0,
        )
        style.configure(
            "Mini.Treeview.Heading",
            background=DARK_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Mini.Treeview",
            background=[("selected", CARD_BG)],
            foreground=[("selected", TEXT_PRIMARY)],
        )

        # --- State ---
        self.csv_path = DEFAULT_CSV_PATH
        self.last_mtime = 0.0

        # --- Watchdog state ---
        self._watch_observer = None
        self._watch_handler = None
        self._watch_dir = None
        self._watch_file = None
        self._watch_pending_id = None  # debounce timer id
        self._watchdog_active = False

        # Refresh orchestration
        self._refresh_in_progress = False  # True while a parse/build thread is running
        self._pending_refresh_id = None  # after() id for debounced UI scheduling
        self._parse_coalesce = (
            False  # set True if a refresh was requested while one is running
        )

        self.trades_df = pd.DataFrame()
        self.completed_trades = (
            []
        )  # list of dicts (symbol, entry_price, exit_price, quantity, pnl, timestamp, side)
        self.current_positions = {}
        self.symbol_stats = {}
        self.stats = {
            "gross_pnl": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }

        # --- Layout ---
        self._build_layout()

        # Initial paint & loop
        self._refresh_from_disk(initial=True)

        # Start watchdog (if installed) for the current file
        if _WATCHDOG_OK and os.path.exists(self.csv_path):
            self._install_file_watch(self.csv_path)

        self._tick_clock()
        self._monitor_file()  # keep as fallback; it becomes a no-op if watchdog is active
        self.root.mainloop()

    # ---------------- UI BUILD ----------------
    def _build_layout(self):

        # Content: left stats + symbols, right chart
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        # Fix left column width so the right chart gets the rest
        left.configure(width=340)
        left.pack_propagate(False)  # don't auto-expand left to its children

        self.clock_frame = tk.Frame(left, bg=self.root.cget("bg"))
        self.clock_frame.pack(anchor="n", pady=(0, 6), fill=tk.X)

        self.clock_digits = tk.Label(
            self.clock_frame,
            text="",
            font=("Segoe UI", 44, "bold"),
            fg="#FFD700",
            bg=self.root.cget("bg"),
        )
        self.clock_ampm = tk.Label(
            self.clock_frame,
            text="",
            font=("Segoe UI", 16, "bold"),  # ← make AM/PM bold
            fg="#FFD700",
            bg=self.root.cget("bg"),
        )

        # Two spacer columns on the sides; cluster lives in the middle
        self.clock_frame.grid_columnconfigure(0, weight=1)  # left spacer
        self.clock_frame.grid_columnconfigure(3, weight=1)  # right spacer

        # Place digits and AM/PM in the middle
        self.clock_digits.grid(row=0, column=1, sticky="e")  # right-align digits
        self.clock_ampm.grid(
            row=0, column=2, sticky="w", padx=(6, 0)
        )  # left-align AM/PM

        # Keep vertical centering call
        self._align_ampm_vertically()
        self.root.after(50, self._align_ampm_vertically)

        # --- Stat tiles (2×2 grid)
        cards = ttk.Frame(left)
        cards.pack(fill=tk.X, padx=8, pady=(0, 6))

        # Make two equal-width columns, allow row growth
        cards.grid_columnconfigure(0, weight=1, uniform="cards")
        cards.grid_columnconfigure(1, weight=1, uniform="cards")

        self.card_gross = self._make_stat_card(cards, "P/L", "$0.00", color="green")
        self.card_trades = self._make_stat_card(cards, "Trades", "0")
        self.card_wr = self._make_stat_card(cards, "WR", "0.0%")
        self.card_pf = self._make_stat_card(
            cards, "PF", "0.00"
        )  # renamed for compactness

        # Place them in a 2×2 grid (wraps automatically)
        for idx, card in enumerate(
            [self.card_gross, self.card_trades, self.card_wr, self.card_pf]
        ):
            r, c = divmod(idx, 2)
            card["frame"].grid(row=r, column=c, sticky="nsew", padx=3, pady=3)

        # Symbols table
        sym_panel = ttk.Frame(left)
        sym_panel.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        sym_card = ttk.Frame(sym_panel, style="Card.TFrame")
        sym_card.pack(fill=tk.BOTH, expand=False)

        cols = ("symbol", "pnl", "trades", "wr")
        self.tree = ttk.Treeview(
            sym_card, columns=cols, show="headings", selectmode="browse"
        )

        self.tree.heading("symbol", text="Symbol")
        self.tree.heading("pnl", text="P/L ($)")
        self.tree.heading("trades", text="Trades")
        self.tree.heading("wr", text="Win %")
        self.tree.column("symbol", width=90, anchor="w", stretch=False)
        self.tree.column("pnl", width=100, anchor="e", stretch=False)
        self.tree.column("trades", width=70, anchor="center", stretch=False)
        self.tree.column("wr", width=70, anchor="center", stretch=False)

        # create scrollbars first
        vsb = ttk.Scrollbar(sym_card, orient="vertical", command=self.tree.yview)

        # wire tree <-> scrollbars
        self.tree.configure(yscrollcommand=vsb.set)

        # pack widgets
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=6)
        vsb.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6), pady=6)

        # tags for row coloring used in _update_symbol_table
        self.tree.tag_configure("pnl_pos", foreground=GREEN)
        self.tree.tag_configure("pnl_neg", foreground=RED)
        self.tree.tag_configure("pnl_neu", foreground=TEXT_PRIMARY)

        # track rows by symbol (used by diffing updater)
        self._tree_iids = {}  # symbol -> iid

        # Chart area
        right = ttk.Frame(content)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        chart_card = ttk.Frame(right, style="Card.TFrame")
        chart_card.pack(fill=tk.BOTH, expand=True)

        # Let Matplotlib manage margins tightly
        self.fig = Figure(
            figsize=(6, 4), dpi=120, facecolor=CARD_BG, constrained_layout=True
        )
        self.ax = self.fig.add_subplot(111, facecolor=CARD_BG)
        # Reusable x-axis locator/formatter (avoid recreating every repaint)
        self._x_locator = mdates.MinuteLocator(interval=15)
        self._x_formatter = FuncFormatter(
            lambda x, pos: mdates.num2date(x).strftime("%I:%M").lstrip("0")
        )

        # Persistent line + holders for the area fills
        self._line_cum = self.ax.plot(
            [], [], linewidth=2, color=TEXT_PRIMARY, zorder=3
        )[0]
        self._fill_polys = []
        # --- HOD/LOD annotation handles (created lazily) ---
        self._hod_marker = None
        self._lod_marker = None
        self._hod_label = None
        self._lod_label = None
        # Set all static axis bits once
        self._apply_static_axis_style()

        # Put the old label into the plot title with minimal padding
        try:
            self.ax.set_title("Running P/L Curve", pad=4)
            # Nudge top/bottom margins tighter than the default constrained layout
            try:
                self.fig.set_constrained_layout_pads(
                    h_pad=0.02, w_pad=0.02, hspace=0.02, wspace=0.02
                )
            except Exception:
                pass
        except Exception:
            pass  # safe on older Matplotlib versions

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_card)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=CARD_BG, highlightthickness=0, bd=0)
        widget.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # --- Controls (anchored at bottom of LEFT column) ---
        controls = ttk.Frame(left)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

        # 1) CSV path row (one full line)
        path_frame = ttk.Frame(controls)
        path_frame.pack(fill=tk.X)

        ttk.Label(path_frame, text="CSV path:", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )

        self.path_var = tk.StringVar(value=self.csv_path)
        path_entry = ttk.Entry(path_frame, textvariable=self.path_var)
        path_entry.grid(row=0, column=1, sticky="ew")
        path_entry.bind(
            "<Return>", lambda e: self._request_data_refresh(reason="enter")
        )

        ttk.Button(path_frame, text="Browse…", command=self._choose_path).grid(
            row=0, column=2, sticky="w", padx=(6, 0)
        )
        path_frame.grid_columnconfigure(1, weight=1)  # make entry stretch

        # 2) Buttons (Copy + Mini)
        btns = ttk.Frame(controls)
        btns.pack(fill=tk.X, pady=(6, 0))

        self.btn_copy = ttk.Button(
            btns, text="🗐 Copy Chart", command=self._copy_chart_to_clipboard
        )
        self.btn_mini = ttk.Button(
            btns, text="▣ Mini Mode", command=self._toggle_mini_mode
        )

        self.btn_copy.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self.btn_mini.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        btns.grid_columnconfigure(0, weight=1, uniform="btns")
        btns.grid_columnconfigure(1, weight=1, uniform="btns")

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(
            self.root, textvariable=self.status_var, style="Muted.TLabel", anchor="w"
        )
        status.pack(fill=tk.X, padx=8, pady=(0, 6))

    def _align_ampm_vertically(self):
        """Center the AM/PM label vertically relative to the big clock digits."""
        try:
            import tkinter.font as tkfont

            dfont = tkfont.Font(font=self.clock_digits["font"])
            afont = tkfont.Font(font=self.clock_ampm["font"])
            d_h = dfont.metrics("ascent") + dfont.metrics("descent")
            a_h = afont.metrics("ascent") + afont.metrics("descent")
            offset = max(0, (d_h - a_h) // 2)
            # push AM/PM down by 'offset' pixels so it sits centered to the digits
            self.clock_ampm.grid_configure(pady=(offset, 0))
            self.clock_digits.grid_configure(pady=0)
        except Exception:
            pass

    def _make_stat_card(self, parent, title, value, color=None):
        # tighter outer padding
        frame = ttk.Frame(parent, style="Card.TFrame", padding=(4, 2))

        # row fills width, but we’ll center the content group inside it
        row = tk.Frame(frame, bg=CARD_BG)
        row.pack(fill=tk.X, expand=True)

        # a small content group that we center
        content = tk.Frame(row, bg=CARD_BG)
        content.pack(anchor="center")  # <— centers the label+number as a unit

        # Small, non-bold label
        title_label = tk.Label(
            content,
            text=f"{title}: ",
            font=("Segoe UI", 10),  # small, NOT bold
            fg=TEXT_MUTED,
            bg=CARD_BG,
        )
        title_label.pack(side=tk.LEFT)

        # Bigger, bold number (only this is colorized)
        value_var = tk.StringVar(value=value)
        value_label = tk.Label(
            content,
            textvariable=value_var,
            font=("Segoe UI", 16, "bold"),  # ← bigger number
            fg=(color if color else TEXT_PRIMARY),
            bg=CARD_BG,
        )
        value_label.pack(side=tk.LEFT)

        return {
            "frame": frame,
            "title_label": title_label,
            "value_label": value_label,
            "value_var": value_var,
        }

    def _set_card(self, card, title, value, color=None):
        card["title_label"].configure(text=f"{title}: ")
        card["value_var"].set(value)
        card["value_label"].configure(fg=(color if color is not None else TEXT_PRIMARY))

    # ---------------- CLOCK & CONTROLS ----------------
    def _tick_clock(self):
        now = datetime.now()  # or datetime.now(TIMEZONE_PACIFIC) if you use tz
        digits = now.strftime("%I:%M:%S").lstrip("0")  # big part
        ampm = now.strftime("%p")  # small part

        # Two-label clock (preferred)
        if hasattr(self, "clock_digits") and hasattr(self, "clock_ampm"):
            self.clock_digits.config(text=digits)
            self.clock_ampm.config(text=ampm)
        # Fallback for old single-label clock (safe to keep)
        elif hasattr(self, "clock_label"):
            self.clock_label.config(text=f"{digits} {ampm}")

        # Mini window clock (leave as-is)
        if getattr(self, "mini_mode", False) and getattr(self, "mini_vars", None):
            try:
                self.mini_vars["clock"].set(f"{digits} {ampm}")
            except Exception:
                pass

        # One-shot mini repaint if queued
        if getattr(self, "_mini_needs_data_repaint", False):
            self._mini_needs_data_repaint = False
            self._refresh_mini()

        # Next tick
        self.root.after(500, self._tick_clock)

    def _choose_path(self):
        chosen = filedialog.askopenfilename(
            title="Select Fidelity CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if chosen:
            self.path_var.set(chosen)
            self.csv_path = chosen
            if _WATCHDOG_OK and os.path.exists(self.csv_path):
                self._install_file_watch(self.csv_path)
            # run the background/ debounced path
            self._request_data_refresh(reason="path chosen")

    # ---------------- FILE WATCH ----------------
    def _monitor_file(self):

        # If watchdog is active, just keep the timer alive and return
        if self._watchdog_active:
            self.root.after(REFRESH_INTERVAL_MS, self._monitor_file)
            return

        # Fallback: simple mtime polling
        try:
            if os.path.exists(self.csv_path):
                mtime = os.path.getmtime(self.csv_path)
                if mtime != self.last_mtime:
                    self.last_mtime = mtime
                    self._request_data_refresh(reason="polling")

        except Exception as e:
            self._set_status(f"File monitor error: {e}")

        self.root.after(REFRESH_INTERVAL_MS, self._monitor_file)

    def _install_file_watch(self, path: str):
        """Install a watchdog observer for `path` (or switch to a new file)."""
        if not _WATCHDOG_OK:
            self._set_status("watchdog not installed — using 1s polling fallback")
            self._watchdog_active = False
            return

        try:
            # Clean up any previous observer
            self._uninstall_file_watch()
        except Exception:
            pass

        if not path:
            self._watchdog_active = False
            return

        watch_file = os.path.abspath(path)
        watch_dir = os.path.dirname(watch_file) or "."

        # Handler that debounces and refreshes on changes to the target file
        app = self

        class _CsvWatchHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.is_directory:
                    return
                if os.path.abspath(event.src_path) == app._watch_file:
                    app._debounce_watchdog_refresh("modified")

            def on_created(self, event):
                if event.is_directory:
                    return
                # file recreated by some tools
                if os.path.abspath(event.src_path) == app._watch_file:
                    app._debounce_watchdog_refresh("created")

            def on_moved(self, event):
                # editors often write to temp and move over the original
                dest = os.path.abspath(getattr(event, "dest_path", event.src_path))
                src = os.path.abspath(event.src_path)
                if src == app._watch_file or dest == app._watch_file:
                    app._watch_file = dest  # track the new path if it changed
                    app._debounce_watchdog_refresh("moved")

            def on_deleted(self, event):
                if os.path.abspath(event.src_path) == app._watch_file:
                    # stay armed; file might be recreated
                    app._debounce_watchdog_refresh("deleted")

        self._watch_file = watch_file
        self._watch_dir = watch_dir
        self._watch_handler = _CsvWatchHandler()
        self._watch_observer = Observer()
        self._watch_observer.schedule(
            self._watch_handler, self._watch_dir, recursive=False
        )
        self._watch_observer.start()
        self._watchdog_active = True
        try:
            self._set_status(f"Watching file changes via watchdog: {self._watch_file}")
        except Exception:
            pass

    def _uninstall_file_watch(self):
        """Stop the watchdog observer if running."""
        try:
            if self._watch_observer is not None:
                self._watch_observer.stop()
                self._watch_observer.join(timeout=1.0)
        finally:
            self._watch_observer = None
            self._watch_handler = None
            self._watch_dir = None
            # keep _watch_file so we can keep path in the UI
            self._watchdog_active = False

    def _debounce_watchdog_refresh(self, reason: str = ""):
        """Coalesce bursts of FS events into a single refresh."""
        # cancel a pending timer if any
        if self._watch_pending_id is not None:
            try:
                self.root.after_cancel(self._watch_pending_id)
            except Exception:
                pass
            self._watch_pending_id = None

        # schedule a single refresh shortly after the last event
        def _go():
            self._watch_pending_id = None
            self._request_data_refresh(reason="watchdog")

        self._watch_pending_id = self.root.after(250, _go)

    def _request_data_refresh(self, reason: str = ""):
        """Coalesce refresh requests and run parsing off the UI thread."""
        # Clear any pending debounce callback id (we're about to run)
        if self._pending_refresh_id is not None:
            try:
                self.root.after_cancel(self._pending_refresh_id)
            except Exception:
                pass
            self._pending_refresh_id = None

        # If a parse is already running, coalesce
        if self._refresh_in_progress:
            self._parse_coalesce = True
            return

        # Start a new worker
        self._refresh_in_progress = True
        self._parse_coalesce = False

        def worker():
            snapshot = None
            err = None
            try:
                # Build a fresh snapshot entirely off the UI thread
                if not self.csv_path or not os.path.exists(self.csv_path):
                    err = "CSV not found. Set the correct path."
                else:
                    df = self._parse_fidelity_csv(self.csv_path)
                    completed, positions = self._build_trades(df)
                    stats = self._compute_stats(completed)
                    sym_stats = self._compute_symbol_stats(completed)
                    snapshot = (df, completed, positions, stats, sym_stats)
            except Exception as e:
                err = f"{e}"

            # Marshal the result back to the UI thread
            def apply():
                # Apply results (or error) on the main thread
                if err:
                    self._set_status(
                        f"Parse error: {err}" if "error" in err.lower() else err
                    )
                    # Paint empty if we truly have nothing
                    if not os.path.exists(self.csv_path):
                        self._paint_empty_chart()
                        self._update_stat_cards()
                        self._update_symbol_table({})
                else:
                    df, completed, positions, stats, sym_stats = snapshot
                    self.trades_df = df
                    self.completed_trades = completed
                    self.current_positions = positions
                    self.stats = stats
                    self.symbol_stats = sym_stats

                    self._update_stat_cards()
                    self._update_symbol_table(self.symbol_stats)
                    self._paint_chart(self.completed_trades)

                    try:
                        when = datetime.fromtimestamp(
                            os.path.getmtime(self.csv_path)
                        ).strftime("%I:%M:%S %p")
                        self._set_status(
                            f"Updated from CSV ({when})"
                            + (f" — {reason}" if reason else "")
                        )
                    except Exception:
                        self._set_status("Updated from CSV")

                    # Ask mini to repaint data once on the next clock tick
                    self._mini_needs_data_repaint = True

                # Mark complete and, if another request came in, run again
                self._refresh_in_progress = False
                if self._parse_coalesce:
                    # Clear the coalesce flag and start again
                    self._parse_coalesce = False
                    # Short defer so the UI can breathe
                    self.root.after(
                        50, lambda: self._request_data_refresh(reason="coalesced")
                    )

            self.root.after(0, apply)

        t = threading.Thread(target=worker, name="ParseThread", daemon=True)
        t.start()

    # ---------------- DATA FLOW ----------------
    def _refresh_from_disk(self, force=False, initial=False):
        if not self.csv_path or not os.path.exists(self.csv_path):
            self._set_status("CSV not found. Set the correct path.")
            self._paint_empty_chart()
            self._update_stat_cards()
            self._update_symbol_table({})
            return

        try:
            df = self._parse_fidelity_csv(self.csv_path)
        except Exception as e:
            self._set_status(f"Parse error: {e}")
            return

        self.trades_df = df
        self.completed_trades, self.current_positions = self._build_trades(df)
        self.stats = self._compute_stats(self.completed_trades)
        self.symbol_stats = self._compute_symbol_stats(self.completed_trades)

        self._update_stat_cards()
        self._update_symbol_table(self.symbol_stats)
        self._paint_chart(self.completed_trades)
        when = datetime.fromtimestamp(os.path.getmtime(self.csv_path)).strftime(
            "%I:%M:%S %p"
        )
        self._set_status(f"Updated from CSV ({when})")

        # Ask mini to repaint data once on the next clock tick (decoupled from 500ms cycle)
        self._mini_needs_data_repaint = True

    # ---------------- CSV PARSER ----------------
    def _parse_fidelity_csv(self, file_path: str) -> pd.DataFrame:
        """
        Parse Fidelity export efficiently:

        - Find header row, then pandas reads from there (no manual StringIO copy).
        - Read only the columns we need (usecols).
        - Vectorized extraction:
            * Price from Status via regex
            * Side from Trade Description via str.contains
            * Quantity via to_numeric (commas handled)
        - Parse 'Order Time' (Eastern) -> convert to Pacific using zoneinfo (DST-safe),
        then drop tz to keep the rest of the app's naive datetimes.
        """
        # 1) Find the header line index once (no full read into memory)
        header_idx = None
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                if ("Symbol" in line) and ("Status" in line):
                    header_idx = i
                    break
        if header_idx is None:
            return pd.DataFrame(
                columns=["Symbol", "Price", "Side", "Quantity", "OrderDateTime"]
            )

        # 2) Read only required columns from disk starting at header
        required_cols = [
            "Symbol",
            "Status",
            "Trade Description",
            "Quantity",
            "Order Time",
        ]
        df = pd.read_csv(
            file_path,
            skiprows=header_idx,  # header row becomes the next line
            header=0,
            usecols=lambda c: c in required_cols,  # keep only what we use
            thousands=",",  # handle "1,000"
            dtype={  # keep as strings for vector ops; Quantity will be parsed later
                "Symbol": "string",
                "Status": "string",
                "Trade Description": "string",
                "Quantity": "string",
                "Order Time": "string",
            },
            engine="c",  # fastest parser
            low_memory=False,
        )

        if df.empty:
            return pd.DataFrame(
                columns=["Symbol", "Price", "Side", "Quantity", "OrderDateTime"]
            )

        # 3) Drop rows without Symbol
        df = df.dropna(subset=["Symbol"]).copy()

        # 4) Vectorized PRICE from Status: "@ $12.34"
        #    Use the precompiled PRICE_RE and str.extract
        price_str = df["Status"].astype("string").fillna("")
        df["Price"] = price_str.str.extract(PRICE_RE, expand=False).astype("float64")

        # 5) Vectorized SIDE from Trade Description
        desc = df["Trade Description"].astype("string").fillna("").str.lower()
        is_buy = desc.str.contains("buy", na=False)
        is_sell = desc.str.contains("sell", na=False)
        df["Side"] = np.where(is_buy, "Buy", np.where(is_sell, "Sell", pd.NA))

        # 6) Vectorized QUANTITY
        qty = pd.to_numeric(
            df["Quantity"].str.replace(",", "", regex=False), errors="coerce"
        )
        df["Quantity"] = qty.fillna(0).astype("int64")

        # 7) Vectorized ORDER TIME (Eastern -> Pacific, DST-safe)
        #    Normalize newlines/spaces, extract components, parse, tz-localize & convert.
        ot = (
            df["Order Time"]
            .astype("string")
            .fillna("")
            .str.replace("\n", " ", regex=False)
            .str.strip()
        )
        parts = ot.str.extract(DT_RE)  # cols: [time, ampm, date]
        has_all = parts.notna().all(axis=1)

        dt_str = (parts[2] + " " + parts[0] + " " + parts[1].str.upper()).where(
            has_all, None
        )
        dt_naive = pd.to_datetime(
            dt_str, format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
        )
        # Localize to Eastern, convert to Pacific, then drop tz to keep naive datetimes downstream
        dt_eastern = dt_naive.dt.tz_localize(
            ZoneInfo("America/New_York"), nonexistent="NaT", ambiguous="NaT"
        )
        dt_pacific = dt_eastern.dt.tz_convert(ZoneInfo("America/Los_Angeles"))
        df["OrderDateTime"] = dt_pacific.dt.tz_localize(None)

        # 8) Final cleanup and ordering
        df = df.dropna(subset=["Price", "Side", "OrderDateTime"]).copy()
        df = df[df["Quantity"] > 0]
        df = df.sort_values("OrderDateTime", kind="mergesort", ignore_index=True)

        return df[["Symbol", "Price", "Side", "Quantity", "OrderDateTime"]]

    # ---------------- P&L CONSTRUCTION ----------------
    def _build_trades(self, df: pd.DataFrame):
        """
        Build realized P&L using FIFO lots per symbol.

        Input df columns (from _parse_fidelity_csv):
        Symbol | Price | Side | Quantity | OrderDateTime

        Returns:
        completed_trades: list[dict]  # (symbol, entry_price, exit_price, quantity, pnl, timestamp, side)
        current_positions: dict[str, dict]  # summary of open lots (for future use/UI)
        """
        if df.empty:
            return [], {}

        # Per-symbol open lots: deque of {"side": "Long"/"Short", "qty": int, "price": float}
        open_lots: dict[str, deque] = {}
        completed: list[dict] = []

        # We’ll also keep a simple summary for current_positions
        positions_summary: dict[str, dict] = {}

        # Ensure chronological order (OrderDateTime is already sorted in parser, but be safe)
        df = df.sort_values("OrderDateTime", kind="mergesort")

        # NEW
        for row in df.itertuples(index=False):
            sym = row.Symbol
            px = float(row.Price)
            side = str(row.Side)
            qty = int(row.Quantity)
            ts = row.OrderDateTime

            if qty <= 0 or side not in ("Buy", "Sell"):
                continue

            # Init per-symbol lot queue
            q = open_lots.setdefault(sym, deque())

            if side == "Buy":
                # Buys CLOSE Short lots first (FIFO), then OPEN Long for any remainder
                remain = qty
                # Close shorts
                while remain > 0 and q and q[0]["side"] == "Short":
                    lot = q[0]
                    take = min(remain, lot["qty"])
                    # Closing short: entry=lot.price (shorted at), exit=px (buy cover)
                    pnl = (lot["price"] - px) * take
                    completed.append(
                        {
                            "symbol": sym,
                            "entry_price": float(lot["price"]),
                            "exit_price": px,
                            "quantity": int(take),
                            "pnl": float(pnl),
                            "timestamp": ts,
                            "side": "Short",
                        }
                    )
                    lot["qty"] -= take
                    remain -= take
                    if lot["qty"] == 0:
                        q.popleft()
                # Any remaining becomes a new Long lot
                if remain > 0:
                    q.append({"side": "Long", "qty": int(remain), "price": px})

            else:  # side == "Sell"
                # Sells CLOSE Long lots first (FIFO), then OPEN Short for any remainder
                remain = qty
                # Close longs
                while remain > 0 and q and q[0]["side"] == "Long":
                    lot = q[0]
                    take = min(remain, lot["qty"])
                    # Closing long: entry=lot.price (bought at), exit=px (sell)
                    pnl = (px - lot["price"]) * take
                    completed.append(
                        {
                            "symbol": sym,
                            "entry_price": float(lot["price"]),
                            "exit_price": px,
                            "quantity": int(take),
                            "pnl": float(pnl),
                            "timestamp": ts,
                            "side": "Long",
                        }
                    )
                    lot["qty"] -= take
                    remain -= take
                    if lot["qty"] == 0:
                        q.popleft()
                # Any remaining becomes a new Short lot
                if remain > 0:
                    q.append({"side": "Short", "qty": int(remain), "price": px})

        # Build a lightweight summary of current open positions (not used elsewhere yet)
        for sym, q in open_lots.items():
            if not q:
                continue
            # Compute net exposure and a simple weighted average by side
            long_qty = sum(l["qty"] for l in q if l["side"] == "Long")
            short_qty = sum(l["qty"] for l in q if l["side"] == "Short")

            # Weighted averages (guard divide-by-zero)
            def wavg(side_name):
                lots = [l for l in q if l["side"] == side_name]
                tot = sum(l["qty"] for l in lots)
                if tot == 0:
                    return 0.0
                return sum(l["price"] * l["qty"] for l in lots) / tot

            positions_summary[sym] = {
                "long_qty": int(long_qty),
                "long_avg": float(wavg("Long")),
                "short_qty": int(short_qty),
                "short_avg": float(wavg("Short")),
            }

        return completed, positions_summary

    def _compute_stats(self, completed):
        if not completed:
            return {
                "gross_pnl": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
            }
        total = sum(x["pnl"] for x in completed)
        wins = [x for x in completed if x["pnl"] > 0]
        losses = [x for x in completed if x["pnl"] < 0]
        gross_profit = sum(x["pnl"] for x in wins)
        gross_loss = abs(sum(x["pnl"] for x in losses))
        pf = (
            (gross_profit / gross_loss)
            if gross_loss > 0
            else (math.inf if gross_profit > 0 else 0.0)
        )
        wr = (len(wins) / len(completed) * 100.0) if completed else 0.0
        return {
            "gross_pnl": total,
            "total_trades": len(completed),
            "win_rate": wr,
            "profit_factor": pf,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }

    def _compute_symbol_stats(self, completed):
        out = {}
        for tr in completed:
            sym = tr["symbol"]
            out.setdefault(sym, {"pnl": 0.0, "trades": 0, "wins": 0})
            out[sym]["pnl"] += tr["pnl"]
            out[sym]["trades"] += 1
            if tr["pnl"] > 0:
                out[sym]["wins"] += 1
        for sym, s in out.items():
            s["win_rate"] = (s["wins"] / s["trades"] * 100.0) if s["trades"] else 0.0
        return out

    # ---------------- UI UPDATE ----------------
    def _update_stat_cards(self):
        import math

        # Read current stats safely
        pnl = float(self.stats.get("gross_pnl", 0.0) or 0.0)
        total_trades = int(self.stats.get("total_trades", 0) or 0)
        win_rate = float(self.stats.get("win_rate", 0.0) or 0.0)
        pf_raw = self.stats.get("profit_factor", None)

        # ---------- Profit Factor text (with edge cases) ----------
        if pf_raw is None:
            pf_text = "—"
        else:
            try:
                pf_val = float(pf_raw)
                if math.isnan(pf_val):
                    pf_text = "—"
                elif math.isinf(pf_val):
                    pf_text = "∞"
                else:
                    pf_text = f"{pf_val:.2f}"
            except Exception:
                pf_text = "—"

        # ---------- Update the four tiles (one line each) ----------
        self._set_card(
            self.card_gross, "P/L", f"${pnl:,.2f}", color=(GREEN if pnl >= 0 else RED)
        )
        self._set_card(self.card_trades, "Trades", f"{total_trades}")
        self._set_card(self.card_wr, "WR", f"{win_rate:.1f}%")
        self._set_card(self.card_pf, "PF", pf_text)

    def _update_symbol_table(self, sym_stats):
        tree = self.tree
        if not sym_stats:
            # remove any lingering rows
            for iid in tree.get_children():
                tree.delete(iid)
            self._tree_iids.clear()
            return

        # Desired order: PnL desc
        items = sorted(sym_stats.items(), key=lambda kv: kv[1]["pnl"], reverse=True)

        seen_symbols = set()
        last_iid = ""
        for sym, s in items:
            seen_symbols.add(sym)

            # format pnl text and tag
            if s["pnl"] > 0:
                pnl_str = f"+{s['pnl']:.2f}"
                tag = "pnl_pos"
            elif s["pnl"] < 0:
                pnl_str = f"{s['pnl']:.2f}"
                tag = "pnl_neg"
            else:
                pnl_str = f"{s['pnl']:.2f}"
                tag = "pnl_neu"

            values = (sym, pnl_str, s["trades"], f"{s['win_rate']:.1f}%")

            if sym in self._tree_iids and self._tree_iids[sym] in tree.get_children(""):
                iid = self._tree_iids[sym]
                # Update in place if values changed
                if tree.item(iid, "values") != values:
                    tree.item(iid, values=values, tags=(tag,))
                # Reorder if necessary (move after last inserted)
                if last_iid != iid:
                    tree.move(iid, "", "end")
                last_iid = iid
            else:
                # Insert new row
                iid = tree.insert("", "end", values=values, tags=(tag,))
                self._tree_iids[sym] = iid
                last_iid = iid

        # Remove rows that are no longer present
        to_delete = [sym for sym in self._tree_iids.keys() if sym not in seen_symbols]
        for sym in to_delete:
            iid = self._tree_iids.pop(sym, None)
            if iid is not None:
                try:
                    tree.delete(iid)
                except Exception:
                    pass

        # --- Adjust visible height to fit data: min 5 rows, max 10 rows
        try:
            row_count = len(tree.get_children(""))
            target_rows = max(5, min(10, row_count))
            tree.configure(height=target_rows)
        except Exception:
            pass

    # ---------------- CHART ----------------
    def _apply_static_axis_style(self):
        # Face + grid
        self.ax.set_facecolor(CARD_BG)
        self.ax.grid(True, color=GRID, alpha=0.6)

        # Labels
        self.ax.set_xlabel("Time (PT)", color=TEXT_MUTED)
        self.ax.set_ylabel("Profit/Loss ($)", color=TEXT_MUTED)

        # Ticks: locator/formatter and styling
        self.ax.xaxis.set_major_locator(self._x_locator)
        self.ax.xaxis.set_major_formatter(self._x_formatter)
        for lbl in self.ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")
            lbl.set_color(TEXT_PRIMARY)
        for lbl in self.ax.get_yticklabels():
            lbl.set_color(TEXT_PRIMARY)

        # Zero baseline (keep a reference; draw only once)
        if not hasattr(self, "_zero_line") or self._zero_line.axes is not self.ax:
            self._zero_line = self.ax.axhline(
                0.0, color=TEXT_MUTED, linewidth=1.2, alpha=0.7, zorder=1
            )

    def _paint_empty_chart(self):
        # Clear axes and recreate the persistent line so future updates work
        self.ax.clear()
        # Remove any existing HOD/LOD annotations
        for h in ("_hod_marker", "_lod_marker", "_hod_label", "_lod_label"):
            obj = getattr(self, h, None)
            try:
                if obj is not None:
                    obj.remove()
            except Exception:
                pass
            setattr(self, h, None)
        # Re-apply static styling (grid/labels/ticks/locator/baseline)
        self._apply_static_axis_style()

        # Reset the line and clear any remembered fills
        self._line_cum = self.ax.plot(
            [], [], linewidth=2, color=TEXT_PRIMARY, zorder=3
        )[0]
        if getattr(self, "_fill_polys", None):
            self._fill_polys.clear()

        self.ax.text(
            0.5,
            0.5,
            "No trades to display",
            color=TEXT_MUTED,
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            fontsize=14,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw_idle()

    def _update_hod_lod(self, ts, cum):
        """Create/update High/Low of Day annotations on the running P/L curve."""
        if not ts or not cum:
            return

        import numpy as np

        # Find indices of max/min cumulative P/L
        arr = np.asarray(cum, dtype=float)
        hi_idx = int(np.argmax(arr))
        lo_idx = int(np.argmin(arr))

        hi_t, hi_v = ts[hi_idx], float(cum[hi_idx])
        lo_t, lo_v = ts[lo_idx], float(cum[lo_idx])

        # Ensure limits are up-to-date (needed for nice offsets)
        self.ax.relim()
        self.ax.autoscale_view()
        y_min, y_max = self.ax.get_ylim()
        y_span = max(1e-9, (y_max - y_min))
        up_offset = 0.03 * y_span
        dn_offset = 0.03 * y_span

        # Create/update markers
        if self._hod_marker is None or self._hod_marker.axes is not self.ax:
            self._hod_marker = self.ax.scatter(
                [], [], s=36, zorder=4, color=GREEN, edgecolors="none"
            )
        self._hod_marker.set_offsets([[hi_t, hi_v]])

        if self._lod_marker is None or self._lod_marker.axes is not self.ax:
            self._lod_marker = self.ax.scatter(
                [], [], s=36, zorder=4, color=RED, edgecolors="none"
            )
        self._lod_marker.set_offsets([[lo_t, lo_v]])

        # Create/update labels
        hod_text = f"HOD  ${hi_v:,.2f}"
        lod_text = f"LOD  ${lo_v:,.2f}"

        if self._hod_label is None or self._hod_label.axes is not self.ax:
            self._hod_label = self.ax.text(
                hi_t,
                hi_v + up_offset,
                hod_text,
                color=GREEN,
                fontsize=9,
                weight="bold",
                ha="center",
                va="bottom",
                zorder=5,
            )
        else:
            self._hod_label.set_position((hi_t, hi_v + up_offset))
            self._hod_label.set_text(hod_text)
            self._hod_label.set_color(GREEN)

        if self._lod_label is None or self._lod_label.axes is not self.ax:
            self._lod_label = self.ax.text(
                lo_t,
                lo_v - dn_offset,
                lod_text,
                color=RED,
                fontsize=9,
                weight="bold",
                ha="center",
                va="top",
                zorder=5,
            )
        else:
            self._lod_label.set_position((lo_t, lo_v - dn_offset))
            self._lod_label.set_text(lod_text)
            self._lod_label.set_color(RED)

    def _paint_chart(self, completed):
        if not completed:
            # If no trades, still show a descriptive title for today
            self.ax.set_title(
                f"Running P/L — {datetime.now().strftime('%B %d, %Y')}",
                color=TEXT_PRIMARY,
                pad=12,
            )
            self.ax.text(
                0.5,
                0.5,
                "No trades to display",
                color=TEXT_MUTED,
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=14,
            )
            self.canvas.draw_idle()
            return

        trades = sorted(completed, key=lambda x: x["timestamp"])
        ts = [t["timestamp"] for t in trades]
        cum = []
        running = 0.0
        for t in trades:
            running += t["pnl"]
            cum.append(running)

        # Title with trading date inferred from first timestamp
        trading_day = min(ts).strftime("%B %d, %Y")
        self.ax.set_title(f"Running P/L — {trading_day}", color=TEXT_PRIMARY, pad=12)

        # Update persistent line instead of re-plotting anew
        self._line_cum.set_data(ts, cum)

        # Adjust limits (small padding)
        self.ax.relim()
        self.ax.autoscale_view()

        # (Optional) leave your fill_between calls as-is for now,
        # or remove them if you want the absolute lightest redraw.

        # Fill green/red areas
        # Remove old fills to avoid stacking
        if getattr(self, "_fill_polys", None):
            for poly in self._fill_polys:
                try:
                    poly.remove()
                except Exception:
                    pass
            self._fill_polys.clear()
        else:
            self._fill_polys = []

        zeros = np.zeros(len(cum))
        times_num = mdates.date2num(ts)
        cum_arr = np.array(cum)
        poly_pos = self.ax.fill_between(
            times_num,
            zeros,
            cum_arr,
            where=(cum_arr >= 0),
            alpha=0.25,
            interpolate=True,
            zorder=2,
            color=GREEN,
        )
        self._fill_polys.append(poly_pos)

        poly_neg = self.ax.fill_between(
            times_num,
            zeros,
            cum_arr,
            where=(cum_arr < 0),
            alpha=0.25,
            interpolate=True,
            zorder=2,
            color=RED,
        )
        self._fill_polys.append(poly_neg)

        # X axis: real times (6:30, 6:45 …)
        start = min(ts).replace(hour=6, minute=30, second=0, microsecond=0)
        last = max(ts)
        minute = last.minute
        rounded_min = ((minute // 15) + 1) * 15
        if rounded_min >= 60:
            end = last.replace(
                hour=min(last.hour + 1, 23), minute=0, second=0, microsecond=0
            )
        else:
            end = last.replace(minute=rounded_min, second=0, microsecond=0)
        if end < start + timedelta(minutes=45):
            end = start + timedelta(hours=1)

        self.ax.set_xlim(start, end)
        # --- Add/update High/Low of Day annotations ---
        self._update_hod_lod(ts, cum)
        self.canvas.draw_idle()

    # ---------------- COPY CHART ----------------
    def _copy_chart_to_clipboard(self):
        """
        Copy the current chart to the Windows clipboard as an image.
        Fast path: pywin32 (CF_DIB) using a single render.
        Fallback: PowerShell + temp file if pywin32/Pillow aren't available.
        """
        try:
            # Render the figure ONCE into PNG bytes (no second render anywhere below)
            buf = io.BytesIO()
            self.fig.savefig(
                buf, format="png", dpi=160, facecolor=CARD_BG, bbox_inches="tight"
            )
            png_bytes = buf.getvalue()
            buf.close()

            if platform.system() != "Windows":
                messagebox.showwarning(
                    "Clipboard", "Image clipboard copy is implemented for Windows only."
                )
                return

            # ---- Fast path: use Windows clipboard (CF_DIB) via pywin32 ----
            try:
                import win32clipboard
                import win32con
                from PIL import Image  # Pillow for PNG->BMP(DIB) conversion
            except Exception:
                win32clipboard = None  # force fallback

            if win32clipboard is not None:
                try:
                    # Convert PNG bytes -> PIL Image -> BMP (in-memory) -> DIB bytes (strip 14-byte BMP header)
                    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                    dib_buf = io.BytesIO()
                    img.save(dib_buf, format="BMP")
                    bmp_bytes = dib_buf.getvalue()
                    dib_buf.close()
                    dib_bytes = bmp_bytes[
                        14:
                    ]  # CF_DIB expects the DIB, not the BMP file header

                    # Put DIB on clipboard
                    win32clipboard.OpenClipboard()
                    try:
                        win32clipboard.EmptyClipboard()
                        win32clipboard.SetClipboardData(win32con.CF_DIB, dib_bytes)
                    finally:
                        win32clipboard.CloseClipboard()

                    messagebox.showinfo("Clipboard", "Chart copied to clipboard.")
                    return
                except Exception as e:
                    # If anything goes wrong, fall through to PowerShell fallback
                    pass

            # ---- Fallback: PowerShell + temp file (works without pywin32/Pillow) ----
            try:
                temp_path = Path.cwd() / "_chart_clip.png"
                with open(temp_path, "wb") as f:
                    f.write(png_bytes)

                import subprocess

                subprocess.run(
                    [
                        "powershell",
                        "-command",
                        "Add-Type -AssemblyName System.Windows.Forms; "
                        f"[System.Windows.Forms.Clipboard]::SetImage([System.Drawing.Image]::FromFile('{str(temp_path)}'))",
                    ],
                    check=False,
                )
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

                messagebox.showinfo("Clipboard", "Chart copied to clipboard.")
            except Exception as e:
                messagebox.showerror("Clipboard", f"Failed to copy chart: {e}")

        except Exception as e:
            messagebox.showerror("Clipboard", f"Failed to render chart: {e}")

        # ---------------- MINI MODE ----------------

    def _toggle_mini_mode(self):
        """Show/Hide the always-on-top ticker with Gross P/L + clock."""
        if self.mini_mode:
            self._destroy_mini()
            self.mini_mode = False
        else:
            self._init_mini()
            self.mini_mode = True
            # immediate paint
            self._refresh_mini()

    def _init_mini(self):
        # Create frameless, always-on-top window
        self.mini_win = tk.Toplevel(self.root)
        self.mini_win.overrideredirect(True)
        self.mini_win.attributes("-topmost", True)
        self.mini_win.configure(bg=CARD_BG)
        # Force first-size compute and first data paint on open
        self._mini_last_n_rows = None
        self._mini_needs_data_repaint = True

        # Geometry
        if self._mini_last_pos:
            self.mini_win.geometry(self._mini_last_pos)
        else:
            # tall enough for ribbon + some rows; will auto-resize later
            self.mini_win.geometry("600x168+100+60")

        # --- WRAPPERS ---
        wrap = tk.Frame(self.mini_win, bg=CARD_BG, padx=8, pady=8)
        wrap.pack(fill=tk.BOTH, expand=True)

        # We'll put the ribbon on top…
        ribbon_row = tk.Frame(wrap, bg=CARD_BG)
        ribbon_row.pack(fill=tk.X)

        # …and the table below
        table_wrap = tk.Frame(wrap, bg=CARD_BG)
        table_wrap.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # --- RIBBON (stats) ---
        self.mini_vars = {
            "pnl": tk.StringVar(value="$0.00"),
            "trades": tk.StringVar(value="0"),
            "winrate": tk.StringVar(value="0.0%"),
            "pf": tk.StringVar(value="0.00"),
            "clock": tk.StringVar(value="--:--:--"),
        }

        def _mk_stat(parent, title, var):
            f = tk.Frame(parent, bg=CARD_BG)
            tk.Label(
                f, text=title, font=("Segoe UI", 9, "bold"), fg="#9aa3a8", bg=CARD_BG
            ).pack(anchor="w")
            val = tk.Label(
                f,
                textvariable=var,
                font=("Segoe UI", 14, "bold"),
                fg=TEXT_PRIMARY,
                bg=CARD_BG,
            )
            val.pack(anchor="w")
            return f, val

        # (after ribbon_row defined)

        # 1) CLOCK (leftmost)
        clk_wrap = tk.Frame(ribbon_row, bg=CARD_BG)
        clk_wrap.pack(side=tk.LEFT, padx=(10, 10))
        clk_val = tk.Label(
            clk_wrap,
            textvariable=self.mini_vars["clock"],
            font=("Segoe UI", 24, "bold"),
            fg="#FFD700",
            bg=CARD_BG,
        )
        clk_val.pack(anchor="w")

        # 2) Gross P/L
        pnl_f, pnl_val = _mk_stat(ribbon_row, "Gross P/L", self.mini_vars["pnl"])
        pnl_f.pack(side=tk.LEFT, padx=10)
        self._mini_blocks["pnl"] = pnl_val

        # 3) Total Trades
        tr_f, _ = _mk_stat(ribbon_row, "Trades", self.mini_vars["trades"])
        tr_f.pack(side=tk.LEFT, padx=10)

        # 4) Win Rate
        wr_f, wr_val = _mk_stat(ribbon_row, "WR", self.mini_vars["winrate"])
        wr_f.pack(side=tk.LEFT, padx=10)
        self._mini_blocks["winrate"] = wr_val

        # 5) Profit Factor
        pf_f, pf_val = _mk_stat(ribbon_row, "PF", self.mini_vars["pf"])
        pf_f.pack(side=tk.LEFT, padx=10)
        self._mini_blocks["pf"] = pf_val

        # --- TABLE (per-symbol) ---
        cols = ("symbol", "pnl", "trades", "winpct")
        self.mini_tree = ttk.Treeview(
            table_wrap,
            columns=cols,
            show="headings",
            height=1,  # will be set dynamically in _refresh_mini
            style="Mini.Treeview",
        )

        # headings
        self.mini_tree.heading("symbol", text="Symbol")
        self.mini_tree.heading("pnl", text="P/L ($)")
        self.mini_tree.heading("trades", text="Trades")
        self.mini_tree.heading("winpct", text="Win %")

        # columns
        self.mini_tree.column("symbol", width=80, anchor="w")
        self.mini_tree.column("pnl", width=110, anchor="e")
        self.mini_tree.column("trades", width=70, anchor="center")
        self.mini_tree.column("winpct", width=70, anchor="center")

        self.mini_tree.tag_configure("pos", foreground=GREEN)
        self.mini_tree.tag_configure("neg", foreground=RED)
        self.mini_tree.tag_configure("neu", foreground=TEXT_PRIMARY)

        self.mini_tree.pack(fill=tk.X)

        # --- DRAGGING / TOGGLE BINDINGS (after everything exists) ---
        for widget in (
            self.mini_win,
            wrap,
            ribbon_row,
            clk_wrap,
            pnl_f,
            tr_f,
            wr_f,
            pf_f,
            table_wrap,
            self.mini_tree,
        ):
            widget.bind("<Button-1>", self._mini_drag_start)
            widget.bind("<B1-Motion>", self._mini_drag_move)

        ribbon_row.bind("<Double-Button-1>", lambda e: self._toggle_mini_mode())
        self.mini_tree.bind("<Double-Button-1>", lambda e: self._toggle_mini_mode())

    def _destroy_mini(self):
        if self.mini_win and self.mini_win.winfo_exists():
            # remember last position like "+1230+40"
            try:
                self._mini_last_pos = (
                    f"+{self.mini_win.winfo_x()}+{self.mini_win.winfo_y()}"
                )
            except Exception:
                pass
            self.mini_win.destroy()
            self._mini_last_n_rows = None
            self._mini_needs_data_repaint = False
            self.mini_vars = None  # if you store mini state here

        self.mini_win = None

    def _mini_drag_start(self, event):
        widget = event.widget
        widget._drag_start_x = event.x
        widget._drag_start_y = event.y

    def _mini_drag_move(self, event):
        win = self.mini_win
        if not win:
            return
        x = win.winfo_x() + event.x - getattr(event.widget, "_drag_start_x", 0)
        y = win.winfo_y() + event.y - getattr(event.widget, "_drag_start_y", 0)
        win.geometry(f"+{x}+{y}")

    def _refresh_mini(self):
        # tolerate early calls / half-initialized states
        if not getattr(self, "mini_mode", False):
            return
        mini_win = getattr(self, "mini_win", None)
        if mini_win is None:
            return
        try:
            if not mini_win.winfo_exists():
                return
        except Exception:
            return

        # some builds may not have the tree yet for a tick or two
        mini_tree = getattr(self, "mini_tree", None)

        s = getattr(self, "stats", {}) or {}

        # --- top ribbon values ---
        pnl = float(s.get("gross_pnl", 0.0) or 0.0)
        trades = int(s.get("total_trades", 0) or 0)

        # winrate may come as 0.53 or 53.0; normalize to percent
        _wr = s.get("win_rate", s.get("winrate", 0.0))
        try:
            winrate = float(_wr or 0.0)
        except Exception:
            winrate = 0.0
        if 0.0 <= winrate <= 1.0:
            winrate *= 100.0  # treat fractions as percentages

        pf = float(s.get("profit_factor", 0.0) or 0.0)

        # text formatting
        pnl_text = f"${abs(pnl):,.2f}"
        if pnl > 0:
            pnl_text = f"+{pnl_text}"
        elif pnl < 0:
            pnl_text = f"-{pnl_text}"

        self.mini_vars["pnl"].set(pnl_text)
        self.mini_vars["trades"].set(f"{trades}")
        self.mini_vars["winrate"].set(f"{winrate:.1f}%")
        self.mini_vars["pf"].set(f"{pf:.2f}" if trades > 0 else "—")
        self.mini_vars["clock"].set(datetime.now().strftime("%I:%M:%S %p").lstrip("0"))

        # pnl color + slight bg tint
        if pnl > 0:
            pnl_fg, bg = GREEN, "#152820"
        elif pnl < 0:
            pnl_fg, bg = RED, "#281515"
        else:
            pnl_fg, bg = TEXT_PRIMARY, CARD_BG

        # safely touch mini stat labels if they exist
        mb = getattr(self, "_mini_blocks", {}) or {}
        if "pnl" in mb:
            try:
                mb["pnl"].configure(fg=pnl_fg)
            except Exception:
                pass
        try:
            mini_win.configure(bg=bg)
        except Exception:
            pass

        # winrate & pf coloring (safe guards)
        wr_fg = GREEN if winrate >= 50.0 else (RED if trades > 0 else TEXT_PRIMARY)
        if "winrate" in mb:
            try:
                mb["winrate"].configure(fg=wr_fg)
            except Exception:
                pass
        if "pf" in mb:
            try:
                mb["pf"].configure(
                    fg=(
                        GREEN
                        if pf > 1.0
                        else (RED if pf < 1.0 and trades > 0 else TEXT_PRIMARY)
                    )
                )
            except Exception:
                pass

        # --- per-symbol table (use self.symbol_stats, not self.stats) ---
        sym_map = getattr(self, "symbol_stats", {}) or {}

        # Build rows and sort ALL by P/L (desc)
        rows = [
            {
                "symbol": str(sym).upper(),
                "pnl": float(d.get("pnl", 0.0) or 0.0),
                "trades": int(d.get("trades", 0) or 0),
                # keep as a plain number in the table (no % sign)
                "win_pct": float(d.get("win_rate", 0.0) or 0.0)
                * (
                    100.0 if 0.0 <= float(d.get("win_rate", 0.0) or 0.0) <= 1.0 else 1.0
                ),
            }
            for sym, d in sym_map.items()
        ]

        rows.sort(key=lambda r: r["pnl"], reverse=True)

        # honor a cap if you set one; otherwise show all
        if getattr(self, "MINI_SYMBOL_ROWS", None) is not None:
            rows = rows[: int(self.MINI_SYMBOL_ROWS)]

        # If tree not built yet, we can stop here after computing rows
        if mini_tree is None:
            return

        # Compute how many rows we're ACTUALLY rendering
        max_rows = getattr(self, "MINI_MAX_SYMBOL_ROWS", None)
        n_rows = max(1, min(len(rows), max_rows or len(rows)))

        # Check the actual widget height too (new Treeview starts with a default height)
        try:
            current_height = int(mini_tree.cget("height"))
        except Exception:
            current_height = None

        if (n_rows != getattr(self, "_mini_last_n_rows", None)) or (
            current_height != n_rows
        ):
            # Resize Treeview height to exactly fit rendered rows
            try:
                mini_tree.configure(height=n_rows)
            except Exception:
                pass

            # Resize the mini window height to fit everything neatly
            try:
                x = mini_win.winfo_x()
                y = mini_win.winfo_y()
                w = max(mini_win.winfo_width(), 560)

                base_h = getattr(self, "MINI_BASE_HEIGHT", 120)
                hdr_h = getattr(self, "MINI_HEADER_HEIGHT", 44)
                row_h = getattr(self, "MINI_ROW_HEIGHT", 22)
                vpad = getattr(self, "MINI_VERTICAL_PADDING", 18)

                h = base_h + hdr_h + n_rows * row_h + vpad
                mini_win.geometry(f"{w}x{h}+{x}+{y}")
            except Exception:
                pass

            self._mini_last_n_rows = n_rows

        # Clear & repopulate with the exact rows we're displaying
        try:
            for iid in mini_tree.get_children():
                mini_tree.delete(iid)
        except Exception:
            pass

        for r in rows:
            pnl_txt = f"{r['pnl']:+,.2f}"
            tag = "pos" if r["pnl"] > 0 else ("neg" if r["pnl"] < 0 else "neu")
            try:
                mini_tree.insert(
                    "",
                    "end",
                    values=(r["symbol"], pnl_txt, r["trades"], f"{r['win_pct']:.1f}"),
                    tags=(tag,),
                )
            except Exception:
                # tolerate insertion issues without crashing the clock
                continue

    # ---------------- MISC ----------------
    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _on_close(self):
        try:
            self._uninstall_file_watch()
        except Exception:
            pass
        try:
            # cancel any pending debounce
            if self._watch_pending_id is not None:
                self.root.after_cancel(self._watch_pending_id)
        except Exception:
            pass
        self.root.destroy()

    # --------------------------------------


if __name__ == "__main__":
    TradingCompanionApp()
