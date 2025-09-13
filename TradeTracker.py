from __future__ import annotations

import io
import json
import csv
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz
import streamlit as st


try:
    import pandas_market_calendars as mcal  # NYSE trading calendar & holidays
except Exception:
    mcal = None

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import base64
import math
import streamlit.components.v1 as components

# --- Incremental cache helpers ---
import os
import hashlib
from functools import lru_cache

try:
    # pip install platformdirs
    from platformdirs import user_config_path
except Exception:
    user_config_path = None


APP_NAME = "TradeTracker"

# -------------------------
# Unified Color Theme (fixed)
# -------------------------
import re as _re

class Theme:
    """Single source of truth for all colors; helpers accept palette keys or hex."""
    palette = {
        "bg_dark":   "#0f1623",
        "bg_panel":  "#111825",
        "ink":       "#cbd5e1",
        "ink_subtle":"#aab6d3",
        "white":     "#ffffff",
        "black":     "#111111",
        "green":     "#16a34a",   # wins
        "red":       "#dc2626",   # losses
        "orange":    "#f59e0b",   # fees
        "gray":      "#cacfde",
        "blue":      "#394556",
    }

    @classmethod
    def hex(cls, name_or_hex: str) -> str:
        s = (name_or_hex or "").strip()
        # palette key
        if s in cls.palette:
            return cls.palette[s]
        # already #RRGGBB
        if s.startswith("#") and _re.fullmatch(r"#[0-9a-fA-F]{6}", s):
            return s
        # bare 6-hex digits -> normalize
        if _re.fullmatch(r"[0-9a-fA-F]{6}", s):
            return "#" + s
        # last resort: return as-is (keeps legacy values working)
        return s

    @classmethod
    def rgba(cls, name_or_hex: str, alpha: float) -> str:
        h = cls.hex(name_or_hex).lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"


# Back-compat alias used throughout the code
COL = Theme

def inject_theme_css():
    """Expose Theme.palette as CSS variables for style blocks."""
    pal = Theme.palette

    def _rgba(hex_color: str, a: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    css = f"""
    <style>
      :root {{
        --bg-dark:   {pal['bg_dark']};
        --bg-panel:  {pal['bg_panel']};
        --ink:       {pal['ink']};
        --ink-subtle:{pal['ink_subtle']};
        --white:     {pal['white']};
        --black:     {pal['black']};
        --green:     {pal['green']};
        --red:       {pal['red']};
        --orange:    {pal['orange']};
        --gray:      {pal['gray']};
        --blue:      {pal['blue']};

        /* alpha variants */
        --white-04:  {_rgba(pal['white'], 0.04)};
        --white-10:  {_rgba(pal['white'], 0.10)};
        --black-15:  {_rgba(pal['black'], 0.15)};
        --ink-12:    {_rgba(pal['ink'], 0.12)};
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


CONFIG_FILENAME = "config.json"

def _config_path() -> Path:
    # store config in the project folder, next to the main script
    return Path(__file__).parent / "config.json"


def load_user_config() -> dict:
    p = _config_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        # Normalize into Path objects
        return {k: Path(v).expanduser() for k, v in data.items()}
    except Exception:
        return {}


def save_user_config(cfg: dict) -> None:
    # Convert all values to forward-slash style before writing
    clean_cfg = {k: str(Path(v).expanduser().resolve().as_posix()) for k, v in cfg.items()}
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean_cfg, indent=2), encoding="utf-8")

# ---- Device-specific config (env > user-config file > fallback) ----
_FALLBACKS = {
    "raw_csvs": r"C:\Users\an\Desktop\Trading\VSCode Apps\history",
    "sec_schedule": r"C:\Users\an\Desktop\Trading\VSCode Apps\sec_rate_schedule.csv",
}

_cfg = load_user_config()

DEFAULT_FOLDER = _cfg.get("raw_csvs", Path("C:/Users/an/Desktop/Trading/VSCode Apps/history"))
SEC_RATE_SCHEDULE_PATH_DEFAULT = _cfg.get(
    "sec_schedule",
    Path("C:/Users/an/Desktop/Trading/VSCode Apps/sec_rate_schedule.csv")
)


TIMEZONE_EASTERN = pytz.timezone("America/New_York")
TIMEZONE_PACIFIC = pytz.timezone("America/Los_Angeles")

# -------- Incremental cache config --------
CACHE_DIR_NAME = ".tt_cache"
FILE_FILLS_DIR = "file_fills"     # per-file fills parquet
DAILY_TRADES_DIR = "daily_trades" # per-day trades parquet
DAILY_STATS_DIR = "daily_stats"   # per-day stats parquet
INDEX_NAME = "index.parquet"      # tracks file -> fills metadata

def _cache_dir_for(folder: Path) -> Path:
    d = folder / CACHE_DIR_NAME
    (d / FILE_FILLS_DIR).mkdir(parents=True, exist_ok=True)
    (d / DAILY_TRADES_DIR).mkdir(parents=True, exist_ok=True)
    (d / DAILY_STATS_DIR).mkdir(parents=True, exist_ok=True)
    return d

def _file_key(path: Path) -> str:
    """Stable filename-friendly key for a CSV path."""
    return hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest()[:16]

def _index_path(cache_dir: Path) -> Path:
    return cache_dir / INDEX_NAME


# --- Formatting helpers ---
fmt_money = lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
fmt_pct = lambda x: f"{x*100:.1f}%"


# --- NEW: tiny helpers for Equity Curves ---

def render_png_export_ui(
    fig: "go.Figure",
    key_prefix: str = "single",
    *,
    date=None,            # datetime.date|datetime.datetime|str "YYYY-MM-DD"
    start=None,           # same types as `date`
    end=None,             # same types as `date`
    basename: str = "equity",
):
    """
    Save/Copy PNG UI with real Save-As (Chromium) + fallback, and
    auto filename from the chart's date or date range.

    Filename rules:
      - if `date` provided  -> {basename}-YYYY-MM-DD.png
      - elif `start`+`end`  -> {basename}-YYYY-MM-DD_to_YYYY-MM-DD.png
      - else                -> {basename}-{key_prefix}.png
    """
    import json, datetime as _dt
    import plotly.graph_objects as go
    import streamlit.components.v1 as components
    from plotly.utils import PlotlyJSONEncoder

    def _to_date(d):
        if d is None:
            return None
        if isinstance(d, (_dt.date, _dt.datetime)):
            return d.date() if isinstance(d, _dt.datetime) else d
        return _dt.date.fromisoformat(str(d))  # accept "YYYY-MM-DD"

    date  = _to_date(date)
    start = _to_date(start)
    end   = _to_date(end)

    if date:
        filename = f"{basename}-{date:%Y-%m-%d}.png"
    elif start and end:
        filename = f"{basename}-{start:%Y-%m-%d}_to_{end:%Y-%m-%d}.png"
    else:
        filename = f"{basename}-{key_prefix}.png"

    # Serialize fig safely
    fig_obj = go.Figure(fig).to_plotly_json()
    fig_json = json.dumps(fig_obj, cls=PlotlyJSONEncoder)

    # Unique element IDs (fixed: no literal braces)
    save_id = f"saveBtn_{key_prefix}"
    copy_id = f"copyBtn_{key_prefix}"
    status_id = f"status_{key_prefix}"
    offscreen_id = f"offscreen_{key_prefix}"

    # Safe JS string for suggested filename
    suggested_js = json.dumps(filename)

    components.html(f"""
    <style>
      .tt-toolbar {{
        display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding-bottom:8px;
      }}
      .tt-btn {{
        appearance: none;
        padding: 8px 12px;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        border: 1px solid rgba(125,125,125,.38);
        background: rgba(127,127,127,.12);
        color: #222; /* readable in light theme */
      }}
      .tt-btn:hover {{ background: rgba(127,127,127,.18); }}
      .tt-btn:active {{ transform: translateY(1px); }}
      .tt-status {{ opacity: .85; font-size: 12px; }}
      @media (prefers-color-scheme: dark) {{
        .tt-btn {{
          border-color: rgba(220,220,220,.28);
          background: rgba(255,255,255,.10);
          color: #fff; /* readable in dark theme */
        }}
        .tt-btn:hover {{ background: rgba(255,255,255,.16); }}
      }}
    </style>

    <div class="tt-toolbar">
      <button id="{save_id}" class="tt-btn">💾 Save PNG</button>
      <button id="{copy_id}" class="tt-btn">📋 Copy to clipboard</button>
      <span id="{status_id}" class="tt-status"></span>
    </div>
    <div id="{offscreen_id}" style="position:absolute; left:-99999px; top:-99999px;"></div>

    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <script>
    (function(){{
      const STATUS = document.getElementById("{status_id}");
      const container = document.getElementById("{offscreen_id}");
      const saveBtn = document.getElementById("{save_id}");
      const copyBtn = document.getElementById("{copy_id}");
      const suggestedName = {suggested_js};
      const fig = {fig_json};

      function setStatus(t) {{ if (STATUS) STATUS.textContent = t; }}

      async function ensurePlotlyReady() {{
        const maxWait = 2500; const start = performance.now();
        while (!window.Plotly) {{
          if (performance.now() - start > maxWait) throw new Error("Plotly not loaded");
          await new Promise(r=>setTimeout(r, 50));
        }}
      }}

      async function getBlob() {{
        await ensurePlotlyReady();
        if (!container.dataset.rendered) {{
          await window.Plotly.newPlot(container, fig.data, fig.layout, {{ staticPlot:true, displayModeBar:false }});
          container.dataset.rendered = "1";
        }}
        const dataUrl = await window.Plotly.toImage(container, {{ format:"png", width:1000, height:600, scale:2 }});
        const res = await fetch(dataUrl);
        return await res.blob();
      }}

      async function copyToClipboard() {{
        try {{
          setStatus("Rendering...");
          const blob = await getBlob();
          if (navigator.clipboard && window.ClipboardItem) {{
            await navigator.clipboard.write([ new ClipboardItem({{ "image/png": blob }}) ]);
            setStatus("Copied!");
            copyBtn.textContent = "✅ Copied";
            setTimeout(()=>{{ copyBtn.textContent = "📋 Copy to clipboard"; setStatus(""); }}, 1600);
          }} else {{
            setStatus("Clipboard API not available");
          }}
        }} catch (err) {{
          console.error(err); setStatus("Copy failed");
        }}
      }}

      async function saveAsFile() {{
        try {{
          setStatus("Rendering...");
          const blob = await getBlob();

          if (window.showSaveFilePicker) {{
            // Chromium/Edge: real OS Save As dialog
            const handle = await window.showSaveFilePicker({{
              suggestedName: suggestedName,
              types: [{{ description: "PNG image", accept: {{ "image/png": [".png"] }} }}],
            }});
            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
          }} else {{
            // Fallback: anchor download (Firefox/Safari)
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = suggestedName;
            document.body.appendChild(a);
            a.click();
            setTimeout(()=>URL.revokeObjectURL(a.href), 2000);
            a.remove();
          }}

          setStatus("Saved!");
          saveBtn.textContent = "✅ Saved";
          setTimeout(()=>{{ saveBtn.textContent = "💾 Save PNG"; setStatus(""); }}, 1600);
        }} catch (err) {{
          console.error(err); setStatus("Save failed");
        }}
      }}

      if (copyBtn) copyBtn.addEventListener('click', copyToClipboard);
      if (saveBtn) saveBtn.addEventListener('click', saveAsFile);
    }})();
    </script>
    """, height=170)


def _ceil_to_next_15min(ts: pd.Timestamp) -> pd.Timestamp:
    """Round UP to the next 15-min mark (tz-aware)."""
    if pd.isna(ts):
        return ts
    # normalize seconds first
    ts = ts.floor("min")
    add = (15 - (ts.minute % 15)) % 15
    if add == 0:
        add = 15
    return (ts + pd.Timedelta(minutes=add)).replace(second=0, microsecond=0)


import numpy as np

@st.cache_data(show_spinner=False)
def daily_running_curve(_trades_df: pd.DataFrame, day: date, trades_sig: tuple) -> pd.DataFrame:
    """
    Build the *realized* intraday running NET P/L curve for a Pacific 'day'.
    Returns columns: ['time' (tz-aware, PT), 'equity'] in chronological order.

    Notes:
    - Uses trade EXIT times, so curve steps when a trade closes.
    - Starts at 6:30 PT at $0.

    Caching: `_trades_df` is ignored for hashing; `trades_sig` is the cache key.
    """
    if _trades_df.empty:
        return pd.DataFrame(columns=["time", "equity"])

    # Filter by Pacific calendar day using EXIT time
    exit_pst = _trades_df["exit_time"].dt.tz_convert(TIMEZONE_PACIFIC)
    mask = exit_pst.dt.date == day
    if not mask.any():
        start_ts = pd.Timestamp(
            year=day.year, month=day.month, day=day.day, hour=6, minute=30, tz=TIMEZONE_PACIFIC
        )
        return pd.DataFrame({"time": [start_ts, start_ts + pd.Timedelta(minutes=15)], "equity": [0.0, 0.0]})

    t = _trades_df.loc[mask, ["profit_net"]].copy()
    t["exit_pst"] = exit_pst.loc[mask]
    t = t.sort_values("exit_pst")

    # Build step-like series: start at 6:30 PT at 0, then step at each exit
    start_ts = pd.Timestamp(
        year=day.year, month=day.month, day=day.day, hour=6, minute=30, tz=TIMEZONE_PACIFIC
    )
    times = pd.DatetimeIndex([start_ts]).append(pd.DatetimeIndex(t["exit_pst"]))
    equity = np.concatenate([[0.0], pd.to_numeric(t["profit_net"], errors="coerce").fillna(0.0).cumsum().to_numpy()])
    return pd.DataFrame({"time": times, "equity": equity})


@st.cache_data(show_spinner=False)
def range_cumulative_curve(_daily_df: pd.DataFrame, start_d: date, end_d: date, daily_sig: tuple) -> pd.DataFrame:
    """
    Build a daily cumulative equity curve between [start_d, end_d], inclusive.
    Returns columns: ['date', 'equity'] where 'equity' is cumsum of NET daily_profit.

    Caching: `_daily_df` is ignored for hashing; `daily_sig` is the cache key.
    """
    if _daily_df.empty:
        return pd.DataFrame(columns=["date", "equity"])

    d = _daily_df.loc[(_daily_df["date"] >= start_d) & (_daily_df["date"] <= end_d), ["date", "daily_profit"]].copy()
    if d.empty:
        return pd.DataFrame(columns=["date", "equity"])

    d = d.sort_values("date", kind="mergesort")  # stable + fast
    d["equity"] = pd.to_numeric(d["daily_profit"], errors="coerce").fillna(0.0).cumsum()
    return d[["date", "equity"]]


@st.cache_data(show_spinner=False)
def _nyse_holidays_year_cached(y: int) -> dict:
    """Return {date: 'Holiday Name'} for NYSE holidays in a given year."""
    # Reuse your existing nyse_holidays_named helper internally
    return nyse_holidays_named(date(y, 1, 1), date(y, 12, 31))

@st.cache_data(show_spinner=False)
def _year_view_html(daily_df: pd.DataFrame, year: int, cols: int = 4, compact: bool = False) -> str:
    if daily_df.empty:
        return "<div>No daily stats available for this year.</div>"

    import datetime as _dt
    from calendar import monthrange

    d = daily_df.copy()
    d["dts"] = pd.to_datetime(d["date"], errors="coerce")
    ydf = d.loc[d["dts"].dt.year == year].copy()

    # Day classifications
    pos_days = set(ydf.loc[ydf["daily_profit"] > 0, "dts"].dt.date.tolist())
    neg_days = set(ydf.loc[ydf["daily_profit"] < 0, "dts"].dt.date.tolist())
    holinames = _nyse_holidays_year_cached(year)
    holi_days = set(holinames.keys())

    # Per-month summaries
    month_sum = (
        ydf.assign(month=ydf["dts"].dt.month)
           .groupby("month", as_index=False)
           .agg(net=("daily_profit", "sum"),
                fees=("fees", "sum"),
                trades=("num_trades", "sum"))
    ).set_index("month")

    # Sizing
    cell_h = 18 if compact else 22
    gap_px = 3 if compact else 6
    font_r = 0.68 if compact else 0.78

    # (named but not required in CSS vars below)
    green  = "#16a34a"
    red    = "#dc2626"
    orange = "#f59e0b"

    css = f"""
        <style>
        /* Grid of month cards */
        .yv2-grid {{
            display: grid;
            grid-template-columns: repeat({cols}, minmax(0, 1fr));
            gap: {gap_px + 10}px;
            margin-top: 8px;
        }}

        /* Month card chrome */
        .yv2-month {{
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0;
            box-shadow: none;
        }}

        .yv2-mname {{
            text-align: center;
            font-weight: 900;
            font-size: .95rem;
            color: var(--ink);
            margin-bottom: 8px;
            letter-spacing: .3px;
        }}

        /* Calendar grid: always 6 weeks (6 x 7 = 42 cells) */
        .yv2-cal {{
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            grid-template-rows: repeat(6, {cell_h}px);
            gap: {gap_px}px;
        }}

        /* Day tiles */
        .yv2-day {{
            display: flex; align-items: center; justify-content: center;
            border-radius: 8px;
            font-size: {font_r}rem;
            line-height: 1;
            border: 1px solid var(--white-10);
        }}

        /* Day states */
        .yv2-day.muted {{ background: var(--white-04); color: var(--ink); }}
        .yv2-day.pos   {{ background: var(--green);   color: var(--white); }}
        .yv2-day.neg   {{ background: var(--red);     color: var(--white); }}
        .yv2-day.holi  {{ background: var(--gray);    color: var(--black); font-weight: 800; }}

        /* Footer: left cluster (Profit, Fees) + right cluster (Trades) */
        .yv2-stats {{
            display: flex;
            flex-wrap: nowrap;
            justify-content: space-between;   /* left cluster vs right cluster */
            align-items: baseline;            /* align currency digits nicely */
            padding: 2px 6px 0 6px;
            font-variant-numeric: tabular-nums;
            min-height: 22px;                 /* reserves space for consistent height */
            gap: 0;                           /* gaps are set on inner clusters */
        }}

        .yv2-stats .left {{
            display: flex;
            align-items: baseline;
            gap: 12px;                        /* spacing between Profit and Fees */
        }}

        .yv2-stats .right {{
            display: flex;
            align-items: baseline;
            gap: 12px;
        }}

        /* Typography + colors for footer values */
        .yv2-net {{ font-weight: 700; }}          /* Profit bold */
        .yv2-net.pos {{ color:#2e7d32; }}
        .yv2-net.neg {{ color:#d9534f; }}

        .yv2-fee {{ color:#ff8c00; font-weight: 400; }}   /* Fees normal */
        .yv2-trades {{ color: var(--ink-subtle); font-weight: 400; white-space: nowrap; }}  /* Trades normal, on right */
        </style>
    """

    def _fmt(x: float) -> str:
        return f"${x:,.2f}"

    months_html = []
    for m in range(1, 12 + 1):
        mname = _dt.date(year, m, 1).strftime("%B")

        # Sunday-first grid with leading blanks
        def _sunfirst(w): return (w + 1) % 7
        lead = _sunfirst(_dt.date(year, m, 1).weekday())
        _, ndays = monthrange(year, m)

        cells = []
        for _ in range(lead):
            cells.append('<div class="yv2-day muted"></div>')

        for daynum in range(1, ndays + 1):
            dt = _dt.date(year, m, daynum)
            klass = ["yv2-day"]
            if dt in holi_days:
                klass.append("holi")
            elif dt in pos_days:
                klass.append("pos")
            elif dt in neg_days:
                klass.append("neg")
            else:
                klass.append("muted")
            title = holinames.get(dt, "")
            cells.append(f'<div class="{" ".join(klass)}" title="{title}">{daynum}</div>')

        # Pad trailing blanks to 42 cells (6 rows x 7 cols)
        pad = 42 - len(cells)
        if pad > 0:
            cells.extend(['<div class="yv2-day muted"></div>'] * pad)

        cal_html = f'<div class="yv2-cal">{"".join(cells)}</div>'

        # Monthly aggregates
        net   = float(month_sum.loc[m, "net"])   if m in month_sum.index else 0.0
        fees  = float(month_sum.loc[m, "fees"])  if m in month_sum.index else 0.0
        trs   = int(month_sum.loc[m, "trades"])  if m in month_sum.index else 0
        net_cls = "pos" if net >= 0 else "neg"

        # Footer: Profit & Fees on the left, Trades on the right
        stats_html = (
            f'<div class="yv2-stats">'
            f'  <div class="left">'
            f'    <span class="yv2-net {net_cls}">{_fmt(net)}</span>'
            f'    <span class="yv2-fee">{_fmt(fees)}</span>'
            f'  </div>'
            f'  <div class="right">'
            f'    <span class="yv2-trades">{trs:,} trades</span>'
            f'  </div>'
            f'</div>'
        )

        months_html.append(
            f'<div class="yv2-month"><div class="yv2-mname">{mname}</div>{cal_html}{stats_html}</div>'
        )

    return css + f'<div class="yv2-grid">{"".join(months_html)}</div>'


@st.cache_data(show_spinner=False)
def _csv_bytes_trades(trades: pd.DataFrame) -> bytes:
    if trades.empty: return b""
    t = trades.copy()
    t["entry_time"] = t["entry_time"].dt.tz_convert(TIMEZONE_PACIFIC).dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    t["exit_time"]  = t["exit_time"].dt.tz_convert(TIMEZONE_PACIFIC).dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    return t.to_csv(index=False).encode("utf-8")

@st.cache_data(show_spinner=False)
def _csv_bytes_daily(daily: pd.DataFrame) -> bytes:
    if daily.empty: return b""
    return daily.to_csv(index=False).encode("utf-8")


# -------------------------
# Utility: caching signature
# -------------------------


def folder_signature(folder: Path) -> Tuple[Tuple[str, float, int], ...]:
    """Return a signature of (filename, mtime, size) for all CSVs in folder.
    This is used as a cache key so we only recompute when files change."""
    entries = []
    for p in sorted(folder.glob("*.csv")):
        try:
            stat = p.stat()
            entries.append((str(p), stat.st_mtime, stat.st_size))
        except FileNotFoundError:
            continue
    return tuple(entries)

def _load_index(cache_dir: Path) -> pd.DataFrame:
    ip = _index_path(cache_dir)
    if ip.exists():
        try:
            return pd.read_parquet(ip)
        except Exception:
            pass
    return pd.DataFrame(columns=["source_file","mtime","size","fills_path","min_date_pst","max_date_pst","rows"])

def _save_index(cache_dir: Path, idx: pd.DataFrame) -> None:
    ip = _index_path(cache_dir)
    idx.to_parquet(ip, index=False)

def _scan_and_update_file_fills(folder: Path, sec_schedule: pd.DataFrame) -> tuple[pd.DataFrame, list[date]]:
    """
    Ensure per-file fills parquet exists & is fresh for each CSV.
    Returns (index_df, affected_dates) where affected_dates are PST dates whose per-day aggregates should be rebuilt.
    Handles: new, modified, and deleted CSVs.
    """
    cache_dir = _cache_dir_for(folder)
    idx = _load_index(cache_dir)

    # Snapshot current *.csv files
    cur_entries = []
    for p in sorted(folder.glob("*.csv")):
        try:
            stt = p.stat()
            cur_entries.append((str(p), stt.st_mtime, stt.st_size))
        except FileNotFoundError:
            continue
    cur = pd.DataFrame(cur_entries, columns=["source_file","mtime","size"])

    affected_dates: set[date] = set()

    # -------- Deletions --------
    if not idx.empty:
        deleted = idx.loc[~idx["source_file"].isin(cur["source_file"] if not cur.empty else [])]
        for row in deleted.itertuples(index=False):
            # collect dates from old shard to rebuild those days without this file
            try:
                if isinstance(row.fills_path, str) and os.path.exists(row.fills_path):
                    old_df = pd.read_parquet(row.fills_path, columns=["date_pst"])
                    if not old_df.empty:
                        ds = pd.to_datetime(old_df["date_pst"], errors="coerce").dt.date.dropna().unique().tolist()
                        affected_dates.update(ds)
                    # remove old shard
                    Path(row.fills_path).unlink(missing_ok=True)
            except Exception:
                pass

        if not deleted.empty:
            idx = idx.loc[~idx["source_file"].isin(deleted["source_file"])].copy()
            _save_index(cache_dir, idx)

    # -------- New/modified --------
    # Left-join by (source_file,mtime,size) to find new or changed files
    merged = cur.merge(idx[["source_file","mtime","size","fills_path"]],
                       on=["source_file","mtime","size"], how="left", indicator=True)
    to_build = merged[merged["_merge"] == "left_only"][["source_file","mtime","size"]]

    if not to_build.empty:
        built_rows = []
        for row in to_build.itertuples(index=False):
            src = Path(row.source_file)

            # Gather dates from old shard (if we had one for this source_file)
            old_dates: set[date] = set()
            prev_rows = idx.loc[idx["source_file"] == str(src)]
            if not prev_rows.empty:
                for pr in prev_rows.itertuples(index=False):
                    try:
                        if isinstance(pr.fills_path, str) and os.path.exists(pr.fills_path):
                            old_df = pd.read_parquet(pr.fills_path, columns=["date_pst"])
                            if not old_df.empty:
                                ds = pd.to_datetime(old_df["date_pst"], errors="coerce").dt.date.dropna().unique().tolist()
                                old_dates.update(ds)
                    except Exception:
                        pass

            # Re-parse this one file
            raw_one = read_single_fidelity_csv(src)
            if raw_one.empty:
                fills_one = pd.DataFrame(columns=[
                    "symbol","side","quantity","signed_qty","price","timestamp",
                    "timestamp_pst","date_pst","fill_value","source_file","trade_date_et",
                    "sec_rate_per_million","sec_fee_per_share"
                ])
            else:
                raw_one["__source_file"] = str(src)
                fills_one = raw_to_fills(raw_one, sec_schedule)

            # Persist new shard
            fp = _write_fills_parquet(cache_dir, src, fills_one)

            # New dates from the rebuilt shard
            new_dates: set[date] = set()
            if "date_pst" in fills_one.columns and not fills_one.empty:
                new_dates.update(pd.to_datetime(fills_one["date_pst"], errors="coerce").dt.date.dropna().unique().tolist())

            # Union old+new -> rebuild all potentially impacted days
            affected_dates.update(old_dates)
            affected_dates.update(new_dates)

            # Summaries for index row
            if not fills_one.empty and "date_pst" in fills_one.columns:
                min_d = pd.to_datetime(fills_one["date_pst"], errors="coerce").dropna().dt.date.min()
                max_d = pd.to_datetime(fills_one["date_pst"], errors="coerce").dropna().dt.date.max()
                nrows = int(len(fills_one))
            else:
                min_d = pd.NaT
                max_d = pd.NaT
                nrows = 0

            built_rows.append({
                "source_file": str(src),
                "mtime": row.mtime,
                "size": row.size,
                "fills_path": str(fp),
                "min_date_pst": min_d if pd.notna(min_d) else pd.NaT,
                "max_date_pst": max_d if pd.notna(max_d) else pd.NaT,
                "rows": nrows,
            })

        # Replace any prior rows for these source_files and append new ones
        new_idx_rows = pd.DataFrame(built_rows)
        if not new_idx_rows.empty:
            idx = idx.loc[~idx["source_file"].isin(new_idx_rows["source_file"])].copy()
            frames = [df for df in (idx, new_idx_rows) if not df.empty]
            idx = pd.concat(frames, ignore_index=True, sort=False) if frames else new_idx_rows

        _save_index(cache_dir, idx)

    return idx, sorted(affected_dates)

def _rebuild_days_from_fills(folder: Path, idx: pd.DataFrame, days: list[date]) -> None:
    """For each PST date, gather all fills from all files for that date, compute trades & daily stats, and persist per-day parquet."""
    if not days:
        return
    cache_dir = _cache_dir_for(folder)
    fills_paths = idx["fills_path"].dropna().unique().tolist()

    # Load all per-file fills lazily then filter per day
    all_fills = []
    for fp in fills_paths:
        try:
            df = pd.read_parquet(fp, columns=[
                "symbol","side","quantity","signed_qty","price","timestamp",
                "timestamp_pst","date_pst","fill_value","source_file","trade_date_et",
                "sec_rate_per_million","sec_fee_per_share"
            ])
            all_fills.append(df)
        except Exception:
            continue

    if not all_fills:
        for d in days:
            (cache_dir / DAILY_TRADES_DIR / f"{d.isoformat()}.parquet").unlink(missing_ok=True)
            (cache_dir / DAILY_STATS_DIR / f"{d.isoformat()}.parquet").unlink(missing_ok=True)
        return

    big = pd.concat(all_fills, ignore_index=True) if len(all_fills) > 1 else all_fills[0]

    # --- dtype guards ---
    if "date_pst" in big.columns:
        big["date_pst"] = pd.to_datetime(big["date_pst"], errors="coerce").dt.date
    else:
        for d in days:
            (cache_dir / DAILY_TRADES_DIR / f"{d.isoformat()}.parquet").unlink(missing_ok=True)
            (cache_dir / DAILY_STATS_DIR / f"{d.isoformat()}.parquet").unlink(missing_ok=True)
        return

    if "timestamp" in big.columns and not pd.api.types.is_datetime64_any_dtype(big["timestamp"]):
        big["timestamp"] = pd.to_datetime(big["timestamp"], errors="coerce")

    for d in days:
        # Subset to that PST date and sort by timestamp to preserve fills order
        subset = big.loc[(big["date_pst"].values == d)].sort_values("timestamp")
        if subset.empty:
            (cache_dir / DAILY_TRADES_DIR / f"{d.isoformat()}.parquet").unlink(missing_ok=True)
            (cache_dir / DAILY_STATS_DIR / f"{d.isoformat()}.parquet").unlink(missing_ok=True)
            continue

        # Compute trades & stats with your existing pure functions
        tdf = fills_to_trades(subset)
        ddf = compute_daily_stats(tdf, subset)

        # 🚫 Drop any per-day numbering so final numbering is assigned globally on load
        tdf = tdf.drop(columns=["trade_id"], errors="ignore")

        tdf_out = cache_dir / DAILY_TRADES_DIR / f"{d.isoformat()}.parquet"
        tdf.to_parquet(tdf_out, index=False)

        ddf_out = cache_dir / DAILY_STATS_DIR / f"{d.isoformat()}.parquet"
        ddf.to_parquet(ddf_out, index=False)


def _load_all_trades_daily_from_cache(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate per-day trades & stats parquet across all days; ALWAYS returns (trades_df, daily_df).
    Never returns None.
    """
    cache_dir = _cache_dir_for(folder)  # ensures dirs exist

    tdir = cache_dir / DAILY_TRADES_DIR
    sdir = cache_dir / DAILY_STATS_DIR

    # Safe defaults with expected schemas
    trades_cols = [
        "trade_id","symbol","direction","entry_time","exit_time","shares",
        "entry_price","exit_price","profit","profit_net","fee_total",
        "cents_per_share","time_in_trade","max_abs_pos"
    ]
    daily_cols = [
        "date","daily_profit","fees","num_trades","wins","losses",
        "win_rate","gross_profit","gross_loss","profit_factor",
        "volume_shares","volume_dollars","day"
    ]

    trades_frames: list[pd.DataFrame] = []
    stats_frames: list[pd.DataFrame] = []

    # Gather trades shards
    if tdir.exists():
        for fp in sorted(tdir.glob("*.parquet")):
            try:
                try:
                    df = pd.read_parquet(fp, columns=trades_cols)
                except Exception:
                    # Newer shards may omit some columns (e.g., trade_id). Read all and align later.
                    df = pd.read_parquet(fp)
                if not isinstance(df, pd.DataFrame):
                    continue
                trades_frames.append(df)
            except Exception:
                # Corrupt/missing shard: skip
                continue

    # Gather daily stats shards
    if sdir.exists():
        for fp in sorted(sdir.glob("*.parquet")):
            try:
                df = pd.read_parquet(fp, columns=daily_cols)
                if not isinstance(df, pd.DataFrame):
                    continue
                stats_frames.append(df)
            except Exception:
                continue

    # Concatenate or return empty frames
    if trades_frames:
        trades = pd.concat(trades_frames, ignore_index=True, sort=False)
    else:
        trades = pd.DataFrame(columns=trades_cols)

    if stats_frames:
        daily = pd.concat(stats_frames, ignore_index=True, sort=False)
    else:
        daily = pd.DataFrame(columns=daily_cols)

    # --- Normalize dtypes & assign global trade_id for uniqueness ---
    if not trades.empty:
        # Ensure datetimes for stable sort
        if "entry_time" in trades and not pd.api.types.is_datetime64_any_dtype(trades["entry_time"]):
            trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
        if "exit_time" in trades and not pd.api.types.is_datetime64_any_dtype(trades["exit_time"]):
            trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")

        # Compact dtypes to save RAM and speed df ops
        if "symbol" in trades:
            trades["symbol"] = trades["symbol"].astype("category")
        if "direction" in trades:
            trades["direction"] = trades["direction"].astype("category")


        # Stable global ordering, then unique IDs
        sort_cols = [c for c in ["entry_time", "symbol", "exit_time", "shares"] if c in trades.columns]
        if sort_cols:
            trades = trades.sort_values(sort_cols, na_position="last").reset_index(drop=True)
        else:
            trades = trades.reset_index(drop=True)

        # Drop any existing per-day IDs before reassigning (if present)
        trades = trades.drop(columns=["trade_id"], errors="ignore")
        import numpy as np  # safe local import
        trades["trade_id"] = np.arange(1, len(trades) + 1)

    # Ensure required columns exist even if empty
    for col in trades_cols:
        if col not in trades.columns:
            trades[col] = pd.Series(dtype="float64") if col != "symbol" else pd.Series(dtype="object")
    trades = trades[trades_cols]

    for col in daily_cols:
        if col not in daily.columns:
            daily[col] = pd.Series(dtype="float64") if col != "date" and col != "day" else pd.Series(dtype="object")
    daily = daily[daily_cols]

    return trades, daily

def _sec_sched_hash(df: pd.DataFrame) -> tuple:
    if df is None or df.empty:
        return ()
    cols = [c for c in ["effective_from","end_date","rate_per_million"] if c in df.columns]
    slim = df[cols].copy()
    # Make serializable & stable
    slim["effective_from"] = pd.to_datetime(slim["effective_from"], errors="coerce").astype("int64")
    if "end_date" in slim:
        slim["end_date"] = pd.to_datetime(slim["end_date"], errors="coerce").astype("int64")
    slim["rate_per_million"] = pd.to_numeric(slim["rate_per_million"], errors="coerce").fillna(0.0).round(10)
    return tuple(map(tuple, slim.fillna(0).to_numpy()))


# -------------------------
# Fidelity CSV parsing
# -------------------------

CSV_COLUMNS = [
    "Symbol",
    "Status",
    "Trade Description",
    "Quantity",
    "Order Time",
    "Trade Type",
]


@st.cache_data(show_spinner=False)
def read_single_fidelity_csv(path: Path) -> pd.DataFrame:
    """Read a single Fidelity CSV file into a raw DataFrame (unparsed),
    robust to multiline fields. Skips first 3 header lines."""
    if not path.exists() or not path.is_file():
        return pd.DataFrame(columns=CSV_COLUMNS)

    b = path.read_bytes()
    text = b.decode("utf-8", errors="replace")
    lines = text.splitlines(True)
    content = "".join(lines[3:])  # drop header lines
    rdr = csv.reader(io.StringIO(content))
    rows = list(rdr)
    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)
    header = rows[0]
    data_rows = rows[1:]
    df = pd.DataFrame(data_rows, columns=header)
    # Normalize friend’s schema: prefer 'Trade Type'
    if "Order Type" in df.columns and "Trade Type" not in df.columns:
        df = df.rename(columns={"Order Type": "Trade Type"})

    # Some files may contain trailing empty rows
    return df.dropna(how="all").reset_index(drop=True)

def _write_fills_parquet(cache_dir: Path, src_csv: Path, fills: pd.DataFrame) -> Path:
    key = _file_key(src_csv)
    out = cache_dir / FILE_FILLS_DIR / f"{key}.parquet"
    # Keep the same columns your app expects in downstream steps
    fills.to_parquet(out, index=False)
    return out

def _read_fills_parquet(cache_dir: Path, src_csv: Path) -> pd.DataFrame:
    key = _file_key(src_csv)
    fp = cache_dir / FILE_FILLS_DIR / f"{key}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    return pd.DataFrame(
        columns=[
            "symbol","side","quantity","signed_qty","price","timestamp",
            "timestamp_pst","date_pst","fill_value","source_file","trade_date_et",
            "sec_rate_per_million","sec_fee_per_share"
        ]
    )


@st.cache_data(show_spinner=False)
def load_all_raw(folder: Path, sig: Tuple[Tuple[str, float, int], ...]) -> pd.DataFrame:
    """Load and concatenate all raw CSVs using the provided signature as cache key."""
    if not Path(folder).exists():
        return pd.DataFrame(columns=CSV_COLUMNS + ["__source_file"])

    frames = []
    for f, _, _ in sig:
        df = read_single_fidelity_csv(Path(f))
        df["__source_file"] = f
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=CSV_COLUMNS + ["__source_file"])

    out = pd.concat(frames, ignore_index=True)

    # ---- Memory/CPU optimizations (behavior-preserving) ----
    # Convert frequently-repeated text columns to categoricals
    cat_cols = ["Symbol", "Status", "Trade Description", "Trade Type", "Order Type"]
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category")

    # Keep order time as fast/compact string dtype and trim whitespace
    if "Order Time" in out.columns:
        # pandas StringDtype is faster/lighter than generic object strings
        out["Order Time"] = out["Order Time"].astype("string").str.strip()

    # Source file is highly repetitive → categorical
    if "__source_file" in out.columns:
        out["__source_file"] = out["__source_file"].astype("category")

    return out


@st.cache_data(show_spinner=False)
def load_sec_rate_schedule(path_str: str) -> pd.DataFrame:
    """
    Load SEC rate schedule CSV with columns like:
      start_date, [end_date], rate_per_million
    Returns a sorted frame with datetime64[ns] keys suitable for merge_asof.
    """
    try:
        df = pd.read_csv(path_str)
    except Exception as e:
        st.error(f"Failed to load SEC rate schedule from {path_str}: {e}")
        return pd.DataFrame(columns=["effective_from", "end_date", "rate_per_million"])

    # Basic column presence
    if "start_date" not in df or "rate_per_million" not in df:
        st.error(
            "SEC schedule must include at least 'start_date' and 'rate_per_million'."
        )
        return pd.DataFrame(columns=["effective_from", "end_date", "rate_per_million"])

    df = df.copy()

    # Convert to datetime64[ns], tz-naive, normalized (00:00:00)
    eff = pd.to_datetime(df["start_date"], errors="coerce")
    eff = eff.dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT")
    eff = eff.dt.normalize()
    df["effective_from"] = eff

    if "end_date" in df:
        endd = pd.to_datetime(df["end_date"], errors="coerce")
        endd = endd.dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT")
        endd = endd.dt.normalize()
        df["end_date"] = endd
    else:
        df["end_date"] = pd.NaT

    df["rate_per_million"] = pd.to_numeric(
        df["rate_per_million"], errors="coerce"
    ).fillna(0.0)

    # Sort by effective_from ascending
    df = df.sort_values("effective_from").reset_index(drop=True)

    return df[["effective_from", "end_date", "rate_per_million"]]


# -------------------------
# Transform raw -> fills
# -------------------------

SIDE_MAP = {
    "sell short": "SELL_SHORT",
    "buy to cover": "BUY_TO_COVER",
    "buy": "BUY",
    "sell": "SELL",
}


@st.cache_data(show_spinner=False)
def raw_to_fills(raw: pd.DataFrame, sec_schedule: pd.DataFrame) -> pd.DataFrame:
    """Parse Fidelity rows into executed fills with clean columns:
    [symbol, side, quantity, signed_qty, price, timestamp, fill_value, source_file]"""
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "quantity",
                "signed_qty",
                "price",
                "timestamp",
                "fill_value",
                "source_file",
            ]
        )

    df = raw.loc[raw["Status"].str.contains("FILLED", na=False)].copy()

    # --- Price (last number in Status) ---
    status_clean = df["Status"].fillna("").str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(
        status_clean.str.extract(r"([0-9]*\.?[0-9]+)(?!.*[0-9])", expand=False),
        errors="coerce",
    )

    # --- Side from Trade Description (prefix match, else contains, else UNKNOWN) ---
    tdesc = df["Trade Description"].fillna("").str.lower().str.strip()

    # prefix-first, then contains, falling back to UNKNOWN
    is_sell_short = tdesc.str.startswith("sell short")
    is_buy_to_cover = tdesc.str.startswith("buy to cover")
    is_buy = tdesc.str.startswith("buy")
    is_sell = tdesc.str.startswith("sell")

    side = np.select(
        [is_sell_short, is_buy_to_cover, is_buy, is_sell],
        ["SELL_SHORT", "BUY_TO_COVER", "BUY", "SELL"],
        default=None,
    )

    # fallback: contains
    unknown = pd.isna(side)
    if unknown.any():
        s = tdesc[unknown]
        side2 = np.select(
            [
                s.str.contains("sell short"),
                s.str.contains("buy to cover"),
                s.str.contains(r"\bbuy\b"),
                s.str.contains(r"\bsell\b"),
            ],
            ["SELL_SHORT", "BUY_TO_COVER", "BUY", "SELL"],
            default="UNKNOWN",
        )
        side[unknown] = side2

    df["side"] = side

    # --- Quantity (support both schemas) ---
    if "Quantity" in df.columns and df["Quantity"].notna().any():
        qsrc = df["Quantity"].astype(str).str.replace(",", "", regex=False)
        qty = qsrc.str.extract(r"(-?\d+)", expand=False)
    else:
        # Friend's CSV: quantity is embedded in Trade Description, e.g. "Sell 4100 at Market"
        desc = df["Trade Description"].astype(str).str.lower().str.replace(",", "", regex=False)

        # Primary: number right after side verb (handles: sell short / buy to cover / buy / sell)
        qty = desc.str.extract(r"(?:sell short|buy to cover|buy|sell)\s+(-?\d+)\b", expand=False)

        # Fallback: last integer in the string (belt & suspenders)
        qty = qty.fillna(desc.str.extract(r"(-?\d+)\b(?!.*\d)", expand=False))

    df["quantity"] = pd.to_numeric(qty, errors="coerce").astype("Int64")


    # --- Timestamp: "hh:mm:ss AM\nMM/DD/YYYY" (Eastern) → Pacific ---
    ts_src = df["Order Time"].astype(str)
    parts = ts_src.str.replace("\r", " ", regex=False).str.replace(
        "\n", " ", regex=False
    )
    parts = parts.str.replace(r"\s+", " ", regex=True).str.strip()
    # Grab two pieces and join for robust parse
    ext = parts.str.extract(r"(\d{1,2}:\d{2}:\d{2}\s*[AP]M)\s+(\d{2}/\d{2}/\d{4})")
    dt = pd.to_datetime(
        ext[0] + " " + ext[1], format="%I:%M:%S %p %m/%d/%Y", errors="coerce"
    )
    # Localize to Eastern then convert to Pacific
    dt = dt.dt.tz_localize(
        TIMEZONE_EASTERN, nonexistent="NaT", ambiguous="NaT"
    ).dt.tz_convert(TIMEZONE_PACIFIC)
    df["timestamp"] = dt

    # --- Signed qty ---
    side_to_sign = {"BUY": 1, "BUY_TO_COVER": 1, "SELL": -1, "SELL_SHORT": -1}
    df["signed_qty"] = (
        df["side"].map(side_to_sign).fillna(0).astype(int) * df["quantity"].abs()
    )

    # --- Other columns ---
    df["symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df["fill_value"] = df["price"] * df["quantity"].abs()
    df["source_file"] = df.get("__source_file", "")

    # --- SEC fee rate join (by ET trade date, effective_from) ---

    # Build PST helpers once so they're always present
    df["timestamp_pst"] = df["timestamp"].dt.tz_convert(TIMEZONE_PACIFIC)
    df["date_pst"] = df["timestamp_pst"].dt.date

    # Always build ET trade date as tz-naive, normalized datetime64[ns]
    df["trade_date_et"] = (
        df["timestamp"]
        .dt.tz_convert(TIMEZONE_EASTERN)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if sec_schedule is None or sec_schedule.empty:
        # No schedule: fill zeros
        df["sec_rate_per_million"] = 0.0
        df["sec_fee_per_share"] = 0.0
    else:
        # Ensure schedule keys are datetime64[ns] and sorted
        rates = sec_schedule.dropna(subset=["effective_from"]).copy()
        rates = rates.sort_values("effective_from")

        # Backward-asof join: most recent effective rate on/before ET trade date
        left = df.sort_values("trade_date_et")
        right = rates  # already sorted

        joined = pd.merge_asof(
            left,
            right,
            left_on="trade_date_et",
            right_on="effective_from",
            direction="backward",
        )

        # If end_date exists and trade_date exceeds it, null out the rate
        if "end_date" in joined.columns:
            mask_out = joined["trade_date_et"] > joined["end_date"]
            joined.loc[mask_out, "rate_per_million"] = np.nan

        # Per-share fee only for SELL-side executions (negative signed_qty)
        joined["sec_fee_per_share"] = np.where(
            joined["signed_qty"] < 0,
            joined["price"] * (joined["rate_per_million"].fillna(0.0) / 1_000_000.0),
            0.0,
        )

        df = joined
        df["sec_rate_per_million"] = df["rate_per_million"].fillna(0.0)
        df["sec_fee_per_share"] = df["sec_fee_per_share"].fillna(0.0)

    # Select columns (defined unconditionally)
    cols = [
        "symbol",
        "side",
        "quantity",
        "signed_qty",
        "price",
        "timestamp",
        "timestamp_pst",
        "date_pst",
        "fill_value",
        "source_file",
        "trade_date_et",
        "sec_rate_per_million",
        "sec_fee_per_share",
    ]

    return (
        df.dropna(subset=["timestamp", "price"])[cols]
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )



# -------------------------
# Trade reconstruction
# -------------------------


@dataclass
class Trade:
    trade_id: int
    symbol: str
    direction: str  # 'long' or 'short'
    entry_time: datetime
    exit_time: datetime | None = None
    entry_qty: int = 0
    exit_qty: int = 0
    entry_value: float = 0.0
    exit_value: float = 0.0
    max_abs_pos: int = 0
    fee_total: float = 0.0

    def add_entry(self, qty: int, price: float):
        self.entry_qty += qty
        self.entry_value += qty * price
        cur_pos = self.entry_qty - self.exit_qty
        self.max_abs_pos = max(self.max_abs_pos, abs(cur_pos))

    def add_exit(self, qty: int, price: float):
        self.exit_qty += qty
        self.exit_value += qty * price
        cur_pos = self.entry_qty - self.exit_qty
        self.max_abs_pos = max(self.max_abs_pos, abs(cur_pos))


@st.cache_data(show_spinner=False)
def fills_to_trades(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "symbol",
                "direction",
                "entry_time",
                "exit_time",
                "shares",
                "entry_price",
                "exit_price",
                "profit",
                "cents_per_share",
                "time_in_trade",
                "max_abs_pos",
            ]
        )

    trades: List[Trade] = []
    current: Dict[str, Trade | None] = {}
    position: Dict[str, int] = {}
    next_id = 1

    # Precompute column indices for fast + safe tuple access
    _cols = {c: i for i, c in enumerate(fills.columns)}

    def _get(row, name, default=None):
        """Tuple-safe column getter by name with a default."""
        idx = _cols.get(name)
        if idx is None:
            return default
        val = row[idx]
        return default if val is None else val

    for row in fills.sort_values("timestamp").itertuples(index=False, name=None):
        # Read fields by name via index map (works even if column order changes)
        sym = _get(row, "symbol")
        ts = _get(row, "timestamp")
        price = float(_get(row, "price", 0.0) or 0.0)
        qty_remaining = int(_get(row, "signed_qty", 0) or 0)

        # Optional columns with sensible defaults (older data may not have this)
        sec_fee_per_share = float(_get(row, "sec_fee_per_share", 0.0) or 0.0)

        # init per-symbol books
        if sym not in position:
            position[sym] = 0
            current[sym] = None

        # consume this fill against the open position/trade
        while qty_remaining != 0:
            pos = position[sym]
            tr = current[sym]

            # open a new trade if flat or none
            if pos == 0 or tr is None:
                direction = "long" if qty_remaining > 0 else "short"
                tr = Trade(
                    trade_id=next_id, symbol=sym, direction=direction, entry_time=ts
                )
                current[sym] = tr
                next_id += 1
                pos = position[sym]  # pos is still 0 here

            if tr.direction == "long":
                if qty_remaining > 0:
                    # add to long entry
                    tr.add_entry(qty_remaining, price)
                    position[sym] = pos + qty_remaining
                    qty_remaining = 0
                else:
                    # sell reduces/flat/closes long
                    sell_qty = min(-qty_remaining, pos)
                    if sell_qty > 0:
                        tr.add_exit(sell_qty, price)
                        # SEC fee accrues on shares sold
                        tr.fee_total += float(sell_qty) * sec_fee_per_share
                        position[sym] = pos - sell_qty
                        qty_remaining += sell_qty  # qty_remaining is negative
                        pos = position[sym]
                    if pos == 0:
                        tr.exit_time = ts
                        trades.append(tr)
                        current[sym] = None
                    # leftover negative qty (if any) flips on next loop
            else:  # short
                if qty_remaining < 0:
                    # add to short entry
                    tr.add_entry(-qty_remaining, price)
                    # SEC fee accrues on shares sold short (entry side for shorts)
                    tr.fee_total += float(-qty_remaining) * sec_fee_per_share
                    position[sym] = pos + qty_remaining  # qty_remaining negative
                    qty_remaining = 0
                else:
                    # buy-to-cover reduces/flat/closes short
                    cover_qty = min(qty_remaining, -pos)
                    if cover_qty > 0:
                        tr.add_exit(cover_qty, price)
                        position[sym] = pos + cover_qty
                        qty_remaining -= cover_qty
                        pos = position[sym]
                    if pos == 0:
                        tr.exit_time = ts
                        trades.append(tr)
                        current[sym] = None
                    # leftover positive qty (if any) flips on next loop

    # Summarize trades
    rows = []
    for tr in trades:
        entry_px = tr.entry_value / tr.entry_qty if tr.entry_qty else np.nan
        exit_px = tr.exit_value / tr.exit_qty if tr.exit_qty else np.nan

        # Gross P/L (unchanged math)
        if tr.direction == "long":
            pnl_gross = tr.exit_value - tr.entry_value
        else:
            pnl_gross = tr.entry_value - tr.exit_value

        # Net P/L = Gross - fees
        pnl_net = pnl_gross - tr.fee_total

        # Use NET for cps by default (your new global default)
        cps_net = (pnl_net / tr.entry_qty * 100.0) if tr.entry_qty else np.nan

        rows.append(
            {
                "trade_id": tr.trade_id,
                "symbol": tr.symbol,
                "direction": tr.direction,
                "entry_time": tr.entry_time,
                "exit_time": tr.exit_time,
                "shares": tr.entry_qty,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "profit": pnl_gross,  # keep GROSS for reference
                "profit_net": pnl_net,  # NEW: default metric elsewhere
                "fee_total": tr.fee_total,  # NEW: show fees where sensible
                "cents_per_share": cps_net,  # NET cps displayed by default
                "time_in_trade": (
                    (tr.exit_time - tr.entry_time)
                    if (tr.exit_time and tr.entry_time)
                    else pd.Timedelta(0)
                ),
                "max_abs_pos": tr.max_abs_pos,
            }
        )

    trades_df = (
        pd.DataFrame(rows)
        .sort_values(["entry_time", "trade_id"])
        .reset_index(drop=True)
    )
    return trades_df


# -------------------------
# Daily stats, overall stats
# -------------------------


@st.cache_data(show_spinner=False)
def compute_daily_stats(
    trades_df: pd.DataFrame, fills_df: pd.DataFrame
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "daily_profit",  # NET (default)
                "fees",  # NEW
                "num_trades",
                "wins",
                "losses",
                "win_rate",
                "gross_profit",  # from GROSS trades
                "gross_loss",
                "profit_factor",
                "volume_shares",
                "volume_dollars",
                "day",
            ]
        )

    trades = trades_df.copy()
    fills = fills_df.copy()

    # Dates in Pacific (reuse precomputed for speed)
    # trades exit_time is PST convertible; fill date is already precomputed
    trades["date"] = trades["exit_time"].dt.tz_convert(TIMEZONE_PACIFIC).dt.date
    if "date_pst" in fills.columns:
        fills["date"] = fills["date_pst"]
    else:
        fills["date"] = fills["timestamp"].dt.tz_convert(TIMEZONE_PACIFIC).dt.date

    # Trade-based metrics
    grp = trades.groupby("date", as_index=False)
    base = grp.agg(
        daily_profit=("profit_net", "sum"),  # NET default
        fees=("fee_total", "sum"),  # NEW daily fees
        num_trades=("trade_id", "count"),
        wins=("profit_net", lambda s: (s > 0).sum()),  # wins by NET P/L
        losses=("profit_net", lambda s: (s < 0).sum()),
        gross_profit=("profit", lambda s: s[s > 0].sum()),  # still report gross
        gross_loss=("profit", lambda s: -s[s < 0].sum()),
    )

    base["win_rate"] = (base["wins"] / base["num_trades"]).astype(float)
    base["profit_factor"] = np.where(
        base["gross_loss"].gt(0), base["gross_profit"] / base["gross_loss"], np.nan
    )

    # Volume from fills
    vol = fills.groupby("date", as_index=False).agg(
        volume_shares=("quantity", lambda s: s.abs().sum()),
        volume_dollars=("fill_value", lambda s: s.abs().sum()),
    )

    out = (
        base.merge(vol, on="date", how="left")
        .assign(day=lambda d: pd.to_datetime(d["date"]).dt.strftime("%a"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Downcast for faster transport/render
    for c in ["daily_profit", "fees", "gross_profit", "gross_loss", "volume_dollars"]:
        out[c] = pd.to_numeric(out[c], errors="coerce", downcast="float")
    for c in ["num_trades", "wins", "losses", "volume_shares"]:
        out[c] = pd.to_numeric(out[c], errors="coerce", downcast="integer")

    return out


# -------------------------
# Calendar utilities
# -------------------------

def nyse_holidays_named(start_dt: date, end_dt: date) -> dict[date, str]:
    names: dict[date, str] = {}

    # Try to load an NYSE calendar (XNYS is the modern code; NYSE for older installs)
    cal = None
    if mcal is not None:
        try:
            cal = mcal.get_calendar("XNYS")
        except Exception:
            try:
                cal = mcal.get_calendar("NYSE")
            except Exception:
                cal = None

    # If we have a calendar, mark weekdays that are NOT in the schedule as closed
    if cal is not None:
        try:
            sched = cal.schedule(start_date=start_dt, end_date=end_dt)
            all_days = pd.date_range(start_dt, end_dt, freq="D")
            open_days = set(pd.to_datetime(sched.index).date)
            for dts in all_days:
                dd = dts.date()
                if dd.weekday() < 5 and dd not in open_days:
                    names[dd] = (
                        US_HOL.get(dd, "Market Holiday")
                        if "US_HOL" in globals() and US_HOL
                        else "Market Holiday"
                    )
            return names
        except Exception:
            # fall through to holidays-only fallback
            pass

    # Fallback: no market calendar — use the holidays package, weekdays only
    if "US_HOL" in globals() and US_HOL:
        for dts in pd.date_range(start_dt, end_dt, freq="D"):
            dd = dts.date()
            if dd.weekday() < 5 and dd in US_HOL:
                names[dd] = US_HOL.get(dd, "Market Holiday")

    return names

# ---- Unified trading-day helpers ----
def _today_pacific() -> date:
    return datetime.now(TIMEZONE_PACIFIC).date()

def _month_end(d: date) -> date:
    # last calendar day of that month
    from calendar import monthrange
    return date(d.year, d.month, monthrange(d.year, d.month)[1])

def trading_days_between(start_d: date, end_d: date) -> int:
    """
    Count market TRADING days (NYSE) between start_d and end_d inclusive.
    Prefers pandas_market_calendars if available; otherwise falls back to
    weekdays minus known/derived closures.
    """
    if end_d < start_d:
        return 0

    # Best: use NYSE schedule if pandas_market_calendars (mcal) is available
    if mcal is not None:
        try:
            cal = mcal.get_calendar("XNYS")
        except Exception:
            try:
                cal = mcal.get_calendar("NYSE")
            except Exception:
                cal = None
        if cal is not None:
            try:
                sched = cal.schedule(start_date=start_d, end_date=end_d)
                return int(len(sched.index))
            except Exception:
                pass  # fall through to weekday/holiday logic

    # Fallback: weekdays minus closures from our holiday/closure map
    closed = set(nyse_holidays_named(start_d, end_d).keys())

    # Ensure Good Friday is excluded even if holidays lib lacks it
    try:
        from dateutil.easter import easter as _easter
        for y in range(start_d.year, end_d.year + 1):
            gf = _easter(y) - timedelta(days=2)
            if start_d <= gf <= end_d and gf.weekday() < 5:
                closed.add(gf)
    except Exception:
        pass

    return sum(
        1
        for n in range((end_d - start_d).days + 1)
        if (start_d + timedelta(days=n)).weekday() < 5
        and (start_d + timedelta(days=n)) not in closed
    )


def build_month_grid(year: int, month: int) -> List[List[date]]:
    """Return a month grid with weeks Sunday→Saturday (5–6 rows)."""
    first = date(year, month, 1)

    # Sunday on/before the first (Monday=0..Sunday=6)
    start = first - timedelta(days=(first.weekday() + 1) % 7)

    # Last day of month
    last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(
        days=1
    )

    # Saturday on/after the last (Saturday=5)
    end = last + timedelta(days=(5 - last.weekday()) % 7)

    grid: List[List[date]] = []
    cur = start
    while cur <= end:
        # Week layout: Sun, Mon, Tue, Wed, Thu, Fri, Sat
        week = [cur + timedelta(days=i) for i in range(7)]
        grid.append(week)
        cur += timedelta(days=7)

    return grid


@st.cache_data(show_spinner=False)
def _sec_rate_help_cached(path_guess: str) -> str:
    return _sec_rate_help_from_csv(path_guess)


def _sec_rate_help_from_csv(path_guess: str) -> str:
    """Return a helper string like 'Current SEC rate: $27.80 per $1M' from the CSV."""
    try:
        sched = load_sec_rate_schedule(path_guess)
    except Exception:
        sched = pd.DataFrame()

    if sched is None or sched.empty:
        return "Current SEC rate: —"

    # Today in ET, schedule uses tz-naive normalized datetimes
    today_et = pd.Timestamp(datetime.now(TIMEZONE_EASTERN).date())

    # Pick the row whose [effective_from, end_date] window contains today (fallbacks included)
    cur = (
        sched[
            (sched["effective_from"] <= today_et)
            & (sched["end_date"].isna() | (today_et <= sched["end_date"]))
        ]
        .sort_values("effective_from")
        .tail(1)
    )

    if cur.empty:
        # fallback: most recent <= today, else last row
        cur = (
            sched[sched["effective_from"] <= today_et]
            .sort_values("effective_from")
            .tail(1)
        )
        if cur.empty:
            cur = sched.tail(1)

    rate = float(cur["rate_per_million"].iloc[0])
    return f"Current SEC rate: ${rate:,.2f} per $1M"


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Trade Tracker", layout="wide")
inject_theme_css()


# Sidebar title (compact, above navigation)
st.sidebar.markdown(
    """
    <style>
      .side-title {
        font-size: 1.3rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 6px 0 6px 0;
        text-wrap: balance;
      }
    </style>
    <div class="side-title">💸&nbsp;Trade Tracker</div>
    """,
    unsafe_allow_html=True,
)

nav = st.sidebar.radio(
    "Navigate",
    (
        "📅 Calendar",
        "📈 Equity Curves",
        "🌄 Daily Stats",
        "🎉 Year View",
        "📒 Trade Log",
        "📊 Overview Dash",
    ),
    index=0,  # default = Calendar
)


# Sidebar controls
with st.sidebar:
    st.header("Settings")

    # Force refresh at the very top
    refresh = st.button("🔄 Force refresh", width="content")

    # Consolidated selectors: stacked with no divider
    # Inside: with st.sidebar:
    folder_str = st.text_input(
        "Raw CSVs",
        value=DEFAULT_FOLDER,
        help="Folder containing your daily Fidelity order-history CSV files.",
    )

    # compute helper from the *current* path (session) or default
    _sec_path_guess = st.session_state.get("sec_path", SEC_RATE_SCHEDULE_PATH_DEFAULT)
    _sec_help = _sec_rate_help_cached(_sec_path_guess)

    sec_path = st.text_input(
        "Fee Schedule",
        value=SEC_RATE_SCHEDULE_PATH_DEFAULT,
        help=_sec_help,  # <- now shows the current rate
        key="sec_path",  # <- ensures helper refreshes after edits
    )

    save_defaults = st.button("💾 Save config file", use_container_width=True)
    if save_defaults:
        save_user_config({"raw_csvs": folder_str, "sec_schedule": sec_path})
        st.success("Saved defaults for this device.")

    folder = Path(folder_str).expanduser()
    sec_schedule = load_sec_rate_schedule(sec_path)


# Load data (with caching keyed by folder signature)
# -------- Incremental loader (per-file + per-day sharded) --------
if not folder.exists():
    st.warning(f"Folder not found: {folder}")
    st.stop()

# Optional hard refresh clears Streamlit caches (kept) AND rebuilds impacted days below
if refresh:
    for fn in (read_single_fidelity_csv, load_all_raw, raw_to_fills, fills_to_trades, compute_daily_stats):
        try:
            fn.clear()
        except Exception:
            pass


cache_dir = _cache_dir_for(folder)

# --- Fast-change detection ---
sig = folder_signature(folder)  # tuple of (file, mtime, size)
sec_hash = _sec_sched_hash(sec_schedule)

last_sig = st.session_state.get("last_sig")
last_sec_hash = st.session_state.get("last_sec_hash")

need_rebuild = (sig != last_sig) or (sec_hash != last_sec_hash) or refresh

if need_rebuild:
    # Step 1: scan & update per-file shards
    idx, affected_dates = _scan_and_update_file_fills(folder, sec_schedule)
    # Step 2: rebuild only changed PST days
    _rebuild_days_from_fills(folder, idx, affected_dates)
    # Persist signatures for next run
    st.session_state["last_sig"] = sig
    st.session_state["last_sec_hash"] = sec_hash

# Step 3: assemble frames from per-day shards (cheap)
trades, daily = _load_all_trades_daily_from_cache(folder)

# Lightweight cache keys (change when data changes)
trades_sig = (
    int(len(trades)),
    str(pd.to_datetime(trades["exit_time"]).min()) if not trades.empty else "",
    str(pd.to_datetime(trades["exit_time"]).max()) if not trades.empty else "",
    float(pd.to_numeric(trades.get("profit_net", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "profit_net" in trades else 0.0,
)

daily_sig = (
    int(len(daily)),
    str(pd.to_datetime(daily["date"]).min()) if not daily.empty else "",
    str(pd.to_datetime(daily["date"]).max()) if not daily.empty else "",
    float(pd.to_numeric(daily.get("daily_profit", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "daily_profit" in daily else 0.0,
)


# (Optional) If first-ever run with no shards yet: fall back once to legacy path so UI still works
if trades.empty and daily.empty:
    sig = folder_signature(folder)
    raw = load_all_raw(folder, sig)
    fills = raw_to_fills(raw, sec_schedule)
    # Build ALL days once, then read from shards next time
    if not fills.empty:
        # Write per-file shard for each source file (bootstrapping cache)
        for src, df in fills.groupby("source_file", dropna=False):
            _write_fills_parquet(cache_dir, Path(src if pd.notna(src) else "unknown.csv"), df)
        # Reindex & rebuild all days present
        idx, affected_dates = _scan_and_update_file_fills(folder, sec_schedule)
        all_days = sorted(pd.to_datetime(fills["date_pst"]).dropna().dt.date.unique())
        _rebuild_days_from_fills(folder, idx, all_days)
        trades, daily = _load_all_trades_daily_from_cache(folder)

@st.cache_data(show_spinner=False)
def _build_trades_display_cached(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    view = trades_df.copy()

    # PST date/day
    exit_dt_pst = view["exit_time"].dt.tz_convert(TIMEZONE_PACIFIC)
    view["date_pst"] = exit_dt_pst.dt.strftime("%b %d, %Y")
    view["day_pst"]  = exit_dt_pst.dt.strftime("%a")

    # PST time strings
    for col in ["entry_time", "exit_time"]:
        view[col] = (
            pd.to_datetime(view[col], errors="coerce")
            .dt.tz_convert(TIMEZONE_PACIFIC)
            .dt.strftime("%I:%M:%S %p")
        )

    profit_emoji = np.where(view["profit_net"] > 0, "🟢", np.where(view["profit_net"] < 0, "🔴", "⚪"))
    view["profit_net_str"] = profit_emoji + " " + view["profit_net"].map(lambda x: f"${x:,.2f}")
    view["fees_str"]       = view["fee_total"].map(lambda x: f"${x:,.2f}")
    view["cps_str"]        = view["cents_per_share"].map(lambda x: f"{x:.2f}¢" if pd.notna(x) else "—")
    view["shares_str"]     = view["shares"].map(lambda x: f"{int(x):,}")

    td_secs = view["time_in_trade"].dt.total_seconds().fillna(0).astype("int64")
    view["duration_str"] = np.where(
        td_secs < 60,
        td_secs.astype(str) + " sec",
        (td_secs // 60).astype(str) + " min " + (td_secs % 60).astype(str) + " sec",
    )
    return view


# --- Precompute Trade Log display fields once per run ---
_trades_disp = _build_trades_display_cached(trades)

@st.cache_data(show_spinner=False)
def _sorted_trades_display(view_sig: tuple, view_df: pd.DataFrame) -> pd.DataFrame:
    # view_sig should reflect the number of rows & last trade id (or date)
    if view_df.empty:
        return view_df
    return view_df.sort_values("trade_id", ascending=False, kind="stable")

# lightweight signature so cache invalidates when trades change
_view_sig = (int(len(_trades_disp)), _trades_disp["trade_id"].max() if not _trades_disp.empty else -1)
_trades_disp = _sorted_trades_display(_view_sig, _trades_disp)


# Sidebar: Exports (only after trades/daily exist)
with st.sidebar:
    st.markdown("---")
    st.caption("Exports")
    if not trades.empty:
        st.download_button(
            "💾 Full Trade Log (CSV)",
            data=_csv_bytes_trades(trades),
            file_name="trade_log.csv",
            mime="text/csv",
            key="dl_trades_sidebar",
            width="content",
        )
    if not daily.empty:
        st.download_button(
            "💾 Full Daily Stats (CSV)",
            data=_csv_bytes_daily(daily),
            file_name="daily_stats.csv",
            mime="text/csv",
            key="dl_daily_sidebar",
            width="content",
        )



# -------------
# Trade Log tab
# -------------

@st.cache_data(show_spinner=False)
def summarize_daily_window_kpis(daily_slice: pd.DataFrame) -> dict:
    """Return KPIs for a slice of the daily stats DataFrame."""
    if daily_slice.empty:
        return dict(net=0.0, fees=0.0, trades=0, wins=0, losses=0, win_rate=0.0, profit_factor=np.nan)
    net = float(daily_slice.get("daily_profit", pd.Series(dtype=float)).sum())
    fees = float(daily_slice.get("fees", pd.Series(dtype=float)).sum()) if "fees" in daily_slice.columns else 0.0
    trades = int(daily_slice.get("num_trades", pd.Series(dtype=int)).sum()) if "num_trades" in daily_slice.columns else 0
    wins = int(daily_slice.get("wins", pd.Series(dtype=int)).sum()) if "wins" in daily_slice.columns else int(0)
    losses = int(daily_slice.get("losses", pd.Series(dtype=int)).sum()) if "losses" in daily_slice.columns else int(0)
    # Win rate: compute from sum of wins/trades to avoid recomputing per day
    win_rate = (wins / trades) if trades else 0.0
    # Profit factor: safe combine if available; else np.nan
    if "gross_profit" in daily_slice.columns and "gross_loss" in daily_slice.columns:
        gp = float(daily_slice["gross_profit"].sum())
        gl = float(daily_slice["gross_loss"].sum())
        pf = (gp / gl) if gl > 0 else np.nan
    else:
        pf = np.nan
    return dict(net=net, fees=fees, trades=trades, wins=wins, losses=losses, win_rate=win_rate, profit_factor=pf)


if nav == "📒 Trade Log":
    st.subheader("📒 Trade Log")

    if trades.empty:
        st.info(
            "No trades parsed yet. Add CSV files to your folder or adjust settings."
        )
    else:
        # ===== CSS: multiple selectors for Streamlit versions + tighter rows =====
        st.markdown(
            """
            <style>
              /* Try multiple targets: old/new Streamlit DOMs */
              .trade-log [data-testid="stDataFrame"],
              .trade-log .stDataFrame {
                height: 80vh !important;       /* responsive target */
              }
              .trade-log [data-testid="stDataFrame"] > div,
              .trade-log .stDataFrame > div {
                height: 100% !important;       /* inner container honors outer height */
              }
              /* Denser rows = more visible */
              .trade-log [data-testid="stDataFrame"] table tbody tr td,
              .trade-log [data-testid="stDataFrame"] table thead tr th,
              .trade-log .stDataFrame table tbody tr td,
              .trade-log .stDataFrame table thead tr th {
                padding-top: 4px !important;
                padding-bottom: 4px !important;
              }
              .trade-log [data-testid="stDataFrame"] table,
              .trade-log .stDataFrame table {
                line-height: 1.15 !important;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # ===== Build table data (from precomputed _trades_disp) =====
        view = _trades_disp

        if view.empty:
            st.info(
                "No trades parsed yet. Add CSV files to your folder or adjust settings."
            )
        else:
            # Build display from precomputed fields (no copies, no tz work here)
            display = view.loc[
                :,
                [
                    "trade_id",
                    "date_pst",
                    "day_pst",
                    "symbol",
                    "shares_str",
                    "entry_price",
                    "exit_price",
                    "entry_time",
                    "exit_time",
                    "profit_net_str",
                    "fees_str",
                    "cps_str",
                    "duration_str",
                    "direction",
                    "profit",  # gross, numeric
                ],
            ].rename(
                columns={
                    "trade_id": "Trade #",
                    "date_pst": "Date",
                    "day_pst": "Day",
                    "symbol": "Ticker",
                    "shares_str": "Shares",
                    "entry_price": "Entry Price",
                    "exit_price": "Exit Price",
                    "entry_time": "Entry Time",
                    "exit_time": "Exit Time",
                    "profit_net_str": "Net Profit",
                    "fees_str": "Fees",
                    "cps_str": "¢/Share (net)",
                    "duration_str": "Duration",
                    "direction": "Direction",
                    "profit": "Gross Profit",
                }
            )

            # Light formatting for the two numeric price columns and gross profit string
            display["Entry Price"] = display["Entry Price"].astype("float")
            display["Exit Price"] = display["Exit Price"].astype("float")
            display["Gross Profit"] = display["Gross Profit"].map(
                lambda x: f"${x:,.2f}"
            )

            # ===== Render (no row/column styling beyond column_config) =====
            st.markdown('<div class="trade-log">', unsafe_allow_html=True)
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Entry Price": st.column_config.NumberColumn(format="%.4f"),
                    "Exit Price": st.column_config.NumberColumn(format="%.4f"),
                    "¢/Share (net)": st.column_config.TextColumn(),
                    "Shares": st.column_config.TextColumn(),
                },
                height=700,
            )
            st.markdown("</div>", unsafe_allow_html=True)


# -------------
# Daily Stats tab
# -------------
elif nav == "🌄 Daily Stats":
    st.subheader("🌄 Daily Stats")

    if daily.empty:
        st.info(
            "No daily stats available yet. Add CSV files to your folder or adjust settings."
        )
    else:
        # ------- Filters (date range + min trades only) -------
        left, mid = st.columns([2, 2])

        with left:
            min_date = daily["date"].min()
            max_date = daily["date"].max()
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

        with mid:
            sort_by = st.selectbox(
                "Sort by",
                [
                    "Date (newest first)",
                    "Date (oldest first)",
                    "P/L (desc)",
                    "P/L (asc)",
                    "Trades (desc)",
                    "Trades (asc)",
                ],
                index=0,  # default to newest first
            )

        # ------- Apply filters -------
        start_dt, end_dt = (
            date_range if isinstance(date_range, tuple) else (date_range, date_range)
        )
        df = daily.copy()
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        sort_map = {
            "P/L (desc)": ("daily_profit", False),
            "P/L (asc)": ("daily_profit", True),
            "Trades (desc)": ("num_trades", False),
            "Trades (asc)": ("num_trades", True),
            "Date (oldest first)": ("date", True),
            "Date (newest first)": ("date", False),
        }
        col, asc = sort_map.get(sort_by, ("date", False))
        df = df.sort_values(col, ascending=asc)

        # ------- Top-line metrics for filtered range -------
        gross_pos = df.loc[df["daily_profit"] > 0, "daily_profit"].sum()
        gross_neg = -df.loc[df["daily_profit"] < 0, "daily_profit"].sum()
        profit_factor = (gross_pos / gross_neg) if gross_neg > 0 else float("nan")
        total_days = len(df)
        net_pl = df["daily_profit"].sum()
        total_fees = df["fees"].sum()

        # Win rate BY DAY (profitable days ÷ total days)
        win_rate = (df["daily_profit"] > 0).mean() if total_days > 0 else 0.0

        # Expectancy (avg P/L per day)
        expectancy = df["daily_profit"].mean() if total_days > 0 else 0.0

        # Largest win / loss (single day)
        largest_win = df["daily_profit"].max() if total_days > 0 else 0.0
        largest_loss = df["daily_profit"].min() if total_days > 0 else 0.0

        # Max drawdown over the FILTERED slice
        # (Sort by date, build equity curve, compute running max and drawdown)
        if total_days > 0:
            eq = df.sort_values("date")["daily_profit"].cumsum()
            run_max = eq.cummax()
            dd_series = eq - run_max
            max_drawdown = float(dd_series.min())  # negative number or 0
        else:
            max_drawdown = 0.0

        # Row 1
        m1, m5, m2, m3, m4 = st.columns(5)
        m1.metric("Net P/L", f"${net_pl:,.2f}")
        m5.metric("Total fees", f"${total_fees:,.2f}")
        m2.metric(
            "Profit factor", f"{profit_factor:.2f}" if pd.notna(profit_factor) else "—"
        )
        m3.metric("Win rate (by day)", f"{win_rate*100:.1f}%")
        m4.metric("Trading Days", total_days)

        # Row 2 (new)
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Expectancy (per day)", f"${expectancy:,.2f}")
        # max_drawdown is negative; render with minus sign
        n2.metric(
            "Max drawdown",
            f"-${abs(max_drawdown):,.2f}" if max_drawdown < 0 else "$0.00",
        )
        n3.metric("Largest winning day", f"${largest_win:,.2f}")
        n4.metric("Largest losing day", f"${largest_loss:,.2f}")

        st.markdown("---")

        # ------- Daily table (high-contrast with green/red shading) -------
        if df.empty:
            st.info("No rows match your current filters.")
        else:
            # --- Build emoji profit + reorder columns for display ---
            # Convert P/L to numeric (we will replace with emoji string)
            pl_vals = df["daily_profit"].astype(float)
            pl_emoji = np.where(pl_vals > 0, "🟢", np.where(pl_vals < 0, "🔴", "⚪"))

            show = df[
                [
                    "date",
                    "day",
                    "daily_profit",
                    "fees",
                    "num_trades",
                    "wins",
                    "losses",
                    "win_rate",
                    "gross_profit",
                    "gross_loss",
                    "profit_factor",
                    "volume_shares",
                    "volume_dollars",
                ]
            ].copy()

            show.rename(
                columns={
                    "date": "Date",
                    "day": "Day",
                    "daily_profit": "P/L",
                    "fees": "Fees",
                    "num_trades": "# Trades",
                    "wins": "Wins",
                    "losses": "Losses",
                    "win_rate": "Win Rate",
                    "gross_profit": "Gross +",
                    "gross_loss": "Gross −",
                    "profit_factor": "PF",
                    "volume_shares": "Volume (sh)",
                    "volume_dollars": "Volume ($)",
                },
                inplace=True,
            )

            # Formats
            show["Date"] = pd.to_datetime(show["Date"]).dt.strftime("%b %d, %Y")
            show["Gross +"] = show["Gross +"].map(lambda x: f"${x:,.2f}")
            show["Gross −"] = show["Gross −"].map(lambda x: f"${x:,.2f}")
            show["Win Rate"] = show["Win Rate"].map(lambda x: f"{x*100:.1f}%")
            show["PF"] = show["PF"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            show["Volume ($)"] = show["Volume ($)"].map(lambda x: f"${x:,.0f}")
            show["Volume (sh)"] = show["Volume (sh)"].map(lambda x: f"{x:,.0f}")
            show["Fees"] = show["Fees"].map(lambda x: f"${x:,.2f}")

            # Emoji P/L like Trade Log (no heavy Styler)
            show["P/L"] = pl_emoji + " " + pl_vals.map(lambda x: f"${x:,.2f}")

            # Reorder: keep Date/Day/P&L first, then #Trades → Win Rate → PF → Volumes
            cols_order = [
                "Date",
                "Day",
                "P/L",
                "Fees",
                "# Trades",
                "Win Rate",
                "PF",
                "Wins",
                "Losses",
                "Gross +",
                "Gross −",
                "Volume (sh)",
                "Volume ($)",
            ]
            show = show[cols_order]

            # Plain dataframe (fast)
            st.dataframe(
                show,
                width="stretch",
                hide_index=True,
                height=600,
            )

        st.markdown("---")

        # ------- Charts -------
        if not df.empty:
            c1, c2 = st.columns([2, 1])

            with c1:
                d = df.sort_values("date")
                pos = d[d["daily_profit"] >= 0]
                neg = d[d["daily_profit"] < 0]

                bar = go.Figure()
                if not pos.empty:
                    bar.add_bar(
                        x=pos["date"],
                        y=pos["daily_profit"],
                        name="Positive",
                        marker_color="#428246",
                    )
                if not neg.empty:
                    bar.add_bar(
                        x=neg["date"],
                        y=neg["daily_profit"],
                        name="Negative",
                        marker_color="#8b2d2d",
                    )

                bar.update_layout(
                    barmode="relative",
                    title="Daily P/L",
                    yaxis_title="P/L ($)",
                    xaxis_title="Date",
                    showlegend=False,
                    height=320,
                    margin=dict(l=20, r=20, t=40, b=10),
                )
                st.plotly_chart(bar, width="content")

            with c2:
                hist = px.histogram(
                    df,
                    x="daily_profit",
                    nbins=30,
                    labels={"daily_profit": "P/L ($)"},
                    title="Distribution",
                )
                hist.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(hist, width="content")

            wk_sum = (
                df.groupby("day", as_index=False)["daily_profit"]
                .sum()
                .assign(sign=lambda x: np.where(x["daily_profit"] >= 0, "pos", "neg"))
            )
            order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            wk_sum["day"] = pd.Categorical(
                wk_sum["day"], categories=order, ordered=True
            )
            wk_sum = wk_sum.sort_values("day")

            wk = px.bar(
                wk_sum,
                x="day",
                y="daily_profit",
                color="sign",
                color_discrete_map={"pos": "#428246", "neg": "#8b2d2d"},
                labels={"daily_profit": "Total P/L ($)", "day": "Weekday"},
                title="Total P/L by Weekday",
            )
            wk.update_layout(
                showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=10)
            )
            st.plotly_chart(wk, width="content")


# -------------
# Year View tab (space-optimized, one-line KPIs, 4×3 only)
# -------------
elif nav == "🎉 Year View":
    st.subheader("🎉 Year View")

    if daily.empty:
        st.info("No daily stats available yet. Add CSV files to your folder or adjust settings.")
    else:
        # ---------- Data prep ----------
        d = daily.copy()
        d["dts"] = pd.to_datetime(d["date"], errors="coerce")
        years = sorted(d["dts"].dt.year.dropna().unique().tolist())
        default_year = years[-1] if years else date.today().year

        # Compact selectbox + KPI classes
        st.markdown(
            """
            <style>
              /* Make the Year select fit its content */
              div[data-testid="stSelectbox"] { width: fit-content !important; }
              div[data-testid="stSelectbox"] > div { width: fit-content !important; }
              div[data-baseweb="select"] { min-width: 0 !important; }

              /* KPI line */
              .yv-line { display:flex; align-items:center; gap:18px; }  /* space between YTD and Fees */
              .yv-kpi  { font-weight:700; }
              .yv-pos  { color:var(--green); font-weight:800; }  /* YTD >= 0 */
              .yv-neg  { color:var(--red); font-weight:800; }  /* YTD <  0 */
              .yv-fee  { color:var(--orange); font-weight:700; }  /* Fees orange */
              .yv-right { text-align:right; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # ---------- One-line top row ----------
        # [ Year selector |  YTD + Fees + Trades  |  Days left (right edge) ]
        c_sel, c_mid, c_right = st.columns([1.05, 6, 1.6], gap="small")

        # Year selector (far left, compact)
        with c_sel:
            year = st.selectbox(
                "Year",
                options=years,
                index=years.index(default_year) if years else 0,
                key="year_view_year",
                label_visibility="collapsed",
            )

        # Slice for chosen year and compute KPIs
        ydf = d[d["dts"].dt.year == year].copy()
        _kpis = summarize_daily_window_kpis(ydf)
        ytd_net, ytd_fees, ytd_trades = _kpis["net"], _kpis["fees"], _kpis["trades"]

        # Trading days remaining in selected year (uses unified helper)
        today = _today_pacific()
        if year < today.year:
            trading_days_left = 0
        else:
            start = (today + timedelta(days=1)) if year == today.year else date(year, 1, 1)
            end   = date(year, 12, 31)
            trading_days_left = trading_days_between(start, end)


        # KPIs: YTD (colored) + Fees (orange) + Trades — one line, centered block
        net_cls = "yv-pos" if ytd_net >= 0 else "yv-neg"
        with c_mid:
            st.markdown(
                f"""
                <div class="yv-line">
                  <span class="yv-kpi {net_cls}">YTD: {fmt_money(ytd_net)}</span>
                  <span class="yv-kpi yv-fee">Fees: {fmt_money(ytd_fees)}</span>
                  <span class="yv-kpi">Trades: {ytd_trades:,}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Days left — far right
        with c_right:
            st.markdown(f'<div class="yv-kpi yv-right">Days left: {trading_days_left}</div>', unsafe_allow_html=True)

        # ---------- Year grid: fixed at 4×3 (no 6×2 toggle) ----------
        html = _year_view_html(daily, year, cols=4, compact=False)
        st.markdown(html, unsafe_allow_html=True)



# -------------
# Equity Curves tab (NEW)
# -------------
elif nav == "📈 Equity Curves":
    st.subheader("📈 Equity Curves")

    if trades.empty:
        st.info("No trades found yet. Add CSV files to your folder or adjust settings.")
    else:
        # ---- Mode toggle state (default = 'single') ----
        if "eq_mode" not in st.session_state:
            st.session_state.eq_mode = "single"  # 'single' or 'range'

        def _preset_dates(which: str) -> tuple[date, date]:
            """Return (start, end) for preset name based on available data in `daily`."""
            if daily.empty:
                today = datetime.now(TIMEZONE_PACIFIC).date()
                return today, today

            min_dt = daily["date"].min()
            max_dt = daily["date"].max()
            today = max_dt

            if which == "Year to date":
                start = date(today.year, 1, 1)
                return max(start, min_dt), today

            if which == "Month to date":
                start = date(today.year, today.month, 1)
                return max(start, min_dt), today

            if which == "First Year":
                # first 365 days from first available date
                end = min(min_dt + timedelta(days=365 - 1), max_dt)
                return min_dt, end

            if which == "First 3 months":
                # first ~90 days from first available date
                end = min(min_dt + timedelta(days=90 - 1), max_dt)
                return min_dt, end

            # Custom fallback; caller will show a picker
            return min_dt, max_dt

        # --- Available dates (by trade EXIT date in PT), newest first ---
        trade_days = (
            trades["exit_time"]
            .dt.tz_convert(TIMEZONE_PACIFIC)
            .dt.date.dropna()
            .unique()
        )
        trade_days = sorted(trade_days)
        latest_day = trade_days[-1]

        # Layout: LEFT = chart, RIGHT = controls (date + buttons)
        left_col, right_col = st.columns([5.0, 1], gap="small")

        # --- Right panel: compact date input (MM/DD/YYYY) ---
        with right_col:
            st.markdown(
                """
                <style>
                div[data-testid="stDateInput"] { width: fit-content !important; }
                div[data-baseweb="input"] { min-width: 0 !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # --- Mode switcher ---
            if st.session_state.eq_mode == "single":
                # Single-day selector
                sel_day = st.date_input(
                    "Date",
                    value=latest_day,
                    min_value=trade_days[0],
                    max_value=latest_day,
                    format="MM/DD/YYYY",
                    key="eq_single_date",
                )

                st.markdown("### ")
                make_range = st.button("Choose custom time range", width="content")
                if make_range:
                    st.session_state.eq_mode = "range"
                    st.rerun()

            else:
                # Range mode with presets
                preset = st.selectbox(
                    "Preset",
                    (
                        "Year to date",
                        "Month to date",
                        "First Year",
                        "First 3 months",
                        "Custom",
                    ),
                    index=0,
                    key="eq_range_preset",
                )
                pstart, pend = _preset_dates(preset)

                if preset == "Custom":
                    min_dt = daily["date"].min() if not daily.empty else pstart
                    max_dt = daily["date"].max() if not daily.empty else pend
                    dater = st.date_input(
                        "Date range",
                        value=(pstart, pend),
                        min_value=min_dt,
                        max_value=max_dt,
                        key="eq_custom_range",
                    )
                    if isinstance(dater, tuple) and len(dater) == 2:
                        pstart, pend = dater

                st.markdown("### ")
                back = st.button("Back to single-day mode", width="content")
                if back:
                    st.session_state.eq_mode = "single"
                    st.rerun()

        # Always NET
        tdf = trades.copy()

        if st.session_state.eq_mode == "single":
            # -------- SINGLE-DAY (existing behavior) --------
            sel_day = st.session_state.get("eq_single_date")

            # cached
            curve = daily_running_curve(trades, sel_day, trades_sig)

            # Guard (no trades that day) -> flat line
            if curve.shape[0] == 2 and curve["equity"].abs().sum() == 0:
                with left_col:
                    st.info("No closed trades on this day. Nothing to plot.")
                    st.line_chart(curve.set_index("time")["equity"])
            else:
                x_start = pd.Timestamp(
                    sel_day.year, sel_day.month, sel_day.day, 6, 30, tz=TIMEZONE_PACIFIC
                )
                last_trade_ts = curve["time"].iloc[-1]
                if last_trade_ts == x_start and float(curve["equity"].iloc[-1]) == 0.0:
                    x_end = x_start + pd.Timedelta(minutes=15)
                else:
                    x_end = _ceil_to_next_15min(last_trade_ts)

                y = curve["equity"]
                x = curve["time"]
                y_pos, y_neg = y.clip(lower=0), y.clip(upper=0)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_pos,
                        mode="lines",
                        line=dict(width=0, shape="linear"),
                        fill="tozeroy",
                        name="Gain",
                        hoverinfo="skip",
                        fillcolor="rgba(46,125,50,0.55)",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_neg,
                        mode="lines",
                        line=dict(width=0, shape="linear"),
                        fill="tozeroy",
                        name="Loss",
                        hoverinfo="skip",
                        fillcolor="rgba(139,45,45,0.55)",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(width=3, color=COL.hex("white"), shape="linear"),
                        name="Running P/L",
                    )
                )

                y_max, y_min = float(y.max()), float(y.min())
                imax, imin = y.idxmax(), y.idxmin()
                x_max = x.loc[imax]
                x_min = x.loc[imin]
                y_max = float(y.loc[imax])
                y_min = float(y.loc[imin])

                if isinstance(x_max, pd.Series):
                    x_max = x_max.iloc[0]
                    y_max = float(y.loc[imax].iloc[0])
                if isinstance(x_min, pd.Series):
                    x_min = x_min.iloc[0]
                    y_min = float(y.loc[imin].iloc[0])

                if y_max != y_min:
                    fig.add_annotation(
                        x=x_max,
                        y=y_max,
                        text=f"High: ${y_max:,.0f}",
                        showarrow=True,
                        arrowhead=2,
                        yshift=10,
                        bgcolor="rgba(0,0,0,0.5)",
                    )
                    fig.add_annotation(
                        x=x_min,
                        y=y_min,
                        text=f"Low: ${y_min:,.0f}",
                        showarrow=True,
                        arrowhead=2,
                        yshift=-10,
                        bgcolor="rgba(0,0,0,0.5)",
                    )

                net_pl = float(y.iloc[-1])  # final running P/L for the day

                title_txt = pd.Timestamp(sel_day).strftime("%B %d, %Y")
                fig.update_layout(
                    title=dict(
                        text=f"Running P/L — {title_txt}  (Net: ${net_pl:,.0f})",
                        x=0.5,
                        xanchor="center",
                    ),
                    xaxis_title="Time (PT)",
                    yaxis_title="Profit/Loss ($)",
                    autosize=False,
                    width=1000,
                    height=600,
                    margin=dict(l=118, r=40, t=80, b=80),
                    plot_bgcolor="#0e1218",
                    paper_bgcolor="#0e1218",
                    font=dict(color="#e9eef7"),
                    showlegend=False,
                )

                fig.update_yaxes(title_standoff=28, automargin=True)
                fig.update_xaxes(range=[x_start, x_end], tickformat="%H:%M")

                with left_col:
                    st.plotly_chart(fig, width=False)

                # ----- PNG export for single-day -----
                with right_col:
                    st.markdown("### ")
                    st.markdown("### ")
                    do_export = st.button("Prepare PNG", key="eq_png_single")
                    if do_export:
                        render_png_export_ui(fig, key_prefix="single", date=sel_day, basename="equity")

        else:
            # ===== Cached builder for RANGE charts (build-on-demand) =====
            @st.cache_data(show_spinner=False)
            def _build_range_equity_fig(_curve_df: pd.DataFrame, start_d: date, end_d: date, col_hex_white: str) -> go.Figure:
                # _curve_df columns expected: ["date", "equity"] from range_cumulative_curve(...)
                if _curve_df is None or _curve_df.empty:
                    return go.Figure()

                y2 = _curve_df["equity"]
                x2 = _curve_df["date"]
                net_pl_range = float(y2.iloc[-1])
                y2_pos, y2_neg = y2.clip(lower=0), y2.clip(upper=0)

                fig2 = go.Figure()
                # green fill above 0
                fig2.add_trace(
                    go.Scatter(
                        x=x2, y=y2_pos, mode="lines",
                        line=dict(width=0, shape="linear"),
                        fill="tozeroy", hoverinfo="skip",
                        fillcolor="rgba(46,125,50,0.55)", showlegend=False,
                    )
                )
                # red fill below 0
                fig2.add_trace(
                    go.Scatter(
                        x=x2, y=y2_neg, mode="lines",
                        line=dict(width=0, shape="linear"),
                        fill="tozeroy", hoverinfo="skip",
                        fillcolor="rgba(139,45,45,0.55)", showlegend=False,
                    )
                )
                # white line on top
                fig2.add_trace(
                    go.Scatter(
                        x=x2, y=y2, mode="lines",
                        line=dict(width=3, color=col_hex_white, shape="linear"),
                        name="Cumulative P/L",
                    )
                )

                # annotate high/low (safe for duplicate indices)
                y2_max, y2_min = float(y2.max()), float(y2.min())
                if y2_max != y2_min:
                    imax, imin = y2.idxmax(), y2.idxmin()

                    x2_max = x2.loc[imax]
                    x2_min = x2.loc[imin]
                    y2_max = float(y2.loc[imax])
                    y2_min = float(y2.loc[imin])

                    if isinstance(x2_max, pd.Series):
                        x2_max = x2_max.iloc[0]
                        y2_max = float(y2.loc[imax].iloc[0])
                    if isinstance(x2_min, pd.Series):
                        x2_min = x2_min.iloc[0]
                        y2_min = float(y2.loc[imin].iloc[0])

                    fig2.add_annotation(
                        x=x2_max, y=y2_max, text=f"High: ${y2_max:,.0f}",
                        showarrow=True, arrowhead=2, yshift=10,
                        bgcolor="rgba(0,0,0,0.5)",
                    )
                    fig2.add_annotation(
                        x=x2_min, y=y2_min, text=f"Low: ${y2_min:,.0f}",
                        showarrow=True, arrowhead=2, yshift=-10,
                        bgcolor="rgba(0,0,0,0.5)",
                    )

                title_txt = f"{start_d.strftime('%b %d, %Y')} → {end_d.strftime('%b %d, %Y')}"
                fig2.update_layout(
                    title=dict(text=f"Cumulative P/L — {title_txt}  (Net: ${net_pl_range:,.0f})", x=0.5, xanchor="center"),
                    xaxis_title="Date",
                    yaxis_title="Cumulative P/L ($)",
                    height=600,
                    margin=dict(l=118, r=40, t=80, b=80),
                    plot_bgcolor="#0e1218",
                    paper_bgcolor="#0e1218",
                    font=dict(color="#e9eef7"),
                    showlegend=False,
                )
                fig2.update_yaxes(title_standoff=28, automargin=True)
                return fig2

            # -------- RANGE MODE (new) --------
            preset = st.session_state.get("eq_range_preset", "Year to date")
            start_d, end_d = _preset_dates(preset)
            # If Custom was chosen earlier, honor the picked dates
            if preset == "Custom":
                dater = st.session_state.get("eq_custom_range")
                if isinstance(dater, tuple) and len(dater) == 2:
                    start_d, end_d = dater

            # Build the cumulative curve dataframe (cached upstream)
            curve_d = range_cumulative_curve(daily, start_d, end_d, daily_sig)

            with left_col:
                if curve_d.empty:
                    st.info("No days in the selected range.")
                else:
                    # Build-on-demand cached figure
                    fig2 = _build_range_equity_fig(curve_d, start_d, end_d, COL.hex("white"))
                    st.plotly_chart(fig2, width="content")

            # --- PNG export for range mode (right column), parallel to single-day export ---
            with right_col:
                st.markdown("### ")
                st.markdown("### ")
                do_export2 = st.button("Prepare PNG", key="eq_png_range")
                if do_export2 and not curve_d.empty:
                    render_png_export_ui(fig2, key_prefix="range", start=start_d, end=end_d, basename="equity")



# -------------
# Calendar tab (fix: show weekday holiday NAMES even when a daily row exists)
# -------------
elif nav == "📅 Calendar":
    st.subheader("📅 Monthly P/L")

    # --- Optional holiday support (python-holidays) ---
    # --- put these near your other imports in the Calendar tab scope ---
    try:
        import holidays as pyhol

        US_HOL = pyhol.US()
    except Exception:
        US_HOL = None

    try:
        from dateutil.easter import easter as _easter
    except Exception:
        _easter = None

    try:
        import pandas_market_calendars as pmc  # pip install pandas-market-calendars
    except Exception:
        pmc = None

    def build_holiday_map(start_d: date, end_d: date) -> dict[date, str]:
        """
        Return {date: 'Holiday Name'} for weekdays in [start_d, end_d].
        - Uses python-holidays for federal names (if available)
        - Ensures Good Friday appears by computing it directly
        - Optionally merges NYSE non-trading weekdays from pandas_market_calendars
        """
        out: dict[date, str] = {}

        # 1) Federal holiday names (readable labels)
        if US_HOL is not None:
            cur = start_d
            while cur <= end_d:
                if cur.weekday() < 5 and cur in US_HOL:
                    out[cur] = str(US_HOL.get(cur))
                cur += timedelta(days=1)

        # 2) Explicitly add Good Friday (market-only closure)
        if _easter is not None:
            for y in range(start_d.year, end_d.year + 1):
                gf = _easter(y) - timedelta(days=2)
                if start_d <= gf <= end_d and gf.weekday() < 5:
                    out[gf] = "Good Friday"

        # 3) NYSE schedule to catch other market-only closures (name fallback)
        if pmc is not None:
            cal = pmc.get_calendar("NYSE")
            sched = cal.schedule(start_date=start_d, end_date=end_d)
            session_dates = {ts.date() for ts in sched.index.to_pydatetime()}

            cur = start_d
            while cur <= end_d:
                if (
                    cur.weekday() < 5
                    and (cur not in session_dates)
                    and (cur not in out)
                ):
                    out[cur] = "Market Closed"
                cur += timedelta(days=1)

        return out

    @st.cache_data(show_spinner=False)
    def build_holiday_map_cached(year: int, month: int) -> dict[date, str]:
        """Cache holiday/closure names for a specific (year, month)."""
        first = date(year, month, 1)
        last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(
            days=1
        )
        return build_holiday_map(first, last)

    if daily.empty:
        st.info("No daily stats available to populate the calendar.")
    else:
        # ---- 4/5 : 1/5 layout for Calendar (left) and Controls (right) ----
        cal_left, cal_right = st.columns([7, 1], gap="medium")

        # ---- date ranges & labels (once) ----
        all_dates = daily["date"].sort_values().tolist()
        min_dt, max_dt = all_dates[0], all_dates[-1]
        years = list(range(min_dt.year, max_dt.year + 1))
        month_labels = [
            (m, pd.to_datetime(f"2000-{m:02d}-01").strftime("%B") + f" ({m})")
            for m in range(1, 13)
        ]
        label_to_month = {lbl: m for m, lbl in month_labels}

        # ---- right controls ----
        with cal_right:
            sel_year = st.selectbox(
                "Year",
                options=years,
                index=years.index(max_dt.year),
            )
            sel_month_label = st.selectbox(
                "Month",
                options=[lbl for _, lbl in month_labels],
                index=max_dt.month - 1,
            )
            sel_month = label_to_month[sel_month_label]

            # Trading days left this year (weekdays not holidays)
            today = _today_pacific()
            end_year = date(today.year, 12, 31)
            remaining = trading_days_between(today + timedelta(days=1), end_year)

            st.markdown("---", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="mini-kpi">
                <div class="mini-kpi__label">Trading days left in Year</div>
                <div class="mini-kpi__val">{remaining}</div>
                </div>
                <style>
                .mini-kpi {{
                    display:inline-block;
                    padding:10px 12px;
                    border:1px solid var(--blue);
                    border-radius:10px;
                    background:var(--bg-dark);
                    margin-top:6px;
                    text-align:center;
                    width:100%;
                }}
                .mini-kpi__label {{
                    font-size:14px;
                    font-weight:700;
                    color:var(--ink-subtle);
                    margin-bottom:6px;
                    text-transform:uppercase;
                    letter-spacing:0.5px;
                }}
                .mini-kpi__val {{
                    font-size:36px;
                    font-weight:900;
                    color:var(--white);
                    line-height:1.2;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Trading days left this MONTH (weekdays not holidays)
            # Month window: first → last day of *this* month
            from calendar import monthrange

            first_month_day = date(today.year, today.month, 1)
            last_month_day = date(
                today.year,
                today.month,
                monthrange(today.year, today.month)[1],
            )

            # start at tomorrow, but never before the first day of the month
            start_month = max(today + timedelta(days=1), first_month_day)

            if start_month > last_month_day:
                remaining_month = 0
            else:
                remaining_month = trading_days_between(start_month, last_month_day)


            # Render a second mini KPI just under the Year KPI
            st.markdown(
                f"""
                <div class="mini-kpi" style="margin-top:8px;">
                  <div class="mini-kpi__label">Trading days left in Month</div>
                  <div class="mini-kpi__val">{remaining_month}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        # ---- compute month boundaries + map data ----
        first = date(sel_year, sel_month, 1)
        last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(
            days=1
        )
        hols_named = build_holiday_map_cached(sel_year, sel_month)  # {date: name}

        # Convert daily index to pure date keys for a clean lookup
        dmap = {pd.to_datetime(r["date"]).date(): r for _, r in daily.iterrows()}

        # ---- header labels & order: Sunday..Saturday + weekly sums ----
        header_labels = [
            "SUNDAY",
            "MONDAY",
            "TUESDAY",
            "WEDNESDAY",
            "THURSDAY",
            "FRIDAY",
            "SATURDAY",
        ]

        # ---- compact high-contrast calendar CSS ----
        st.markdown(
            """
            <style>
            .cal-wrap { overflow-x: auto; }
            .cal-card {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0;
            box-shadow: none;
            }

            .cal-header {
                display: flex; align-items: flex-end; justify-content: space-between;
                gap: 12px; margin: 2px 2px 12px 2px;
            }
            .cal-title {
                font-size: 26px; font-weight: 900; color: var(--ink);
                letter-spacing: .3px;
            }
            /* P/L tint used inside the month title */
            .pl-pos { color: var(--green); }
            .pl-neg { color: var(--red); }
            .pl-neu { color: var(--gray); }

            .cal-month-fee {
                font-weight: 900; color: var(--orange);
                white-space: nowrap; font-size: 16px; opacity: .95;
            }
            .cal-month-fee .amt { color: var(--orange); }

            /* Table skeleton */
            table.cal { border-collapse: separate; border-spacing: 0; width: 100%; table-layout: fixed; }
            table.cal th, table.cal td { border: 1px solid var(--white-10); }

            /* Header row */
            table.cal th {
                background: var(--bg-dark);
                color: var(--ink);
                font-weight: 900;
                letter-spacing: .3px;
                font-size: 12px;
                padding: 8px 8px;
                text-align: left;
            }

            /* Cells */
            table.cal td {
            vertical-align: top;
            border: 2px solid var(--white-10);
            padding: 8px;
            background: var(--white-04);
            color: var(--ink);
            }

            /* Fixed cell height (tweak 92–104px to taste) */
            :root { --cal-cell-h: 70px; }

            table.cal td > .cell {
            height: var(--cal-cell-h);
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            overflow: hidden;            /* prevent cell growth */
            }

            /* Make all states use the same fixed inner height */
            .pos > .cell,
            .neg > .cell,
            .hol > .cell,
            .neu > .cell,
            .out > .cell,
            .wsum > .cell { height: var(--cal-cell-h); }

            /* Text compactness so it fits the fixed box */
            .daynum { font-weight: 900; font-size: 13px; line-height: 1.1; opacity: .95; }
            .pnl    { margin-top: 4px; font-weight: 900; font-size: 13px; line-height: 1.15; }
            .fees   { margin-top: 2px; font-size: 12px; line-height: 1.1; color: var(--orange); font-weight: 800; }
            .trades { font-size: 11px; line-height: 1.1; opacity: .9; margin-top: 2px;
                    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

            /* Day states */
            .pos { background-color: var(--green) !important; color: var(--white) !important; }
            .pos .daynum,.pos .pnl,.pos .trades,.pos .fees { color: var(--white) !important; }

            .neg { background-color: var(--red) !important; color: var(--white) !important; }
            .neg .daynum,.neg .pnl,.neg .trades,.neg .fees { color: var(--white) !important; }

            .hol { background-color: var(--gray) !important; color: var(--black) !important; }
            .hol .daynum,.hol .pnl,.hol .trades,.hol .fees { color: var(--black) !important; }

            /* Neutral tones */
            .neu { background-color: var(--white-04) !important; color: var(--ink) !important; }
            .out { background-color: var(--bg-dark) !important; color: var(--ink-subtle) !important; }

            /* Weekly sums column */
            .wsum { font-weight: 900; }
            .wsum .fees { font-weight: 900; }
            .wsum.pos { background-color: var(--green) !important; color: var(--white) !important; }
            .wsum.neg { background-color: var(--red) !important; color: var(--white) !important; }
            .wsum.neu { background-color: var(--white-04) !important; color: var(--ink) !important; }

            /* Spacer between grid and weekly sums (no visible border) */
            .gap { width: 10px; background: transparent; border: none !important; }

            /* Subtle hover feedback to prevent visual overlap feeling */
            table.cal td:hover { outline: 2px solid var(--white-10); outline-offset: -2px; }
            </style>
            """,
            unsafe_allow_html=True,
        )


        # ---- grid & cells ----
        grid = build_month_grid(sel_year, sel_month)

        def cell_html(d: date) -> tuple[str, float, float, int]:
            """Return (html, pnl, fees, trades) for a given day.

            IMPORTANT: We check holidays FIRST so names show up even if a daily row exists.
            """
            if d.month != sel_month:
                return '<td class="out"><div class="cell"></div></td>', 0.0, 0.0, 0

            # 1) Holiday override (weekday + named holiday)
            if d.weekday() < 5 and d in hols_named:
                name = hols_named[d]
                html = (
                    f'<td class="hol"><div class="cell">'
                    f'  <div class="daynum">{d.day}</div>'
                    f'  <div class="trades"><b>{name}</b></div>'
                    f'</div></td>'
                )
                return html, 0.0, 0.0, 0

            # 2) Normal day (use data if present)
            rec = dmap.get(d)
            if rec is not None:
                pnl = float(rec.get("daily_profit", 0.0)) or 0.0
                trades = int(rec.get("num_trades", 0)) or 0
                fees_val = float(rec.get("fees", 0.0)) or 0.0
                cls = "pos" if pnl > 0 else ("neg" if pnl < 0 else "neu")
                pnl_str = f"${pnl:,.2f}".replace("$-", "-$")
                fees_line = (
                    f'<div class="fees">Fees: ${fees_val:,.2f}</div>'
                    if fees_val > 0
                    else ""
                )
                html = (
                    f'<td class="{cls}"><div class="cell">'
                    f'  <div class="daynum">{d.day}</div>'
                    f'  <div class="pnl">{pnl_str}</div>'
                    f"  {fees_line}"
                    f'  <div class="trades">{trades} Trades</div>'
                    f'</div></td>'
                )
                return html, pnl, fees_val, trades

            # 3) Empty weekday/weekend in month
            return (
                f'<td class="neu"><div class="cell"><div class="daynum">{d.day}</div></div></td>',
                0.0, 0.0, 0,
            )


        header_cells = [f"<th>{lbl}</th>" for lbl in header_labels] + [
            '<th class="gap"></th>',
            "<th>WEEKLY SUMS</th>",
        ]
        body_rows = []
        for week in grid:
            tds = []
            wk_pnl = 0.0
            wk_trades = 0
            wk_fees = 0.0
            for d in week:
                h, p, f, t = cell_html(d)
                tds.append(h)
                if d.month == sel_month:
                    wk_pnl += p
                    wk_fees += f
                    wk_trades += t

            tds.append('<td class="gap"></td>')
            sum_cls = "pos" if wk_pnl > 0 else ("neg" if wk_pnl < 0 else "neu")
            wsum_str = f"${wk_pnl:,.2f}".replace("$-", "-$")
            fees_line = (
                f'<div class="fees">Fees: ${wk_fees:,.2f}</div>' if wk_fees > 0 else ""
            )
            tds.append(
                f'<td class="wsum {sum_cls}"><div class="cell">'
                f'  <div class="pnl">{wsum_str}</div>'
                f"  {fees_line}"
                f'  <div class="trades">{wk_trades} Trades</div>'
                f'</div></td>'
            )

            body_rows.append("<tr>" + "".join(tds) + "</tr>")

        # ---- Title: "Month YYYY — $XXXXX.XX" + right-justified Monthly Fees (if any) ----
        month_slice = daily[(daily["date"] >= first) & (daily["date"] <= last)]
        monthly_total = month_slice["daily_profit"].sum()
        monthly_fees = month_slice["fees"].sum()
        monthly_str = f"${monthly_total:,.2f}".replace("$-", "-$")
        title_month = pd.to_datetime(f"{sel_year}-{sel_month:02d}-01").strftime("%B %Y")

        pl_cls = (
            "pl-pos"
            if monthly_total > 0
            else ("pl-neg" if monthly_total < 0 else "pl-neu")
        )
        title_left_html = f'{title_month} — <span class="{pl_cls}">{monthly_str}</span>'
        title_right_html = (
            f'<div class="cal-month-fee">Month Fees: <span class="amt">${monthly_fees:,.2f}</span></div>'
            if monthly_fees > 0
            else ""
        )

        # ---- render in left column ----
        html = [
            '<div class="cal-wrap"><div class="cal-card">',
            f'<div class="cal-header"><div class="cal-title">{title_left_html}</div>{title_right_html}</div>',
            '<table class="cal">',
            "<thead><tr>",
            *header_cells,
            "</tr></thead>",
            "<tbody>",
            *body_rows,
            "</tbody>",
            "</table>",
            "</div></div>",
        ]
        with cal_left:
            st.markdown("".join(html), unsafe_allow_html=True)


# ---------------------
# Overview Dashboard tab
# ---------------------
else:
    st.subheader("📊 Overview")

    if trades.empty or daily.empty:
        st.info("No trades found yet.")
    else:
        # ---- Equity curve from daily stats (cumulative P/L) ----
        dcurve = daily[["date", "daily_profit"]].copy().sort_values("date")
        dcurve["equity"] = dcurve["daily_profit"].cumsum()

        # ---- Precompute totals used in right rail ----
        net_pl = float(dcurve["equity"].iloc[-1]) if not dcurve.empty else 0.0  # total net

        total_trades = len(trades)
        wins = int((trades["profit"] > 0).sum())
        losses = int((trades["profit"] < 0).sum())
        scratches = int((trades["profit"] == 0).sum())
        win_rate_trades = wins / total_trades if total_trades else 0.0
        gross_profit = trades.loc[trades["profit"] > 0, "profit"].sum()
        gross_loss = -trades.loc[trades["profit"] < 0, "profit"].sum()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("nan")

        # --- New: aggregates for Overview tiles ---
        total_fees_all = float(daily["fees"].sum()) if not daily.empty else 0.0

        today = datetime.now(TIMEZONE_PACIFIC).date()
        ytd_start = date(today.year, 1, 1)
        net_pl_ytd = (
            float(daily.loc[daily["date"] >= ytd_start, "daily_profit"].sum())
            if not daily.empty
            else 0.0
        )

        # ---- 3/4 vs 1/4 layout ----
        left, right = st.columns([3, 1], gap="large")

        with left:
            # ---- Top-line tiles row (3 tiles) ----
            row1a, row1b, row1c = st.columns(3)
            row1a.metric("Account Net P/L", f"${net_pl:,.2f}")
            row1b.metric("Total Fees", f"${total_fees_all:,.2f}")
            row1c.metric("Net P/L (YTD)", f"${net_pl_ytd:,.2f}")

            y = dcurve["equity"]
            x = dcurve["date"]

            # Separate positive vs negative areas
            y_pos, y_neg = y.clip(lower=0), y.clip(upper=0)

            fig = go.Figure()

            # Green fill above 0
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_pos,
                    mode="lines",
                    line=dict(width=0),
                    fill="tozeroy",
                    fillcolor="rgba(46,125,50,0.55)",  # semi-transparent green
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # Red fill below 0
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_neg,
                    mode="lines",
                    line=dict(width=0),
                    fill="tozeroy",
                    fillcolor="rgba(139,45,45,0.55)",  # semi-transparent red
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # White line on top
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=3, color=COL.hex("white")),
                    name="Cumulative P/L",
                )
            )

            fig.update_layout(
                yaxis_title="Cumulative P/L ($)",
                xaxis_title="Date",
                height=420,
                margin=dict(l=20, r=20, t=30, b=10),
                plot_bgcolor="#0e1218",
                paper_bgcolor="#0e1218",
                font=dict(color="#e9eef7"),
                showlegend=False,
            )

            st.plotly_chart(fig, width="content")

        with right:
            st.markdown("#### Outcomes")
            pie = px.pie(
                names=["Wins", "Losses", "Scratch"],
                values=[wins, losses, scratches],
                hole=0.55,
                color=["Wins", "Losses", "Scratch"],
                color_discrete_map={
                    "Wins": "#428246",  # green
                    "Losses": "#8b2d2d",  # red
                    "Scratch": "#6b7687",  # neutral gray
                },
            )
            pie.update_layout(
                showlegend=True, height=220, margin=dict(l=0, r=0, t=10, b=10)
            )
            st.plotly_chart(pie, width="content")

            st.markdown("#### Totals")

            colA, colB = st.columns(2)

            with colA:
                st.metric("Total trades", total_trades)
                st.metric("Win rate (by trade)", f"{win_rate_trades*100:.1f}%")

            with colB:
                st.metric(
                    "Profit factor",
                    f"{profit_factor:.2f}" if pd.notna(profit_factor) else "—",
                )
                st.metric("Unique trading days", daily.shape[0])

        st.markdown("---")

        # ---- Leaders & Laggards section ----
        sec_left, sec_right = st.columns([1, 1], gap="large")

        # LEFT: Top/Bottom tickers
        with sec_left:
            st.markdown("#### Top & Bottom Tickers (Total P/L)")

            tick_sum = (
                trades.groupby("symbol", observed=True)["profit"].sum().sort_values(ascending=False)
            )
            if tick_sum.empty:
                st.info("No tickers to display yet.")
            else:
                t_left, t_right = st.columns(2)
                with t_left:
                    st.markdown("##### 🏆 Top 5")
                    top5 = (
                        tick_sum.head(5).rename("Total P/L").map(lambda x: f"${x:,.2f}")
                    )
                    st.table(top5)

                with t_right:
                    st.markdown("##### ⚠️ Bottom 5")
                    bottom5 = (
                        tick_sum.tail(5).rename("Total P/L").map(lambda x: f"${x:,.2f}")
                    )
                    st.table(bottom5)

        # RIGHT: Best/Worst 5 days
        with sec_right:
            st.markdown("#### Best & Worst Days")

            daily_idxed = daily.copy()
            daily_idxed["date"] = pd.to_datetime(daily_idxed["date"])

            best5 = daily_idxed.nlargest(5, "daily_profit")[
                ["date", "daily_profit"]
            ].copy()
            worst5 = daily_idxed.nsmallest(5, "daily_profit")[
                ["date", "daily_profit"]
            ].copy()

            def _fmt_day_table(df):
                if df.empty:
                    return pd.DataFrame({"Date": [], "P/L": []})
                out = df.copy()
                out["Date"] = pd.to_datetime(out["date"]).dt.strftime("%b %d, %Y")
                out["P/L"] = out["daily_profit"].map(lambda x: f"${x:,.2f}")
                return out[["Date", "P/L"]]

            bw_left, bw_right = st.columns(2)
            with bw_left:
                st.markdown("##### 🟢 Best 5 Days")
                st.table(_fmt_day_table(best5))
            with bw_right:
                st.markdown("##### 🔴 Worst 5 Days")
                st.table(_fmt_day_table(worst5))

        st.markdown("---")

        # ---- Interesting Stats (concise risk & consistency snapshot) ----
        # Daily slice
        daily_sorted = daily.sort_values("date")
        eq = daily_sorted["daily_profit"].cumsum()
        run_max = eq.cummax()
        dd_series = eq - run_max
        max_drawdown = (
            float(dd_series.min()) if not dd_series.empty else 0.0
        )  # negative or 0

        expectancy_day = daily["daily_profit"].mean() if not daily.empty else 0.0
        median_day = daily["daily_profit"].median() if not daily.empty else 0.0
        std_day = daily["daily_profit"].std() if not daily.empty else 0.0

        # Activity stats
        avg_trades_per_day = trades.shape[0] / daily.shape[0] if daily.shape[0] else 0.0
        busiest_day_trades = 0
        if not trades.empty:
            trades_per_day = (
                trades["exit_time"]
                .dt.tz_convert(TIMEZONE_PACIFIC)
                .dt.date.value_counts()
            )
            busiest_day_trades = (
                int(trades_per_day.max()) if not trades_per_day.empty else 0
            )

        # Long vs short counts
        long_count = int((trades["direction"] == "long").sum())
        short_count = int((trades["direction"] == "short").sum())

        # Row 1: risk stats
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Expectancy (per day)", f"${expectancy_day:,.2f}")
        g2.metric("Median day P/L", f"${median_day:,.2f}")
        g3.metric("Daily P/L stdev", f"${std_day:,.2f}")
        g4.metric(
            "Max drawdown",
            f"-${abs(max_drawdown):,.2f}" if max_drawdown < 0 else "$0.00",
        )

        # --- Weekly & Monthly rollups for "best" metrics ---
        daily_idxed = daily.copy()
        daily_idxed["date"] = pd.to_datetime(daily_idxed["date"])

        # Weekly (Sun–Sat periods). Change to "W-MON" if you prefer Mon–Sun.
        weekly = (
            daily_idxed.assign(week=daily_idxed["date"].dt.to_period("W-SUN"))
            .groupby("week", as_index=False)["daily_profit"]
            .sum()
        )
        # Add week start/end for display
        weekly["week_start"] = weekly["week"].dt.start_time.dt.date
        weekly["week_end"] = weekly["week"].dt.end_time.dt.date
        weekly_sorted = weekly.sort_values("daily_profit", ascending=False)

        # Monthly
        monthly = (
            daily_idxed.assign(month=daily_idxed["date"].dt.to_period("M"))
            .groupby("month", as_index=False)["daily_profit"]
            .sum()
        )
        monthly["month_label"] = monthly["month"].dt.strftime("%B %Y")
        monthly_sorted = monthly.sort_values("daily_profit", ascending=False)

        # Best week & month rows (guard against empties)
        best_week = weekly_sorted.iloc[0] if not weekly_sorted.empty else None
        best_month = monthly_sorted.iloc[0] if not monthly_sorted.empty else None

        # Row 2: Best week/month + activity
        h1, h2, h3, h4 = st.columns(4)
        if best_week is not None:
            h1.metric(
                "Best week (sum P/L)",
                f"${best_week['daily_profit']:,.2f}",
                f"{best_week['week_start']} → {best_week['week_end']}",
            )
        else:
            h1.metric("Best week (sum P/L)", "—")

        if best_month is not None:
            h2.metric(
                "Best month (sum P/L)",
                f"${best_month['daily_profit']:,.2f}",
                best_month["month_label"],
            )
        else:
            h2.metric("Best month (sum P/L)", "—")

        h3.metric("Avg trades / day", f"{avg_trades_per_day:.1f}")
        h4.metric("Busiest day (trades)", f"{busiest_day_trades}")

        st.markdown("---")

        # ---- Composition & counts ----
        cA, cB, cC = st.columns(3)
        with cA:
            st.markdown("#### Position Mix")
            st.metric("Long trades", long_count)
            st.metric("Short trades", short_count)

        with cB:
            st.markdown("#### Gross Totals")
            st.metric("Total gains (winning trades)", f"${gross_profit:,.2f}")
            st.metric("Total losses (losing trades)", f"${gross_loss:,.2f}")

        with cC:
            st.markdown("#### Streaks (by trade)")
            # Longest trade-level streaks
            trade_sign = np.sign(trades.sort_values("exit_time")["profit"]).values

            def calc_streak(arr, positive=True):
                best = run = 0
                for v in arr:
                    ok = v > 0 if positive else v < 0
                    run = run + 1 if ok else 0
                    best = max(best, run)
                return best

            st.metric("Longest win streak", calc_streak(trade_sign, True))
            st.metric("Longest losing streak", calc_streak(trade_sign, False))


# -------------------------
# End of app
# -------------------------
