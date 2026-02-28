from __future__ import annotations
# -*- coding: utf-8 -*-
import itertools
import math
import os
import re
import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _escape_html(x) -> str:
    """HTML-escape helper for safe rendering inside st.markdown(unsafe_allow_html=True)."""
    try:
        return html.escape("" if x is None else str(x), quote=True)
    except Exception:
        return ""


def _get_num(row, key: str, default: float = 0.0) -> float:
    """Safe numeric getter for pandas Series/dict-like rows."""
    try:
        v = row.get(key, default)  # type: ignore[attr-defined]
    except Exception:
        try:
            v = row[key]  # type: ignore[index]
        except Exception:
            v = default
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return float(default)
        # pandas NA / numpy NaN
        if hasattr(pd, "isna") and pd.isna(v):  # type: ignore[arg-type]
            return float(default)
        return float(v)
    except Exception:
        return float(default)
import pandas as pd

from pandas.io.formats.style import Styler
import streamlit as st

# ----------------------------
# Page config (wide)
# ----------------------------
st.set_page_config(
    page_title="Profit Mix Optimizer",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# RTL + responsive theme CSS
# ----------------------------
RTL_CSS = """
<style>
/* RTL baseline */
html, body, [class*="css"]  {
  direction: rtl;
  text-align: right;
}

/* Keep sliders LTR so the thumb and ticks behave naturally */
div[data-baseweb="slider"]{
  direction: ltr !important;
}
div[data-baseweb="slider"] *{
  direction: ltr !important;
}

/* Header / cards */
.profit-title {
  font-size: 34px;
  font-weight: 800;
  margin-bottom: 2px;
}
.profit-subtitle {
  font-size: 15px;
  opacity: 0.85;
  margin-top: 0px;
  margin-bottom: 18px;
}
.kpi-card {
  border-radius: 18px;
  padding: 14px 16px;
  border: 1px solid rgba(120,120,120,0.20);
  background: rgba(255,255,255,0.55);
}

/* Luxury alternative cards (mobile friendly, no horizontal scroll) */
.alt-card{
  border-radius: 20px;
  padding: 16px 16px 14px 16px;
  border: 1px solid rgba(120,120,120,0.22);
  background: rgba(255,255,255,0.62);
  box-shadow: 0 10px 26px rgba(0,0,0,0.06);
  margin: 10px 0 14px 0;
}
.alt-head{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap:12px;
  margin-bottom:10px;
}
.alt-title{
  font-size:18px;
  font-weight:800;
}
.alt-score{
  font-size:13px;
  opacity:0.85;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(120,120,120,0.22);
  background: rgba(250,250,250,0.6);
  white-space: nowrap;
}
.alt-mini{
  display:flex;
  flex-direction:column;
  gap:8px;
}
.alt-row{
  display:grid;
  grid-template-columns: 120px 1fr;
  gap:10px;
  align-items:start;
}
.alt-k{
  font-size:12px;
  opacity:0.75;
  line-height:1.35;
}
.alt-v{
  font-size:13px;
  line-height:1.55;
  word-break: break-word;
  overflow-wrap: anywhere;
}
@media (prefers-color-scheme: dark) {
  .alt-card{ background: rgba(22,22,28,0.72); border: 1px solid rgba(255,255,255,0.12); box-shadow: 0 10px 26px rgba(0,0,0,0.28); }
  .alt-score{ border: 1px solid rgba(255,255,255,0.12); background: rgba(40,40,52,0.55); }
}
@media (max-width: 520px){
  .alt-row{ grid-template-columns: 98px 1fr; }
  .alt-title{ font-size:17px; }
}
@media (prefers-color-scheme: dark) {
  .kpi-card { background: rgba(30,30,30,0.55); border: 1px solid rgba(255,255,255,0.12); }
}
/* Mobile: prefer dark background */
@media (max-width: 768px) {
  .stApp {
    background: #0f1116;
    color: #e8e8e8;
  }
  .kpi-card { background: rgba(25,25,30,0.7); border: 1px solid rgba(255,255,255,0.12); }
}

/* Make dataframe headers RTL */
div[data-testid="stDataFrame"] *{
  direction: rtl;
  text-align: right;
}

/* Wider text columns */

/* ----------------------------
   Wall-Street "lux" results UI
   ---------------------------- */
.lux-shell { max-width: 1040px; margin: 0 auto; }
.lux-hero {
  background:
    radial-gradient(1200px 260px at 86% 0%, rgba(220, 194, 120, 0.22) 0%, rgba(220, 194, 120, 0.0) 60%),
    linear-gradient(180deg, #0b1220 0%, #0b1220 44%, rgba(11,18,32,0.0) 44%);
  padding: 18px 16px 0;
  border-radius: 18px;
  margin-bottom: 14px;
}
.lux-title { color:#f2f4f7; font-weight: 900; font-size: 22px; margin: 0 0 8px; }
.lux-subtitle { color: rgba(242,244,247,0.78); font-size: 13px; margin: 0 0 14px; }
.lux-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 900px){ .lux-grid { grid-template-columns: 1fr; } }

.lux-card {
  background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
  border: 1px solid #e6eaf2;
  border-radius: 18px;
  box-shadow: 0 16px 34px rgba(16,24,40,0.08);
  padding: 14px 14px 12px;
  overflow: hidden;
}
.lux-card-header { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom: 10px; }
.lux-name { font-weight: 900; font-size: 16px; color:#101828; }
.lux-score { font-size: 12.5px; color:#475467; }
.lux-score b { color:#0b1220; }

.lux-weights { border: 1px solid #eef2f7; background: #f8fafc; border-radius: 14px; padding: 10px; }
.w-item { display:flex; align-items:flex-start; gap:10px; padding:7px 0; border-bottom: 1px dashed #e6eaf2; }
.w-item:last-child{ border-bottom: none; padding-bottom: 0; }
.w-pct {
  min-width: 62px;
  font-weight: 900;
  letter-spacing: .2px;
  color:#0b1220;
  background: rgba(220, 194, 120, 0.22);
  border: 1px solid rgba(220, 194, 120, 0.55);
  border-radius: 999px;
  text-align:center;
  padding: 3px 8px;
  font-size: 12px;
}
.w-name { flex: 1; min-width: 0; color:#101828; font-size: 13px; line-height: 1.25; word-break: break-word; overflow-wrap: anywhere; }
.w-name .w-track { display:block; color:#667085; font-size: 12px; margin-top: 2px; }

.lux-kpis { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
@media (max-width: 520px){ .lux-kpis { grid-template-columns: 1fr; } }
.lux-kpi { border: 1px solid #eef2f7; border-radius: 12px; padding: 8px 10px; background: #ffffff; }
.lux-kpi .k { color:#667085; font-size: 11.5px; margin-bottom: 2px; white-space: nowrap; }
.lux-kpi .v { color:#101828; font-weight: 800; font-size: 13px; }
.lux-adv { margin-top: 10px; color:#475467; font-size: 12.5px; }
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

# ----------------------------
# Password Gate
# ----------------------------
def _check_password() -> bool:
    """
    Password gate with Streamlit session_state.
    Recommended: set APP_PASSWORD in Streamlit secrets.
    """
    if st.session_state.get("auth_ok", False):
        return True

    correct = None
    # Prefer secrets; fallback to env; final fallback hardcoded (demo only)
    if hasattr(st, "secrets") and "APP_PASSWORD" in st.secrets:
        correct = str(st.secrets["APP_PASSWORD"])
    else:
        correct = os.getenv("APP_PASSWORD", "1234")

    st.markdown('<div class="profit-title">🔒 כניסה</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="profit-subtitle">האפליקציה מוגנת בסיסמה. הזן סיסמה כדי להמשיך.</div>',
        unsafe_allow_html=True,
    )
    pwd = st.text_input("סיסמה", type="password", placeholder="••••••••")

    c1, c2 = st.columns([1, 6])
    with c1:
        go = st.button("כניסה", use_container_width=True)
    if go:
        if pwd == correct:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("סיסמה שגויה.")

    st.stop()

_check_password()

# ----------------------------
# Constants / helpers
# ----------------------------
FUNDS_FILE = "קרנות השתלמות.xlsx"
SERVICE_FILE = "ציוני שירות.xlsx"

PARAM_ALIASES = {
    "stocks": ["סך חשיפה למניות", "מניות"],
    "foreign": ["סך חשיפה לנכסים המושקעים בחו\"ל", "סך חשיפה לנכסים המושקעים בחו׳ל", "חו\"ל", "חו׳ל"],
    "fx": ["חשיפה למט\"ח", "מט\"ח", "מט׳׳ח"],
    "illiquid": ["נכסים לא סחירים", "לא סחירים", "לא-סחיר", "לא סחיר"],
    "sharpe": ["מדד שארפ", "שארפ"],
    "israel_assets": ["נכסים בארץ", "נכסים בישראל", "בארץ", "ישראל"],
}

DISPLAY_NAMES = {
    "foreign": "יעד חו״ל (%)",
    "israel": "יעד ישראל (%)",
    "stocks": "יעד מניות (%)",
    "fx": "יעד מט״ח (%)",
    "illiquid": "מקסימום לא־סחיר (%)",
    "sharpe": "שארפ",
    "service": "שירות",
    "score": "Score (סטייה)",
}

def _to_float(x) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    s = s.replace(",", "")
    s = s.replace("%", "")
    s = s.replace("−", "-")
    s = re.sub(r"[^\d\.\-]+", "", s)
    if s in ("", "-", "."):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def _match_param(row_name: str, key: str) -> bool:
    rn = str(row_name).strip()
    for a in PARAM_ALIASES[key]:
        if a in rn:
            return True
    return False

def _extract_manager(fund_name: str) -> str:
    """
    Heuristic: take first token up to 'קרן'/'השתלמות'/'-' etc.
    Works for names like 'מנורה השתלמות כללי', 'כלל השתלמות כללי'.
    """
    name = str(fund_name).strip()
    # common splitters
    for splitter in [" קרן", " השתלמות", " -", "-", "  "]:
        if splitter in name:
            head = name.split(splitter)[0].strip()
            if head:
                return head
    # fallback: first word
    return name.split()[0] if name.split() else name

@dataclass
class FundRecord:
    track: str
    fund: str
    manager: str
    stocks: float
    foreign: float
    fx: float
    illiquid: float
    sharpe: float
    service: float

def _load_service_scores(path: str) -> Dict[str, float]:
    try:
        df = pd.read_excel(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    if "provider" not in df.columns or "score" not in df.columns:
        # Try first two columns
        df = df.iloc[:, :2].copy()
        df.columns = ["provider", "score"]
    out = {}
    for _, r in df.iterrows():
        p = str(r["provider"]).strip()
        sc = _to_float(r["score"])
        if p and not math.isnan(sc):
            out[p] = float(sc)
    return out

@st.cache_data(show_spinner=False)
def load_funds_long(funds_path: str, service_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    svc = _load_service_scores(service_path)
    xls = pd.ExcelFile(funds_path)
    records: List[Dict] = []
    for sh in xls.sheet_names:
        # Skip 'ניהול אישי' / IRA sheets (personal management), if present
        sh_str = str(sh)
        sh_low = sh_str.lower()
        if re.search(r"ניהול\s*אישי", sh_str) or re.search(r"(^|[^a-z])ira([^a-z]|$)", sh_low):
            continue
        df = pd.read_excel(xls, sheet_name=sh, header=None)
        if df.empty:
            continue
        # Expect first row as headers: first cell 'פרמטר'
        # Column 0: parameter names, columns 1..n: funds
        header_row = df.iloc[0].tolist()
        if not str(header_row[0]).strip().startswith("פרמטר"):
            # try find 'פרמטר' row
            idxs = df.index[df.iloc[:, 0].astype(str).str.contains("פרמטר", na=False)].tolist()
            if not idxs:
                continue
            df = df.iloc[idxs[0]:].reset_index(drop=True)
            header_row = df.iloc[0].tolist()
        fund_names = [c for c in header_row[1:] if str(c).strip() and str(c).strip() != "nan"]
        if not fund_names:
            continue

        # Build mapping param->row index
        param_col = df.iloc[1:, 0].astype(str).tolist()
        def row_for(key: str) -> Optional[int]:
            for i, rn in enumerate(param_col, start=1):
                if _match_param(rn, key):
                    return i
            return None

        ridx_stocks = row_for("stocks")
        ridx_foreign = row_for("foreign")
        ridx_fx = row_for("fx")
        ridx_ill = row_for("illiquid")
        ridx_sharpe = row_for("sharpe")

        if ridx_foreign is None and ridx_stocks is None:
            # Not a relevant sheet
            continue

        for j, fname in enumerate(fund_names, start=1):
            manager = _extract_manager(fname)
            rec = {
                "track": sh,
                "fund": str(fname).strip(),
                "manager": manager,
                "stocks": _to_float(df.iloc[ridx_stocks, j]) if ridx_stocks is not None else np.nan,
                "foreign": _to_float(df.iloc[ridx_foreign, j]) if ridx_foreign is not None else np.nan,
                "fx": _to_float(df.iloc[ridx_fx, j]) if ridx_fx is not None else np.nan,
                "illiquid": _to_float(df.iloc[ridx_ill, j]) if ridx_ill is not None else np.nan,
                "sharpe": _to_float(df.iloc[ridx_sharpe, j]) if ridx_sharpe is not None else np.nan,
            }
            # keep only rows that have at least foreign or stocks numeric
            if all(math.isnan(rec[k]) for k in ["foreign", "stocks", "fx", "illiquid", "sharpe"]):
                continue
            rec["service"] = float(svc.get(manager, 50.0))  # placeholder default
            records.append(rec)

    df_long = pd.DataFrame.from_records(records)
    # Clean NaNs to float
    for c in ["stocks", "foreign", "fx", "illiquid", "sharpe", "service"]:
        if c in df_long.columns:
            df_long[c] = pd.to_numeric(df_long[c], errors="coerce")

    return df_long, svc

# ----------------------------
# Load embedded files
# ----------------------------
if not os.path.exists(FUNDS_FILE):
    st.error(f"לא נמצא קובץ הנתונים '{FUNDS_FILE}' בתיקיית הפרויקט. העלה אותו לשורש הריפו ב-GitHub.")
    st.stop()
if not os.path.exists(SERVICE_FILE):
    st.error(f"לא נמצא קובץ השירות '{SERVICE_FILE}' בתיקיית הפרויקט. העלה אותו לשורש הריפו ב-GitHub.")
    st.stop()

df_long, service_map = load_funds_long(FUNDS_FILE, SERVICE_FILE)

# Basic validation
n_tracks = df_long["track"].nunique() if not df_long.empty else 0
n_records = len(df_long)

st.markdown('<div class="profit-title">📊 Profit Mix Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="profit-subtitle">כלי לחיפוש תמהיל השקעות אופטימלי בין מסלולי קרנות השתלמות, על בסיס קובץ הנתונים המובנה. '
    'מגדירים יעד חשיפות, מגבלות וקריטריוני דירוג – ומקבלים 3 חלופות שונות.</div>',
    unsafe_allow_html=True
)
st.info(f"✅ זוהו **{n_tracks}** מסלולי השקעה בקובץ | ✅ זוהו **{n_records}** קופות (מנהל×מסלול)")

if df_long.empty:
    st.error("לא הצלחתי לזהות טבלאות בקובץ. ודא שבכל גיליון יש שורה ראשונה 'פרמטר' ואז עמודות של קופות.")
    st.stop()

# ----------------------------
# UI
# ----------------------------
tab1, tab2, tab4, tab3 = st.tabs(["הגדרות יעד", "תוצאות (3 חלופות)", "השוואת מסלולים", "פירוט חישוב / שקיפות"])

# Session defaults
def _init_state():
    st.session_state.setdefault("n_funds", 2)
    st.session_state.setdefault("mix_policy", "מותר לערבב מנהלים")
    st.session_state.setdefault("step", 5)
    st.session_state.setdefault("primary_rank", "דיוק")
    st.session_state.setdefault("targets", {"foreign": 30.0, "stocks": 40.0, "fx": 25.0, "illiquid": 20.0})
    st.session_state.setdefault("include", {"foreign": True, "stocks": True, "fx": False, "illiquid": False})
    st.session_state.setdefault("constraint", {
        "foreign": ("רך", "בדיוק"),
        "stocks": ("רך", "בדיוק"),
        "fx": ("רך", "לפחות"),
        "illiquid": ("קשיח", "לכל היותר"),
    })
    st.session_state.setdefault("score_weights", {"foreign": 1.0, "stocks": 1.0, "fx": 1.0, "illiquid": 1.0})
    st.session_state.setdefault("last_results", None)
    st.session_state.setdefault("last_note", "")

_init_state()


def _weights_detail(weights, funds_str: str, tracks_str: str) -> str:
    """Return human-readable weights string aligned with funds/tracks."""
    try:
        ws = list(weights) if isinstance(weights, (list, tuple, np.ndarray)) else []
    except Exception:
        ws = []
    funds = [s.strip() for s in (funds_str or "").split("|") if s.strip()]
    tracks = [s.strip() for s in (tracks_str or "").split("|") if s.strip()]
    n = max(len(ws), len(funds), len(tracks))
    parts = []
    for i in range(n):
        w = ws[i] if i < len(ws) else None
        fund = funds[i] if i < len(funds) else ""
        track = tracks[i] if i < len(tracks) else ""
        w_txt = f"{w:.0f}%" if isinstance(w, (int, float, np.floating)) else "?"
        label = fund
        if track:
            label = f"{fund} ({track})" if fund else f"({track})"
        parts.append(f"{w_txt} {label}".strip())
    return " | ".join(parts) if parts else ""


def _weights_items(weights, funds_str: str, tracks_str: str) -> List[Dict[str, str]]:
    """Structured weights for premium UI. Designed to never overflow on mobile."""
    try:
        ws = list(weights) if isinstance(weights, (list, tuple, np.ndarray)) else []
    except Exception:
        ws = []
    funds = [s.strip() for s in (funds_str or "").split("|") if s.strip()]
    tracks = [s.strip() for s in (tracks_str or "").split("|") if s.strip()]
    n = max(len(ws), len(funds), len(tracks))
    items: List[Dict[str, str]] = []
    for i in range(n):
        w = ws[i] if i < len(ws) else None
        fund = funds[i] if i < len(funds) else ""
        track = tracks[i] if i < len(tracks) else ""
        pct = "?" if not isinstance(w, (int, float, np.floating)) else f"{int(round(float(w)))}%"
        items.append({"pct": pct, "fund": fund, "track": track})
    return items


def _weights_short(weights):
    """Return compact weights string like '40% / 30% / 30%'."""
    if weights is None:
        return ""
    try:
        w = [float(x) for x in weights]
    except Exception:
        return ""
    if not w:
        return ""
    return " / ".join([f"{int(round(x))}%" for x in w])

def _weights_for_n(n: int, step: int) -> List[Tuple[int, ...]]:
    step = max(1, int(step))
    if n == 1:
        return [(100,)]
    if n == 2:
        return [(w, 100 - w) for w in range(0, 101, step)]
    # n == 3
    out = []
    for w1 in range(0, 101, step):
        for w2 in range(0, 101 - w1, step):
            w3 = 100 - w1 - w2
            if w3 % step == 0:
                out.append((w1, w2, w3))
    return out

def _apply_israel_rule(targets: Dict[str, float]) -> Dict[str, float]:
    # If user sets Israel target, convert to foreign internally; we expose only foreign in UI (per your latest note),
    # but keep the rule here for consistency.
    return targets

def _compute_mix_metrics(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # arr shape (n_funds, n_metrics), weights (n_funds,)
    return np.nansum(arr * weights[:, None], axis=0)

def _hard_ok(value: float, target: float, mode: str) -> bool:
    # mode: "בדיוק", "לפחות", "לכל היותר"
    if math.isnan(value):
        return False
    if mode == "בדיוק":
        return abs(value - target) < 1e-9
    if mode == "לפחות":
        return value + 1e-9 >= target
    if mode == "לכל היותר":
        return value - 1e-9 <= target
    return True

def _soft_distance(value: float, target: float) -> float:
    if math.isnan(value) or math.isnan(target):
        return 0.0
    return abs(value - target) / 100.0  # normalize

def _make_advantage(primary: str, row: Dict, base_row: Optional[Dict]=None) -> str:
    if primary == "דיוק":
        return f"הכי מדויק ליעד, סטייה כוללת {row['score']:.4f}"
    if primary == "שארפ":
        if base_row is None:
            return f"שארפ משוקלל גבוה ({float(row.get('sharpe', row.get('sharpe_weighted', 0.0))):.2f})"
        delta = float(row.get("sharpe", row.get("sharpe_weighted", 0.0))) - float(base_row.get("sharpe", base_row.get("sharpe_weighted", 0.0)))
        return f"שארפ גבוה יותר ב-{delta:.2f} תוך סטייה {row['score']:.4f}"
    if primary == "שירות":
        if base_row is None:
            return f"ציון שירות משוקלל הגבוה ביותר ({float(row.get('service', row.get('שירות', 0.0)) or 0.0):.1f})"
        delta = _get_num(row, "service", 0.0) - _get_num(base_row, "service", 0.0)
        return f"שירות גבוה יותר ב-{delta:.1f} תוך סטייה {row['score']:.4f}"
    return f"חלופה חזקה לפי {primary}"

def _prefilter_candidates(df: pd.DataFrame, include: Dict[str, bool], targets: Dict[str, float], cap: int) -> pd.DataFrame:
    # Quick score for single fund closeness to targets (soft only) to reduce search space
    # Keep those with smallest sum of deviations for selected soft metrics (foreign/stocks/fx)
    keys = [k for k, v in include.items() if v and k in ["foreign", "stocks", "fx", "illiquid"]]
    if not keys:
        keys = ["foreign", "stocks"]
    tmp = df.copy()
    score = np.zeros(len(tmp), dtype=float)
    for k in keys:
        score += np.abs(tmp[k].fillna(0.0).to_numpy() - float(targets.get(k, 0.0))) / 100.0
    tmp["_single_score"] = score
    tmp = tmp.sort_values("_single_score", ascending=True).head(cap).drop(columns=["_single_score"])
    return tmp

def find_best_solutions(
    df: pd.DataFrame,
    n_funds: int,
    step: int,
    mix_policy: str,
    include: Dict[str, bool],
    constraint: Dict[str, Tuple[str, str]],
    targets: Dict[str, float],
    primary_rank: str,
    max_solutions_scan: int = 60000,
) -> Tuple[pd.DataFrame, str]:
    """
    Returns a dataframe of candidate solutions (many), and a note string.
    We use a stable/rigorous scan but with a prefilter cap to keep Streamlit Cloud responsive.
    """
    # Validate include targets
    targets = {k: float(v) for k, v in targets.items()}

    # Pre-filter to keep search manageable
    cap = 80 if n_funds == 2 else 55 if n_funds == 3 else 120
    df_scan = _prefilter_candidates(df, include, targets, cap=cap)

    weights_list = _weights_for_n(n_funds, step)
    if not weights_list:
        return pd.DataFrame(), "לא נמצאו משקלים אפשריים. נסה צעד קטן יותר (למשל 5%)."

    # Decide columns for objective
    metric_keys = ["foreign", "stocks", "fx", "illiquid"]
    active_soft = [k for k in metric_keys if include.get(k, False)]
    if not active_soft:
        active_soft = ["foreign", "stocks"]

    # Hard constraints: any metric marked "קשיח"
    hard_keys = []
    for k in metric_keys:
        hardness, mode = constraint.get(k, ("רך", "בדיוק"))
        if hardness == "קשיח":
            hard_keys.append((k, mode))

    # Arrays
    A = df_scan[["foreign", "stocks", "fx", "illiquid", "sharpe", "service"]].to_numpy(dtype=float)
    # indices map to df rows
    records = df_scan.reset_index(drop=True)

    solutions = []
    scanned = 0
    # group by manager if needed
    if mix_policy == "אותו מנהל בלבד":
        groups = list(records.groupby("manager").groups.values())
        combos_iter = []
        for idxs in groups:
            if len(idxs) >= n_funds:
                combos_iter.append(itertools.combinations(list(idxs), n_funds))
        combo_source = itertools.chain.from_iterable(combos_iter)
    else:
        combo_source = itertools.combinations(range(len(records)), n_funds)

    for combo in combo_source:
        combo = tuple(combo)
        arr = A[list(combo), :]  # (n, 6)
        # quick skip if all nan for key metrics
        if np.all(np.isnan(arr[:, 0:4])):
            continue

        for w in weights_list:
            scanned += 1
            if scanned > max_solutions_scan:
                break

            weights = np.array(w, dtype=float) / 100.0
            mix = _compute_mix_metrics(arr[:, 0:6], weights)
            foreign, stocks, fx, illiq, sharpe, service = mix.tolist()
            israel = 100.0 - foreign if not math.isnan(foreign) else np.nan

            # Hard constraints
            ok = True
            for k, mode in hard_keys:
                val = {"foreign": foreign, "stocks": stocks, "fx": fx, "illiquid": illiq}.get(k, np.nan)
                tgt = targets.get(k, 0.0)
                if not _hard_ok(val, tgt, mode):
                    ok = False
                    break
            if not ok:
                continue

            # Score for "דיוק": sum of normalized deviations for active soft keys (even if they are hard, score still informative)
            score = 0.0
            for k in active_soft:
                val = {"foreign": foreign, "stocks": stocks, "fx": fx, "illiquid": illiq}.get(k, np.nan)
                score += _soft_distance(val, targets.get(k, 0.0))

            # Gather labels
            fund_labels = [records.loc[i, "fund"] for i in combo]
            track_labels = [records.loc[i, "track"] for i in combo]
            managers = [records.loc[i, "manager"] for i in combo]
            manager_set = " | ".join(sorted(set(managers)))

            solutions.append({
                "combo": combo,
                "weights": w,
                "מנהלים": manager_set,
                "מסלולים": " | ".join(track_labels),
                "קופות": " | ".join(fund_labels),
                "חו״ל (%)": foreign,
                "ישראל (%)": israel,
                "מניות (%)": stocks,
                "מט״ח (%)": fx,
                "לא־סחיר (%)": illiq,
                "שארפ משוקלל": sharpe,
                "שירות משוקלל": service,
                "score": score,
            })
        if scanned > max_solutions_scan:
            break

    if not solutions:
        return pd.DataFrame(), f"לא נמצאו פתרונות שעומדים במגבלות. נסה לרכך מגבלות קשיחות או להגדיל צעד/מספר קופות."

    df_sol = pd.DataFrame(solutions)

    note = f"נסרקו {min(scanned, max_solutions_scan):,} קומבינציות (לאחר סינון מוקדם ל-{len(records)} קופות)."

    # Sorting by primary rank for candidate ordering
    if primary_rank == "דיוק":
        df_sol = df_sol.sort_values(["score", "שארפ משוקלל", "שירות משוקלל"], ascending=[True, False, False])
    elif primary_rank == "שארפ":
        df_sol = df_sol.sort_values(["שארפ משוקלל", "score"], ascending=[False, True])
    elif primary_rank == "שירות":
        df_sol = df_sol.sort_values(["שירות משוקלל", "score"], ascending=[False, True])
    else:
        df_sol = df_sol.sort_values(["score"], ascending=[True])

    return df_sol, note

def pick_three_distinct(df_sol: pd.DataFrame, primary_rank: str) -> pd.DataFrame:
    """
    Always return 3 solutions with distinct manager sets.
    #1: best by primary_rank ordering already in df_sol
    #2: best by Sharpe (distinct managers)
    #3: best by Service (distinct managers)
    """
    if df_sol.empty:
        return df_sol

    picked = []
    used_manager_sets = set()

    def manager_key(row) -> str:
        return str(row["מנהלים"]).strip()

    # 1) primary
    for _, r in df_sol.iterrows():
        mk = manager_key(r)
        if mk not in used_manager_sets:
            picked.append(r)
            used_manager_sets.add(mk)
            break

    base = picked[0] if picked else None

    # 2) Sharpe
    df_sh = df_sol.sort_values(["שארפ משוקלל", "score"], ascending=[False, True])
    for _, r in df_sh.iterrows():
        mk = manager_key(r)
        if mk not in used_manager_sets:
            picked.append(r)
            used_manager_sets.add(mk)
            break

    # 3) Service
    df_sv = df_sol.sort_values(["שירות משוקלל", "score"], ascending=[False, True])
    for _, r in df_sv.iterrows():
        mk = manager_key(r)
        if mk not in used_manager_sets:
            picked.append(r)
            used_manager_sets.add(mk)
            break

    # Fill if still missing (rare)
    if len(picked) < 3:
        for _, r in df_sol.iterrows():
            mk = manager_key(r)
            if mk not in used_manager_sets:
                picked.append(r)
                used_manager_sets.add(mk)
            if len(picked) == 3:
                break

    df_out = pd.DataFrame(picked).reset_index(drop=True)

    # Add "חלופה" + "יתרון"
    rows = []
    for i in range(len(df_out)):
        row = df_out.iloc[i].to_dict()
        row['משקלים (פירוט)'] = _weights_detail(row.get('weights'), row.get('קופות',''), row.get('מסלולים',''))
        row['weights_items'] = _weights_items(row.get('weights'), row.get('קופות',''), row.get('מסלולים',''))
        row['משקלים'] = _weights_short(row.get('weights'))
        if i == 0:
            row["חלופה"] = "חלופה 1 (דירוג ראשי)"
            row["יתרון"] = _make_advantage(primary_rank, row)
        elif i == 1:
            row["חלופה"] = "חלופה 2 (שארפ)"
            row["יתרון"] = _make_advantage("שארפ", row, base_row=base.to_dict() if base is not None else None)
        else:
            row["חלופה"] = "חלופה 3 (שירות)"
            row["יתרון"] = _make_advantage("שירות", row, base_row=base.to_dict() if base is not None else None)
        rows.append(row)
    return pd.DataFrame(rows)

def _color_rows(df: pd.DataFrame, targets: Dict[str, float], constraint: Dict[str, Tuple[str, str]]) -> 'Styler':
    # Conditional formatting via Styler (works in st.dataframe as static if use st.dataframe? better use st.dataframe without styler.
    # We'll use st.dataframe normally and add per-cell highlights by HTML-free notes in KPI cards.
    return df.style

def _render_kpi_cards(alt_rows: pd.DataFrame):
    """Premium Wall-Street style summary cards (no horizontal scroll)."""
    if alt_rows.empty:
        return

    st.markdown(
        """
        <div class="lux-shell">
          <div class="lux-hero">
            <div class="lux-title">תוצאות מובילות</div>
            <div class="lux-subtitle">כדי לבחור בפועל — פשוט הגדירו את אחוזי החלוקה בין המסלולים כפי שמופיע בכל חלופה.</div>
            <div class="lux-grid">
        """,
        unsafe_allow_html=True,
    )

    for i in range(min(3, len(alt_rows))):
        r = alt_rows.iloc[i]
        items = r.get('weights_items') if isinstance(r, dict) else None
        # When r is a Series, weights_items may exist in its index.
        items = items if items is not None else (r['weights_items'] if 'weights_items' in r else [])

        weights_html = "".join(
            [
                f"<div class='w-item'><div class='w-pct'>{_escape_html(it.get('pct',''))}</div><div class='w-name'>{_escape_html(it.get('fund',''))}<span class='w-track'>{_escape_html(it.get('track',''))}</span></div></div>"
                for it in (items or [])
            ]
        )

        st.markdown(
            f"""
            <div class="lux-card">
              <div class="lux-card-header">
                <div class="lux-name">{_escape_html(str(r['חלופה']))} <span class="pill">{_escape_html(str(r.get('יתרון','')))}</span></div>
                <div class="lux-score">Score: <b>{float(r['score']):.4f}</b></div>
              </div>
              <div class="lux-weights">{weights_html}</div>
              <div class="lux-kpis">
                <div class="lux-kpi"><div class="k">חו״ל</div><div class="v">{float(r['חו״ל (%)']):.2f}%</div></div>
                <div class="lux-kpi"><div class="k">מניות</div><div class="v">{float(r['מניות (%)']):.2f}%</div></div>
                <div class="lux-kpi"><div class="k">מט״ח</div><div class="v">{float(r['מט״ח (%)']):.2f}%</div></div>
                <div class="lux-kpi"><div class="k">לא־סחיר</div><div class="v">{float(r['לא־סחיר (%)']):.2f}%</div></div>
              </div>
              <div class="lux-adv">שארפ: <b>{float(r['שארפ משוקלל']):.2f}</b> · שירות: <b>{float(r['שירות משוקלל']):.1f}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div></div></div>", unsafe_allow_html=True)


def _md_to_html_lines(md_text: str) -> str:
    """Convert simple markdown-ish lines (used in weights detail) into safe HTML lines."""
    if not md_text:
        return ""
    lines = [ln.strip() for ln in str(md_text).splitlines() if ln.strip()]
    out = []
    for ln in lines:
        ln = ln.lstrip("- ").strip()
        ln = (
            ln.replace("&", "&amp;")
              .replace("<", "&lt;")
              .replace(">", "&gt;")
        )
        out.append(ln)
    return "<br>".join(out)


def _render_alt_card(rr: dict, idx: int):
    """Render one alternative as a premium, Wall‑Street style card (RTL + mobile safe)."""
    title = rr.get("חלופה", f"חלופה {idx}")
    score = rr.get("score", np.nan)
    advantage = rr.get("יתרון", "")
    items = rr.get("weights_items") or []

    def _fmt_pct(x):
        try:
            return f"{float(x):.2f}%"
        except Exception:
            return "—"

    def _fmt_num(x, fmt="{:.2f}"):
        try:
            return fmt.format(float(x))
        except Exception:
            return "—"

    weights_html = "".join(
        [
            f"<div class='w-item'><div class='w-pct'>{_escape_html(it.get('pct',''))}</div><div class='w-name'>{_escape_html(it.get('fund',''))}<span class='w-track'>{_escape_html(it.get('track',''))}</span></div></div>"
            for it in items
        ]
    )

    score_str = "—" if (score is None or (isinstance(score, float) and np.isnan(score))) else f"{float(score):.4f}"

    html = f"""
    <div class=\"lux-card\">
      <div class=\"lux-card-header\">
        <div class=\"lux-name\">{_escape_html(str(title))} <span class=\"pill\">{_escape_html(str(advantage))}</span></div>
        <div class=\"lux-score\">Score: <b>{score_str}</b></div>
      </div>

      <div class=\"lux-weights\">{weights_html}</div>

      <div class=\"lux-kpis\">
        <div class=\"lux-kpi\"><div class=\"k\">חו״ל</div><div class=\"v\">{_fmt_pct(rr.get('חו״ל (%)'))}</div></div>
        <div class=\"lux-kpi\"><div class=\"k\">ישראל</div><div class=\"v\">{_fmt_pct(rr.get('ישראל (%)'))}</div></div>
        <div class=\"lux-kpi\"><div class=\"k\">מניות</div><div class=\"v\">{_fmt_pct(rr.get('מניות (%)'))}</div></div>
        <div class=\"lux-kpi\"><div class=\"k\">מט״ח</div><div class=\"v\">{_fmt_pct(rr.get('מט״ח (%)'))}</div></div>
        <div class=\"lux-kpi\"><div class=\"k\">לא־סחיר</div><div class=\"v\">{_fmt_pct(rr.get('לא־סחיר (%)'))}</div></div>
        <div class=\"lux-kpi\"><div class=\"k\">שארפ</div><div class=\"v\">{_fmt_num(rr.get('שארפ משוקלל'), '{:.2f}')}</div></div>
        <div class=\"lux-kpi\"><div class=\"k\">שירות</div><div class=\"v\">{_fmt_num(rr.get('שירות משוקלל'), '{:.1f}')}</div></div>
        <div class=\"lux-kpi\"><div class=\"k\">סה"כ</div><div class=\"v\">{score_str}</div></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ----------------------------
# Tab 1: Inputs (main, not sidebar)
# ----------------------------
with tab1:
    st.subheader("הגדרות בסיס")
    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.2, 1.2])

    with c1:
        st.session_state["n_funds"] = st.selectbox(
            "כמה קופות לשלב?",
            options=[1, 2, 3],
            index=[1, 2, 3].index(st.session_state["n_funds"]),
        )
    with c2:
        st.session_state["mix_policy"] = st.selectbox(
            "מדיניות מנהלים",
            options=["מותר לערבב מנהלים", "אותו מנהל בלבד"],
            index=0 if st.session_state["mix_policy"] == "מותר לערבב מנהלים" else 1,
        )
    with c3:
        st.session_state["step"] = st.selectbox(
            "צעד משקלים (%)",
            options=[1, 2, 5, 10, 20],
            index=[1, 2, 5, 10, 20].index(st.session_state["step"]),
            help="בצעד קטן החיפוש יסודי יותר אך כבד יותר.",
        )
    with c4:
        st.session_state["primary_rank"] = st.selectbox(
            "דירוג ראשי",
            options=["דיוק", "שארפ", "שירות"],
            index=["דיוק", "שארפ", "שירות"].index(st.session_state["primary_rank"]),
        )

    st.divider()
    st.subheader("יעדים ומגבלות – סט אחד לכל המשתנים")

    # One unified set: include + target + hard/soft + inequality mode
    rows = []
    mcols = st.columns([1.2, 1.2, 1.2, 1.0, 1.0])
    with mcols[0]:
        st.markdown("**משתנה**")
    with mcols[1]:
        st.markdown("**לכלול בדירוג**")
    with mcols[2]:
        st.markdown("**יעד (%)**")
    with mcols[3]:
        st.markdown("**קשיחות**")
    with mcols[4]:
        st.markdown("**כיוון**")

    def metric_row(key: str, label: str, default_mode: str):
        cols = st.columns([1.2, 1.2, 1.2, 1.0, 1.0])
        with cols[0]:
            st.write(label)
        with cols[1]:
            inc = st.checkbox(" ", value=st.session_state["include"].get(key, False), key=f"inc_{key}")
        with cols[2]:
            val = st.slider(
                " ", min_value=0.0, max_value=120.0 if key in ("foreign", "fx") else 100.0,
                value=float(st.session_state["targets"].get(key, 0.0)),
                step=0.5, key=f"tgt_{key}",
                label_visibility="collapsed"
            )
        with cols[3]:
            hard = st.selectbox(
                " ", options=["רך", "קשיח"],
                index=0 if st.session_state["constraint"].get(key, ("רך", default_mode))[0] == "רך" else 1,
                key=f"hard_{key}",
                label_visibility="collapsed"
            )
        with cols[4]:
            mode = st.selectbox(
                " ", options=["בדיוק", "לפחות", "לכל היותר"],
                index=["בדיוק", "לפחות", "לכל היותר"].index(st.session_state["constraint"].get(key, ("רך", default_mode))[1]),
                key=f"mode_{key}",
                label_visibility="collapsed"
            )
        st.session_state["include"][key] = inc
        st.session_state["targets"][key] = float(val)
        st.session_state["constraint"][key] = (hard, mode)

    metric_row("foreign", "חו״ל", "בדיוק")
    metric_row("stocks", "מניות", "בדיוק")
    metric_row("fx", "מט״ח", "לפחות")
    metric_row("illiquid", "לא־סחיר", "לכל היותר")

    st.divider()
    st.subheader("הרצה")
    run = st.button("חשב 3 חלופות", type="primary", use_container_width=True)

    if run:
        with st.spinner("מריץ חיפוש יסודי..."):
            sols, note = find_best_solutions(
                df=df_long,
                n_funds=st.session_state["n_funds"],
                step=st.session_state["step"],
                mix_policy=st.session_state["mix_policy"],
                include=st.session_state["include"],
                constraint=st.session_state["constraint"],
                targets=_apply_israel_rule(st.session_state["targets"]),
                primary_rank=st.session_state["primary_rank"],
                max_solutions_scan=90000 if st.session_state["n_funds"] <= 2 else 70000,
            )
            st.session_state["last_note"] = note
            if sols.empty:
                st.session_state["last_results"] = None
            else:
                top3 = pick_three_distinct(sols, st.session_state["primary_rank"])
                st.session_state["last_results"] = {
                    "solutions_all": sols.head(5000),  # keep limited for transparency tab
                    "top3": top3,
                }
        if st.session_state["last_results"] is None:
            st.error("לא נמצאו פתרונות.")
        else:
            st.success("מוכן! עבור לטאב 'תוצאות'.")

# ----------------------------
# Tab 2: Results (show full table immediately)
# ----------------------------
with tab2:
    st.subheader("תוצאות (3 חלופות)")
    if st.session_state.get("last_results") is None:
        st.info("כדי לראות תוצאות, עבור לטאב 'הגדרות יעד' ולחץ 'חשב 3 חלופות'.")
    else:
        top3 = st.session_state["last_results"]["top3"].copy()

        # KPI cards
        _render_kpi_cards(top3)

        st.markdown("#### פירוט לכל חלופה")
        st.caption(st.session_state.get("last_note", ""))

        # Render each alternative as a compact mini-table card (mobile friendly)
        for i, rr in enumerate(top3.to_dict(orient="records"), start=1):
            _render_alt_card(rr, i)


# ----------------------------
# Tab 4: Compare tracks / funds
# ----------------------------
def _fmt_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{float(x):.2f}%"
    except Exception:
        return "—"


def _fmt_num(x: float, decimals: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "—"


def _render_compare_cards(df_sel: pd.DataFrame, title: str = "השוואה"):
    if df_sel is None or df_sel.empty:
        st.info("בחר לפחות מסלול אחד להשוואה.")
        return

    # In some Streamlit Cloud deployments, HTML injected via `st.markdown(..., unsafe_allow_html=True)`
    # can still end up escaped (showing the tags as text). We render the compare cards using
    # `components.html`, which is robust and keeps the design pixel-consistent.

    st.markdown(f"#### {_escape_html(title)}")

    import streamlit.components.v1 as components

    css = """
    <style>
      :root{--cmp-card:#0f1624;--cmp-border:rgba(255,255,255,.10);--cmp-muted:rgba(255,255,255,.70);
            --cmp-shadow:0 14px 40px rgba(0,0,0,.38);}
      .cmp-wrap{direction:rtl;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;}
      .cmp-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:10px;}
      @media(max-width:1100px){.cmp-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
      @media(max-width:760px){.cmp-grid{grid-template-columns:1fr;}}
      .cmp-card{background:var(--cmp-card);border:1px solid var(--cmp-border);border-radius:18px;box-shadow:var(--cmp-shadow);padding:14px 14px 12px;}
      .cmp-head{display:flex;flex-direction:column;gap:4px;margin-bottom:10px;}
      .cmp-fund{font-weight:900;font-size:15px;line-height:1.25;letter-spacing:.2px;}
      .cmp-sub{font-size:12.5px;color:var(--cmp-muted);line-height:1.35;}
      .cmp-kpis{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:10px 0 12px;}
      .cmp-kpi{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:10px;}
      .cmp-kpi .k{font-size:11.5px;color:var(--cmp-muted);margin-bottom:2px;font-weight:900;}
      .cmp-kpi .v{font-weight:900;font-size:15px;}
      .cmp-mini{width:100%;border-collapse:separate;border-spacing:0 8px;}
      .cmp-mini td{padding:7px 10px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);}
      .cmp-mini td:first-child{border-radius:12px 0 0 12px;color:var(--cmp-muted);font-size:12px;font-weight:900;white-space:nowrap;}
      .cmp-mini td:last-child{border-radius:0 12px 12px 0;text-align:left;font-size:13px;font-weight:900;white-space:nowrap;}
      @media(max-width:520px){.cmp-card{padding:12px 12px 10px;}}
    </style>
    """

    cards = []
    for _, r in df_sel.iterrows():
        fund = _escape_html(str(r.get("fund", "")))
        track = _escape_html(str(r.get("track", "")))
        manager = _escape_html(str(r.get("manager", "")))
        service = _fmt_num(r.get("service", float('nan')), 1)
        sharpe = _fmt_num(r.get("sharpe", float('nan')), 2)
        stocks = _fmt_pct(r.get("stocks", float('nan')))
        foreign = _fmt_pct(r.get("foreign", float('nan')))
        fx = _fmt_pct(r.get("fx", float('nan')))
        ill = _fmt_pct(r.get("illiquid", float('nan')))
        cards.append(
            f"""
            <div class='cmp-card'>
              <div class='cmp-head'>
                <div class='cmp-fund'>{fund}</div>
                <div class='cmp-sub'>מסלול: {track} · מנהל: {manager}</div>
              </div>
              <div class='cmp-kpis'>
                <div class='cmp-kpi'><div class='k'>ציון שירות</div><div class='v'>{service}</div></div>
                <div class='cmp-kpi'><div class='k'>Sharpe</div><div class='v'>{sharpe}</div></div>
              </div>
              <table class='cmp-mini'>
                <tr><td>מניות</td><td>{stocks}</td></tr>
                <tr><td>חו\"ל</td><td>{foreign}</td></tr>
                <tr><td>חשיפת מט\"ח</td><td>{fx}</td></tr>
                <tr><td>לא-סחיר</td><td>{ill}</td></tr>
              </table>
            </div>
            """
        )

    body = """<div class='cmp-wrap'><div class='cmp-grid'>{cards}</div></div>""".format(cards="\n".join(cards))
    rows = (len(df_sel) + 2) // 3
    height = int(90 + rows * 270)
    components.html(css + body, height=height, scrolling=False)


with tab4:
    st.subheader("השוואת מסלולי השקעה")
    st.caption("בחר קופות להשוואה ותקבל כרטיסיות/מיני-טבלאות מעוצבות ללא גלילה לרוחב (גם במובייל).")

    mode = st.radio(
        "מצב השוואה",
        ["קופות ספציפיות (מנהל×מסלול)", "ממוצע לפי מסלול בלבד"],
        horizontal=True,
        key="compare_mode",
    )

    if mode.startswith("קופות"):
        df_opt = df_long.copy()
        df_opt["_key"] = df_opt.apply(lambda r: f"{r.get('fund','')} | {r.get('track','')} | {r.get('manager','')}", axis=1)
        options = df_opt["_key"].tolist()
        default_sel = options[:3] if len(options) >= 3 else options
        selected = st.multiselect(
            "חיפוש ובחירה להשוואה",
            options,
            default=default_sel,
            key="compare_funds",
        )
        df_sel = df_opt[df_opt["_key"].isin(selected)].drop(columns=["_key"], errors="ignore")

        # Nice-to-have: quick sort by service/sharpe
        sort_by = st.selectbox("מיין לפי", ["ציון שירות", "Sharpe", "מניות", "חו״ל", "לא-סחיר"], index=0)
        sort_map = {
            "ציון שירות": "service",
            "Sharpe": "sharpe",
            "מניות": "stocks",
            "חו״ל": "foreign",
            "לא-סחיר": "illiquid",
        }
        col = sort_map.get(sort_by, "service")
        df_sel = df_sel.sort_values(col, ascending=False, kind="mergesort")

        _render_compare_cards(df_sel, "כרטיסיות השוואה")

    else:
        tracks = sorted(df_long["track"].dropna().unique().tolist())
        default_tracks = tracks[:3] if len(tracks) >= 3 else tracks
        selected_tracks = st.multiselect(
            "חיפוש ובחירה של מסלולים",
            tracks,
            default=default_tracks,
            key="compare_tracks",
        )
        if not selected_tracks:
            st.info("בחר לפחות מסלול אחד.")
        else:
            df_sel = (
                df_long[df_long["track"].isin(selected_tracks)]
                .groupby("track", as_index=False)
                .agg(
                    stocks=("stocks", "mean"),
                    foreign=("foreign", "mean"),
                    fx=("fx", "mean"),
                    illiquid=("illiquid", "mean"),
                    sharpe=("sharpe", "mean"),
                    service=("service", "mean"),
                )
            )
            df_sel["fund"] = df_sel["track"]
            df_sel["manager"] = "ממוצע"
            _render_compare_cards(df_sel[["fund", "track", "manager", "stocks", "foreign", "fx", "illiquid", "sharpe", "service"]], "ממוצע מסלול")

# ----------------------------
# Tab 3: Transparency
# ----------------------------
with tab3:
    st.subheader("פירוט חישוב / שקיפות")
    st.caption("כדי לא להעמיס – הפירוט מוצג בתוך Expander.")
    with st.expander("לחץ להצגת פירוט"):
        st.write("**פרטי קלט:**")
        st.json({
            "מספר קופות": st.session_state["n_funds"],
            "מדיניות מנהלים": st.session_state["mix_policy"],
            "צעד משקלים": st.session_state["step"],
            "דירוג ראשי": st.session_state["primary_rank"],
            "כולל בדירוג": st.session_state["include"],
            "יעדים": st.session_state["targets"],
            "קשיחות/כיוון": st.session_state["constraint"],
            "הערת ריצה": st.session_state.get("last_note", ""),
        }, expanded=False)

        if st.session_state.get("last_results") is None:
            st.info("אין פתרונות להצגה.")
        else:
            st.markdown("**דוגמאות מתוך רשימת המועמדים (עד 200 שורות):**")
            cand = st.session_state["last_results"]["solutions_all"].head(200).copy()
            # show only relevant cols
            cand = cand[[
                "מנהלים", "קופות", "מסלולים",
                "חו״ל (%)", "מניות (%)", "מט״ח (%)", "לא־סחיר (%)",
                "שארפ משוקלל", "שירות משוקלל", "score", "weights"
            ]].copy()
            cand["משקלים"] = cand["weights"].apply(lambda w: " / ".join([f"{int(x)}%" for x in w]) if isinstance(w, (tuple, list)) else str(w))
            cand = cand.drop(columns=["weights"]).rename(columns={"score": "Score (סטייה)"})
            st.dataframe(cand, use_container_width=True, hide_index=True, column_config={
                "קופות": st.column_config.TextColumn(width="large"),
                "מסלולים": st.column_config.TextColumn(width="large"),
            })

st.caption("© Profit Mix Optimizer – חישוב ממוצע משוקלל על בסיס הקובץ המובנה. ישראל = 100 − חו״ל.")