import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math
import os
from difflib import get_close_matches
import pydeck as pdk

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Comparable Adjustment Explorer", page_icon="📈", layout="wide")

# ===================== CONSTANTS =====================
Y_COL_CANDIDATES = ["Sold Price", "Sale Price", "SoldPrice", "Close Price"]

# Feature set (Bedrooms removed; Basement Y/N added)
FEATURE_SYNONYMS = {
    "SqFt Finished": [
        "SqFt -Total Finished","GLA","Square Feet","Living Area","Total Finished SqFt",
        "Finished SqFt","Sq Ft Finished","Gross Living Area","Above Grade Finished Area"
    ],
    "Above Grade Finished": [
        "Above Grade Finished","Above Grade Finished Area","AGLA","Above Grade Living Area",
        "GLA","Gross Living Area","Living Area","Square Feet","SqFt -Total Finished","Total Finished SqFt"
    ],
    "Basement SqFt Finished": [
        "Below Grade Finished","Below Grade Finished.","Basement SqFt Finished","Basement Finished SqFt",
        "Finished Basement SqFt","Basement Finished Area","Bsmt Fin SqFt","Basement Fin Sq Ft",
        "Basement Sq Ft Finished","Below Grade Finished Area","Below-Grade Finished Area",
        "BGFA","Finished Below Grade","Below Grade Finished, SqFt"
    ],
    "Basement Y/N": [
        "Basement Y/N","Basement Yes/No","Basement","Bsmt Y/N","Has Basement","Basement Present",
        "Basement Exists","BasementYn","Basement_YN","Basement?:","Basement?","Bsmt"
    ],
    "Garage Spaces": [
        "Garage Spaces","Garage","# Garage Spaces","Garage Y/N","Garage YN","Garage Spots","Garage Stalls"
    ],
}

DATE_COL_CANDIDATES = [
    "Close Date","Closing Date","Sold Date","Contract Date","List Date","Sale Date","COE Date","Listing Date"
]

# ===================== HELPERS =====================
# ---- Coordinate detection helpers ----
def _is_lat_col(name: str) -> bool:
    n = str(name).lower()
    if "lot" in n:  # avoid false positives like "Lot Size"
        return False
    return ("lat" in n) or ("y coord" in n) or (n.strip() in {"y","latitude"})

def _is_lon_col(name: str) -> bool:
    n = str(name).lower()
    if "loan" in n:  # avoid "loan amount"
        return False
    return ("lon" in n) or ("lng" in n) or ("long" in n) or ("x coord" in n) or (n.strip() in {"x","longitude"})

def find_lat_lon_cols(df: pd.DataFrame) -> tuple[str|None, str|None]:
    lat_col = lon_col = None
    for c in df.columns:
        if lat_col is None and _is_lat_col(c):
            lat_col = c
        if lon_col is None and _is_lon_col(c):
            lon_col = c
    return lat_col, lon_col

def build_map_dataframe(original_df: pd.DataFrame, work_filtered_with_idx: pd.DataFrame,
                        lat_col: str, lon_col: str, y_col: str, x_col: str):
    """
    Align coordinates from the original df to the filtered working set using the preserved 'index' column.
    Returns a DataFrame with columns: ['lat','lon','price','feature'] plus original index.
    """
    # Pull lat/lon from original rows corresponding to the filtered 'index'
    coords = original_df.loc[:, [lat_col, lon_col]].copy()
    coords.columns = ["_lat", "_lon"]
    # Map coordinates onto the working set by original row index
    out = work_filtered_with_idx.merge(
        coords, left_on="index", right_index=True, how="left"
    )
    # Clean and drop missing coords
    out["_lat"] = pd.to_numeric(out["_lat"], errors="coerce")
    out["_lon"] = pd.to_numeric(out["_lon"], errors="coerce")
    out = out.dropna(subset=["_lat","_lon"]).copy()
    # Build friendly columns for tooltips
    out["lat"] = out["_lat"].astype(float)
    out["lon"] = out["_lon"].astype(float)
    out["price"] = out[y_col].astype(float)
    out["feature"] = out[x_col].astype(float)
    return out[["index","lat","lon","price","feature"]]

def clean_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == "object":
        s = s.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def map_yes_no_to_binary(series: pd.Series) -> pd.Series:
    if series.dtype != "object":
        return series
    s = series.astype(str).str.strip().str.upper()
    yn = {"Y":1,"YES":1,"TRUE":1,"T":1,"1":1,"N":0,"NO":0,"FALSE":0,"F":0,"0":0}
    mapped = s.map(yn)
    out = series.copy()
    out.loc[mapped.notna()] = mapped.loc[mapped.notna()].astype(int)
    return out

def looks_discrete_integer(s: pd.Series, max_unique=12, tol=0.01) -> bool:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return False
    if s.nunique() <= max_unique:
        near_int = (s - s.round()).abs() <= tol
        return bool(near_int.all())
    return False

def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    for c in candidates:
        if c in cols: return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map: return lower_map[c.lower()]
    m = get_close_matches(candidates[0], cols, n=1, cutoff=0.6)
    return m[0] if m else None

def _tokens(s: str) -> set[str]:
    return set(str(s).lower().replace("_"," ").replace("-"," ").replace(".","").split())

def _viable_numeric(series: pd.Series, min_numeric_share=0.6) -> bool:
    s = clean_numeric(map_yes_no_to_binary(series))
    share = s.notna().mean() if len(s) else 0
    return share >= min_numeric_share and np.nanstd(s) > 0

def resolve_feature_column(df: pd.DataFrame, label: str) -> str | None:
    if label not in FEATURE_SYNONYMS:
        return None
    syns = FEATURE_SYNONYMS[label]
    cols = list(df.columns)

    required = {
        "SqFt Finished": {"sqft"} | {"gla","living","area","finished"},
        "Above Grade Finished": {"above","grade"} | {"gla","living","area","finished","sqft"},
        "Basement SqFt Finished": {"finished"} | {"fin"} | {"sqft"},
        "Basement Y/N": {"basement"} | {"yn","y/n","yes","no","present","exists","has"},
        "Garage Spaces": {"garage"} | {"space","spaces","stalls","spots"},
    }.get(label, set())

    if label == "Basement SqFt Finished":
        def _is_bg_finished(col: str) -> bool:
            t = _tokens(col)
            return (("basement" in t) or ({"below","grade"}.issubset(t))) and ({"finished","fin"}.intersection(t) or "sqft" in t)
        narrowed = [c for c in cols if _is_bg_finished(c)]
        if narrowed: cols = narrowed

    def _score(col: str) -> float:
        t = _tokens(col)
        score = 0.0
        for s in syns:
            ts = _tokens(s)
            score = max(score, len(t & ts) / max(1, len(ts)))
        for s in syns:
            if str(col).lower().startswith(str(s).lower()[:6]):
                score += 0.15
        return score

    for s in syns:
        if s in cols and _viable_numeric(df[s]): 
            return s
    lower_map = {c.lower(): c for c in cols}
    for s in syns:
        if s.lower() in lower_map and _viable_numeric(df[lower_map[s.lower()]]):
            return lower_map[s.lower()]

    cand = [c for c in cols if (not required) or required.issubset(_tokens(c))] if required else cols
    cand = sorted(cand, key=lambda c: _score(c), reverse=True)
    for c in cand:
        if _viable_numeric(df[c]):
            return c

    for s in syns:
        m = get_close_matches(s, cols, n=3, cutoff=0.82)
        for c in m:
            if (not required or required & _tokens(c)) and _viable_numeric(df[c]):
                return c
    return None

def regression_slope(x: np.ndarray, y: np.ndarray):
    if len(x) < 2 or np.nanstd(x) == 0:
        return np.nan, np.nan, np.nan
    m, b = np.polyfit(x, y, 1)
    r = pd.Series(x).corr(pd.Series(y))
    r2 = r*r if pd.notna(r) else np.nan
    return m, b, r2

def _is_gla_like(name: str) -> bool:
    n = name.lower()
    return (("gla" in n) or ("living" in n) or ("gross" in n) or ("total" in n) or ("above" in n and "grade" in n)) and ("basement" not in n and "below" not in n)

def compute_stats(df, y_col, x_col):
    x = df[x_col].values
    y = df[y_col].values
    m, _, r2 = regression_slope(x, y)
    median_ppsf = np.nan
    if _is_gla_like(x_col) or ("sqft" in x_col.lower() and "basement" not in x_col.lower() and "below" not in x_col.lower()):
        w = df[df[x_col] > 0].copy()
        median_ppsf = np.nan if w.empty else (w[y_col] / w[x_col]).median()
    return dict(slope=m, r2=r2, median_ppsf=median_ppsf, n=len(df))

def compute_binary_stats(df, y_col, x_col):
    present = sorted(df[x_col].dropna().astype(int).unique().tolist())
    has_both = (present == [0,1])
    res = {"n": len(df), "has_both": has_both}
    mean_no = df.loc[df[x_col]==0, y_col].mean() if 0 in present else np.nan
    mean_yes = df.loc[df[x_col]==1, y_col].mean() if 1 in present else np.nan
    res["mean_no"] = mean_no
    res["mean_yes"] = mean_yes
    if has_both:
        slope, _, r2 = regression_slope(df[x_col].values, df[y_col].values)
        res["slope"] = slope
        res["r2"] = r2
    else:
        res["slope"] = np.nan
        res["r2"] = np.nan
    return res

def fig_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()
# ---- Helpers for narrative labeling and rounding ----
def normalize_feature_label_for_narrative(label: str) -> str:
    """
    If the chosen feature is GLA-like (SqFt Finished, Above Grade Finished, etc.),
    return 'Gross Living Area'. Otherwise return the original label.
    """
    l = (label or "").lower()
    # Exclude basement terms from GLA relabeling
    if "basement" in l or "below" in l:
        return label
    gla_clues = [
        "gla", "gross living", "living area",
        "sqft", "sq ft", "square feet",
        "total finished", "above grade", "finished"
    ]
    if any(k in l for k in gla_clues):
        return "Gross Living Area"
    return label

def round_to_nearest_5_dollars(x: float) -> float:
    """Round a dollar amount to the nearest $5 (handles negatives and NaN)."""
    if x is None:
        return np.nan
    try:
        if np.isnan(x) or np.isinf(x):
            return np.nan
    except Exception:
        pass
    return float(np.round(float(x) / 5.0) * 5.0)

# ======== DATE DETECTOR (stricter; avoids IDs like "List Number") ========
def find_first_date_col(df: pd.DataFrame) -> str | None:
    bad_name_snippets = ["list number","mls#","mls #","mls id","listing id","list no","record id","id"]
    for c in df.columns:
        name = str(c).lower().strip()
        if not any(k in name for k in ["date","close","sold","sale","contract","coe","closing","list date","listing date"]):
            continue
        if any(b in name for b in bad_name_snippets):
            continue
        s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        if s.notna().mean() < 0.5:
            continue
        yrs = s.dropna().dt.year
        if yrs.empty or yrs.min() < 1990 or yrs.max() > 2100:
            continue
        return c
    return None

# ===================== GREEDY REMOVER =====================
def greedy_remove_toward_target(df: pd.DataFrame, y_col: str, x_col: str, target_slope: float, max_removals: int):
    df = df.copy().reset_index(drop=False)
    df = df[['index', y_col, x_col]].dropna()
    if df.empty: return df, [], []
    cur_x = df[x_col].values; cur_y = df[y_col].values
    m, _, _ = regression_slope(cur_x, cur_y)
    removed_info = []
    direction = np.sign(target_slope - (m if not np.isnan(m) else 0))
    if direction == 0: return df, removed_info, []
    for step in range(max_removals):
        n = len(df)
        if n <= 2: break
        cur_x = df[x_col].values; cur_y = df[y_col].values
        m_cur, _, _ = regression_slope(cur_x, cur_y)
        if direction > 0 and m_cur >= target_slope - 1e-9: break
        if direction < 0 and m_cur <= target_slope + 1e-9: break
        best_idx = None; best_diff = None; best_new_m = None
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            x_try = cur_x[mask]; y_try = cur_y[mask]
            if len(x_try) < 2 or np.nanstd(x_try) == 0: continue
            m_try, _, _ = regression_slope(x_try, y_try)
            if direction > 0 and m_try < m_cur - 1e-9: continue
            if direction < 0 and m_try > m_cur + 1e-9: continue
            diff = abs(m_try - target_slope)
            if best_diff is None or diff < best_diff:
                best_diff = diff; best_idx = i; best_new_m = m_try
        if best_idx is None: break
        row = df.iloc[best_idx]
        removed_info.append({
            "orig_index": int(row['index']),
            y_col: float(row[y_col]),
            x_col: float(row[x_col]),
            "step": step + 1,
            "slope_after_removal": float(best_new_m)
        })
        df = df.drop(df.index[best_idx]).reset_index(drop=True)
    return df, removed_info, []

# ===================== FILTERS & OUTLIERS =====================
def filter_data(df_in: pd.DataFrame, y_col: str, x_col: str, date_col: str | None,
                price_rng, x_rng, date_rng, is_binary_x: bool) -> pd.DataFrame:
    out = df_in.copy()
    out = out[(out[y_col] >= price_rng[0]) & (out[y_col] <= price_rng[1])]
    if not is_binary_x:
        out = out[(out[x_col] >= x_rng[0]) & (out[x_col] <= x_rng[1])]
    if date_col and (date_col in out.columns) and date_rng is not None:
        d = pd.to_datetime(out[date_col], errors="coerce")
        mask = (d >= pd.to_datetime(date_rng[0])) & (d <= pd.to_datetime(date_rng[1]))
        out = out[mask]
    return out

def flag_outliers(df: pd.DataFrame, y_col: str, x_col: str, is_binary_x: bool, is_sqft_like: bool) -> pd.Series:
    s = pd.Series(False, index=df.index)
    if len(df) < 8:
        return s
    y = df[y_col]
    if is_binary_x:
        q1, q3 = y.quantile(0.25), y.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        s = (y < lo) | (y > hi)
    else:
        if is_sqft_like:
            xx = df[x_col]
            ppsf = y / xx.replace(0, np.nan)
            med = ppsf.median(skipna=True); std = ppsf.std(skipna=True)
            if pd.notna(std) and std > 0:
                s = (ppsf > med + 2*std) | (ppsf < med - 2*std)
        else:
            x = df[x_col]
            if np.nanstd(x) > 0:
                m, b, _ = regression_slope(x.values, y.values)
                resid = y - (m*x + b)
                sd = resid.std()
                if pd.notna(sd) and sd > 0:
                    s = resid.abs() > 2*sd
    return s.fillna(False)

# ===================== PLOTTING =====================
def make_scatter_figure(
    df, y_col, x_col, title,
    int_ticks=None, jitter_width=0.08, xtick_labels=None,
    flagged_mask=None, hide_flagged=False
):
    fig, ax = plt.subplots(figsize=(7.8,5.2))
    if df.empty:
        ax.set_title(title + " (no data)")
        return fig

    x_true = df[x_col].values
    y = df[y_col].values

    # jitter for discrete
    if int_ticks is not None and len(int_ticks) > 0:
        rng = np.random.default_rng(42)
        x_plot = x_true + rng.uniform(-jitter_width, jitter_width, size=len(x_true))
        ax.set_xticks(int_ticks)
        if xtick_labels:
            ax.set_xticklabels(xtick_labels)
        else:
            ax.set_xticklabels([str(int(t)) for t in int_ticks])
    else:
        x_plot = x_true

    if flagged_mask is not None:
        keep_mask = ~flagged_mask if hide_flagged else np.ones(len(df), dtype=bool)
        flagged_to_plot = (~keep_mask) if hide_flagged else flagged_mask
    else:
        keep_mask = np.ones(len(df), dtype=bool)
        flagged_to_plot = np.zeros(len(df), dtype=bool)

    ax.scatter(x_plot[keep_mask], y[keep_mask], s=32, label="Comps")
    if flagged_to_plot.any():
        ax.scatter(x_plot[flagged_to_plot], y[flagged_to_plot], marker='x', s=80, color='red', label='Flagged')

    x_shown = x_true[keep_mask]; y_shown = y[keep_mask]
    if len(x_shown) >= 2 and np.nanstd(x_shown) > 0:
        m, b, _ = regression_slope(x_shown, y_shown)
        xline = np.linspace(np.nanmin(x_shown), np.nanmax(x_shown), 200)
        ax.plot(xline, m*xline + b, color='red', linewidth=2, label=f"Fit: ${m:,.0f}/unit")

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(axis="y", linestyle="--", color="#e5e5e5", linewidth=0.8)
    fig.tight_layout()
    return fig

def make_bar_figure_binary(df, y_col, x_col, title):
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    if df.empty:
        ax.set_title(title + " (no data)")
        return fig
    g0 = df[df[x_col] == 0][y_col]
    g1 = df[df[x_col] == 1][y_col]
    means = [g0.mean(), g1.mean()]
    ns = [len(g0), len(g1)]
    ses = []
    for g in (g0, g1):
        if len(g) > 1:
            ses.append(g.std(ddof=1) / np.sqrt(len(g)))
        else:
            ses.append(0.0)
    ax.bar([0,1], means, yerr=ses, capsize=6)
    ax.set_xticks([0,1]); ax.set_xticklabels(["No","Yes"])
    ax.set_ylabel(y_col)
    ax.set_title(title + " — means ± SE")
    ax.grid(axis="y", linestyle="--", color="#e5e5e5", linewidth=0.8)
    for i, n in enumerate(ns):
        ax.text(i, means[i], f" n={n}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    return fig

# ===================== EXCEL REPORT =====================
def build_excel_report(orig_fig, adj_fig, adj_clean_fig, kept_df, removed_df, summary_df):
    from pandas import ExcelWriter
    output = BytesIO()
    try:
        import xlsxwriter  # noqa: F401
        img_orig = BytesIO(fig_bytes(orig_fig))
        img_adj = BytesIO(fig_bytes(adj_fig))
        img_adj_clean = BytesIO(fig_bytes(adj_clean_fig))
        with ExcelWriter(output, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            kept_df.to_excel(writer, sheet_name="Kept", index=False)
            removed_df.to_excel(writer, sheet_name="Removed", index=False)
            wb = writer.book
            ws = wb.add_worksheet("Charts")
            writer.sheets["Charts"] = ws
            ws.write(0, 0, "Original")
            ws.insert_image(1, 0, "original.png", {"image_data": img_orig, "x_scale": 0.9, "y_scale": 0.9})
            ws.write(35, 0, "Adjusted (with removed)")
            ws.insert_image(36, 0, "adjusted.png", {"image_data": img_adj, "x_scale": 0.9, "y_scale": 0.9})
            ws.write(70, 0, "Adjusted (clean)")
            ws.insert_image(71, 0, "adjusted_clean.png", {"image_data": img_adj_clean, "x_scale": 0.9, "y_scale": 0.9})
        output.seek(0)
        return output.getvalue(), "xlsxwriter"
    except ModuleNotFoundError:
        with ExcelWriter(output, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            kept_df.to_excel(writer, sheet_name="Kept", index=False)
            removed_df.to_excel(writer, sheet_name="Removed", index=False)
            pd.DataFrame([{"Note": "Charts not embedded because 'xlsxwriter' is not installed. Install it: pip install xlsxwriter"}]).to_excel(writer, sheet_name="Charts_Notes", index=False)
        output.seek(0)
        return output.getvalue(), "openpyxl"

# ===================== AI NARRATIVE (neutral, formal, appraisal-style) =====================
def infer_market_context(df: pd.DataFrame):
    geo_cols = ["City","Municipality","County","State","ST","Zip","ZIP","Postal Code","Neighborhood",
                "Subdivision","MLS Area","Area","Address","Street Address","Location","Region"]
    found = {}
    for c in df.columns:
        cl = str(c).strip()
        if cl in geo_cols or any(k in cl.lower() for k in ["city","county","state","zip","postal","neigh","subdiv","area","address","location","region"]):
            vc = df[cl].dropna().astype(str)
            if not vc.empty:
                found[cl] = vc.value_counts().head(3).index.tolist()
    city = state = county = zipc = None
    for k, v in found.items():
        lk = k.lower()
        if "city" in lk and v: city = v[0]
        if lk in ("state","st") and v: state = v[0]
        if "county" in lk and v: county = v[0].replace(" County","")
        if ("zip" in lk or "postal" in lk) and v: zipc = v[0]
    parts = []
    if city: parts.append(str(city))
    if state and state.upper() not in ("NAN","") and state not in parts: parts.append(str(state))
    if not parts and county: parts.append(f"{county} County")
    if not parts and zipc: parts.append(f"ZIP {zipc}")
    location_str = ", ".join(parts) if parts else None

    date_col = find_first_date_col(df)
    timeframe = None
    if date_col:
        d = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not d.empty:
            start = d.min().strftime("%b %Y"); end = d.max().strftime("%b %Y")
            timeframe = start if start == end else f"{start}–{end}"
    return {"location": location_str, "timeframe": timeframe}

def ai_summary(feature_label, y_col, stats_before, stats_after, context, sample_rows):
    """
    Generates a formal appraisal-style paragraph.
    Enforces:
      - 'Gross Living Area' wording for GLA-like features
      - Coefficient reported rounded to the nearest $5
    """
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return "Set OPENAI_API_KEY to enable the AI market narrative."
        client = OpenAI(api_key=key)

        # Prefer 'after' stats; fall back to 'before'
        slope_use = stats_after.get("slope", np.nan)
        if np.isnan(slope_use):
            slope_use = stats_before.get("slope", np.nan)
        r2_use = stats_after.get("r2", np.nan)
        if np.isnan(r2_use):
            r2_use = stats_before.get("r2", np.nan)

        mppsf_after = stats_after.get("median_ppsf", np.nan)
        mppsf_before = stats_before.get("median_ppsf", np.nan)
        mppsf_val = mppsf_after if not np.isnan(mppsf_after) else mppsf_before
        mppsf_available = not np.isnan(mppsf_val)

        # Normalize labeling and rounding for the narrative
        narr_label = normalize_feature_label_for_narrative(feature_label)
        slope_rounded_5 = round_to_nearest_5_dollars(slope_use)
        coeff_phrase = None if np.isnan(slope_rounded_5) else f"${slope_rounded_5:,.0f} per additional square foot"

        # Optional context (location, timeframe)
        loc = context.get("location") or ""
        tf = context.get("timeframe") or ""
        where_when = ", ".join([p for p in [loc, tf] if p]).strip(", ")

        system_rules = (
            "You are a certified residential appraiser writing a professional adjustment explanation for an appraisal report. "
            "Use concise, formal language (4–6 sentences). State that regression analysis of comparable sales in the subject’s "
            "competitive market area was used. Refer to the feature as the exact string provided. Explain that the analysis isolated "
            "the contributory effect of the feature while accounting for other market drivers (e.g., location, condition, amenities). "
            "State that the data set was screened for accuracy and that outliers or atypical sales were removed to reflect typical "
            "market behavior. Conclude that the resulting coefficient represents the market-supported rate of change in sale price "
            "attributable to differences in the feature, providing a credible, data-driven basis for the applied adjustment. "
            "Do NOT mention targets, parameter tuning, admin tools, UI controls, or internal processes. "
            "Use dollars with thousands separators for any monetary figures."
        )

        user_payload = {
            "market_context": where_when,                 # e.g., "Louisville, KY, Jan 2024–Oct 2025"
            "y_axis_label": y_col,                        # "Sold Price"
            "feature_label_exact": narr_label,            # enforced label (e.g., "Gross Living Area")
            "coefficient_literal_phrase": coeff_phrase,   # e.g., "$50 per additional square foot"
            "r2_value": None if (r2_use is None or np.isnan(r2_use)) else float(r2_use),
            "median_price_per_sqft_available": bool(mppsf_available),
            "median_price_per_sqft": None if not mppsf_available else float(mppsf_val),
            "instructions": (
                "Use 'feature_label_exact' verbatim when naming the feature. "
                "If 'coefficient_literal_phrase' is provided, use it verbatim when stating the coefficient. "
                "Optionally include 'market_context' in the opening sentence if present."
            ),
            "example_open": (
                f"The adjustment for {narr_label} was developed using regression analysis applied to a data set of "
                "comparable properties from within the subject’s competitive market area."
            ),
        }

        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": str(user_payload)},
            ],
        )
        return chat.choices[0].message.content.strip()

    except Exception as e:
        return f"AI narrative unavailable: {e}"


# ===================== FILE LOADER =====================
@st.cache_data(show_spinner=False)
def load_df(uploaded):
    name = getattr(uploaded, "name", "")
    if name.lower().endswith((".csv", ".txt")):
        return pd.read_csv(uploaded, sep=None, engine="python")
    return pd.read_excel(uploaded)

# ===================== UI =====================
st.title("Comparable Adjustment Explorer")
st.write("Upload a CSV or Excel. Y-axis is always **Sold Price**. Choose one comparison feature.")

uploaded = st.file_uploader("Upload data file", type=["csv","xlsx","xls"])
if uploaded is None:
    st.info("Waiting for a file…"); st.stop()

df = load_df(uploaded)
df.columns = [c.strip() for c in df.columns]

# Y locked
y_col = pick_column(df, Y_COL_CANDIDATES)
if not y_col:
    st.error("Could not find a Sold Price column. Try renaming to one of: " + ", ".join(Y_COL_CANDIDATES))
    st.stop()
st.success(f"Y-axis locked to: {y_col}")

# Feature choice
feature_label = st.selectbox("Compare Sold Price against:", list(FEATURE_SYNONYMS.keys()), index=0)
x_col = resolve_feature_column(df, feature_label)
if not x_col:
    st.error(f"Could not map “{feature_label}” to any column in your file.\nTry one similar to: {FEATURE_SYNONYMS[feature_label]}")
    st.stop()

# Clean + prep
work = df[[y_col, x_col]].copy()
work[y_col] = clean_numeric(work[y_col])
work[x_col] = clean_numeric(map_yes_no_to_binary(work[x_col]))
work = work.dropna(subset=[y_col, x_col]).reset_index(drop=False)
if work.empty:
    st.error("No usable data after cleaning."); st.stop()

# Binary detection
uniq = np.sort(work[x_col].dropna().unique())
is_binary = len(uniq) <= 2 and set(np.unique(uniq).astype(int)) <= {0,1}
intish = looks_discrete_integer(work[x_col]) if not is_binary else True
if is_binary:
    int_ticks = [0,1]; xtick_labels = ["No","Yes"]
else:
    int_ticks = np.sort(work[x_col].round().astype(int).unique()) if intish else None
    xtick_labels = None

# ===== Inline filter panel =====
date_col = find_first_date_col(df)
work_for_filters = work.copy()
global_date_min = global_date_max = None
if date_col:
    # Map dates to working rows
    all_dates = pd.to_datetime(df[date_col], errors="coerce")
    work_for_filters[date_col] = all_dates.iloc[work_for_filters['index']].values
    # IMPORTANT: slider bounds use the FULL file's date range (not just working set)
    if all_dates.notna().any():
        global_date_min = all_dates.min()
        global_date_max = all_dates.max()

price_min, price_max = float(work_for_filters[y_col].min()), float(work_for_filters[y_col].max())
x_min, x_max = float(work_for_filters[x_col].min()), float(work_for_filters[x_col].max())

with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        price_rng = st.slider(
            "Sold Price range", min_value=price_min, max_value=price_max,
            value=(price_min, price_max), step=max(1.0, (price_max-price_min)/200.0)
        )
    with c2:
        x_rng = st.slider(
            f"{x_col} range", min_value=x_min, max_value=x_max,
            value=(x_min, x_max), step=(1.0 if is_binary else max(1.0, (x_max-x_min)/200.0)),
            disabled=is_binary
        )
    with c3:
        if date_col and global_date_min is not None and global_date_max is not None:
            date_rng = st.date_input(
                f"{date_col} window",
                value=(global_date_min.date(), global_date_max.date()),
                min_value=global_date_min.date(), max_value=global_date_max.date()
            )
        else:
            date_rng = None

# Apply filters (date filter applies to the mapped working rows)
work_filt = filter_data(
    df_in=work_for_filters, y_col=y_col, x_col=x_col, date_col=date_col,
    price_rng=price_rng, x_rng=(x_min, x_max) if is_binary else x_rng,
    date_rng=date_rng, is_binary_x=is_binary
)
if work_filt.empty:
    st.error("No rows match the current filters."); st.stop()

# -------------------- Map slide (filtered comps) --------------------
with st.expander("Map of filtered comps", expanded=False):
    lat_col, lon_col = find_lat_lon_cols(df)
    if not lat_col or not lon_col:
        st.info("No latitude/longitude columns were detected. Add coordinates to your file (e.g., 'Latitude' and 'Longitude') to enable the map.")
    else:
        try:
            # Build a map dataframe aligned to the current filtered set
            map_df = build_map_dataframe(
                original_df=df,
                work_filtered_with_idx=work_filt[["index", x_col, y_col]].copy(),
                lat_col=lat_col, lon_col=lon_col,
                y_col=y_col, x_col=x_col
            )
            if map_df.empty:
                st.info("No mappable rows in the current filters.")
            else:
                # Deck.gl scatter plot
                # Color-blind friendly blue; radius scales with price lightly
                price_min = float(map_df["price"].min())
                price_max = float(map_df["price"].max())
                radius = np.interp(map_df["price"], [price_min, price_max], [40, 140]).astype(int)

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df.assign(radius=radius),
                    get_position='[lon, lat]',
                    get_radius="radius",
                    get_fill_color=[52, 136, 189, 160],  # teal-blue with alpha
                    pickable=True,
                )

                view_state = pdk.ViewState(
                    latitude=float(map_df["lat"].mean()),
                    longitude=float(map_df["lon"].mean()),
                    zoom=10, pitch=0
                )

                tooltip = {
                    "html": "<b>Price:</b> ${price}<br/><b>{x_name}:</b> {feature}",
                    "style": {"backgroundColor": "white", "color": "black"}
                }
                # Dynamic label for the feature
                tooltip["html"] = tooltip["html"].replace("{x_name}", x_col)

                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

                # Small table preview (optional)
                st.caption("Sample of mapped comps")
                st.dataframe(
                    map_df.rename(columns={"price":"Sold Price", "feature": x_col})
                          .drop(columns=["index"])
                          .head(25)
                )
        except Exception as e:
            st.warning(f"Map could not be rendered: {e}")


# Outlier hints
is_sqft_like = _is_gla_like(x_col) or ("sqft" in x_col.lower() and "basement" not in x_col.lower() and "below" not in x_col.lower())
flagged_mask = flag_outliers(work_filt, y_col, x_col, is_binary_x=is_binary, is_sqft_like=is_sqft_like)
hide_flagged = st.checkbox("Hide flagged outliers (visual only)", value=False)
exclude_flagged = st.checkbox("Exclude flagged outliers from analysis & exports", value=False)

# Which data to use for stats/adjustments
work_used = work_filt.copy()
if exclude_flagged:
    work_used = work_used[~flagged_mask].copy()

# Baseline stats
if is_binary:
    bin_stats = compute_binary_stats(work_used, y_col, x_col)
else:
    stats0 = compute_stats(work_used, y_col, x_col)

left, right = st.columns([2, 1], gap="large")
with left:
    if is_binary:
        default_target = 0.0 if not bin_stats["has_both"] or np.isnan(bin_stats["slope"]) else float(round(bin_stats["slope"]))
    else:
        m0, _, _ = regression_slope(work_used[x_col].values, work_used[y_col].values)
        default_target = 0.0 if np.isnan(m0) else float(round(m0))

    target = st.number_input(f"Target price per +1 {feature_label}", value=default_target, step=1.0)
    n_total = len(work_used)
    default_max = max(1, min(math.floor(n_total * 0.25), 200))
    max_removals = st.slider("Max removals allowed", min_value=0, max_value=max(0, min(200, n_total - 2)), value=default_max)

with right:
    st.subheader("Current stats (filtered)")
    if is_binary:
        if bin_stats["has_both"]:
            st.metric("Avg difference (Yes − No)", f"${bin_stats['slope']:,.0f}")
            st.metric("R²", f"{bin_stats['r2']:.3f}" if not np.isnan(bin_stats['r2']) else "—")
        else:
            st.metric("Avg difference (Yes − No)", "—")
            st.caption("Need both Yes and No to compute a difference.")
        st.caption(f"Means — No: {('$'+format(bin_stats['mean_no'],',.0f')) if not np.isnan(bin_stats['mean_no']) else '—'}   |   Yes: {('$'+format(bin_stats['mean_yes'],',.0f')) if not np.isnan(bin_stats['mean_yes']) else '—'}")
        st.caption(f"Comps: {bin_stats['n']} • Flagged: {int(flagged_mask.sum())}")
    else:
        st.metric("Price per +1", f"${stats0['slope']:,.0f}" if not np.isnan(stats0['slope']) else "$nan")
        st.metric("R²", f"{stats0['r2']:.3f}" if not np.isnan(stats0['r2']) else "nan")
        if not np.isnan(stats0["median_ppsf"]):
            st.metric("Median $/sq ft", f"${stats0['median_ppsf']:,.0f}")
        st.caption(f"Comps: {stats0['n']} • Flagged: {int(flagged_mask.sum())}")

st.markdown("---")
use_bar_for_binary = st.checkbox("Show binary as mean bar chart (+ error bars)", value=is_binary)
use_ai = st.checkbox("Generate market narrative (AI)")

# ===================== MAIN ACTION =====================
if st.button("Adjust to Target"):
    kept_df, removed_info, _ = greedy_remove_toward_target(
        work_used[['index', y_col, x_col]].copy(), y_col, x_col, target, max_removals
    )

    # Removed overlay points come from filtered set (pre-exclude) for context
    orig_indexed = work_filt.set_index('index')
    removed_pts = []
    for r in removed_info:
        idx = r['orig_index']
        if idx in orig_indexed.index:
            removed_pts.append({x_col: float(orig_indexed.loc[idx][x_col]),
                                y_col: float(orig_indexed.loc[idx][y_col])})

    kept_plot = kept_df[[x_col, y_col]].copy()

    c1, c2 = st.columns([2, 1], gap="large")

    # --- Original (FILTERED) ---
    with c1:
        if is_binary and use_bar_for_binary:
            orig_fig = make_bar_figure_binary(work_filt, y_col, x_col, f"{feature_label}")
        else:
            orig_fig = make_scatter_figure(
                work_filt[[x_col, y_col]], y_col, x_col,
                f"Original comps — {feature_label} (filtered)",
                int_ticks=int_ticks, xtick_labels=xtick_labels,
                flagged_mask=flagged_mask.values, hide_flagged=hide_flagged
            )
        st.pyplot(orig_fig)
    with c2:
        if is_binary:
            bs_all = compute_binary_stats(work_filt, y_col, x_col)
            st.subheader("Original stats (filtered)")
            if bs_all["has_both"]:
                st.metric("Avg difference (Yes − No)", f"${bs_all['slope']:,.0f}")
                st.metric("R²", f"{bs_all['r2']:.3f}" if not np.isnan(bs_all['r2']) else "—")
            else:
                st.metric("Avg difference (Yes − No)", "—")
                st.caption("Need both Yes and No to compute a difference.")
            st.caption(f"Means — No: {('$'+format(bs_all['mean_no'],',.0f')) if not np.isnan(bs_all['mean_no']) else '—'}   |   Yes: {('$'+format(bs_all['mean_yes'],',.0f')) if not np.isnan(bs_all['mean_yes']) else '—'}")
            st.caption(f"Comps: {bs_all['n']}  •  Flagged: {int(flagged_mask.sum())}")
        else:
            s0 = compute_stats(work_filt, y_col, x_col)
            st.subheader("Original stats (filtered)")
            st.metric("Price per +1", f"${s0['slope']:,.0f}" if not np.isnan(s0['slope']) else "$nan")
            st.metric("R²", f"{s0['r2']:.3f}" if not np.isnan(s0['r2']) else "nan")
            if not np.isnan(s0["median_ppsf"]):
                st.metric("Median $/sq ft", f"${s0['median_ppsf']:,.0f}")
            st.caption(f"Comps: {s0['n']}  •  Flagged: {int(flagged_mask.sum())}")

    st.divider()

    # --- Adjusted (after removals) ---
    with c1:
        if is_binary and use_bar_for_binary:
            adj_fig = make_bar_figure_binary(kept_plot, y_col, x_col, f"{feature_label} (after removals)")
        else:
            adj_fig = make_scatter_figure(
                kept_plot, y_col, x_col,
                f"Adjusted comps — {feature_label} (after removals)",
                int_ticks=int_ticks, xtick_labels=xtick_labels
            )
        st.pyplot(adj_fig)
    with c2:
        if is_binary:
            bs_kept = compute_binary_stats(kept_plot, y_col, x_col)
            st.subheader("Adjusted stats")
            if bs_kept["has_both"]:
                st.metric("Avg difference (Yes − No)", f"${bs_kept['slope']:,.0f}")
                st.metric("R²", f"{bs_kept['r2']:.3f}" if not np.isnan(bs_kept['r2']) else "—")
            else:
                st.metric("Avg difference (Yes − No)", "—")
            st.caption(f"Removed comps: {len(removed_info)}")
        else:
            sA = compute_stats(kept_plot, y_col, x_col)
            st.subheader("Adjusted stats")
            st.metric("Price per +1", f"${sA['slope']:,.0f}" if not np.isnan(sA['slope']) else "$nan")
            st.metric("R²", f"{sA['r2']:.3f}" if not np.isnan(sA['r2']) else "nan")
            if not np.isnan(sA["median_ppsf"]):
                st.metric("Median $/sq ft", f"${sA['median_ppsf']:,.0f}")
            st.caption(f"Removed comps: {len(removed_info)}")

    st.divider()

    # --- Adjusted (clean preview) ---
    with c1:
        adj_clean_fig = make_scatter_figure(
            kept_plot, y_col, x_col, f"Adjusted comps — {feature_label}",
            int_ticks=int_ticks, xtick_labels=xtick_labels
        )
        st.pyplot(adj_clean_fig)
    with c2:
        st.subheader("Final table")
        if is_binary:
            bs_final = compute_binary_stats(kept_plot, y_col, x_col)
            table_rows = {
                "Comps kept": [len(kept_plot)],
                "Removed": [len(removed_info)],
                "Avg difference (Yes−No)": [f"${bs_final['slope']:,.0f}" if bs_final["has_both"] and not np.isnan(bs_final["slope"]) else "—"],
                "R²": [f"{bs_final['r2']:.3f}" if bs_final["has_both"] and not np.isnan(bs_final["r2"]) else "—"],
            }
        else:
            sA = compute_stats(kept_plot, y_col, x_col)
            table_rows = {
                "Comps kept": [len(kept_plot)],
                "Removed": [len(removed_info)],
                "Price per +1": [f"${sA['slope']:,.0f}" if not np.isnan(sA['slope']) else "$nan"],
                "R²": [f"{sA['r2']:.3f}" if not np.isnan(sA['r2']) else "nan"],
            }
            if not np.isnan(sA["median_ppsf"]):
                table_rows["Median $/sq ft"] = [f"${sA['median_ppsf']:,.0f}"]
        st.table(pd.DataFrame(table_rows))

        # ------- Downloads -------
        st.subheader("Downloads")
        removed_df = pd.DataFrame(removed_info)
        kept_csv = kept_df[[x_col, y_col]].to_csv(index=False).encode()
        removed_csv = removed_df.to_csv(index=False).encode()

        if is_binary:
            bs_all = compute_binary_stats(work_filt, y_col, x_col)
            bs_final = compute_binary_stats(kept_plot, y_col, x_col)
            summary_df = pd.DataFrame([{
                "feature_label": feature_label,
                "mapped_feature_column": x_col,
                "original_diff_yes_minus_no": None if not bs_all["has_both"] or np.isnan(bs_all["slope"]) else float(bs_all["slope"]),
                "original_r2": None if not bs_all["has_both"] or np.isnan(bs_all["r2"]) else float(bs_all["r2"]),
                "target_slope": float(target),
                "final_diff_yes_minus_no": None if not bs_final["has_both"] or np.isnan(bs_final["slope"]) else float(bs_final["slope"]),
                "final_r2": None if not bs_final["has_both"] or np.isnan(bs_final["r2"]) else float(bs_final["r2"]),
                "removed_count": len(removed_info),
                "flagged_excluded": bool(exclude_flagged)
            }])
        else:
            s0 = compute_stats(work_filt, y_col, x_col)
            sA = compute_stats(kept_plot, y_col, x_col)
            summary_df = pd.DataFrame([{
                "feature_label": feature_label,
                "mapped_feature_column": x_col,
                "original_slope": None if np.isnan(s0["slope"]) else float(s0["slope"]),
                "original_r2": None if np.isnan(s0["r2"]) else float(s0["r2"]),
                "target_slope": float(target),
                "final_slope": None if np.isnan(sA["slope"]) else float(sA["slope"]),
                "final_r2": None if np.isnan(sA["r2"]) else float(sA["r2"]),
                "original_median_ppsf": None if np.isnan(s0["median_ppsf"]) else float(s0["median_ppsf"]),
                "final_median_ppsf": None if np.isnan(sA["median_ppsf"]) else float(sA["median_ppsf"]),
                "removed_count": len(removed_info),
                "flagged_excluded": bool(exclude_flagged)
            }])

        st.download_button("Kept comps CSV", kept_csv, file_name="kept_comps.csv")
        st.download_button("Removed comps CSV", removed_csv, file_name="removed_rows.csv")
        st.download_button("Summary CSV", summary_df.to_csv(index=False).encode(), file_name="adjustment_summary.csv")

        st.download_button("Original chart PNG", fig_bytes(orig_fig), file_name="original.png")
        st.download_button("Adjusted chart PNG", fig_bytes(adj_fig), file_name="adjusted_with_removed.png")
        st.download_button("Adjusted clean PNG", fig_bytes(adj_clean_fig), file_name="adjusted_clean.png")

        excel_bytes, engine_used = build_excel_report(
            orig_fig, adj_fig, adj_clean_fig,
            kept_df[[x_col, y_col]], pd.DataFrame(removed_info), summary_df
        )
        st.download_button("Download Excel report (.xlsx)", data=excel_bytes,
                           file_name="adjustment_report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if engine_used == "openpyxl":
            st.info("Charts weren’t embedded because `xlsxwriter` isn’t installed. Install it: `pip install xlsxwriter`")

        # ---- AI Narrative (optional) ----
        if use_ai:
            st.subheader("Market narrative (AI)")
            context = infer_market_context(df)
            preview_cols = [c for c in [x_col, y_col] if c in df.columns]
            sample_rows = df[preview_cols].dropna().head(100).copy()
            if is_binary:
                bs_all = compute_binary_stats(work_filt, y_col, x_col)
                bs_final = compute_binary_stats(kept_plot, y_col, x_col)
                s_before = {"slope": bs_all.get("slope", np.nan), "r2": bs_all.get("r2", np.nan), "median_ppsf": np.nan}
                s_after  = {"slope": bs_final.get("slope", np.nan), "r2": bs_final.get("r2", np.nan), "median_ppsf": np.nan}
            else:
                s_before = compute_stats(work_filt, y_col, x_col)
                s_after  = compute_stats(kept_plot, y_col, x_col)
            narrative = ai_summary(
                feature_label=feature_label,
                y_col=y_col,
                stats_before=s_before,
                stats_after=s_after,
                context=context,
                sample_rows=sample_rows
            )
            st.write(narrative)
