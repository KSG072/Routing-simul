# -*- coding: utf-8 -*-
# Matplotlib only (no seaborn), one chart per figure, no explicit colors.
# Keep all existing plots; add Queueing/E2E versions. No saving, only plt.show().

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# --- color mapping for same k (jump) ---
DEFAULT_COLORS = plt.rcParams.get("axes.prop_cycle", None)
DEFAULT_COLORS = (DEFAULT_COLORS.by_key().get("color", None) if DEFAULT_COLORS else None) \
                 or [f"C{i}" for i in range(10)]
_k2color = {}
def color_for_k(k: int) -> str:
    if k not in _k2color:
        _k2color[k] = DEFAULT_COLORS[len(_k2color) % len(DEFAULT_COLORS)]
    return _k2color[k]

# ---- Load data ----
rate = 160
path = Path(f"comparison_results_with_jump_{rate}.csv")
df = pd.read_csv(path, low_memory=False)

# Ensure numeric types
num_cols = [
    "isl path length",
    "gsl path length",
    "path length diff",
    "isl prop delay",
    "gsl prop delay",
    "isl queue delay",
    "gsl queue delay",
    "jump count",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Derive E2E (if queue columns exist)
if "isl queue delay" in df.columns and "gsl queue delay" in df.columns:
    df["isl e2e delay"] = df["isl prop delay"] + df["isl queue delay"]
    df["gsl e2e delay"] = df["gsl prop delay"] + df["gsl queue delay"]

# Helper: ECDF
def ecdf(series):
    x = np.asarray(series.dropna(), dtype=float)
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

# ---------- filter view for all plots except the 3 exempted ones ----------
if "jump count" in df.columns:
    df_ge1 = df[pd.to_numeric(df["jump count"], errors="coerce") >= 1].copy()
else:
    df_ge1 = df.copy()  # fallback

# =========================
# [Original] 1) CDF of path length (ISL vs GSL)  ← EXEMPT (no filter)
# =========================
x1, y1 = ecdf(df["isl path length"])
x2, y2 = ecdf(df["gsl path length"])
plt.figure(figsize=(8, 5))
if x1.size: plt.plot(x1, y1, label="ISL Path Length")
if x2.size: plt.plot(x2, y2, label="GSL Path Length")
plt.xlabel("Path Length")
plt.ylabel("CDF")
plt.title("CDF of Path Lengths (ISL vs GSL)")
plt.legend(); plt.show()

# =========================
# [Original] 2) Bar chart: distribution of path length (ISL vs GSL)  ← EXEMPT (no filter)
# =========================
isl_counts = df["isl path length"].dropna().astype(int).value_counts().sort_index()
gsl_counts = df["gsl path length"].dropna().astype(int).value_counts().sort_index()
bins = sorted(set(isl_counts.index).union(set(gsl_counts.index)))
indices = np.arange(len(bins))
isl_vals = isl_counts.reindex(bins, fill_value=0).values
gsl_vals = gsl_counts.reindex(bins, fill_value=0).values
width = 0.4
plt.figure(figsize=(10, 5))
plt.bar(indices - width/2, isl_vals, width=width, label="ISL Path Length")
plt.bar(indices + width/2, gsl_vals, width=width, label="GSL Path Length")
plt.xticks(indices, bins, rotation=0)
plt.xlabel("Path Length (integer)")
plt.ylabel("Count")
plt.title("Distribution of Path Lengths (Bar Chart)")
plt.legend(); plt.show()

# =========================
# [NEW] Bar chart: distribution of gain of hop count (ISL−GSL)  ← FILTERED (jump≥1)
# =========================
if "path length diff" in df_ge1.columns:
    gdiff = pd.to_numeric(df_ge1["path length diff"], errors="coerce").dropna().astype(int)
    counts = gdiff.value_counts().sort_index()
    bins = counts.index.to_list()
    vals = counts.values
    idx = np.arange(len(bins))
    plt.figure(figsize=(10, 5))
    plt.bar(idx, vals)
    plt.xticks(idx, bins, rotation=0)
    plt.xlabel("Gain of hop count (ISL − GSL)")
    plt.ylabel("Count")
    plt.title("Distribution of Gain of Hop Count (jump ≥ 1)")
    plt.show()
else:
    print("[skip] 'path length diff' column not found — cannot plot gain-of-hop-count distribution.")

# =========================
# [Original] 3) Bar chart: distribution of jump count  ← EXEMPT (no filter)
# =========================
if "jump count" in df.columns:
    jc = df["jump count"].dropna().astype(int).value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(jc.index.astype(int), jc.values)
    plt.xlabel("Number of relay nodes in path")
    plt.ylabel("Number of paths")
    plt.title("Distribution of relay count per path")
    plt.show()

# =========================
# [Original] 4) CDF of propagation delay (ISL vs GSL)  ← FILTERED (jump≥1)
# =========================
x3, y3 = ecdf(df_ge1["isl prop delay"])
x4, y4 = ecdf(df_ge1["gsl prop delay"])
plt.figure(figsize=(8, 5))
if x3.size: plt.plot(x3, y3, label="ISL Propagation Delay")
if x4.size: plt.plot(x4, y4, label="GSL Propagation Delay")
plt.xlabel("Propagation Delay (ms)")
plt.ylabel("CDF")
plt.title("CDF of Propagation Delay (ISL vs GSL) — jump ≥ 1")
plt.legend(); plt.show()

# =========================
# [Original] 5) Ratio vs Path Length Diff for Prop: (ISL-GSL)/jump  ← FILTERED (jump≥1)
# =========================
ratio_prop = (df_ge1["isl prop delay"] - df_ge1["gsl prop delay"]) / df_ge1["jump count"].replace({0: np.nan})
tmp_prop = pd.DataFrame({"pl_diff": df_ge1["path length diff"], "ratio": ratio_prop}).dropna()
try:
    tmp_prop["pl_diff"] = tmp_prop["pl_diff"].astype(int)
except Exception:
    pass
stats_prop = tmp_prop.groupby("pl_diff")["ratio"].agg(["min", "max", "mean", "median"]).reset_index().sort_values("pl_diff")
plt.figure(figsize=(10, 6))
if not stats_prop.empty:
    plt.plot(stats_prop["pl_diff"], stats_prop["min"], label="min")
    plt.plot(stats_prop["pl_diff"], stats_prop["max"], label="max")
    plt.plot(stats_prop["pl_diff"], stats_prop["mean"], label="mean")
    plt.plot(stats_prop["pl_diff"], stats_prop["median"], label="median")
plt.xlabel("Gain of hop count")
plt.ylabel("Prop. Delay gain per ground node")
plt.title("Aggregate of Ratio vs Path Length Diff (Propagation) — jump ≥ 1")
plt.legend(); plt.show()

# =========================
# [NEW] Subplots: distribution of gain of hop count grouped by jump count  ← FILTERED (jump≥1)
# =========================
sub = df_ge1[["jump count", "path length diff"]].copy()
sub["jump count"] = pd.to_numeric(sub["jump count"], errors="coerce")
sub["path length diff"] = pd.to_numeric(sub["path length diff"], errors="coerce")
sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
# 이미 df_ge1가 jump≥1이므로 아래 라인은 의미상 중복이지만 안전하게 유지
sub = sub[sub["jump count"] > 0]
sub["jump"] = sub["jump count"].round().astype(int)
sub["pldiff"] = sub["path length diff"].round().astype(int)
if sub.empty:
    print("[skip] No data for per-jump histogram of gain-of-hop-count (jump ≥ 1)")
else:
    y_min, y_max = int(sub["pldiff"].min()), int(sub["pldiff"].max())
    bin_edges = np.arange(y_min - 0.5, y_max + 1.5, 1)
    jumps = sorted(sub["jump"].unique().tolist())
    n = len(jumps)
    ncols = 4 if n >= 4 else n
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0*ncols, 3.0*nrows), sharex=True, sharey=True)
    if isinstance(axes, np.ndarray): axes = axes.ravel().tolist()
    else: axes = [axes]
    for idx, k in enumerate(jumps):
        ax = axes[idx]
        grp = sub.loc[sub["jump"] == k, "pldiff"]
        ax.hist(grp, bins=bin_edges)
        ax.set_title(f"jump={k} (N={grp.shape[0]})", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        row = idx // ncols; col = idx % ncols
        if col == 0: ax.set_ylabel("Count")
        if row == nrows - 1: ax.set_xlabel("Gain of hop count (ISL − GSL)")
    for j in range(len(jumps), len(axes)): axes[j].set_visible(False)
    fig.suptitle("Distribution of Gain of Hop Count by Jump Count (jump ≥ 1)", y=0.98)
    plt.tight_layout(); plt.show()

# =========================
# [Original] 7) One figure: stats vs pldiff grouped by jump (Propagation)  ← FILTERED (jump≥1)
# =========================
def compute_stats_for_jump(df_in, k):
    sub = df_in.copy()
    for col in ["isl prop delay", "gsl prop delay", "jump count", "path length diff"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub[sub["jump count"] == k]
    if sub.empty: return None
    sub["ratio"] = (sub["isl prop delay"] - sub["gsl prop delay"]) / sub["jump count"]
    sub["pl_diff"] = pd.to_numeric(sub["path length diff"], errors="coerce")
    sub = sub.dropna(subset=["pl_diff", "ratio"])
    if sub.empty: return None
    sub["pl_diff"] = sub["pl_diff"].round().astype(int)
    stats = sub.groupby("pl_diff")["ratio"].agg(["min","max","mean","median"]).reset_index().sort_values("pl_diff")
    return stats

fig, ax = plt.subplots(figsize=(11, 7))
has_any = False
for k in [1, 2, 3, 4]:
    stats_k = compute_stats_for_jump(df_ge1, k)  # ← filtered
    if stats_k is None or stats_k.empty: continue
    has_any = True
    col = color_for_k(k)
    ax.plot(stats_k["pl_diff"], stats_k["min"], linestyle=":",  marker="o", color=col, label=f"jump={k} • min")
    ax.plot(stats_k["pl_diff"], stats_k["max"], linestyle="-",  marker="s", color=col, label=f"jump={k} • max")
    ax.plot(stats_k["pl_diff"], stats_k["mean"], linestyle="--", marker="^", color=col, label=f"jump={k} • mean")
    ax.plot(stats_k["pl_diff"], stats_k["median"], linestyle="-.", marker="D", color=col, label=f"jump={k} • median")
if has_any:
    ax.set_xlabel("Gain of hop count")
    ax.set_ylabel("Prop. delay gain per ground node")
    ax.set_title("Stats vs Path Length Diff by Jump Count (Propagation) — jump ≥ 1")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout(); plt.show()
else:
    plt.close(fig); print("[skip] No data for propagation jump-wise stats (jump ≥ 1)")

# =========================
# [Original] 8) Stats of (pldiff / jump) by jump  ← already requires jump>0 (kept)
# =========================
sub = df_ge1[["jump count", "path length diff"]].copy()
sub["jump count"] = pd.to_numeric(sub["jump count"], errors="coerce")
sub["path length diff"] = pd.to_numeric(sub["path length diff"], errors="coerce")
sub = sub.dropna(subset=["jump count", "path length diff"])
sub = sub[sub["jump count"] > 0]
sub["jump"] = sub["jump count"].round().astype(int)
sub["pldiff_per_jump"] = sub["path length diff"] / sub["jump count"]
stats8 = sub.groupby("jump")["pldiff_per_jump"].agg(["min","max","mean","median","count"]).reset_index().sort_values("jump")
plt.figure(figsize=(10, 6))
if not stats8.empty:
    plt.plot(stats8["jump"], stats8["min"],    linestyle=":",  marker="o", label="min")
    plt.plot(stats8["jump"], stats8["max"],    linestyle="-",  marker="s", label="max")
    plt.plot(stats8["jump"], stats8["mean"],   linestyle="--", marker="^", label="mean")
    plt.plot(stats8["jump"], stats8["median"], linestyle="-.", marker="D", label="median")
plt.xlabel("Num of relay node per path")
plt.ylabel("Gain of hop count per relay")
plt.title("Stats of (Path Length Diff) / Relay Count (jump ≥ 1)")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# =========================
# [NEW] Heatmaps (filtered): x = gain of hop count, y = E2E delay  ← FILTERED (jump≥1)
# =========================
# e2e columns already prepared above if available
pldiff = pd.to_numeric(df_ge1.get("path length diff"), errors="coerce")
e2e_isl = pd.to_numeric(df_ge1.get("isl e2e delay"), errors="coerce")
e2e_gsl = pd.to_numeric(df_ge1.get("gsl e2e delay"), errors="coerce")

def _heatmap_pldiff_vs_y(x_series, y_series, title_y: str):
    sub = pd.DataFrame({"x": x_series, "y": y_series}).replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        print(f"[skip] No data for heatmap: {title_y} (jump ≥ 1)"); return
    x_int = sub["x"].round().astype(int)
    x_min, x_max = int(x_int.min()), int(x_int.max())
    x_bins = np.arange(x_min - 0.5, x_max + 1.5, 1)
    y_clip = sub["y"].to_numpy()
    q1, q99 = np.nanpercentile(y_clip, [1, 99])
    y_clip = np.clip(y_clip, q1, q99)
    n_ybins = 80
    y_bins = np.linspace(y_clip.min(), y_clip.max(), n_ybins + 1)
    H, xedges, yedges = np.histogram2d(x_int, y_clip, bins=[x_bins, y_bins])
    plt.figure(figsize=(11, 6))
    plt.imshow(H.T, origin="lower", aspect="auto", interpolation="nearest",
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = plt.colorbar(); cbar.set_label("Count")
    plt.xticks(np.arange(x_min, x_max + 1, 1))
    plt.xlabel("Gain of hop count (ISL − GSL)")
    plt.ylabel(title_y)
    plt.title(f"Distribution: E2E vs Gain of Hop Count • {title_y} (jump ≥ 1)")
    plt.tight_layout(); plt.show()

# (1) ISL E2E vs gain
if "isl e2e delay" in df_ge1.columns:
    _heatmap_pldiff_vs_y(pldiff, e2e_isl, "ISL E2E Delay (ms)")
else:
    print("[skip] isl e2e delay not found")
# (2) GSL E2E vs gain
if "gsl e2e delay" in df_ge1.columns:
    _heatmap_pldiff_vs_y(pldiff, e2e_gsl, "GSL E2E Delay (ms)")
else:
    print("[skip] gsl e2e delay not found")
# (3) (ISL − GSL) E2E diff vs gain
if ("isl e2e delay" in df_ge1.columns) and ("gsl e2e delay" in df_ge1.columns):
    _heatmap_pldiff_vs_y(pldiff, (e2e_isl - e2e_gsl), "E2E Delay Difference (ISL − GSL, ms)")
else:
    print("[skip] e2e diff: need both isl/gsl e2e")

# =========================
# [NEW] Heatmaps (filtered): x = gain of hop count, y = Queueing delay  ← FILTERED (jump≥1)
# =========================
if ("isl queue delay" in df_ge1.columns) and ("gsl queue delay" in df_ge1.columns):
    pldiff_q = pd.to_numeric(df_ge1.get("path length diff"), errors="coerce")
    q_isl    = pd.to_numeric(df_ge1.get("isl queue delay"), errors="coerce")
    q_gsl    = pd.to_numeric(df_ge1.get("gsl queue delay"), errors="coerce")

    # (1) ISL Queueing vs gain
    _heatmap_pldiff_vs_y(pldiff_q, q_isl, "ISL Queueing Delay (ms)")

    # (2) GSL Queueing vs gain
    _heatmap_pldiff_vs_y(pldiff_q, q_gsl, "GSL Queueing Delay (ms)")

    # (3) (ISL − GSL) Queueing 차이 vs gain
    _heatmap_pldiff_vs_y(pldiff_q, (q_isl - q_gsl), "Queueing Delay Difference (ISL − GSL, ms)")
else:
    print("[skip] queueing heatmaps: need both 'isl queue delay' and 'gsl queue delay'")


# ============================================================
# [ADDED] Queueing Delay & E2E suites — now accept df_in and use df_ge1  ← FILTERED (jump≥1)
# ============================================================
def plot_metric_suite(df_in: pd.DataFrame, metric_title, isl_col, gsl_col, y_unit_label):
    if isl_col not in df_in.columns or gsl_col not in df_in.columns:
        print(f"[skip] {metric_title}: missing {isl_col} or {gsl_col}"); return
    # A) ECDF (ISL vs GSL)
    x_a, y_a = ecdf(df_in[isl_col]); x_b, y_b = ecdf(df_in[gsl_col])
    plt.figure(figsize=(8, 5))
    if x_a.size: plt.plot(x_a, y_a, label="ISL")
    if x_b.size: plt.plot(x_b, y_b, label="GSL")
    plt.xlabel(f"{metric_title} ({y_unit_label})"); plt.ylabel("CDF")
    plt.title(f"CDF of {metric_title} (ISL vs GSL) — jump ≥ 1"); plt.legend(); plt.show()
    # B) Ratio vs Path Length Diff: (ISL-GSL)/jump
    jump = df_in["jump count"].replace({0: np.nan})
    ratio = (pd.to_numeric(df_in[isl_col], errors="coerce") -
             pd.to_numeric(df_in[gsl_col], errors="coerce")) / pd.to_numeric(jump, errors="coerce")
    tmp = pd.DataFrame({"pl_diff": pd.to_numeric(df_in["path length diff"], errors="coerce"),
                        "ratio": ratio}).dropna()
    try: tmp["pl_diff"] = tmp["pl_diff"].astype(int)
    except Exception: pass
    stats = tmp.groupby("pl_diff")["ratio"].agg(["min","max","mean","median"]).reset_index().sort_values("pl_diff")
    plt.figure(figsize=(10, 6))
    if not stats.empty:
        plt.plot(stats["pl_diff"], stats["min"],    label="min")
        plt.plot(stats["pl_diff"], stats["max"],    label="max")
        plt.plot(stats["pl_diff"], stats["mean"],   label="mean")
        plt.plot(stats["pl_diff"], stats["median"], label="median")
    plt.xlabel("Gain of hop count"); plt.ylabel(f"{metric_title} gain per ground node")
    plt.title(f"Aggregate of Ratio vs Path Length Diff • {metric_title} — jump ≥ 1")
    plt.legend(); plt.show()
    # C) Jump-wise stats (k=1..4)
    fig2, ax2 = plt.subplots(figsize=(11, 7)); have_any = False
    for k in [1,2,3,4]:
        sub = df_in.copy(); sub["jump count"] = pd.to_numeric(sub["jump count"], errors="coerce")
        sub = sub[sub["jump count"] == k]
        if sub.empty: continue
        have_any = True
        ratio_k = (pd.to_numeric(sub[isl_col], errors="coerce") -
                   pd.to_numeric(sub[gsl_col], errors="coerce")) / k
        pl = pd.to_numeric(sub["path length diff"], errors="coerce")
        tmpk = pd.DataFrame({"pl_diff": pl, "ratio": ratio_k}).dropna()
        if tmpk.empty: continue
        tmpk["pl_diff"] = tmpk["pl_diff"].round().astype(int)
        statk = tmpk.groupby("pl_diff")["ratio"].agg(["min","max","mean","median"]).reset_index().sort_values("pl_diff")
        col = color_for_k(k)
        ax2.plot(statk["pl_diff"], statk["min"],    linestyle=":",  marker="o", color=col, label=f"jump={k} • min")
        ax2.plot(statk["pl_diff"], statk["max"],    linestyle="-",  marker="s", color=col, label=f"jump={k} • max")
        ax2.plot(statk["pl_diff"], statk["mean"],   linestyle="--", marker="^", color=col, label=f"jump={k} • mean")
        ax2.plot(statk["pl_diff"], statk["median"], linestyle="-.", marker="D", color=col, label=f"jump={k} • median")
    if have_any:
        ax2.set_xlabel("Gain of hop count"); ax2.set_ylabel(f"{metric_title} gain per ground node")
        ax2.set_title(f"Stats vs Path Length Diff by Jump Count • {metric_title} — jump ≥ 1")
        ax2.grid(True, alpha=0.3); ax2.legend(ncol=2); plt.tight_layout(); plt.show()
    else:
        plt.close(fig2); print(f"[skip] No data for {metric_title} jump-wise stats (jump ≥ 1)")

# --- Queueing Delay (filtered) ---
if "isl queue delay" in df.columns and "gsl queue delay" in df.columns:
    plot_metric_suite(df_ge1, "Queueing Delay", "isl queue delay", "gsl queue delay", "ms")
else:
    print("[skip] Queueing Delay plots: queue delay columns not found")

# --- E2E Delay (prop + queue) (filtered) ---
if "isl e2e delay" in df.columns and "gsl e2e delay" in df.columns:
    plot_metric_suite(df_ge1, "E2E Delay", "isl e2e delay", "gsl e2e delay", "ms")
else:
    print("[skip] E2E plots: e2e columns not found (need both prop and queue delays)")

# ============================================================
# [ADDED] Suites with X-axis = ISL Path Length (jump ≥ 1)
#   - Ratio vs ISL hops: (ISL - GSL)/jump
#   - Jump-wise stats vs ISL hops
# ============================================================
def plot_metric_suite_by_isl_hops(df_in: pd.DataFrame, metric_title, isl_col, gsl_col, y_unit_label):
    # 컬럼 체크
    req = {"isl path length", isl_col, gsl_col}
    if not req.issubset(df_in.columns):
        missing = req - set(df_in.columns)
        print(f"[skip] {metric_title} by ISL hops: missing {missing}")
        return

    # A) Ratio vs ISL Path Length: (ISL-GSL)/jump, x=isl path length(정수)
    isl_hops = pd.to_numeric(df_in["isl path length"], errors="coerce")
    jump = pd.to_numeric(df_in["jump count"], errors="coerce").replace({0: np.nan})
    ratio = (pd.to_numeric(df_in[isl_col], errors="coerce") -
             pd.to_numeric(df_in[gsl_col], errors="coerce")) / jump

    tmp = pd.DataFrame({
        "isl_hops": isl_hops,
        "ratio": ratio
    }).replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(10, 6))
    if not tmp.empty:
        tmp["isl_hops"] = tmp["isl_hops"].round().astype(int)
        stats = (tmp.groupby("isl_hops")["ratio"]
                    .agg(["min","max","mean","median"])
                    .reset_index().sort_values("isl_hops"))
        plt.plot(stats["isl_hops"], stats["min"],    label="min")
        plt.plot(stats["isl_hops"], stats["max"],    label="max")
        plt.plot(stats["isl_hops"], stats["mean"],   label="mean")
        plt.plot(stats["isl_hops"], stats["median"], label="median")
    plt.xlabel("ISL Path Length (integer)")
    plt.ylabel(f"{metric_title} gain per ground node")
    plt.title(f"Aggregate of Ratio vs ISL Path Length • {metric_title} — jump ≥ 1")
    plt.legend()
    plt.show()

    # B) Jump-wise stats (k=1..4): (ISL-GSL)/k vs ISL Path Length
    fig2, ax2 = plt.subplots(figsize=(11, 7))
    have_any = False
    for k in [1, 2, 3, 4]:
        sub = df_in.copy()
        sub["jump count"] = pd.to_numeric(sub["jump count"], errors="coerce")
        sub = sub[sub["jump count"] == k]
        if sub.empty:
            continue
        have_any = True
        isl_h = pd.to_numeric(sub["isl path length"], errors="coerce")
        ratio_k = (pd.to_numeric(sub[isl_col], errors="coerce") -
                   pd.to_numeric(sub[gsl_col], errors="coerce")) / k
        tmpk = pd.DataFrame({"isl_hops": isl_h, "ratio": ratio_k}) \
                 .replace([np.inf, -np.inf], np.nan).dropna()
        if tmpk.empty:
            continue
        tmpk["isl_hops"] = tmpk["isl_hops"].round().astype(int)
        statk = (tmpk.groupby("isl_hops")["ratio"]
                     .agg(["min","max","mean","median"])
                     .reset_index().sort_values("isl_hops"))
        col = color_for_k(k)  # 같은 k는 같은 색
        ax2.plot(statk["isl_hops"], statk["min"],    linestyle=":",  marker="o", color=col, label=f"jump={k} • min")
        ax2.plot(statk["isl_hops"], statk["max"],    linestyle="-",  marker="s", color=col, label=f"jump={k} • max")
        ax2.plot(statk["isl_hops"], statk["mean"],   linestyle="--", marker="^", color=col, label=f"jump={k} • mean")
        ax2.plot(statk["isl_hops"], statk["median"], linestyle="-.", marker="D", color=col, label=f"jump={k} • median")

    if have_any:
        ax2.set_xlabel("ISL Path Length (integer)")
        ax2.set_ylabel(f"{metric_title} gain per ground node")
        ax2.set_title(f"Stats vs ISL Path Length by Jump Count • {metric_title} — jump ≥ 1")
        ax2.grid(True, alpha=0.3)
        ax2.legend(ncol=2)
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig2)
        print(f"[skip] No data for {metric_title} jump-wise stats by ISL hops (jump ≥ 1)")

# ---- CALLS (jump≥1 필터 df_ge1 사용) ----
# Queueing Delay: x = ISL Path Length
if {"isl queue delay","gsl queue delay","isl path length"}.issubset(df.columns):
    plot_metric_suite_by_isl_hops(df_ge1, "Queueing Delay", "isl queue delay", "gsl queue delay", "ms")
else:
    print("[skip] Queueing-by-ISL-hops: needed columns not found")

# E2E Delay: x = ISL Path Length
if {"isl e2e delay","gsl e2e delay","isl path length"}.issubset(df.columns):
    plot_metric_suite_by_isl_hops(df_ge1, "E2E Delay", "isl e2e delay", "gsl e2e delay", "ms")
else:
    print("[skip] E2E-by-ISL-hops: needed columns not found")

# (원하면) Propagation Delay도 동일 방식으로 볼 수 있어요:
# if {"isl prop delay","gsl prop delay","isl path length"}.issubset(df.columns):
#     plot_metric_suite_by_isl_hops(df_ge1, "Propagation Delay", "isl prop delay", "gsl prop delay", "ms")

