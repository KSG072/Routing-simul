#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새 CSV 형식용 멀티 그래프 생성 스크립트

입력 예)
  python new_multi_graphs.py infinite_Q_with_GSL_1.csv limited_Q_with_GSL_1.csv

그래프:
  1) Path Length 분포 (생성률별 subplot)
  2) Jump Count 분포 (생성률별 subplot)
  3) Jump Count 별 Cross Counts 분포  [jump>=1]
  4) Jump Count 별 Path Length Diff 분포 (ISL Path Length - Path Length)  [jump>=1]
  5) (jump별) x=path_length_diff 에 대해 {e2e, expected(result), expected(isl)} 의 min/max/mean 라인  [jump>=1]
  6) (cross별) x=path_length_diff 에 대해 {e2e, expected(result), expected(isl)} 의 min/max/mean 라인  [jump>=1, cross=0 포함]
  7) 3D 막대: (jump, cross) → 평균 {Propagation, Queueing, E2E}

조건:
  - 모든 데이터는 Status == 'success'
  - 3,4,5번 그래프는 jump_count >= 1만 포함
  - 5,6: 같은 jump는 같은 색 유지
"""

import sys, re, ast
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import time
from tqdm.auto import tqdm

# 진행바 켜고 끄는 스위치
USE_TQDM = True

def _rate_key(x):
    try:
        return float(x)
    except Exception:
        return x

def iter_progress(iterable, desc="", unit=""):
    try:
        return tqdm(iterable, desc=desc, unit=unit)
    except Exception:
        return iterable


def log(msg):
    print(msg, flush=True)



# -------------------------
# 설정값
# -------------------------
MAX_SUBPLOTS_PER_ROW = 4      # 생성률/점프별 서브플롯에서 최대 4열
MAX_JUMP_SUBPLOTS    = 10     # (3)(4)에서 jump별 subplot 최대 개수 (너무 많으면 상위 jump만 표시)
N_BINS_HIST          = None   # None이면 정수 bin으로 맞춤 (Path Length 등 정수형)
ALPHA_BARS           = 0.85   # 3D 막대 투명도
VIEW_ELEV            = 22     # 3D 뷰 elevation
VIEW_AZIM            = -60    # 3D 뷰 azimuth (시계방향으로 돌리고 싶으면 감소 방향으로 조정)
# ==== [글로벌 저장 옵션] ====
SAVE_DIR = Path(r"C:\Users\김태성\PycharmProjects\ground-satellite routing\results")  # ← 원하는 폴더
SAVE_FMT = "png"   # "png" / "pdf" 등
SAVE_DPI = 200
DEFAULT_SAVE_MODE = "show"   # "show" | "save" | "both"

def _safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-=+.,() ]+", "_", s.strip())
    s = re.sub(r"\s+", "_", s)
    return s

def save_or_show(fig, base_name: str, rate: str | None = None, save_mode: str = DEFAULT_SAVE_MODE):
    """
    save_mode: "show" (화면표시만), "save" (파일저장만), "both" (둘 다)
    파일명은 "<base_name>__rate=<r>.<fmt>" 형태로 저장
    """
    mode = (save_mode or "show").lower()
    if mode not in {"show", "save", "both"}:
        mode = "show"

    if "save" in mode:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        fname = base_name if rate is None else f"{base_name}__rate={rate}"
        fname = _safe_filename(fname) + f".{SAVE_FMT}"
        fig.savefig(SAVE_DIR / fname, dpi=SAVE_DPI, bbox_inches="tight")

    if mode in {"show", "both"}:
        plt.show()
    else:
        plt.close(fig)
# -------------------------
# 유틸
# -------------------------
def parse_gen_rate_from_name(p: Path) -> str:
    """파일명에서 맨 끝 숫자(예: ..._160.csv → '160')를 생성률로 추출; 없으면 basename 반환"""
    m = re.search(r'_(\d+)\.csv$', p.name)
    return m.group(1) if m else p.stem

def count_string_items(cell) -> int:
    """result 컬럼에서 문자열 토큰(지상노드)을 count = jump_count로 사용"""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return 0
    if isinstance(cell, list):
        seq = cell
    else:
        s = str(cell)
        try:
            seq = ast.literal_eval(s)
        except Exception:
            # 따옴표로 감싼 토큰 수를 세는 폴백
            return len(re.findall(r"(?:'[^']*'|\"[^\"]*\")", s))
    return sum(isinstance(x, str) for x in seq)

def int_bins_from_series(x: pd.Series):
    """정수값 시리즈를 정수 중심 bin으로 표현하기 위한 bin 엣지 생성"""
    x = pd.to_numeric(x, errors="coerce").dropna().astype(int)
    if x.empty:
        return None
    xmin, xmax = int(x.min()), int(x.max())
    # 정수 중심 bin: n-0.5 ~ n+0.5
    return np.arange(xmin - 0.5, xmax + 1.5, 1)

def ecdf(series: pd.Series):
    v = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if v.size == 0:
        return np.array([]), np.array([])
    v.sort()
    y = np.arange(1, v.size + 1) / v.size
    return v, y

def color_cycle():
    prop = plt.rcParams.get("axes.prop_cycle", None)
    return (prop.by_key().get("color", None) if prop else None) or [f"C{i}" for i in range(10)]

_COLORS = color_cycle()
_k2color: Dict[int, str] = {}
def color_for_k(k: int) -> str:
    if k not in _k2color:
        _k2color[k] = _COLORS[len(_k2color) % len(_COLORS)]
    return _k2color[k]

# 기존 color_cycle / _COLORS / color_for_k 아래에 추가
_CROSS_COLORS = _COLORS  # 같은 팔레트 공유
_c2color: Dict[int, str] = {}

def color_for_cross(c: int) -> str:
    if c not in _c2color:
        _c2color[c] = _CROSS_COLORS[len(_c2color) % len(_CROSS_COLORS)]
    return _c2color[c]

# -------------------------
# 로딩 & 전처리
# -------------------------
def load_all(files: List[str]) -> pd.DataFrame:
    start = time.perf_counter()
    frames = []
    total_rows = 0
    kept_rows = 0

    log(f"[Load] 시작: {len(files)}개 파일")
    for f in iter_progress(files, desc="[Load] files", unit="file"):
        p = Path(f)
        df = pd.read_csv(p, low_memory=False)
        orig = len(df)
        total_rows += orig

        # Status==success 필터
        if "Status" in df.columns:
            df = df[df["Status"].astype(str).str.strip().str.lower() == "success"]
        kept = len(df); kept_rows += kept
        log(f"  - {p.name}: {orig} → {kept} (success)")

        # (아래는 기존 전처리 그대로)
        rename_map = {
            "Path Length": "path_len",
            "Detour counts": "detour_counts",
            "cross counts": "cross_counts",
            "e2e delay": "e2e_delay",
            "expected delay(result)": "exp_delay_result",
            "expected delay(isl)": "exp_delay_isl",
            "Queuing Delay": "queue_delay",
            "Propagation Delay": "prop_delay",
            "Transmission Delay": "tx_delay",
            "ISL Path Length": "isl_path_len",
            "isl path length": "isl_path_len",
            "result": "result",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "result" in df.columns:
            df["jump_count"] = df["result"].map(count_string_items)
        else:
            df["jump_count"] = 0

        for c in ["path_len","detour_counts","cross_counts",
                  "e2e_delay","exp_delay_result","exp_delay_isl",
                  "queue_delay","prop_delay","tx_delay","isl_path_len","jump_count"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "isl_path_len" in df.columns and "path_len" in df.columns:
            df["path_length_diff"] = df["isl_path_len"] - df["path_len"]
        else:
            df["path_length_diff"] = np.nan

        df["gen_rate"] = parse_gen_rate_from_name(p)
        df["source"]   = p.name
        frames.append(df)

    if not frames:
        raise SystemExit("입력 CSV가 없습니다.")

    out = pd.concat(frames, ignore_index=True)
    elapsed = time.perf_counter() - start
    log(f"[Load] 완료: 총 {total_rows}행 → success {kept_rows}행, concat 후 {len(out)}행, {elapsed:.2f}s")
    return out

# -------------------------
# (1) Path Length 분포 — 생성률별 subplot
# -------------------------
def plot_path_length_by_rate(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    rates = sorted(df["gen_rate"].unique().tolist())
    n = len(rates)
    ncols = min(MAX_SUBPLOTS_PER_ROW, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0*ncols, 3.0*nrows), sharex=False, sharey=False)
    axes = np.array(axes).ravel().tolist() if isinstance(axes, np.ndarray) else [axes]

    for i, r in enumerate(rates):
        ax = axes[i]
        sub = df[df["gen_rate"] == r]
        if sub.empty:
            ax.set_visible(False); continue
        bins = int_bins_from_series(sub["path_len"]) if N_BINS_HIST is None else N_BINS_HIST
        ax.hist(sub["path_len"].dropna(), bins=bins)
        ax.set_title(f"Rate={r}", fontsize=10)
        ax.set_xlabel("Path Length"); ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)

    for j in range(len(rates), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Path Length Distribution by Generation Rate", y=0.98)
    plt.tight_layout()
    save_or_show(fig, base_name="path_length__by_rate", rate=None, save_mode=save_mode)

# -------------------------
# (2) Jump Count 분포 — 생성률별 subplot
# -------------------------
def plot_jump_count_by_rate(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    rates = sorted(df["gen_rate"].unique().tolist())
    n = len(rates)
    ncols = min(MAX_SUBPLOTS_PER_ROW, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0*ncols, 3.0*nrows), sharex=False, sharey=False)
    axes = np.array(axes).ravel().tolist() if isinstance(axes, np.ndarray) else [axes]

    for i, r in enumerate(rates):
        ax = axes[i]
        sub = df[df["gen_rate"] == r]
        if sub.empty:
            ax.set_visible(False); continue
        bins = int_bins_from_series(sub["jump_count"]) if N_BINS_HIST is None else N_BINS_HIST
        ax.hist(sub["jump_count"].dropna(), bins=bins)
        ax.set_title(f"Rate={r}", fontsize=10)
        ax.set_xlabel("Jump Count"); ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)

    for j in range(len(rates), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Jump Count Distribution by Generation Rate", y=0.98)
    plt.tight_layout()
    save_or_show(fig, base_name="jump_count__by_rate", rate=None, save_mode=save_mode)

# -------------------------
# (3) Jump Count 별 Cross Counts 분포 — jump>=1
# -------------------------
def plot_cross_by_jump(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    """
    생성률(gen_rate)별로 분리해서,
    각 생성률에 대해 jump_count >= 1인 행만 모아
    jump 별로 cross_counts 분포(히스토그램)를 subplot으로 그림.
    - cross=0 포함
    - 각 막대 상단에 count 주석 표시
    - 저장/표시는 save_mode 및 save_or_show 헬퍼 사용
    """
    # gen_rate 정렬(숫자 우선 정렬)
    rates = sorted(
        df["gen_rate"].dropna().unique().tolist(),
        key=lambda s: (int(s) if str(s).isdigit() else float("inf"), str(s))
    )
    if not rates:
        print("[skip] plot_cross_by_jump: no gen_rate found")
        return

    for r in rates:
        sub_rate = df[df["gen_rate"] == r].copy()
        if sub_rate.empty:
            continue

        # jump>=1만 사용
        sub = sub_rate[pd.to_numeric(sub_rate["jump_count"], errors="coerce") >= 0].copy()
        if sub.empty:
            print(f"[skip] rate={r}: no rows with jump_count >= 1")
            continue

        # 사용할 점프 리스트 (너무 많으면 상위 MAX_JUMP_SUBPLOTS개)
        jumps = sorted(sub["jump_count"].dropna().astype(int).unique().tolist())
        if len(jumps) > MAX_JUMP_SUBPLOTS:
            jumps = jumps[:MAX_JUMP_SUBPLOTS]

        n = len(jumps)
        ncols = min(MAX_SUBPLOTS_PER_ROW, n) if n > 0 else 1
        nrows = int(np.ceil(n / ncols)) if n > 0 else 1

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(4.0 * ncols, 3.0 * nrows),
            sharex=True, sharey=True
        )
        axes = np.array(axes).ravel().tolist() if isinstance(axes, np.ndarray) else [axes]

        for i, k in enumerate(jumps):
            ax = axes[i]
            grp = pd.to_numeric(sub.loc[sub["jump_count"] == k, "cross_counts"], errors="coerce").dropna()

            bins = int_bins_from_series(grp) if N_BINS_HIST is None else N_BINS_HIST
            if bins is None or grp.empty:
                ax.set_visible(False)
                continue

            # 히스토그램과 주석(막대 상단에 count)
            counts, edges, patches = ax.hist(grp, bins=bins)
            for rect, c in zip(patches, counts):
                if c > 0:
                    x = rect.get_x() + rect.get_width() / 2.0
                    y = rect.get_height()
                    ax.text(x, y, f"{int(c)}", ha="center", va="bottom", fontsize=8, rotation=0)

            ax.set_title(f"jump={k} (N={grp.shape[0]})", fontsize=10)
            if i % ncols == 0:
                ax.set_ylabel("Count")
            if i // ncols == nrows - 1:
                ax.set_xlabel("Cross Counts")
            ax.grid(True, axis="y", alpha=0.3)

        # 남는 축 숨김
        for j in range(len(jumps), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Cross Counts Distribution by Jump (jump ≥ 1) — generation rate={r}", y=0.98)
        plt.tight_layout()

        # 저장 또는 표시
        if "save_or_show" in globals():
            save_or_show(fig, base_name="cross_by_jump", rate=str(r), save_mode=save_mode)
        else:
            plt.show()

# -------------------------
# (4) Jump Count 별 Path Length Diff 분포 — jump>=1 & path_length_diff 존재
# -------------------------
def plot_pldiff_by_jump(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    rates = sorted(df["gen_rate"].dropna().unique().tolist(), key=_rate_key)
    for r in rates:
        sub_rate = df[df["gen_rate"] == r]
        sub = sub_rate[(pd.to_numeric(sub_rate["jump_count"], errors="coerce") >= 1) &
                       (pd.to_numeric(sub_rate["path_length_diff"], errors="coerce").notna())].copy()
        if sub.empty:
            print(f"[skip] pldiff-by-jump: no data for rate={r}")
            continue

        jumps = sorted(sub["jump_count"].dropna().astype(int).unique().tolist())
        if len(jumps) > MAX_JUMP_SUBPLOTS:
            jumps = jumps[:MAX_JUMP_SUBPLOTS]

        n = len(jumps)
        ncols = min(MAX_SUBPLOTS_PER_ROW, n) if n > 0 else 1
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0*ncols, 3.0*nrows), sharex=True, sharey=True)
        axes = np.array(axes).ravel().tolist() if isinstance(axes, np.ndarray) else [axes]

        for i, k in enumerate(jumps):
            ax = axes[i]
            grp = sub.loc[sub["jump_count"] == k, "path_length_diff"].round().astype(int)
            bins = int_bins_from_series(grp) if N_BINS_HIST is None else N_BINS_HIST
            ax.hist(grp.dropna(), bins=bins)
            ax.set_title(f"jump={k} (N={grp.shape[0]})", fontsize=10)
            if i % ncols == 0: ax.set_ylabel("Count")
            if i // ncols == nrows - 1: ax.set_xlabel("ISL PathLen − PathLen")
            ax.grid(True, axis="y", alpha=0.3)

        for j in range(len(jumps), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Path Length Diff Distribution by Jump (jump ≥ 1) — rate={r}", y=0.98)
        plt.tight_layout()
        save_or_show(fig, base_name="pldiff_by_jump", rate=r, save_mode=save_mode)

def plot_pldiff_by_one_jump(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    """
    (개정 버전)
    - 모집단: jump_count == 1 인 행만 사용
    - subplot: generation rate(gen_rate) 별 1개
    - 각 subplot 안에서 x축=path_length_diff(정수 bin), cross=0과 cross=1 분포를
      막대 2개로 나란히 표시 (같은 x-bin에서 좌/우로 살짝 오프셋)
    """
    rates = sorted(df["gen_rate"].dropna().unique().tolist(), key=_rate_key)
    if not rates:
        print("[skip] pldiff-by-jump: no gen_rate found")
        return

    # 공통 준비: 숫자형으로 캐스팅
    df = df.copy()
    df["jump_count"]       = pd.to_numeric(df["jump_count"], errors="coerce")
    df["path_length_diff"] = pd.to_numeric(df["path_length_diff"], errors="coerce")
    df["cross_counts"]     = pd.to_numeric(df["cross_counts"], errors="coerce")

    # jump==1만
    df = df[(df["jump_count"] == 1) & (df["path_length_diff"].notna())]

    if df.empty:
        print("[skip] pldiff-by-jump: no rows with jump_count==1 and valid pldiff")
        return

    # subplot 레이아웃
    n = len(rates)
    ncols = min(MAX_SUBPLOTS_PER_ROW, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.5*ncols, 3.4*nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).ravel().tolist() if isinstance(axes, np.ndarray) else [axes]

    for i, r in enumerate(rates):
        ax = axes[i]
        sub_r = df[df["gen_rate"] == r]
        if sub_r.empty:
            ax.set_visible(False)
            continue

        # 두 그룹: cross=0 / cross=1 (없으면 빈 시리즈)
        grp0 = sub_r[sub_r["cross_counts"] == 0]["path_length_diff"].dropna()
        grp1 = sub_r[sub_r["cross_counts"] == 1]["path_length_diff"].dropna()

        if grp0.empty and grp1.empty:
            ax.set_visible(False)
            continue

        # bin: 두 그룹의 합집합으로 정수 중심 bin 만들기
        if N_BINS_HIST is None:
            both = pd.concat([grp0, grp1]) if not grp0.empty else grp1
            if grp0.empty and not grp1.empty:
                both = grp1
            elif grp1.empty and not grp0.empty:
                both = grp0
            # 정수 bin으로 변환
            both_int = both.round().astype(int)
            edges = np.arange(both_int.min() - 0.5, both_int.max() + 1.5, 1)
        else:
            edges = N_BINS_HIST  # 사용자가 수를 지정했다면 그대로 사용

        # 공통 bin으로 히스토그램 카운트
        cnt0, edges0 = np.histogram(grp0, bins=edges)
        cnt1, edges1 = np.histogram(grp1, bins=edges)
        # bin center 계산
        centers = (edges[:-1] + edges[1:]) / 2.0

        # 막대 너비와 오프셋
        width = 0.45  # 각 막대 폭
        ax.bar(centers - width/2, cnt0, width=width, label=f"cross=0 (N={grp0.shape[0]})",
               color=color_for_cross(0))
        ax.bar(centers + width/2, cnt1, width=width, label=f"cross=1 (N={grp1.shape[0]})",
               color=color_for_cross(1))

        ax.set_title(f"rate={r}  •  jump=1", fontsize=10)
        ax.set_xlabel("ISL PathLen − PathLen")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    # 남는 축 숨김
    for j in range(len(rates), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Path Length Diff Distribution for jump=1 (cross=0 vs cross=1)", y=0.98)
    plt.tight_layout()

    # 저장/표시
    save_or_show(fig, base_name="pldiff_dist_jump1_cross01_by_rate", rate="all", save_mode=save_mode)

# -------------------------
# 공통: (x=path_length_diff)에서 min/max/mean 라인 그리기
# -------------------------
def _plot_stats_lines_by_group(
    df_in: pd.DataFrame,
    xcol: str,
    ycol: str,
    group_col: str,
    group_values: List[int],
    title: str,
    xlabel: str,
    ylabel: str,
    show_minmax: bool = True,
    save_base: str | None = None,
    rate: str | None = None,
    same_color_by_group: bool = True,
    color_picker=None,  # ← 추가: 그룹값 -> 색상 반환 함수 (예: color_for_k, color_for_cross)
    save_mode: str = DEFAULT_SAVE_MODE
):
    fig = plt.figure(figsize=(11, 7))
    has_any = False
    for g in iter_progress(group_values, desc=f"[Lines] {ycol} by {group_col}", unit="grp"):
        sub = df_in[df_in[group_col] == g].copy()
        if sub.empty: continue
        x = pd.to_numeric(sub[xcol], errors="coerce")
        y = pd.to_numeric(sub[ycol], errors="coerce")
        tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if tmp.empty: continue
        tmp["x"] = tmp["x"].round().astype(int)
        stats = tmp.groupby("x")["y"].agg(["min","max","mean"]).reset_index().sort_values("x")
        if stats.empty: continue
        has_any = True
        if color_picker is not None and same_color_by_group:
            col = color_picker(int(g))
        elif same_color_by_group and group_col == "jump_count":
            col = color_for_k(int(g))
        elif same_color_by_group and group_col == "cross_counts":
            col = color_for_cross(int(g))
        else:
            col = None

        label_base = f"{group_col}={g}"
        if show_minmax:
            plt.plot(stats["x"], stats["min"],  linestyle=":",  marker="o",  label=f"{label_base} • min",  color=col)
            plt.plot(stats["x"], stats["max"],  linestyle="-",  marker="s",  label=f"{label_base} • max",  color=col)
            plt.plot(stats["x"], stats["mean"], linestyle="--", marker="^",  label=f"{label_base} • mean", color=col)
        else:
            plt.plot(stats["x"], stats["mean"], linestyle="--", marker="^",  label=f"{label_base} • mean", color=col)

    if has_any:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        base = save_base or f"stats_lines__{group_col}__{ycol}"
        save_or_show(fig, base_name=base, rate=rate, save_mode=save_mode)
    else:
        plt.close()

# -------------------------
# (5) jump별 라인: x=path_length_diff, y∈{e2e, exp(result), exp(isl)} — jump>=1
# -------------------------
def plot_lines_by_jump(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    rates = sorted(df["gen_rate"].dropna().unique().tolist(), key=_rate_key)
    targets = [
        ("e2e_delay",         "E2E Delay (ms)"),
        ("exp_delay_result",  "Expected Delay (result, ms)"),
        ("exp_delay_isl",     "Expected Delay (ISL, ms)")
    ]

    for r in rates:
        sub_rate = df[df["gen_rate"] == r]
        sub = sub_rate[(pd.to_numeric(sub_rate["jump_count"], errors="coerce") >= 1) &
                       (pd.to_numeric(sub_rate["path_length_diff"], errors="coerce").notna())].copy()
        if sub.empty:
            print(f"[skip] lines-by-jump: no data for rate={r}")
            continue

        jumps = sorted(sub["jump_count"].dropna().astype(int).unique().tolist())
        for ycol, yname in targets:
            if ycol not in sub.columns:
                print(f"[skip] {yname}: column '{ycol}' not found for rate={r}");
                continue
            _plot_stats_lines_by_group(
                sub, xcol="path_length_diff", ycol=ycol,
                group_col="jump_count", group_values=jumps,
                title=f"{yname} vs Gain of Hop Count by Jump (jump ≥ 1) — rate={r}",
                xlabel="ISL PathLen − PathLen", ylabel=f"{yname}",
                same_color_by_group=True,
                save_base=f"lines_by_jump__{ycol}",
                rate=r, save_mode=save_mode
            )


# -------------------------
# (6) cross별 라인: x=path_length_diff, y∈{e2e, exp(result), exp(isl)} — jump>=1, cross=0 포함
# -------------------------
def plot_lines_by_cross(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    rates = sorted(df["gen_rate"].dropna().unique().tolist(), key=_rate_key)
    targets = [
        ("e2e_delay",         "E2E Delay (ms)"),
        ("exp_delay_result",  "Expected Delay (result, ms)"),
        ("exp_delay_isl",     "Expected Delay (ISL, ms)")
    ]

    for r in rates:
        sub_rate = df[df["gen_rate"] == r]
        sub = sub_rate[(pd.to_numeric(sub_rate["jump_count"], errors="coerce") >= 1) &
                       (pd.to_numeric(sub_rate["path_length_diff"], errors="coerce").notna())].copy()
        if sub.empty:
            print(f"[skip] lines-by-cross: no data for rate={r}")
            continue

        # 숫자형 보정
        for c in ["cross_counts", "path_length_diff", "e2e_delay",
                  "exp_delay_result", "exp_delay_isl"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        # pldiff 정수 bin
        sub["pldiff_int"] = sub["path_length_diff"].round().astype("Int64")

        # cross 목록 (0 포함)
        crosses = sorted(sub["cross_counts"].dropna().astype(int).unique().tolist())

        # ─────────────────────────────────────────────
        # 1) 기존 3개 지표: helper 이용(색상은 기존 로직 유지)
        # ─────────────────────────────────────────────
        for ycol, yname in targets:
            if ycol not in sub.columns:
                print(f"[skip] {yname}: column '{ycol}' not found for rate={r}")
                continue
            _plot_stats_lines_by_group(
                sub, xcol="path_length_diff", ycol=ycol,
                group_col="cross_counts", group_values=crosses,
                title=f"{yname} vs Gain of Hop Count by Cross (jump ≥ 1, cross≥0) — rate={r}",
                xlabel="ISL PathLen − PathLen", ylabel=f"{yname}",
                same_color_by_group=True,
                save_base=f"lines_by_cross__{ycol}",
                rate=r, save_mode=save_mode,
                show_minmax=False
            )

        # ─────────────────────────────────────────────
        # 2) 추가 지표 A: e2e_delay - exp_delay_result  (우회로 인한 추가지연)
        #    → cross별 같은 색 유지(color_for_cross), 'mean'만 라인으로 그림
        # ─────────────────────────────────────────────
        sub["extra_delay_due_to_detour"] = sub["e2e_delay"] - sub["exp_delay_result"]
        dfA = (sub.dropna(subset=["extra_delay_due_to_detour", "pldiff_int", "cross_counts"])
                  .groupby(["cross_counts", "pldiff_int"], dropna=False)["extra_delay_due_to_detour"]
                  .mean().reset_index().sort_values(["cross_counts", "pldiff_int"]))

        if not dfA.empty:
            fig, ax = plt.subplots(figsize=(11, 7))
            for c in crosses:
                seg = dfA[dfA["cross_counts"] == c]
                if seg.empty:
                    continue
                ax.plot(
                    seg["pldiff_int"].astype(int),
                    seg["extra_delay_due_to_detour"],
                    marker="o", linestyle="-",
                    color=color_for_cross(int(c)),
                    label=f"cross={c}"
                )
            ax.set_xlabel("ISL PathLen − PathLen")
            ax.set_ylabel("우회로 인한 추가지연")
            ax.set_title(f"우회로 인한 추가지연 vs Gain of Hop Count by Cross — rate={r}")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2)
            plt.tight_layout()
            if "save_or_show" in globals():
                save_or_show(fig, base_name="lines_by_cross__extra_delay", rate=r, save_mode=save_mode)
            else:
                plt.show()

        # ─────────────────────────────────────────────
        # 3) 추가 지표 B: exp_delay_isl - e2e_delay  (ISL 경로 사용 시 이득)
        #    → cross별 같은 색 유지(color_for_cross), 'mean'만 라인으로 그림
        # ─────────────────────────────────────────────
        sub["isl_benefit"] = sub["exp_delay_isl"] - sub["e2e_delay"]
        dfB = (sub.dropna(subset=["isl_benefit", "pldiff_int", "cross_counts"])
                  .groupby(["cross_counts", "pldiff_int"], dropna=False)["isl_benefit"]
                  .mean().reset_index().sort_values(["cross_counts", "pldiff_int"]))

        if not dfB.empty:
            fig, ax = plt.subplots(figsize=(11, 7))
            for c in crosses:
                seg = dfB[dfB["cross_counts"] == c]
                if seg.empty:
                    continue
                ax.plot(
                    seg["pldiff_int"].astype(int),
                    seg["isl_benefit"],
                    marker="o", linestyle="-",
                    color=color_for_cross(int(c)),
                    label=f"cross={c}"
                )
            ax.set_xlabel("ISL PathLen − PathLen")
            ax.set_ylabel("ISL 경로 사용 시 이득")
            ax.set_title(f"ISL 경로 사용 시 이득 vs Gain of Hop Count by Cross — rate={r}")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2)
            plt.tight_layout()
            if "save_or_show" in globals():
                save_or_show(fig, base_name="lines_by_cross__isl_benefit", rate=r, save_mode=save_mode)
            else:
                plt.show()


# -------------------------
# (7) 3D 막대: (jump, cross) → 평균 {Propagation, Queueing, E2E}
# -------------------------
def plot_3d_bars(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    """
    (jump_count, cross_counts) → 평균 {Propagation, Queueing, E2E} 3D 막대그래프
    - 각 생성률(gen_rate)별로 개별 그림 생성
    - 막대 투명도: ALPHA_BARS (전역)
    - 시계방향 회전: VIEW_AZIM (전역, 값 감소 방향이 시계방향)
    - 저장/표시는 save_or_show() 사용
    """
    # 생성률 정렬(숫자면 숫자 기준, 아니면 문자열)
    rates = sorted(
        df["gen_rate"].dropna().unique().tolist(),
        key=lambda s: (int(s) if str(s).isdigit() else float('inf'), str(s))
    )
    if not rates:
        print("[skip] No gen_rate to plot 3D bars.")
        return

    need_cols = {"jump_count", "cross_counts", "prop_delay", "queue_delay", "e2e_delay"}
    missing = need_cols - set(df.columns)
    if missing:
        print(f"[skip] 3D bars: missing columns {missing}")
        return

    for r in rates:
        sub = df[df["gen_rate"] == r].copy()
        if sub.empty:
            continue

        # 숫자형 보정
        for c in ["jump_count", "cross_counts", "prop_delay", "queue_delay", "e2e_delay"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        # 그룹 평균 (jump, cross)
        gb = (
            sub.groupby(["jump_count", "cross_counts"], dropna=False)
               .agg(mean_prop=("prop_delay", "mean"),
                    mean_queue=("queue_delay", "mean"),
                    mean_e2e=("e2e_delay", "mean"))
               .reset_index()
               .dropna(how="any", subset=["jump_count", "cross_counts"])
        )
        if gb.empty:
            print(f"[skip] 3D bars: empty after group for rate={r}")
            continue

        # 축 카테고리(정수)
        js = np.sort(gb["jump_count"].dropna().astype(int).unique())
        cs = np.sort(gb["cross_counts"].dropna().astype(int).unique())
        if js.size == 0 or cs.size == 0:
            print(f"[skip] 3D bars: no categories for rate={r}")
            continue

        # (jump, cross) → 값 매핑
        def _grid(metric_col: str):
            J, C = np.meshgrid(js, cs, indexing="ij")
            H = np.full(J.shape, np.nan, dtype=float)
            m = {(int(row.jump_count), int(row.cross_counts)): float(row[metric_col])
                 for _, row in gb.dropna(subset=[metric_col]).iterrows()}
            for i, jv in enumerate(js):
                for k, cv in enumerate(cs):
                    H[i, k] = m.get((jv, cv), np.nan)
            return J, C, H

        # 공통 그리기 루틴
        def _bar3d(J, C, H, title, base_name):
            # NaN 제외한 셀만 바
            mask = ~np.isnan(H)
            if not np.any(mask):
                print(f"[skip] 3D bars ({title}): no finite values (rate={r})")
                return

            # 막대 폭/깊이
            dx = dy = 0.6

            # bar3d에 들어갈 평면 좌표/높이
            xs = J[mask].ravel() - dx/2
            ys = C[mask].ravel() - dy/2
            zs = np.zeros(xs.size, dtype=float)
            hs = H[mask].ravel()

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.bar3d(xs, ys, zs, dx, dy, hs, alpha=ALPHA_BARS)  # 색은 기본 팔레트 사용

            ax.set_xlabel("Jump Count")
            ax.set_ylabel("Cross Counts (including 0)")
            ax.set_zlabel("Mean Delay (ms)")
            ax.set_title(f"{title} vs (Jump, Cross) — rate={r}")

            # 정수 tick
            ax.set_xticks(js)
            ax.set_yticks(cs)

            # 시야각(시계방향 = azim 감소)
            ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

            plt.tight_layout()
            save_or_show(fig, base_name=base_name, rate=str(r), save_mode=save_mode)

        # 메트릭별로 그림 생성
        for col, title, base in [
            ("mean_prop",  "3D Bars: Propagation Delay", "bars3d_prop"),
            ("mean_queue", "3D Bars: Queueing Delay",    "bars3d_queue"),
            ("mean_e2e",   "3D Bars: E2E Delay",         "bars3d_e2e"),
        ]:
            J, C, H = _grid(col)
            _bar3d(J, C, H, title, base)


# (8)  cross별 CDF (jump≥1)
def plot_cdf_diff_isl_minus_e2e_by_cross(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    need_cols = {"exp_delay_isl", "e2e_delay", "cross_counts", "jump_count"}
    if not need_cols.issubset(df.columns):
        print(f"[skip] CDF(diff): missing columns {need_cols - set(df.columns)}");
        return

    rates = sorted(df["gen_rate"].dropna().unique().tolist(), key=_rate_key)
    for r in rates:
        sub0 = df[df["gen_rate"] == r].copy()
        sub0["jump_count"]   = pd.to_numeric(sub0["jump_count"], errors="coerce")
        sub0["cross_counts"] = pd.to_numeric(sub0["cross_counts"], errors="coerce")
        sub0["exp_delay_isl"] = pd.to_numeric(sub0["exp_delay_isl"], errors="coerce")
        sub0["e2e_delay"]     = pd.to_numeric(sub0["e2e_delay"], errors="coerce")
        sub = sub0[(sub0["jump_count"] >= 1)].dropna(subset=["cross_counts","exp_delay_isl","e2e_delay"])
        if sub.empty:
            print(f"[skip] CDF(diff) by cross: no data for rate={r}");
            continue

        crosses = sorted(sub["cross_counts"].astype(int).unique().tolist())  # 0 포함

        fig = plt.figure(figsize=(9, 6))
        for c in iter_progress(crosses, desc=f"[CDF] isl-e2e by cross (rate={r})", unit="cross"):
            grp = sub[sub["cross_counts"] == c]
            diff = grp["exp_delay_isl"] - grp["e2e_delay"]
            x, y = ecdf(diff)
            if x.size == 0:
                continue
            plt.plot(x, y, label=f"cross={c} (N={len(grp)})", color=color_for_cross(int(c)))

        plt.xlabel("Δ delay = Expected(ISL) − E2E (ms)")
        plt.ylabel("CDF")
        plt.title(f"CDF of Δ delay by Cross Count (jump ≥ 1) — rate={r}")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        save_or_show(fig, base_name="cdf_diff_isl_minus_e2e_by_cross", rate=r, save_mode=save_mode)

# ---------- (추가) cross 전용 색상 매핑 ----------
_cross2color = {}
def color_for_cross(c: int) -> str:
    global _cross2color, _COLORS
    if not _cross2color:
        # 기본 팔레트 준비 (없으면 C0..C9)
        try:
            colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
        except Exception:
            colors = []
        _COLORS = colors or [f"C{i}" for i in range(10)]
    if c not in _cross2color:
        _cross2color[c] = _COLORS[len(_cross2color) % len(_COLORS)]
    return _cross2color[c]


# ---------- (추가) 생성률별 cross count 라인 그래프 (x=pldiff, y=평균 딜레이) ----------
def plot_pldiff_distribution_by_cross(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    """
    각 생성률(gen_rate)마다,
      x = path_length_diff(= ISL PathLen - PathLen, 정수 bin),
      그룹 = cross_counts (0 포함, 같은 cross는 같은 색),
      y = 'count' (빈도)  ← 꺾은선 평균 → 막대 분포로 변경
    를 그린 막대그래프를 각각 생성한다.

    - jump_count >= 1 데이터만 사용
    - path_length_diff가 NaN이면 해당 rate는 스킵
    - 범례는 기존처럼 cross=값 으로 유지
    """
    need_cols = {"gen_rate", "cross_counts", "path_length_diff", "jump_count"}
    miss = need_cols - set(df.columns)
    if miss:
        print(f"[skip] plot_cross_lines_by_rate: missing columns {miss}")
        return

    # 생성률 정렬(숫자 우선 → 문자열)
    rates = sorted(
        df["gen_rate"].dropna().unique().tolist(),
        key=lambda s: (int(s) if str(s).isdigit() else float('inf'), str(s))
    )
    if not rates:
        print("[skip] plot_cross_lines_by_rate: no gen_rate")
        return

    for r in rates:
        sub_r = df[df["gen_rate"] == r].copy()
        # jump>=1 & pldiff 유효
        sub_r = sub_r[pd.to_numeric(sub_r["jump_count"], errors="coerce") >= 1]
        sub_r = sub_r[pd.to_numeric(sub_r["path_length_diff"], errors="coerce").notna()]
        if sub_r.empty:
            print(f"[skip] rate={r}: no data (jump>=1 & pldiff present)")
            continue

        # 숫자형 보정 + pldiff 정수 bin
        sub_r["cross_counts"]   = pd.to_numeric(sub_r["cross_counts"], errors="coerce")
        sub_r["pldiff_int"]     = pd.to_numeric(sub_r["path_length_diff"], errors="coerce").round().astype("Int64")

        # 교차표: index=pldiff, columns=cross, values=count
        ct = (sub_r
              .dropna(subset=["pldiff_int", "cross_counts"])
              .groupby(["pldiff_int", "cross_counts"])
              .size()
              .reset_index(name="count"))

        if ct.empty:
            print(f"[skip] rate={r}: empty counts table")
            continue

        pivot = (ct
                 .pivot(index="pldiff_int", columns="cross_counts", values="count")
                 .fillna(0)
                 .sort_index())

        x_vals  = pivot.index.astype(int).to_numpy()
        crosses = list(pd.Series(pivot.columns).dropna().astype(int).sort_values().to_list())
        if not crosses:
            print(f"[skip] rate={r}: no cross columns after pivot")
            continue

        # 막대 폭/오프셋 계산 (그룹 막대)
        n_groups   = len(crosses)
        width_full = 0.9
        bar_w      = width_full / max(n_groups, 1)

        fig, ax = plt.subplots(figsize=(11, 7))
        for i, c in enumerate(crosses):
            # 각 cross의 x 위치를 살짝 옮겨서 그룹 막대 구성
            x_pos = x_vals + (i - (n_groups - 1) / 2.0) * bar_w
            y_vals = pivot.get(c, pd.Series([0]*len(x_vals), index=pivot.index)).to_numpy()
            ax.bar(x_pos, y_vals, width=bar_w, label=f"cross={c}", color=color_for_cross(int(c)))

        ax.set_xlabel("Gain of hop count (ISL PathLen − PathLen)")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution by Cross vs Gain of Hop Count — generation rate={r}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(x) for x in x_vals])
        ax.legend(ncol=2)
        plt.tight_layout()

        # 저장/표시
        if "save_or_show" in globals():
            save_or_show(fig, base_name="cross_pldiff_hist", rate=str(r), save_mode=save_mode)
        else:
            plt.show()

def plot_3d_negative_ratio_by_jump_cross(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    """
    각 생성률(gen_rate)별로,
      X = jump_count (※ jump>=1만 사용),
      Y = cross_counts (0 포함),
      Z = 비율( expected(isl) − e2e < 0 )을 3D 막대그래프로 표현.

    - 전체(overall) 비율도 jump>=1로 필터된 모집단 기준으로 계산/표시.
    """

    need = {"gen_rate", "jump_count", "cross_counts", "exp_delay_isl", "e2e_delay"}
    missing = need - set(df.columns)
    if missing:
        print(f"[skip] plot_3d_negative_ratio_by_jump_cross: missing columns {missing}")
        return

    # gen_rate 정렬(숫자로 끝나는 파일명을 쓰는 경우 보기 좋게 정렬)
    rates = sorted(
        df["gen_rate"].dropna().unique().tolist(),
        key=lambda s: (int(s) if str(s).isdigit() else float('inf'), str(s))
    )
    if not rates:
        print("[skip] no gen_rate")
        return

    for r in rates:
        sub = df[df["gen_rate"] == r].copy()
        if sub.empty:
            continue

        # 숫자화
        for c in ["jump_count", "cross_counts", "exp_delay_isl", "e2e_delay"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        # ===== 핵심 변경: jump_count >= 1만 사용 =====
        sub = sub[(sub["jump_count"] >= 1)].dropna(subset=["exp_delay_isl", "e2e_delay", "jump_count", "cross_counts"])
        if sub.empty:
            print(f"[skip] rate={r}: no rows after filtering jump_count>=1")
            continue

        # 음수 여부 플래그
        sub["diff_isl_minus_e2e"] = sub["exp_delay_isl"] - sub["e2e_delay"]
        sub["is_negative"] = (sub["diff_isl_minus_e2e"] < 0).astype(int)

        # --- 전체(해당 rate, jump>=1 모집단) 음수 비율 ---
        overall_total = int(sub.shape[0])
        overall_neg   = int(sub["is_negative"].sum())
        overall_ratio = (overall_neg / overall_total) if overall_total > 0 else 0.0

        # --- (jump, cross) 버킷별 비율 ---
        gb = (sub.groupby(["jump_count", "cross_counts"], dropna=False)
                .agg(total=("is_negative", "size"),
                     neg=("is_negative", "sum"))
                .reset_index())
        if gb.empty:
            print(f"[skip] rate={r}: empty groups (after jump>=1 filter)")
            continue

        gb["ratio"] = gb["neg"] / gb["total"]

        # 축 값: jump는 1부터, cross는 0 포함
        js = np.sort(gb["jump_count"].dropna().astype(int).unique())
        cs = np.sort(gb["cross_counts"].dropna().astype(int).unique())
        if js.size == 0 or cs.size == 0:
            print(f"[skip] rate={r}: no categories")
            continue

        # 격자에 값 매핑
        J, C = np.meshgrid(js, cs, indexing="ij")
        H = np.full(J.shape, np.nan, dtype=float)
        val_map = {(int(row.jump_count), int(row.cross_counts)): float(row.ratio)
                   for _, row in gb.iterrows()}
        for i, jv in enumerate(js):
            for k, cv in enumerate(cs):
                H[i, k] = val_map.get((jv, cv), np.nan)

        H_plot = np.nan_to_num(H, nan=0.0)

        # 3D 막대 렌더링
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        dx = dy = 0.6
        xs, ys, zs, hs = [], [], [], []
        for i, jv in enumerate(js):
            for k, cv in enumerate(cs):
                xs.append(jv - dx/2)
                ys.append(cv - dy/2)
                zs.append(0.0)
                hs.append(H_plot[i, k])

        ax.bar3d(xs, ys, zs, dx, dy, hs, alpha=ALPHA_BARS)

        ax.set_xlabel("Jump Count (≥ 1)")
        ax.set_ylabel("Cross Counts (including 0)")
        ax.set_zlabel("Ratio (exp(ISL) − E2E < 0)")
        ax.set_title(
            f"3D Bars: Negative Ratio by Jump × Cross — generation rate={r} (jump≥1)\n"
            f"(overall: {overall_ratio:.2%}  =  {overall_neg}/{overall_total})"
        )
        ax.set_xticks(js); ax.set_yticks(cs)
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        plt.tight_layout()

        if "save_or_show" in globals():
            save_or_show(fig, base_name="bars3d_neg_ratio_jge1", rate=str(r), save_mode=save_mode)
        else:
            plt.show()

def plot_3d_mean_abs_gap_on_negative_by_jump_cross(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    """
    각 generation rate 별로,
      X = jump_count (※ jump>=1만 사용),
      Y = cross_counts (0 포함),
      Z = mean( | e2e_delay − exp_delay_isl | )  ← 단, negative 케이스( exp(isl) − e2e < 0 )만 집계

    그래프 제목에 전체(해당 rate, jump>=1 모집단) 대비
    negative 표본 비율과 overall mean |gap|도 같이 표기.
    """
    need = {"gen_rate", "jump_count", "cross_counts", "exp_delay_isl", "e2e_delay"}
    missing = need - set(df.columns)
    if missing:
        print(f"[skip] plot_3d_mean_abs_gap_on_negative_by_jump_cross: missing columns {missing}")
        return

    # gen_rate 정렬(숫자 우선)
    rates = sorted(
        df["gen_rate"].dropna().unique().tolist(),
        key=lambda s: (int(s) if str(s).isdigit() else float("inf"), str(s))
    )
    if not rates:
        print("[skip] no gen_rate")
        return

    for r in rates:
        sub = df[df["gen_rate"] == r].copy()
        if sub.empty:
            continue

        # 숫자화
        for c in ["jump_count", "cross_counts", "exp_delay_isl", "e2e_delay"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        # ----- 필터: jump_count >= 1 -----
        sub = sub[(sub["jump_count"] >= 1)].dropna(subset=["e2e_delay", "exp_delay_isl", "jump_count", "cross_counts"])
        if sub.empty:
            print(f"[skip] rate={r}: no rows after filtering jump_count>=1")
            continue

        # diff = e2e − exp(isl) ; negative 정의는 exp(isl) − e2e < 0  <=>  diff > 0
        sub["diff"] = sub["e2e_delay"] - sub["exp_delay_isl"]
        neg = sub[sub["diff"] > 0].copy()
        if neg.empty:
            print(f"[skip] rate={r}: no negative cases (exp(isl) − e2e < 0)")
            continue

        neg["abs_gap"] = neg["diff"].abs()

        # overall 통계(제목 표기용) — 모수: jump>=1 모집단
        overall_total = int(sub.shape[0])           # jump>=1 모집단 크기
        overall_neg   = int(neg.shape[0])           # negative 표본 수
        overall_ratio = (overall_neg / overall_total) if overall_total > 0 else 0.0
        overall_mean_abs = float(neg["abs_gap"].mean())

        # (jump, cross) 그룹 평균 |gap|
        gb = (neg.groupby(["jump_count", "cross_counts"], dropna=False)
                .agg(mean_abs=("abs_gap", "mean"),
                     n=("abs_gap", "size"))
                .reset_index())
        if gb.empty:
            print(f"[skip] rate={r}: empty groups")
            continue

        # 축 값들
        js = np.sort(gb["jump_count"].dropna().astype(int).unique())
        cs = np.sort(gb["cross_counts"].dropna().astype(int).unique())
        if js.size == 0 or cs.size == 0:
            print(f"[skip] rate={r}: no categories")
            continue

        # 격자에 평균값 매핑
        J, C = np.meshgrid(js, cs, indexing="ij")
        H = np.full(J.shape, np.nan, dtype=float)
        val_map = {(int(row.jump_count), int(row.cross_counts)): float(row.mean_abs) for _, row in gb.iterrows()}
        for i, jv in enumerate(js):
            for k, cv in enumerate(cs):
                H[i, k] = val_map.get((jv, cv), np.nan)

        H_plot = np.nan_to_num(H, nan=0.0)

        # --- 3D bar ---
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        dx = dy = 0.6
        xs, ys, zs, heights = [], [], [], []
        for i, jv in enumerate(js):
            for k, cv in enumerate(cs):
                xs.append(jv - dx/2)
                ys.append(cv - dy/2)
                zs.append(0.0)
                heights.append(H_plot[i, k])

        ax.bar3d(xs, ys, zs, dx, dy, heights, alpha=ALPHA_BARS)

        ax.set_xlabel("Jump Count (≥ 1)")
        ax.set_ylabel("Cross Counts (including 0)")
        ax.set_zlabel("Mean |E2E − Expected(ISL)| (ms)  [negative only]")
        ax.set_title(
            f"3D Bars: Mean |E2E − Expected(ISL)| on Negative • generation rate={r}\n"
            f"(overall negative ratio: {overall_ratio:.2%}  =  {overall_neg}/{overall_total},  "
            f"overall mean |gap| = {overall_mean_abs:.3f} ms)"
        )
        ax.set_xticks(js); ax.set_yticks(cs)
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        plt.tight_layout()

        if "save_or_show" in globals():
            save_or_show(fig, base_name="bars3d_mean_abs_gap_negonly_jge1", rate=str(r), save_mode=save_mode)
        else:
            plt.show()

def plot_rate_delay_lines(df: pd.DataFrame, save_mode: str = DEFAULT_SAVE_MODE):
    """
    x축: generation rate
    y축: ms
    legend: E2E, Expected (result), Expected (ISL)
    집계: rate별 평균 (mean)

    - Status==success 필터는 load_all()에서 이미 수행된다고 가정
    - 세 컬럼이 모두 있을 때만 그립니다.
    """
    needed = ["gen_rate", "e2e_delay", "exp_delay_result", "exp_delay_isl"]
    if not set(needed).issubset(df.columns):
        missing = [c for c in needed if c not in df.columns]
        print(f"[skip] plot_rate_delay_lines: missing columns {missing}")
        return

    # 숫자형 변환
    df_loc = df.copy()
    df_loc["gen_rate_num"]     = pd.to_numeric(df_loc["gen_rate"], errors="coerce")
    df_loc["e2e_delay"]        = pd.to_numeric(df_loc["e2e_delay"], errors="coerce")
    df_loc["exp_delay_result"] = pd.to_numeric(df_loc["exp_delay_result"], errors="coerce")
    df_loc["exp_delay_isl"]    = pd.to_numeric(df_loc["exp_delay_isl"], errors="coerce")

    df_loc = df_loc.dropna(subset=["gen_rate_num"])

    if df_loc.empty:
        print("[skip] plot_rate_delay_lines: no rows with valid generation rate")
        return

    # rate별 평균 집계
    agg = (df_loc
           .groupby("gen_rate_num", as_index=False)
           .agg(e2e_mean=("e2e_delay", "mean"),
                exp_res_mean=("exp_delay_result", "mean"),
                exp_isl_mean=("exp_delay_isl", "mean"))
           .sort_values("gen_rate_num"))

    if agg.empty:
        print("[skip] plot_rate_delay_lines: no aggregated rows")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(agg["gen_rate_num"], agg["e2e_mean"],      marker="o", label="E2E")
    ax.plot(agg["gen_rate_num"], agg["exp_res_mean"],  marker="s", label="Expected (result)")
    ax.plot(agg["gen_rate_num"], agg["exp_isl_mean"],  marker="^", label="Expected (ISL)")

    ax.set_xlabel("Generation Rate")
    ax.set_ylabel("Delay (ms)")
    ax.set_title("Average Delays vs Generation Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    # 저장 또는 표시
    save_or_show(fig, base_name="rate_vs_delay_lines", rate="all", save_mode=save_mode)


def run(input_files):
    df = load_all(input_files)
    if df.empty:
        print("빈 데이터입니다."); return
    # by_rate 그래프(이미 생성률별 subplot)
    save_mode = DEFAULT_SAVE_MODE
    plot_path_length_by_rate(df, save_mode=save_mode)
    plot_jump_count_by_rate(df, save_mode=save_mode)
    # # 나머지는 각 함수에서 생성률별로 따로 그림
    plot_cross_by_jump(df, save_mode=save_mode)
    plot_pldiff_by_jump(df, save_mode=save_mode)
    plot_pldiff_by_one_jump(df, save_mode=save_mode)
    # plot_pldiff_distribution_by_cross(df, save_mode=DEFAULT_SAVE_MODE)
    # # plot_lines_by_jump(df, save_mode=save_mode)
    plot_lines_by_cross(df, save_mode=save_mode)
    # plot_cdf_diff_isl_minus_e2e_by_cross(df, save_mode=save_mode)
    plot_3d_bars(df, save_mode=save_mode)
    # plot_3d_negative_ratio_by_jump_cross(df, save_mode=DEFAULT_SAVE_MODE)
    # plot_3d_mean_abs_gap_on_negative_by_jump_cross(df, save_mode=DEFAULT_SAVE_MODE)
    plot_rate_delay_lines(df, save_mode=DEFAULT_SAVE_MODE)


if __name__ == "__main__":
    import os
    from glob import glob

    # 1) 직접 명시하는 방식 (권장: 확실함)
    BASE = r"C:\Users\김태성\PycharmProjects\ground-satellite routing\results"
    RATES = [1] # 필요한 생성률만
    input_files = []
    for r in RATES:
        # 둘 다 GSL 결과(버퍼 제한/무제한)라면 이렇게 추가
        # input_files.append(os.path.join(BASE, f"infinite_Q_with_GSL_{r}.csv"))
        input_files.append(os.path.join(BASE, f"limited_Q_with_GSL_{r}.csv"))

    # 2) 또는 패턴으로 자동 수집 (원하면 주석 해제)
    # input_files = sorted(glob(os.path.join(BASE, "*_with_GSL_*.csv")))

    print("[INFO] files:", input_files)
    run(input_files)

