import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.ticker import PercentFormatter


def hop_distance():
    # --- 설정: CSV 파일 경로 ---
    file_path = '../results/legacy/path length_10000iterations.csv'

    # --- CSV 읽어서 각 컬럼 리스트에 저장 ---
    min_vals = []
    max_vals = []

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 숫자 형 변환 필요시 int() 또는 float() 사용
            min_vals.append(float(row['min_length']))
            max_vals.append(float(row['max_length']))

    # --- 오름차순 정렬 ---
    min_vals.sort()
    max_vals.sort()

    # --- CDF 계산 (i번째 값이 전체에서 차지하는 비율) ---
    n_min = len(min_vals)
    n_max = len(max_vals)

    cdf_min = [i / n_min for i in range(1, n_min + 1)]
    cdf_max = [i / n_max for i in range(1, n_max + 1)]

    # --- 그래프 그리기 ---
    plt.figure(figsize=(8, 6))
    plt.step(min_vals, cdf_min, where='post', label='min_length')
    plt.step(max_vals, cdf_max, where='post', label='max_length')

    plt.xlabel('Length')
    plt.ylabel('Cumulative Distribution Function')
    plt.title('CDF of min_length vs. max_length')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.patches as patches

from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

def load_heatmap(gen_rate, t, N, M, data):
    def draw_cross_split_cell(ax, row, col, values):
        x, y = row, col  # ⬅ row가 가로축 (satellite index), col이 세로축 (orbit index)

        # 중심점
        cx, cy = x + 0.5, y + 0.5

        # 각 영역 정의
        top    = [(x, y + 1), (x + 1, y + 1), (cx, cy)]
        bottom = [(x + 1, y), (x, y), (cx, cy)]
        left   = [(x, y), (x, y + 1), (cx, cy)]
        right  = [(x + 1, y + 1), (x + 1, y), (cx, cy)]

        regions = [top, bottom, left, right]
        base_color = 'red'

        for i, region in enumerate(regions):
            triangle = patches.Polygon(
                region, closed=True,
                facecolor=base_color,
                alpha=values[i],
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(triangle)

        # 셀 테두리
        rect = patches.Rectangle((x, y), 1, 1, fill=False, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

    # 시각화 영역 설정
    fig, ax = plt.subplots(figsize=(N + 2, M))  # ⬅ 가로축이 N

    ax.set_xlim(0, N)
    ax.set_ylim(0, M)
    ax.set_aspect('equal')
    ax.set_title(f'generated packet per user ={gen_rate} time={t}')

    # 눈금 및 라벨
    ax.set_xticks(np.arange(N) + 0.5)
    ax.set_xticklabels(np.arange(N))
    ax.set_yticks(np.arange(M) + 0.5)
    ax.set_yticklabels(np.arange(M))
    ax.set_xlabel('satellite index')  # ⬅ x축
    ax.set_ylabel('orbit index')      # ⬅ y축

    # 셀 그리기 (col = 세로축 index, row = 가로축 index)
    for n in range(N):
        for m in range(M):
            draw_cross_split_cell(ax, n, M - 1 - m, data[n][m])  # y축 뒤집기 위해 M - 1 - m 사용

    # 컬러바 (투명도 표현용)
    cmap = ListedColormap([[0, 0, 1, alpha] for alpha in np.linspace(0, 1, 256)])  # 파란색 투명도 변화
    sm = ScalarMappable(cmap=cmap)
    sm.set_array([])

    # cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    # cbar.set_label('Queue Buffer Load Status')
    # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    # cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'])

    plt.tight_layout()
    plt.show()



import re
from typing import Dict, List
import pandas as pd

# ========= 사용자 설정 =========



# ========= 유틸: node_id → 대륙 추정 =========
def guess_continent(node_id: str) -> str:
    """
    node_id의 접두/키워드로 대륙 추정.
    예) asia-1606, eu-12, na-3, south-america-7 ...
    규칙에 안 맞으면 'unknown' 반환.
    """
    s = str(node_id).strip().lower()

    patterns: Dict[str, List[str]] = {
        "N.america": [
            r"^na\b", r"^n\.?america", r"^north[_\- ]?america"
        ],
        "S.america": [
            r"^sa\b", r"^s\.?america", r"^south[_\- ]?america"
        ],
        "europe": [
            r"^eu\b", r"^europe"
        ],
        "asia": [
            r"^as\b", r"^asia"
        ],
        "africa": [
            r"^af\b", r"^africa"
        ],
        "oceania": [
            r"^oc\b", r"^oceania", r"^australia", r"^au\b"
        ],
    }
    for cont, pats in patterns.items():
        for p in pats:
            if re.search(p, s):
                return cont
    # 대시 이전 접두어가 대륙명인 경우도 커버(예: africa-123)
    head = s.split("-")[0]
    simple = {
        "na": "N.america", "sa": "S.america", "eu": "europe",
        "as": "asia", "af": "africa", "oc": "oceania",
        "namerica": "N.america", "samerica": "S.america",
        "europe": "europe", "asia": "asia", "africa": "africa", "oceania": "oceania",
    }
    return simple.get(head, "unknown")


# ========= 데이터 로드 & 전처리 =========
def load_counts(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    needed = {"node_id", "total_counts", "success_counts", "drop_counts", "avg_queuing_delay", "delay_portion"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {sorted(miss)}")

    # 수치형 변환
    for c in ["total_counts", "success_counts", "drop_counts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # ★ 지연 지표가 있으면 숫자 변환
    for c in ["avg_queuing_delay", "delay_portion"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 대륙라벨
    df["continent"] = df["node_id"].map(guess_continent)

    # unknown 제거(그래프 대상에서 제외)
    before = len(df)
    df = df[df["continent"] != "unknown"].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"[info] unknown 대륙으로 분류되어 제외된 노드: {removed}개")

    # 순서 정렬
    df["continent"] = pd.Categorical(df["continent"], categories=CONTINENTS_ORDER, ordered=True)
    return df


# ========= 그래프 #1: 대륙별 total_counts 박스플롯 =========
def plot_box_total_by_continent(df: pd.DataFrame):
    data = [df.loc[df["continent"] == cont, "total_counts"].to_numpy()
            for cont in CONTINENTS_ORDER if cont in df["continent"].unique()]
    labels = [cont for cont in CONTINENTS_ORDER if cont in df["continent"].unique()]
    if not data:
        print("[skip] 박스플롯: 표시할 대륙 데이터가 없습니다.")
        return

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, labels=labels, showmeans=True)
    plt.title("Distribution of total_counts by Continent")
    plt.ylabel("total_counts")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========= 그래프 #2: 대륙별 노드 막대(성공/드롭 스택 + 주석) =========
def plot_per_continent_node_bars(df: pd.DataFrame):
    """
    2×3 서브플롯으로 각 대륙의 노드별 막대그래프:
      - x: node_id (많으면 간격/표식 줄임)
      - bar 높이: total_counts
      - 내부 stacked: success(파랑) + drop(빨강) (+ other(회색) if any)
      - 막대 상단: total_counts 수치
      - 내부: success%, drop% 표시
    """
    continents = [c for c in CONTINENTS_ORDER if c in df["continent"].unique()]
    if not continents:
        print("[skip] 막대그래프: 표시할 대륙 데이터가 없습니다.")
        return

    n = len(continents)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5*ncols, 3.8*nrows), sharey=False)
    axes = axes.ravel()

    for idx, cont in enumerate(continents):
        ax = axes[idx]
        sub = df[df["continent"] == cont].copy()
        # 노드가 많으면 total_counts 내림차순으로 정렬
        sub = sub.sort_values("total_counts", ascending=False).reset_index(drop=True)

        x = np.arange(len(sub))
        total = sub["total_counts"].to_numpy(dtype=float)
        suc   = sub["success_counts"].to_numpy(dtype=float)
        drp   = sub["drop_counts"].to_numpy(dtype=float)
        other = np.clip(total - (suc + drp), a_min=0, a_max=None)

        # 배경: total (연한 회색)
        ax.bar(x, total, color="#dddddd", width=0.8, label="total")

        # 내부 스택: success(파랑), drop(빨강), other(연한회색 더 진하게)
        bottom = np.zeros_like(total)
        ax.bar(x, suc,   bottom=bottom, color="tab:blue",  width=0.8, label="success")
        bottom = bottom + suc
        ax.bar(x, drp,   bottom=bottom, color="tab:red",   width=0.8, label="drop")
        bottom = bottom + drp
        if other.sum() > 0:
            ax.bar(x, other, bottom=bottom, color="#aaaaaa", width=0.8, label="other")

        # 막대 상단 total 값
        ymax = max(total.max() * 1.12, 1.0)
        for xi, t in zip(x, total):
            ax.text(xi, t + ymax*0.01, f"{int(t)}", ha="center", va="bottom", fontsize=8, rotation=0)

        # 내부 비율 텍스트(success, drop)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_pct = np.where(total > 0, suc / total, 0.0)
            d_pct = np.where(total > 0, drp / total, 0.0)

        # 너무 낮은 막대 내부엔 텍스트 생략(겹침 방지)
        height_thresh = ymax * 0.04

        for xi, s, d, t, s_p, d_p in zip(x, suc, drp, total, s_pct, d_pct):
            if s > height_thresh:
                ax.text(xi, s/2, f"{int(round(s_p*100))}%", color="white",
                        ha="center", va="center", fontsize=7)
            if d > height_thresh:
                ax.text(xi, s + d/2, f"{int(round(d_p*100))}%", color="white",
                        ha="center", va="center", fontsize=7)

        # 라벨/축 설정
        ax.set_title(f"{cont} (N={len(sub)})", fontsize=11)
        ax.set_ylim(0, ymax)
        # node_id 레이블이 많으면 일부만 표시
        labels = sub["node_id"].astype(str).tolist()
        step = max(1, len(labels) // 30)
        shown_labels = [lab if (i % step == 0) else "" for i, lab in enumerate(labels)]
        ax.set_xticks(x)
        ax.set_xticklabels(shown_labels, rotation=90, fontsize=7)
        ax.set_ylabel("total_counts")
        ax.grid(axis="y", alpha=0.3)

        if idx == 0:
            ax.legend(ncol=3, fontsize=8)

    # 남는 축 비우기
    for j in range(len(continents), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Continent Node Bars: total with success/drop split", y=0.98)
    plt.tight_layout()
    plt.show()

def plot_per_continent_node_metric_like_graph2(
    df: pd.DataFrame,
    y_var: str = "avg_queuing_delay",   # "avg_queuing_delay" 또는 "delay_portion"
    portion_as_pct: bool = False,       # delay_portion을 %로 보고 싶으면 True
):
    """
    그래프2와 레이아웃/정렬/라벨링은 동일, 단지 y값만 선택 지표로 바꿔서 표시.
      - x: graph2와 동일(대륙별로 node_id, total_counts 내림차순 정렬)
      - y: y_var 컬럼 값
    """
    if y_var not in df.columns:
        print(f"[skip] '{y_var}' 컬럼이 CSV에 없습니다.")
        return

    continents = [c for c in CONTINENTS_ORDER if c in df["continent"].unique()]
    if not continents:
        print("[skip] 표시할 대륙 데이터가 없습니다.")
        return

    # 그래프2와 동일한 레이아웃
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5*ncols, 3.8*nrows), sharey=False)
    axes = axes.ravel()

    for idx, cont in enumerate(continents):
        ax = axes[idx]
        sub = df[df["continent"] == cont].copy()

        # ★ 그래프2와 동일한 x축: total_counts 내림차순 정렬
        sub = sub.sort_values("total_counts", ascending=False).reset_index(drop=True)

        x = np.arange(len(sub))
        y = sub[y_var].to_numpy(dtype=float)

        # delay_portion을 %로 보고 싶으면 변환
        if y_var == "delay_portion" and portion_as_pct:
            y = y * 100.0

        # 단일 막대(그래프2의 스택 대신, 값만 바뀐 막대)
        ax.bar(x, y, color="tab:purple", width=0.8, label=y_var)

        # y축 상단 여유 및 라벨(그래프2의 스타일을 비슷하게)
        if len(y) > 0 and np.isfinite(y).any():
            ymax = max(np.nanmax(y) * 1.12, 1e-12)
            ax.set_ylim(0, ymax)
            # 상단 값 표기
            for xi, val in zip(x, y):
                if np.isfinite(val):
                    ax.text(xi, val + ymax * 0.01, f"{val:.3g}", ha="center", va="bottom", fontsize=7)

        # x축 라벨 간격 처리 (그래프2 동일)
        labels = sub["node_id"].astype(str).tolist()
        step = max(1, len(labels) // 30)
        shown_labels = [lab if (i % step == 0) else "" for i, lab in enumerate(labels)]
        ax.set_xticks(x)
        ax.set_xticklabels(shown_labels, rotation=90, fontsize=7)

        # y축 라벨/포맷
        if y_var == "delay_portion" and portion_as_pct:
            ax.set_ylabel("delay_portion (%)")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        else:
            ax.set_ylabel(y_var)

        ax.set_title(f"{cont} (N={len(sub)})", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # 범례는 첫 축에만 (옵션)
        if idx == 0:
            ax.legend(fontsize=8)

    # 남는 축 비우기 (그래프2 동일)
    for j in range(len(continents), len(axes)):
        axes[j].set_visible(False)

    title = f"Per-Continent Node Bars: {y_var}"
    if y_var == "delay_portion" and portion_as_pct:
        title += " (%)"
    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    plt.show()

# ========= 메인 =========
if __name__ == "__main__":
    CSV_PATH = r"relay counts/relay_counts_before_and_after_rate_320.csv"  # 네 파일 경로로 변경
    CONTINENTS_ORDER = ["N.america", "S.america", "europe", "asia", "africa", "oceania"]
    df = load_counts(CSV_PATH)
    if df.empty:
        print("[warn] 빈 데이터입니다. CSV 경로/포맷을 확인하세요.")
    else:
        # # 1) 대륙별 total_counts 박스플롯 (1개)
        # plot_box_total_by_continent(df)
        # # 2) 대륙별 노드 막대 (2×3 서브플롯 합계 6개) -> 총 7개 그래프
        # plot_per_continent_node_bars(df)
        # # 3-변형) y값을 delay_portion(%)로
        plot_per_continent_node_metric_like_graph2(df, y_var="delay_portion", portion_as_pct=True)
