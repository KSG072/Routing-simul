import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

from parameters.PARAMS import PACKET_SIZE_BITS, GIGA

QOS_ORDER   = [0, 1, 2]
QOS_COLORS  = {0:'red', 1:'orange', 2:'black'}
QOS_LABELS  = {0: "High", 1: "Middle", 2: "Low"}
QOS_MARKERS = {0: "o", 1: "s", 2: "^"}  # 평균선에만 마커
# ---------- 분류 유틸(자급자족) ----------
_GROUND_RE = re.compile(r'^[A-Za-z]+-\d+$')  # 지상 노드: 영문자-숫자
_NUMERIC_RE = re.compile(r'^\d+$')  # 숫자 ID (위성)

# QoS 문자열 ↔ 코드 매핑 (필요 시 사용)
_QOS_LABELS = {0: "High", 1: "Middle", 2: "Low"}
_QOS_TO_CODE = {v.lower(): k for k, v in _QOS_LABELS.items()}  # "high" -> 0

def _apply_tick_style(ax, x_minor_div=5, y_minor_div=5,
                      major_len=7, minor_len=4, major_w=0.8, minor_w=0.8,
                      minor_color='0.6'):
    """
    - major/minor tick: 안쪽(in)
    - 주/보조 눈금은 '주 축(왼쪽, 아래쪽)'에만 표시 (top/right = False)
    - 보조 눈금 색상 회색(minor_color)
    """
    # 보조 눈금 위치(주 눈금 사이 등분)
    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_div))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor_div))

    # 주 눈금: 아래/왼쪽만, 안쪽 방향
    ax.tick_params(axis='both', which='major', direction='in',
                   length=major_len, width=major_w,
                   top=False, right=False)

    # 보조 눈금: 아래/왼쪽만, 안쪽 방향, 회색
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=minor_len, width=minor_w,
                   colors=minor_color,   # 회색 눈금
                   top=False, right=False)

def plot_arrival_rate_avg_over_start_at(
    base_dir,                                  # CSV들이 있는 폴더
    rates=(40, 80, 120, 160, 200, 240, 280, 320, 360),
    filename_tpl="result_{rate}.csv",
    metric="e2e_delay",                        # "e2e_delay" | "drop_rate"
    bin_size=50,                               # start_at 구간 간격(ms)
    start_range=(0, 600),                      # x축 범위(ms)
    success_label="success",                   # 성공 Status 라벨
    percent=False,                             # drop_rate를 %로 표시할지
    figsize=(10, 6),
    save_path=None,                            # 저장 경로 (None이면 화면표시)
    show=True
):
    """
    주어진 CSV들(파일명 템플릿: limited_Q_with_GSL_{rate}.csv)을 읽어,
    start_at(=Time(ms) - 각 파일의 Time(ms) 최소값)을 기준으로
    arrival rate별 구간(bin) 평균 지표를 그립니다.

    헤더 포맷(필요 컬럼):
      - "Time (ms)" (int)
      - "e2e delay" (float)  [metric="e2e_delay"일 때]
      - "Status" (str)       [metric="drop_rate"일 때]
    """
    # 라벨 표준화용 리네임(필요한 최소 컬럼만)
    rename_map = {
        "Time (ms)": "time_ms",
        "e2e delay": "e2e_delay",
        "Status": "Status",
    }

    fig, ax = plt.subplots(figsize=figsize)
    any_plotted = False

    for rate in rates:
        path = Path(base_dir) / filename_tpl.format(rate=rate)
        if not path.exists():
            print(f"[WARN] 파일 없음: {path}")
            continue

            # 1) 먼저 읽고
        df = pd.read_csv(path, low_memory=False)

        # (선택) 헤더 공백/BOM 제거
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

        # 2) 그 다음, 존재하는 컬럼만 필터해서 rename
        rename_map = {
            "Time (ms)": "time_ms",
            "e2e delay": "e2e_delay",
            "Status": "Status",
        }
        rename_only_existing = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=rename_only_existing)

        # 필수 컬럼 체크
        if "time_ms" not in df.columns:
            raise ValueError(f"{path.name} 에 'Time (ms)' 컬럼이 없습니다.")
        if metric == "e2e_delay" and "e2e_delay" not in df.columns:
            raise ValueError(f"{path.name} 에 'e2e delay' 컬럼이 없습니다.")
        if metric == "drop_rate" and "Status" not in df.columns:
            raise ValueError(f"{path.name} 에 'Status' 컬럼이 없습니다.")

        # 숫자 변환
        df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
        if metric == "e2e_delay":
            df["e2e_delay"] = pd.to_numeric(df["e2e_delay"], errors="coerce")

        # start_at 계산(파일별 0점)
        df["start_at"] = df["time_ms"] - df["time_ms"].min()

        # x축 범위 필터
        lo, hi = start_range
        m = (df["start_at"] >= lo) & (df["start_at"] <= hi)
        df = df[m]
        if df.empty:
            print(f"[INFO] 범위 {start_range} 내 데이터가 없어 스킵: rate={rate}")
            continue

        # 구간 bin
        df["tbin"] = (df["start_at"] // bin_size) * bin_size
        # 표시용 x좌표: bin 중앙값(가독성)
        df["tbin_center"] = df["tbin"] + bin_size / 2.0

        if metric == "e2e_delay":
            # ✅ success만 사용
            if "Status" not in df.columns:
                raise ValueError(f"{path.name} 에 'Status' 컬럼이 없습니다. (e2e_delay 계산은 success만 사용)")
            status_clean = df["Status"].astype(str).str.strip().str.lower()
            df_success = df[status_clean == str(success_label).lower()]  # default 'success'
            grp = (df_success
                   .dropna(subset=["e2e_delay"])
                   .groupby("tbin_center", as_index=False)["e2e_delay"]
                   .mean())
            if grp.empty:
                continue
            ax.plot(grp["tbin_center"], grp["e2e_delay"], label=f"{rate}", linewidth=1.6)
            any_plotted = True

        elif metric == "drop_rate":
            # 각 bin별 드롭 비율 = (Status != success) / 전체
            status_clean = df["Status"].astype(str).str.strip().str.lower()
            df["_is_drop"] = (status_clean != str(success_label).lower()).astype(int)
            g_total = df.groupby("tbin_center", as_index=False)["_is_drop"].count().rename(columns={"_is_drop": "total"})
            g_drop  = df.groupby("tbin_center", as_index=False)["_is_drop"].sum().rename(columns={"_is_drop": "drop"})
            g = pd.merge(g_total, g_drop, on="tbin_center", how="left")
            g["rate"] = g["drop"] / g["total"].where(g["total"] > 0, np.nan)
            y = g["rate"] * (100.0 if percent else 1.0)
            ax.plot(g["tbin_center"], y, label=f"{rate}", linewidth=1.6)
            any_plotted = True

        else:
            raise ValueError("metric 은 'e2e_delay' 또는 'drop_rate' 중 하나여야 합니다.")

    if not any_plotted:
        print("[INFO] 그릴 데이터가 없습니다.")
        return None, None

    # 축/범례/타이틀
    ax.set_xlim(start_range)
    ax.set_xlabel(f"start_at (ms)   [bin={bin_size}ms]")
    if metric == "e2e_delay":
        ax.set_ylabel("Average E2E Delay (ms)")
        ax.set_title("Average E2E Delay by Arrival Rate over start_at")
    else:
        ax.set_ylabel("Drop Rate (%)" if percent else "Drop Rate (fraction)")
        ax.set_title("Drop Rate by Arrival Rate over start_at")

    ax.grid(True, alpha=0.3)
    ax.legend(title="arrival rate", ncol=3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax

def analyze_files_cdf(base_name, indices, directory='.', qos=None, reserve_top=0.86,
                      target_range=(200, 600)):
    """
    각 파일(=Np 값) 별로 'success' 레코드만 모아서
      - Path Length (hops: Path Length - 1) CDF
      - End-to-End Delay(Queuing+Propagation+Transmission) CDF
    를 그립니다.

    target_range: (lo, hi)로 해석하며 Time (ms) 가 lo 이상, hi 미만인 행만 포함.
    """
    # QoS 파라미터 정규화
    qos_code = None
    if qos is not None:
        if isinstance(qos, str):
            key = qos.strip().lower()
            if key not in _QOS_TO_CODE:
                raise ValueError(f"Unknown qos='{qos}'. Use 0/1/2 or High/Middle/Low.")
            qos_code = _QOS_TO_CODE[key]
        elif isinstance(qos, (int, np.integer)):
            if qos not in (0, 1, 2):
                raise ValueError("qos must be 0,1,2 or 'High'/'Middle'/'Low'.")
            qos_code = int(qos)
        else:
            raise ValueError("qos must be int or str.")

    required = {'Path Length', 'Queuing Delay', 'Propagation Delay',
                'Transmission Delay', 'Status', 'Time (ms)'}
    if qos_code is not None:
        required = required | {'QoS'}

    path_by_idx = {}
    delay_by_idx = {}
    labels = []

    lo, hi = target_range

    for idx in indices:
        filename = os.path.join(directory, f"{base_name}{idx}.csv")
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue

        df = pd.read_csv(filename)
        # 컬럼명 정리(BOM/공백 제거)
        df.columns = [c.strip().replace("\ufeff","") for c in df.columns]

        if not required.issubset(df.columns):
            print(f"File {filename} is missing required columns {required - set(df.columns)}. Skipping.")
            continue

        # Time(ms) 필터 (lo 이상, hi 미만)
        df['Time (ms)'] = pd.to_numeric(df['Time (ms)'], errors='coerce')
        df = df[(df['Time (ms)'] >= lo) & (df['Time (ms)'] < hi)]
        if df.empty:
            print(f"[{idx}] no rows in Time(ms) range {target_range}. Skipping.")
            continue

        df_ok = df[df['Status'] == 'success']
        if qos_code is not None and 'QoS' in df.columns:
            df_ok = df_ok[df_ok['QoS'] == qos_code]

        if df_ok.empty:
            print(f"[{idx}] no success rows in Time(ms) range {target_range} (qos={qos}). Skipping.")
            continue

        # 숫자 변환 (안전)
        for col in ['Path Length', 'Queuing Delay', 'Propagation Delay', 'Transmission Delay']:
            df_ok[col] = pd.to_numeric(df_ok[col], errors='coerce')

        path_vals = (df_ok['Path Length'] - 1).dropna().to_numpy()
        delay_vals = (df_ok['Queuing Delay'] + df_ok['Propagation Delay'] + df_ok['Transmission Delay']).dropna().to_numpy()

        if len(path_vals) == 0 and len(delay_vals) == 0:
            print(f"[{idx}] no valid numeric rows after filtering. Skipping.")
            continue

        label = f"Np={idx}"
        path_by_idx[label] = path_vals
        delay_by_idx[label] = delay_vals
        labels.append(label)

    if not labels:
        print("No valid data to plot for CDF.")
        return

    def _ecdf(values: np.ndarray):
        v = np.sort(values)
        y = np.arange(1, len(v) + 1) / len(v)
        return v, y

    # Path Length CDF
    if any(len(path_by_idx[l]) > 0 for l in labels):
        fig, ax = plt.subplots()
        for lab in labels:
            vals = path_by_idx.get(lab, None)
            if vals is None or len(vals) == 0:
                continue
            x, y = _ecdf(vals)
            ax.step(x, y, where='post', label=lab)

        ax.set_xlabel("Path Length (hops)")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        _apply_tick_style(ax, x_minor_div=5, y_minor_div=0)
        ax.grid(True)

        ncol = min(6, max(1, len(labels)))
        fig.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=ncol, frameon=True)
        plt.show()
    else:
        print("No Path Length data for CDF.")

    # Delay CDF
    if any(len(delay_by_idx[l]) > 0 for l in labels):
        fig, ax = plt.subplots()
        for lab in labels:
            vals = delay_by_idx.get(lab, None)
            if vals is None or len(vals) == 0:
                continue
            x, y = _ecdf(vals)
            ax.step(x, y, where='post', label=lab)

        ax.set_xlabel("End-to-End Delay")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        _apply_tick_style(ax, x_minor_div=5, y_minor_div=0)
        ax.grid(True)

        ncol = min(5, max(1, len(labels)))
        fig.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=ncol, frameon=True)
        plt.show()
    else:
        print("No Delay data for CDF.")

def _place_legend(ax, ncol=3, legend_order=None, reserve_top=0.85):
    """
    reserve_top: 0~1 사이 figure 비율. 이 값만큼을 subplot 영역 상한으로 잡아,
                 위쪽(1.0 ~ reserve_top) 구간을 legend 전용 공간으로 예약합니다.
                 예) 0.85면 위쪽 15%는 legend 영역.
    """
    fig = ax.figure
    handles, labels = ax.get_legend_handles_labels()

    # 지정한 순서대로 정렬
    if legend_order:
        rank = {lab: i for i, lab in enumerate(legend_order)}
        order = sorted(range(len(labels)), key=lambda i: rank.get(labels[i], 10**9))
        handles = [handles[i] for i in order]
        labels  = [labels[i]  for i in order]

    # 1) 먼저 subplot들을 예약영역 내에 배치 (legend 공간 비워둠)
    fig.tight_layout(rect=[0, 0, 1, reserve_top])

    # 2) 비워둔 위쪽 공간(figure 좌표) 중앙에 legend 배치
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=ncol, frameon=True)

def analyze_files_qos(base_name, indices, directory='.',
                  show_min_max=False,
                  legend_order=None,
                  reserve_top=0.85):  # legend 전용 상단 공간 비율
    # QoS별 통계 저장
    stats = {
        qlabel: {
            "path_length_avg": [],
            "path_length_min": [],
            "path_length_max": [],
            "delay_avg": [],
            "delay_min": [],
            "delay_max": [],
            "drop_prob": []
        } for qlabel in QOS_LABELS.values()
    }

    x_ticks = []
    required_columns = {
        'Path Length', 'Queuing Delay', 'Propagation Delay',
        'Transmission Delay', 'Status', 'QoS'
    }

    for idx in indices:
        filename = os.path.join(directory, f"{base_name}{idx}.csv")
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue

        df = pd.read_csv(filename)

        if not required_columns.issubset(df.columns):
            print(f"File {filename} is missing required columns. Skipping.")
            continue

        x_ticks.append(idx)

        for q in QOS_ORDER:
            label = QOS_LABELS[q]
            df_q = df[df['QoS'] == q]

            if df_q.empty:
                for k in ["path_length_avg","path_length_min","path_length_max",
                          "delay_avg","delay_min","delay_max","drop_prob"]:
                    stats[label][k].append(np.nan)
                continue

            success_df = df_q[df_q['Status'] == 'success']
            total_cnt = len(df_q)
            success_ratio = len(success_df) / total_cnt if total_cnt > 0 else 0.0
            drop_prob = 1.0 - success_ratio

            if not success_df.empty:
                path_length = success_df['Path Length'] - 1
                total_delay = (success_df['Queuing Delay']
                               + success_df['Propagation Delay']
                               + success_df['Transmission Delay'])

                stats[label]["path_length_avg"].append(path_length.mean())
                stats[label]["path_length_min"].append(path_length.min())
                stats[label]["path_length_max"].append(path_length.max())

                stats[label]["delay_avg"].append(total_delay.mean())
                stats[label]["delay_min"].append(total_delay.min())
                stats[label]["delay_max"].append(total_delay.max())
            else:
                stats[label]["path_length_avg"].append(np.nan)
                stats[label]["path_length_min"].append(np.nan)
                stats[label]["path_length_max"].append(np.nan)
                stats[label]["delay_avg"].append(np.nan)
                stats[label]["delay_min"].append(np.nan)
                stats[label]["delay_max"].append(np.nan)

            stats[label]["drop_prob"].append(drop_prob)

    if not x_ticks:
        print("No valid files to plot.")
        return

    # legend 열 개수: show_min_max=True면 라벨이 많으니 3열이 보기 좋음
    legend_ncol = 3

    # ----------------------
    # Plot 1: Path Length (제목 없음)
    # ----------------------
    fig, ax = plt.subplots()
    avg_lines = {}
    for q in QOS_ORDER:
        label = QOS_LABELS[q]
        (line,) = ax.plot(
            x_ticks, stats[label]["path_length_avg"],
            marker=QOS_MARKERS[q], color=QOS_COLORS[q], label=f"{label} (avg)"
        )
        avg_lines[label] = line

    if show_min_max:
        for q in QOS_ORDER:
            label = QOS_LABELS[q]
            color = avg_lines[label].get_color()
            ax.plot(x_ticks, stats[label]["path_length_min"], linestyle='--', alpha=0.7,
                    label=f"{label} (min)", color=color)
            ax.plot(x_ticks, stats[label]["path_length_max"], linestyle=':',  alpha=0.7,
                    label=f"{label} (max)", color=color)

    ax.set_xlabel(r"$N_{p}$")
    ax.set_ylabel("Path Length")
    ax.set_xticks(x_ticks)
    ax.set_xlim(min(x_ticks), max(x_ticks))
    # ax.set_ylim(bottom=0, top=30)
    ax.margins(y=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    _apply_tick_style(ax, x_minor_div=0, y_minor_div=5)
    ax.grid(True)

    _place_legend(ax, ncol=legend_ncol, legend_order=legend_order, reserve_top=reserve_top)
    plt.show()

    # ----------------------
    # Plot 2: End-to-End Delay (제목 없음)
    # ----------------------
    fig, ax = plt.subplots()
    avg_lines = {}
    for q in QOS_ORDER:
        label = QOS_LABELS[q]
        (line,) = ax.plot(
            x_ticks, stats[label]["delay_avg"],
            marker=QOS_MARKERS[q], color=QOS_COLORS[q], label=f"{label} (avg)"
        )
        avg_lines[label] = line

    if show_min_max:
        for q in QOS_ORDER:
            label = QOS_LABELS[q]
            color = avg_lines[label].get_color()
            ax.plot(x_ticks, stats[label]["delay_min"], linestyle='--', alpha=0.7,
                    label=f"{label} (min)", color=color)
            ax.plot(x_ticks, stats[label]["delay_max"], linestyle=':',  alpha=0.7,
                    label=f"{label} (max)", color=color)

    ax.set_xlabel(r"$N_{p}$")
    ax.set_ylabel("Delay")
    ax.set_xticks(x_ticks)
    ax.set_xlim(min(x_ticks), max(x_ticks))
    # ax.set_ylim(0, 200)
    ax.margins(y=0)

    _apply_tick_style(ax, x_minor_div=0, y_minor_div=5)
    ax.grid(True)

    _place_legend(ax, ncol=legend_ncol, legend_order=legend_order, reserve_top=reserve_top)
    plt.show()

    # ----------------------
    # Plot 3: Drop Probability (제목 없음)
    # ----------------------
    fig, ax = plt.subplots()
    for q in QOS_ORDER:
        label = QOS_LABELS[q]
        ax.plot(
            x_ticks, stats[label]["drop_prob"],
            marker=QOS_MARKERS[q], color=QOS_COLORS[q], label=f"{label}"
        )

    ax.set_xlabel(r"$N_{p}$")
    ax.set_ylabel("Probability")
    ax.set_xticks(x_ticks)
    ax.set_xlim(min(x_ticks), max(x_ticks))
    ax.set_ylim(bottom=0)
    _apply_tick_style(ax, x_minor_div=0, y_minor_div=0)
    ax.grid(True)

    _place_legend(ax, ncol=legend_ncol, legend_order=legend_order, reserve_top=reserve_top)
    plt.show()
def analyze_files_overall(base_name, indices, directory='.', target_range=(200, 600)):
    """
    각 파일별로 (Time(ms) in [lo, hi)) 범위의 레코드만 사용하여
    - Path Length (success만)
    - e2e delay (success만)
    - Drop Probability (전체 대비 success 비율로 계산)
    를 집계/시각화합니다.
    """
    stats = {
        "index": [],
        "path_length_avg": [],
        "path_length_min": [],
        "path_length_max": [],
        "delay_avg": [],
        "delay_min": [],
        "delay_max": [],
        "drop_prob": []
    }

    required_columns = {'Path Length', 'e2e delay', 'Status', 'Time (ms)'}
    lo, hi = target_range

    for idx in indices:
        filename = os.path.join(directory, f"{base_name}{idx}.csv")
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue

        df = pd.read_csv(filename)
        df.columns = [c.strip().replace("\ufeff","") for c in df.columns]

        if not required_columns.issubset(df.columns):
            print(f"File {filename} is missing required columns. Skipping.")
            continue

        # 시간 필터
        df['Time (ms)'] = pd.to_numeric(df['Time (ms)'], errors='coerce')
        df = df[(df['Time (ms)'] >= lo) & (df['Time (ms)'] < hi)]
        if df.empty:
            print(f"[{idx}] no rows in Time(ms) range {target_range}. Skipping.")
            continue

        # Drop prob 은 전체 분모에 '범위 내 전체 행' 사용
        success_df = df[df['Status'] == 'success']
        total_len = len(df)
        success_ratio = len(success_df) / total_len if total_len > 0 else 0

        stats["index"].append(idx)

        if not success_df.empty:
            # 숫자 변환
            success_df['Path Length'] = pd.to_numeric(success_df['Path Length'], errors='coerce')
            success_df['e2e delay']   = pd.to_numeric(success_df['e2e delay'],   errors='coerce')

            path_length = (success_df['Path Length'] - 1).dropna()
            total_delay = success_df['e2e delay'].dropna()

            stats["path_length_avg"].append(path_length.mean() if not path_length.empty else np.nan)
            stats["path_length_min"].append(path_length.min()  if not path_length.empty else np.nan)
            stats["path_length_max"].append(path_length.max()  if not path_length.empty else np.nan)

            stats["delay_avg"].append(total_delay.mean() if not total_delay.empty else np.nan)
            stats["delay_min"].append(total_delay.min()  if not total_delay.empty else np.nan)
            stats["delay_max"].append(total_delay.max()  if not total_delay.empty else np.nan)
        else:
            stats["path_length_avg"].append(float('nan'))
            stats["path_length_min"].append(float('nan'))
            stats["path_length_max"].append(float('nan'))
            stats["delay_avg"].append(float('nan'))
            stats["delay_min"].append(float('nan'))
            stats["delay_max"].append(float('nan'))

        stats["drop_prob"].append(1 - success_ratio)

    x_ticks = stats["index"]
    if not x_ticks:
        print("No valid files to plot.")
        return

    # Path Length
    plt.figure()
    plt.plot(x_ticks, stats["path_length_avg"], marker='o', color='orange', label='Average')
    plt.plot(x_ticks, stats["path_length_min"], marker='^', linestyle='--', color='green', label='Min')
    # plt.plot(x_ticks, stats["path_length_max"], marker='v', linestyle='--', color='red', label='Max')
    plt.title("Path Length")
    plt.xlabel(r"$N_{p}$")
    plt.ylabel("Path Length")
    plt.xticks(x_ticks)
    plt.xlim(min(x_ticks), max(x_ticks))
    plt.legend(loc='center right')
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # End-to-End Delay
    plt.figure()
    plt.plot(x_ticks, stats["delay_avg"], marker='o', color='orange', label='Average')
    plt.plot(x_ticks, stats["delay_min"], marker='^', linestyle='--', color='green', label='Min')
    plt.plot(x_ticks, stats["delay_max"], marker='v', linestyle='--', color='red', label='Max')
    plt.xlabel(r"Packet Arrival Rate (Mbps)")
    plt.ylabel("End-to-End Delay (ms)")
    plt.xticks(x_ticks)
    plt.xlim(min(x_ticks), max(x_ticks))
    plt.legend(loc='center right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Drop Probability
    plt.figure()
    plt.plot(x_ticks, stats["drop_prob"], marker='o', color='red')
    plt.xlabel(r"Packet Arrival Rate (Mbps)")
    plt.ylabel("Drop Probability")
    plt.xticks(x_ticks)
    plt.xlim(min(x_ticks), max(x_ticks))
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_delay_components_bar(base_name, indices, directory='.',
                                 agg='mean',        # 'mean' 또는 'median'
                                 reserve_top=0.85,  # 범례를 위 바깥으로 빼기 위한 여백
                                 bar_width=0.25,
                                 target_range=(200, 600)):
    """
    각 Np(파일 인덱스)별로 success 행만 집계하여
    Queuing / Propagation / Transmission 지연의 집계값(평균/중앙값)을
    '3개 막대'로 묶은 그룹드 바 차트로 그립니다.

    target_range: (lo, hi)로 해석하며 Time (ms) 가 lo 이상, hi 미만인 행만 포함.
    """
    required = {'Queuing Delay', 'Propagation Delay', 'Transmission Delay', 'Status', 'Time (ms)'}

    x_labels = []
    q_vals, p_vals, t_vals = [], [], []

    lo, hi = target_range

    for idx in indices:
        filename = os.path.join(directory, f"{base_name}{idx}.csv")
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue

        df = pd.read_csv(filename)
        df.columns = [c.strip().replace("\ufeff","") for c in df.columns]

        if not required.issubset(df.columns):
            print(f"File {filename} is missing required columns {required - set(df.columns)}. Skipping.")
            continue

        # 시간 필터
        df['Time (ms)'] = pd.to_numeric(df['Time (ms)'], errors='coerce')
        df = df[(df['Time (ms)'] >= lo) & (df['Time (ms)'] < hi)]
        if df.empty:
            print(f"[{idx}] no rows in Time(ms) range {target_range}. Skipping.")
            continue

        df_ok = df[df['Status'] == 'success']
        if df_ok.empty:
            print(f"[{idx}] no success rows in Time(ms) range {target_range}. Skipping.")
            continue

        # 숫자 변환
        for c in ['Queuing Delay', 'Propagation Delay', 'Transmission Delay']:
            df_ok[c] = pd.to_numeric(df_ok[c], errors='coerce')

        if agg == 'median':
            q = df_ok['Queuing Delay'].median()
            p = df_ok['Propagation Delay'].median()
            t = df_ok['Transmission Delay'].median()
        else:  # default mean
            q = df_ok['Queuing Delay'].mean()
            p = df_ok['Propagation Delay'].mean()
            t = df_ok['Transmission Delay'].mean()

        x_labels.append(idx)
        q_vals.append(q)
        p_vals.append(p)
        t_vals.append(t)

    if not x_labels:
        print("No valid data to plot.")
        return

    x = np.arange(len(x_labels))
    fig, ax = plt.subplots()

    ax.bar(x - bar_width, q_vals, width=bar_width, label='Queuing')
    ax.bar(x,             p_vals, width=bar_width, label='Propagation')
    ax.bar(x + bar_width, t_vals, width=bar_width, label='Transmission')

    ax.set_xlabel(r"Packet Arrival Rate (Mbps)")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_labels])
    ax.set_ylim(bottom=0, top=60)
    ax.margins(y=0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=True)
    plt.show()

def analyze_files_overall_boxplot(base_name, indices, directory='.',
                                  target_range=(200, 600),
                                  plot_path_length=True,
                                  plot_e2e_delay=True,
                                  whis=(5, 95),          # 상하위 백분위로 수염 길이 설정
                                  notch=False,
                                  showfliers=False,      # 아웃라이어 점 표시 여부
                                  widths=0.6,
                                  reserve_top=0.90,      # 위쪽 여백(범례 대신 여백만)
                                  annotate_counts=True   # 각 상자 위에 표본 수 N 표시
                                  ):
    """
    각 파일(=Np/arrival rate)에서 Time(ms) ∈ [lo, hi) & Status=='success' 인 행만 뽑아
    - Path Length(= 'Path Length' - 1) 분포
    - e2e delay(= 'e2e delay') 분포
    를 rate 별 박스플롯으로 그립니다.

    Parameters
    ----------
    base_name : str
        파일 접두어 (예: 'limited_Q_with_GSL_')
    indices : iterable[int]
        파일 인덱스들 (예: (40, 80, 120, ...))
    directory : str
        CSV 폴더 경로
    target_range : (lo, hi)
        Time (ms) 가 lo 이상, hi 미만인 행만 포함
    plot_path_length : bool
        Path Length 박스플롯 표시 여부
    plot_e2e_delay : bool
        e2e delay 박스플롯 표시 여부
    whis : tuple[int,int] | float | str
        matplotlib boxplot의 whis 인자 (수염 범위)
    showfliers : bool
        아웃라이어 표시 여부
    """

    # 어떤 지표를 그릴지에 따라 필요한 컬럼 구성
    base_required = {'Status', 'Time (ms)'}
    pl_required   = {'Path Length'}
    dly_required  = {'e2e delay'}

    # rate별 데이터 컨테이너 (지표별로 독립적으로 모음: 비어있는 rate는 스킵)
    pl_labels,  pl_data,  pl_counts  = [], [], []
    dly_labels, dly_data, dly_counts = [], [], []

    lo, hi = target_range

    for idx in indices:
        filename = os.path.join(directory, f"{base_name}{idx}.csv")
        if not os.path.exists(filename):
            print(f"[WARN] File {filename} not found. Skipping.")
            continue

        df = pd.read_csv(filename, low_memory=False)
        # BOM/공백 제거
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

        # 공통/필요 컬럼 체크
        needed = set(base_required)
        if plot_path_length: needed |= pl_required
        if plot_e2e_delay:   needed |= dly_required

        if not needed.issubset(df.columns):
            missing = needed - set(df.columns)
            print(f"[WARN] File {filename} missing columns: {missing}. Skipping.")
            continue

        # Time 필터
        df['Time (ms)'] = pd.to_numeric(df['Time (ms)'], errors='coerce')
        df = df[(df['Time (ms)'] >= lo) & (df['Time (ms)'] < hi)]
        if df.empty:
            print(f"[INFO] [{idx}] no rows in Time(ms) range {target_range}. Skipping.")
            continue

        # success만
        df_ok = df[df['Status'].astype(str).str.strip().str.lower() == 'success']
        if df_ok.empty:
            print(f"[INFO] [{idx}] no success rows in range {target_range}. Skipping.")
            continue

        # -------- Path Length 분포 --------
        if plot_path_length:
            s = pd.to_numeric(df_ok['Path Length'], errors='coerce') - 1
            s = s.dropna()
            if len(s) > 0:
                pl_labels.append(idx)
                pl_data.append(s.to_numpy())
                pl_counts.append(len(s))

        # -------- e2e delay 분포 --------
        if plot_e2e_delay:
            s = pd.to_numeric(df_ok['e2e delay'], errors='coerce').dropna()
            if len(s) > 0:
                dly_labels.append(idx)
                dly_data.append(s.to_numpy())
                dly_counts.append(len(s))

    # -------- 그리기: Path Length --------
    if plot_path_length and pl_labels:
        x = np.arange(len(pl_labels))
        fig, ax = plt.subplots()

        bp = ax.boxplot(pl_data, positions=x, widths=widths, whis=whis,
                        notch=notch, showfliers=showfliers, patch_artist=True)

        # (선택) 채색/스타일 약간 가독성
        for patch in bp['boxes']:
            patch.set(facecolor='#dddddd', edgecolor='black', alpha=0.8)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set(color='black')

        ax.set_xlabel(r"Packet Arrival Rate (Mbps)")
        ax.set_ylabel("Path Length (hops)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in pl_labels])
        ax.margins(y=0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        # _apply_tick_style(ax, x_minor_div=0, y_minor_div=5)

        # 표본 수 N 주석
        if annotate_counts:
            ymax = max([np.nanmax(v) if len(v) else np.nan for v in pl_data])
            ytxt = ymax + (0.03 * (ymax if np.isfinite(ymax) else 1))
            for xi, n in zip(x, pl_counts):
                ax.text(xi, ytxt, f"N={n}", ha='center', va='bottom', fontsize=9)

        fig.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
        plt.show()
    elif plot_path_length:
        print("[INFO] No Path Length data for boxplot.")

    # -------- 그리기: e2e delay --------
    if plot_e2e_delay and dly_labels:
        x = np.arange(len(dly_labels))
        fig, ax = plt.subplots()

        bp = ax.boxplot(dly_data, positions=x, widths=widths, whis=whis,
                        notch=notch, showfliers=showfliers, patch_artist=True)

        for patch in bp['boxes']:
            patch.set(facecolor='#dddddd', edgecolor='black', alpha=0.8)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set(color='black')

        ax.set_xlabel(r"Packet Arrival Rate (Mbps)")
        ax.set_ylabel("End-to-End Delay (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in dly_labels])
        ax.margins(y=0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        # _apply_tick_style(ax, x_minor_div=0, y_minor_div=5)

        if annotate_counts:
            ymax = max([np.nanmax(v) if len(v) else np.nan for v in dly_data])
            ytxt = ymax + (0.03 * (ymax if np.isfinite(ymax) else 1))
            for xi, n in zip(x, dly_counts):
                ax.text(xi, ytxt, f"N={n}", ha='center', va='bottom', fontsize=9)

        fig.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
        plt.show()
    elif plot_e2e_delay:
        print("[INFO] No e2e delay data for boxplot.")

# ================== 멀티 디렉토리 지원: overall 캐시 + 통합 플로팅 ==================

def compute_or_load_overall_csv(base_name, indices, directory='.',
                                target_range=(200, 800),
                                cache_name='overall.csv',
                                force_recompute=False):
    """
    directory 내부에 cache_name(overall.csv)이 있으면 읽고, 없거나 force_recompute=True면 계산해서 저장.
    반환: pd.DataFrame(columns=['index','path_length_avg','path_length_min','path_length_max',
                               'delay_avg','delay_min','delay_max','drop_prob'])
    """
    cache_path = Path(directory) / cache_name
    if cache_path.exists() and not force_recompute:
        try:
            df_cached = pd.read_csv(cache_path)
            needed = {'index','path_length_avg','path_length_min','path_length_max',
                      'delay_avg','delay_min','delay_max','drop_prob'}
            if needed.issubset(df_cached.columns):
                return df_cached
            else:
                print(f"[WARN] {cache_path} 컬럼이 부족하여 재계산합니다.")
        except Exception as e:
            print(f"[WARN] {cache_path} 읽기 실패({e}). 재계산합니다.")

    # ---- 없으면 계산 (기존 analyze_files_overall 계산 로직 재사용) ----
    required_columns = {'Path Length', 'e2e delay', 'Status', 'Time (ms)'}
    if target_range is None:
        # 전체 파일에서 time 범위 자동 결정
        lo = None
        hi = None
    else:
        lo, hi = target_range
    stats = {
        "index": [], "path_length_avg": [], "path_length_min": [], "path_length_max": [],
        "delay_avg": [], "delay_min": [], "delay_max": [], "drop_prob": [], "generated": [], "success": [],"throughput": []
    }

    for idx in indices:
        filename = os.path.join(directory, f"{base_name}{idx}.csv")
        if not os.path.exists(filename):
            print(f"[SKIP] File {filename} not found.")
            continue

        df = pd.read_csv(filename, low_memory=False)
        df.columns = [c.strip().replace("\ufeff","") for c in df.columns]
        if not required_columns.issubset(df.columns):
            print(f"[SKIP] {filename} 컬럼 부족: {required_columns - set(df.columns)}")
            continue

        # 시간 필터
        df['Time (ms)'] = pd.to_numeric(df['Time (ms)'], errors='coerce')
        if target_range is None:
            # time이 0인 시점부터 마지막까지
            lo = df['Time (ms)'].min()
            # 마지막 row의 time + e2e delay
            last_row = df.iloc[-1]
            last_time = last_row['Time (ms)']
            last_delay = last_row['e2e delay'] if 'e2e delay' in last_row else 0
            hi = last_time + last_delay
        df = df[(df['Time (ms)'] >= lo) & (df['Time (ms)'] < hi)]
        if df.empty:
            print(f"[{idx}] no rows in Time(ms) range ({lo}, {hi}). Skipping.")
            continue

        success_df = df[df['Status'] == 'success']
        total_len = len(df)

        success_ratio = len(success_df) / total_len if total_len > 0 else 0.0

        stats["index"].append(idx)
        if not success_df.empty:
            success_df['Path Length'] = pd.to_numeric(success_df['Path Length'], errors='coerce')
            success_df['e2e delay']   = pd.to_numeric(success_df['e2e delay'],   errors='coerce')

            path_length = (success_df['Path Length'] - 1).dropna()
            total_delay = success_df['e2e delay'].dropna()

            stats["path_length_avg"].append(path_length.mean() if not path_length.empty else np.nan)
            stats["path_length_min"].append(path_length.min()  if not path_length.empty else np.nan)
            stats["path_length_max"].append(path_length.max()  if not path_length.empty else np.nan)

            stats["delay_avg"].append(total_delay.mean() if not total_delay.empty else np.nan)
            stats["delay_min"].append(total_delay.min()  if not total_delay.empty else np.nan)
            stats["delay_max"].append(total_delay.max()  if not total_delay.empty else np.nan)
        else:
            stats["path_length_avg"].append(np.nan)
            stats["path_length_min"].append(np.nan)
            stats["path_length_max"].append(np.nan)
            stats["delay_avg"].append(np.nan)
            stats["delay_min"].append(np.nan)
            stats["delay_max"].append(np.nan)

        stats["drop_prob"].append(1 - success_ratio)
        stats["throughput"].append(((len(success_df)/(hi-lo))*1000)/GIGA*PACKET_SIZE_BITS)
        stats['generated'].append(total_len)
        stats['success'].append(len(success_df))

    df_out = pd.DataFrame(stats)
    # index 정렬(있을 경우)
    if not df_out.empty:
        df_out = df_out.sort_values('index').reset_index(drop=True)

    # 캐시 저장
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(cache_path, index=False, encoding='utf-8-sig')
    return df_out


def _plot_overall_from_many(
    dir_to_df, dir_names,
    legend_in_one=True,
    reserve_top=0.90,
    show_minmax=True
):
    """
    dir_to_df: {dir_name: DataFrame(overall.csv)}
    dir_names: 범례/색상 순서 제어용 리스트 (colors는 이 순서로 배정)
    legend_in_one: True면 라벨을 "DIR (avg)" 형태로 하나의 범례로 처리.
                   False면 '색=디렉토리', '스타일=통계치' 두 개의 범례를 분리(고급).
    show_minmax: True면 min/max도 함께 표시, False면 avg만 표시
    """
    if not dir_to_df:
        print("No dataframes to plot.")
        return

    # 디렉토리별 색 배정
    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])
    colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(dir_names)}

    # 통계치별 스타일 프리셋
    STYLES = {
        'avg': dict(linestyle='-',  marker='o'),
        'min': dict(linestyle='--', marker='^'),
        'max': dict(linestyle=':',  marker='v'),
    }

    # ===== Plot 1: Path Length =====
    fig1, ax1 = plt.subplots()
    for name in dir_names:
        df = dir_to_df.get(name)
        if df is None or df.empty:
            continue
        x = df['index'].to_numpy()
        stat_keys = [('path_length_avg','avg')]
        if show_minmax:
            stat_keys += [('path_length_min','min'), ('path_length_max','max')]
        for stat_key, label_suffix in stat_keys:
            if stat_key not in df.columns:
                continue
            y = df[stat_key].to_numpy()
            style = STYLES[label_suffix]
            label = f"{name} ({label_suffix})" if legend_in_one else None
            ax1.plot(x, y, color=colors[name], label=label, **style)

    ax1.set_xlabel(r"Packet Arrival Rate (Mbps)$")
    ax1.set_ylabel("Path Length (hops)")
    ax1.set_xticks(sorted(np.unique(np.concatenate([df['index'].to_numpy() for df in dir_to_df.values() if not df.empty]))))
    ax1.grid(True)
    _apply_tick_style(ax1, x_minor_div=0, y_minor_div=5)

    if legend_in_one:
        fig1.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
        ax1.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.99), frameon=True)
    else:
        color_handles = [plt.Line2D([0],[0], color=colors[name], lw=2, label=name) for name in dir_names]
        style_labels = ['avg']
        if show_minmax:
            style_labels += ['min','max']
        style_handles = [plt.Line2D([0],[0], color='k', lw=2, **STYLES[k], label=k) for k in style_labels]
        fig1.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
        leg1 = fig1.legend(handles=color_handles, loc='upper left', bbox_to_anchor=(0.15, 0.99), ncol=2, frameon=True, title="Directory")
        leg2 = fig1.legend(handles=style_handles, loc='upper right', bbox_to_anchor=(0.85, 0.99), ncol=3, frameon=True, title="Statistic")
        ax1.add_artist(leg1)

    plt.show()

    # ===== Plot 2: End-to-End Delay =====
    fig2, ax2 = plt.subplots()
    for name in dir_names:
        df = dir_to_df.get(name)
        if df is None or df.empty:
            continue
        x = df['index'].to_numpy()
        stat_keys = [('delay_avg','avg')]
        if show_minmax:
            stat_keys += [('delay_min','min'), ('delay_max','max')]
        for stat_key, label_suffix in stat_keys:
            if stat_key not in df.columns:
                continue
            y = df[stat_key].to_numpy()
            style = STYLES[label_suffix]
            label = f"{name} ({label_suffix})" if legend_in_one else None
            ax2.plot(x, y, color=colors[name], label=label, **style)

    ax2.set_xlabel(r"Packet Arrival Rate (Mbps)")
    ax2.set_ylabel("End-to-End Delay (ms)")
    ax2.set_xticks(sorted(np.unique(np.concatenate([df['index'].to_numpy() for df in dir_to_df.values() if not df.empty]))))
    ax2.grid(True)
    _apply_tick_style(ax2, x_minor_div=0, y_minor_div=5)

    if legend_in_one:
        fig2.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
        ax2.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.99), frameon=True)
    else:
        color_handles = [plt.Line2D([0],[0], color=colors[name], lw=2, label=name) for name in dir_names]
        style_labels = ['avg']
        if show_minmax:
            style_labels += ['min','max']
        style_handles = [plt.Line2D([0],[0], color='k', lw=2, **STYLES[k], label=k) for k in style_labels]
        fig2.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
        leg1 = fig2.legend(handles=color_handles, loc='upper left', bbox_to_anchor=(0.15, 0.99), ncol=2, frameon=True, title="Directory")
        leg2 = fig2.legend(handles=style_handles, loc='upper right', bbox_to_anchor=(0.85, 0.99), ncol=3, frameon=True, title="Statistic")
        ax2.add_artist(leg1)

    plt.show()

    # ===== Plot 3: Drop Probability =====
    fig3, ax3 = plt.subplots()
    for name in dir_names:
        df = dir_to_df.get(name)
        if df is None or df.empty:
            continue
        x = df['index'].to_numpy()
        y = df['drop_prob'].to_numpy() if 'drop_prob' in df.columns else None
        if y is None:
            continue
        ax3.plot(x, y, color=colors[name], linestyle='-', marker='o', label=name)

    ax3.set_xlabel(r"Packet Arrival Rate (Mbps)")
    ax3.set_ylabel("Drop Probability")
    ax3.set_xticks(sorted(np.unique(np.concatenate([df['index'].to_numpy() for df in dir_to_df.values() if not df.empty]))))
    ax3.set_ylim(bottom=0)
    ax3.grid(True)
    _apply_tick_style(ax3, x_minor_div=0, y_minor_div=0)

    fig3.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
    ax3.legend(ncol=min(4, len(dir_names)), loc='upper center', bbox_to_anchor=(0.5, 0.99), frameon=True, title="Directory")
    plt.show()

    # ===== Plot 4: Throughput =====
    fig4, ax4 = plt.subplots()
    for name in dir_names:
        df = dir_to_df.get(name)
        if df is None or df.empty:
            continue
        x = df['index'].to_numpy()
        y = df['throughput'].to_numpy() if 'throughput' in df.columns else None
        if y is None:
            continue
        ax4.plot(x, y, color=colors[name], linestyle='-', marker='o', label=name)

    ax4.set_xlabel(r"Packet Arrival Rate (Mbps)")
    ax4.set_ylabel("Throughput (Gbps)")
    ax4.set_xticks(
        sorted(np.unique(np.concatenate([df['index'].to_numpy() for df in dir_to_df.values() if not df.empty]))))
    # ax4.set_ylim(bottom=50, top=200)
    ax4.grid(True)
    _apply_tick_style(ax4, x_minor_div=0, y_minor_div=0)

    fig4.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
    ax4.legend(ncol=min(4, len(dir_names)), loc='upper center', bbox_to_anchor=(0.5, 0.99), frameon=True,
               title="Directory")
    plt.show()


def analyze_files_overall_multi(base_name,
                                indices,
                                directories,   # 경로 리스트
                                dir_names,     # 같은 길이의 라벨 리스트
                                target_range=(200, 600),
                                cache_name='overall.csv',
                                force_recompute=False,
                                legend_in_one=True,
                                show_minmax=True):

    """
    여러 디렉토리에 대해:
      1) 각 디렉토리에서 overall.csv가 없으면 계산 후 저장(있으면 스킵)
      2) 각 디렉토리의 overall.csv를 읽어와
      3) 한 그래프 위에 디렉토리별로 비교(색=디렉토리, 스타일=통계치)

    directories, dir_names 길이는 같아야 함.
    """
    assert len(directories) == len(dir_names), "directories와 dir_names의 길이가 같아야 합니다."

    dir_to_df = {}
    for d, name in zip(directories, dir_names):
        df_overall = compute_or_load_overall_csv(
            base_name=base_name,
            indices=indices,
            directory=d,
            target_range=target_range,
            cache_name=cache_name,
            force_recompute=force_recompute
        )
        if df_overall is None or df_overall.empty:
            print(f"[INFO] {name}: 사용할 데이터가 없습니다.")
            continue
        dir_to_df[name] = df_overall

    _plot_overall_from_many(dir_to_df, dir_names=dir_names, legend_in_one=legend_in_one, show_minmax=show_minmax)


def _is_ground_id(x) -> bool:
    if pd.isna(x):
        return False
    return bool(_GROUND_RE.match(str(x).strip()))

def _extract_nodes_from_result(result_cell: str):
    """
    result 문자열에서 '지상노드(영문자-숫자)' 또는 '숫자' 토큰을 추출해 리스트로 반환.
    예: "[12, 'GR-1', 34]" -> ["12", "GR-1", "34"]
    """
    if pd.isna(result_cell):
        return []
    s = str(result_cell)
    return re.findall(r'[A-Za-z]+-\d+|\d+', s)

def _get_detour_flag(row) -> bool:
    """
    Detour mode는 항상 존재한다고 하셨으므로 그것만 사용.
    True/False/문자/숫자 모두 허용.
    """
    val = row.get('Detour mode', None)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    sval = str(val).strip().lower()
    return sval in ('true', 't', '1', 'yes', 'y')

def _classify_row(row) -> str:
    """
    새 분류 흐름:
    1) Drop Direction이 지상노드 => 'before g'
    2) else:
         - result 내 지상노드가 있고, result[-2]가 지상노드 => 'after g'
         - else Detour mode True => 'detouring'
         - else => 's to s'
    """
    drop_dir = row.get('Drop Direction', None)
    if _is_ground_id(drop_dir):
        return 'before g'

    tokens = _extract_nodes_from_result(row.get('result', None))
    if len(tokens) >= 2 and _is_ground_id(tokens[-2]):
        return 'after g'

    return 'detouring' if _get_detour_flag(row) else 's to s'
def _classify_row_status(row) -> str:
    """
    새 분류 흐름:
    1) Drop Direction이 지상노드 => 'before g'
    2) else:
         - result 내 지상노드가 있고, result[-2]가 지상노드 => 'after g'
         - else Detour mode True => 'detouring'
         - else => 's to s'
    """
    status = row.get('Status', None)
    if status == 'drop':
        return 'saturated'
    elif status == 'failure':
        return 'link fail'
    else: # status == 'expired'
        return 'expired'

# ---------- 메인 플로팅 ----------
def plot_drop_class_stacked_by_index(
    base_dir='.',
    indices=(40,80,120,160,200,240,280,320,360),
    filename_tpl='result_{idx}_filtered(success).csv',
    classes=('before g','after g','s to s','detouring'),  # other는 기본 제외
    colors=None,          # dict: class -> color
    figsize=(10,6),
    annotate_total=True,  # 막대 위에 총 건수 N 표시
    annotate_pct=True,    # 각 세그먼트 안에 % 표시(너무 얇으면 자동 생략)
    pct_threshold=0.06,   # 막대 대비 세그먼트 비율이 이 값 미만이면 %표기 생략
    edgecolor='black',
    save_path=None,
    show=True
):
    """
    각 index 파일의 'drop_class'를 현장에서 계산하고, index마다 총 건수 높이의 누적막대를 그림.
      - 누적 순서/색은 classes, colors로 제어
      - 높이: 각 클래스 카운트(총합=막대 높이)
      - 내부 라벨: 클래스 비율(%) (옵션)
    """
    # 색상 기본값
    if colors is None:
        colors = {
            'saturated': '#1f77b4',  # 파랑
            'expired': '#ff7f0e',  # 주황
            'link fail': '#2ca02c',  # 초록
            'before g': '#1f77b4',  # 파랑
            'after g' : '#ff7f0e',  # 주황
            's to s'  : '#2ca02c',  # 초록
            'detouring': '#d62728', # 빨강
            'other'   : '#7f7f7f',  # 회색(사용 안함)
        }

    xs      = []
    totals  = []
    stacks  = {cls: [] for cls in classes}

    for idx in indices:
        path = Path(base_dir) / filename_tpl.format(idx=idx)
        if not path.exists():
            print(f"[WARN] not found: {path}")
            continue

        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().replace("\ufeff","") for c in df.columns]

        # 분류
        df['drop_class'] = df.apply(_classify_row, axis=1)
        # df['drop_class'] = df.apply(_classify_row_status, axis=1)

        # 카운트(필요 클래스만)
        vc = df['drop_class'].value_counts()
        total = int(vc.sum())
        if total == 0:
            print(f"[INFO] [{idx}] no rows.")
            continue

        xs.append(idx)
        totals.append(total)
        for cls in classes:
            stacks[cls].append(int(vc.get(cls, 0)))

        # 참고: other가 생기면 알려주기(기본 막대에는 포함 안하지만 알림)
        if 'other' in vc and vc['other'] > 0:
            print(f"[NOTE] [{idx}] 'other' count = {int(vc['other'])} (not plotted)")

    if not xs:
        print("[INFO] nothing to plot.")
        return None, None

    # --- 그리기 ---
    x = np.arange(len(xs))
    fig, ax = plt.subplots(figsize=figsize)

    bottoms = np.zeros_like(x, dtype=float)
    for cls in classes:
        h = np.array(stacks[cls], dtype=float)
        ax.bar(x, h, bottom=bottoms, width=0.7, label=cls,
               color=colors.get(cls, '#cccccc'), edgecolor=edgecolor)
        # 퍼센트 라벨
        if annotate_pct:
            for xi, hh, bot, tot in zip(x, h, bottoms, totals):
                if tot <= 0 or hh <= 0: continue
                frac = hh / tot
                if frac >= pct_threshold:
                    ax.text(xi, bot + hh/2, f"{frac*100:.0f}%",
                            ha='center', va='center', fontsize=9, color='white')
        bottoms += h

    # 총 건수 라벨
    if annotate_total:
        for xi, tot in zip(x, totals):
            ax.text(xi, tot, f"N={tot}", ha='center', va='bottom', fontsize=9)

    # 축/격자/범례
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in xs])
    ax.set_xlabel("index (arrival rate)")
    ax.set_ylabel("count (total drops)")
    ax.set_ylim(0, max(totals) * 1.12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(title="class", ncol=min(4, len(classes)))

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

# 사용 예시
if __name__ == '__main__':
    indices = (40, 80, 120, 160, 200, 240, 280, 320, 360)
    # indices = (40, 80, 160, 200, 240, 280, 360)
    # indices_cdf = (40, 120, 200, 280, 360)
    # legend 원하는 순서 예시: 평균 먼저, 그 다음 min/max
    legend_order = [
        "High (max)", "High (avg)", "High (min)",
        "Middle (max)", "Middle (avg)", "Middle (min)",
        "Low (max)", "Low (avg)", "Low (min)"
    ]
    # # CDF (200~600ms 범위만)
    # analyze_files_cdf(
    #     base_name='limited_Q_with_GSL_',
    #     indices='./prop ver1.7.2_nosig_avghist',
    #     directory='.',
    #     target_range=(200, 600)
    # )
    #
    # Delay Components Bar (200~600ms 범위만)
    # analyze_delay_components_bar(
    #     base_name='result_',
    #     indices=(40, 80, 120, 160, 200, 240, 280, 320, 360),
    #     directory='./prop(251001)',
    #     agg='mean',
    #     reserve_top=0.84,
    #     target_range=(0, 1000)
    # )
    # analyze_delay_components_bar(
    #     base_name='result_',
    #     indices=(40, 80, 120, 160, 200, 240, 280, 320, 360),
    #     directory='./tmc(latest)',
    #     agg='mean',
    #     reserve_top=0.84,
    #     target_range=(0, 1000)
    # )

    # 비교 대상 디렉토리들
    directories = [
        r"./dijkstra(no detour)",
        r"./tmc(latest)",
        r"./prop(251001)",
    ]
    dir_names = [
        "dijkstra",
        "TMC",
        "prop",
    ]

    analyze_files_overall_multi(
        base_name='result_',
        indices=indices,
        # indices=[5,10,15,20],
        directories=directories,
        dir_names=dir_names,
        target_range=(0,1000),
        cache_name="overall.csv",   # 디렉토리별 캐시 파일명
        force_recompute=False,      # True면 캐시 무시하고 재계산
        legend_in_one=True,         # True: "DIR (avg)" 식 1개 범례 / False: 색-디렉토리, 스타일-통계치로 분리
        show_minmax=False
    )

    # plot_drop_class_stacked_by_index(
    #     base_dir='./prop_NCC 2',
    #     indices=(40, 80, 120, 160, 200, 240, 280, 320, 360),
    #     # indices=[5,10,15,20],
    #     filename_tpl='result_{idx}_filtered(success).csv',
    #     # classes 순서/색 바꾸고 싶으면 여기서 지정
    #     classes=('before g','after g','s to s','detouring'),
    #     # classes=('saturated', 'expired', 'link fail'),
    #     # colors={'before g':'#4e79a7','after g':'#f28e2b','s to s':'#59a14f','detouring':'#e15759'},
    #     annotate_total=True,
    #     annotate_pct=True,
    #     pct_threshold=0.06,  # 세그먼트가 6% 미만이면 %라벨 생략
    #     save_path=None, show=True
    # )
    # plot_drop_class_stacked_by_index(
    #     base_dir='./prop ver1.7.2_nosig_avghist',
    #     indices=(40, 80, 120, 160, 200, 240, 280, 320, 360),
    #     # indices=[5, 10, 15, 20],
    #     filename_tpl='result_{idx}_filtered(success).csv',
    #     # classes 순서/색 바꾸고 싶으면 여기서 지정
    #     classes=('before g', 'after g', 's to s', 'detouring'),
    #     # classes=('saturated', 'expired', 'link fail'),
    #     # colors={'before g':'#4e79a7','after g':'#f28e2b','s to s':'#59a14f','detouring':'#e15759'},
    #     annotate_total=True,
    #     annotate_pct=True,
    #     pct_threshold=0.06,  # 세그먼트가 6% 미만이면 %라벨 생략
    #     save_path=None, show=True
    # )


    # # Overall (200~600ms 범위만)
    # analyze_files_overall(
    #     base_name='result_',
    #     indices=(40, 80, 120, 160, 200, 240, 280, 320, 360),
    #     directory='./tmc data rate rollback',
    #     target_range=(200, 600)
    # )
    #
    # analyze_files_overall_boxplot(
    #     base_name='result_',
    #     indices=indices,
    #     directory='./prop_softassume',
    #     target_range=(0, 600),
    #     plot_path_length=True,
    #     plot_e2e_delay=True,
    #     whis=(5, 95),  # 수염을 5~95 백분위로
    #     showfliers=True,  # 아웃라이어 숨김
    #     annotate_counts=True  # 각 박스 위에 표본 수 표시
    # )
    #
    # # 생성된 시간에 따른 드롭율, 딜레이
    # plot_arrival_rate_avg_over_start_at('./prop_softassume', indices, start_range=(0, 600), metric="drop_rate")
    # plot_arrival_rate_avg_over_start_at('./prop_softassume', indices, start_range=(0, 600), metric="e2e_delay")


    # 전체 QoS 합산
    # analyze_files_cdf(
    #     base_name='infinite_Q_with_GSL_',
    #     indices=indices_cdf,
    #     qos=None,  # 전체 QoS 합산
    #     reserve_top=0.95  # legend 공간. 필요시 0.78~0.90 조정
    # )

    # analyze_files_overall(
    #     base_name='infinite_Q_with_GSL_',
    #     indices=indices,
    # )

    # analyze_delay_components_bar(
    #     base_name="infinite_Q_with_GSL_",
    #     indices=indices,
    #     agg='mean',  # 'median' 으로 바꾸면 중앙값
    #     reserve_top=0.84
    # )
    # analyze_files_qos(
    #     base_name='seogwon_results_only_ISL_',
    #     indices=indices,
    #     show_min_max=True,
    #     legend_order=legend_order,   # <- 순서 커스터마이즈
    #     reserve_top=0.85
    # )
    # analyze_files_qos(
    #     base_name='seogwon_results_with_GSL_',
    #     indices=indices,
    #     show_min_max=True,
    #     legend_order=legend_order,  # <- 순서 커스터마이즈
    #     reserve_top=0.85
    # )
    # 전체 QoS 합산
