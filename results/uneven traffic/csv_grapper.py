import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

QOS_ORDER   = [0, 1, 2]
QOS_COLORS  = {0:'red', 1:'orange', 2:'black'}
QOS_LABELS  = {0: "High", 1: "Middle", 2: "Low"}
QOS_MARKERS = {0: "o", 1: "s", 2: "^"}  # 평균선에만 마커

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
    filename_tpl="limited_Q_with_GSL_{rate}.csv",
    metric="e2e_delay",                        # "e2e_delay" | "drop_rate"
    bin_size=50,                               # start_at 구간 간격(ms)
    start_range=(0, 700),                      # x축 범위(ms)
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

        df = pd.read_csv(path, low_memory=False).rename(
            columns={k: v for k, v in rename_map.items() if k in df.columns}
        )

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
            grp = df.dropna(subset=["e2e_delay"]).groupby("tbin_center", as_index=False)["e2e_delay"].mean()
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

def analyze_files_cdf(base_name, indices, directory='.', qos=None, reserve_top=0.86):
    """
    각 파일(=Np 값) 별로 'success' 레코드만 모아서
      - Path Length (hops: Path Length - 1) CDF
      - End-to-End Delay(Queuing+Propagation+Transmission) CDF
    를 그립니다.

    범례: Np 값 (예: "Np=10")
    제목: 없음
    Legend는 그래프 바깥 위쪽(예약 공간)에 배치하여 겹침 방지.

    Parameters
    ----------
    base_name : str
        파일 접두어 (예: 'seogwon_results_only_ISL_')
    indices : iterable[int]
        파일 인덱스들 (예: (10, 14, 18, ...))
    directory : str
        CSV 폴더 경로
    qos : None | int | str
        None이면 QoS 구분 없이 전체 success를 집계.
        0/1/2 또는 'High'/'Middle'/'Low' (대소문자 무관)로 특정 QoS만 필터링.
    reserve_top : float
        0~1 사이. legend가 올라갈 위쪽 여백 비율. (값을 낮추면 여백 더 큼)
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
                'Transmission Delay', 'Status'}
    if qos_code is not None:
        required = required | {'QoS'}

    # 수집 버킷: Np 라벨 -> np.array(values)
    path_by_idx = {}
    delay_by_idx = {}
    labels = []  # 실제 곡선이 있는 Np만 범례에 사용

    for idx in indices:
        filename = os.path.join(directory, f"{base_name}{idx}.csv")
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue

        df = pd.read_csv(filename)
        if not required.issubset(df.columns):
            print(f"File {filename} is missing required columns {required - set(df.columns)}. Skipping.")
            continue

        df_ok = df[df['Status'] == 'success']
        if qos_code is not None:
            df_ok = df_ok[df_ok['QoS'] == qos_code]

        if df_ok.empty:
            print(f"[{idx}] no success rows (qos={qos}). Skipping.")
            continue

        # 값 준비
        path_vals = (df_ok['Path Length'] - 1).dropna().to_numpy()
        delay_vals = (
            df_ok['Queuing Delay'] + df_ok['Propagation Delay'] + df_ok['Transmission Delay']
        ).dropna().to_numpy()

        if len(path_vals) == 0 and len(delay_vals) == 0:
            print(f"[{idx}] no valid numeric rows. Skipping.")
            continue

        label = f"Np={idx}"
        path_by_idx[label] = path_vals
        delay_by_idx[label] = delay_vals
        labels.append(label)

    if not labels:
        print("No valid data to plot for CDF.")
        return

    # ECDF 유틸
    def _ecdf(values: np.ndarray):
        v = np.sort(values)
        y = np.arange(1, len(v) + 1) / len(v)
        return v, y

    # ---------- CDF Plot: Path Length ----------
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
        # ax.set_xlim(0, 30)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        _apply_tick_style(ax, x_minor_div=5, y_minor_div=0)
        ax.grid(True)


        ncol = min(6, max(1, len(labels)))
        # 위쪽 legend 공간 예약
        fig.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=ncol, frameon=True)
        plt.show()
    else:
        print("No Path Length data for CDF.")

    # ---------- CDF Plot: End-to-End Delay ----------
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
        # ax.set_xlim(0, 250)
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
def analyze_files_overall(base_name, indices, directory='.'):
    stats = {
        "index": [],
        "path_length_avg": [],
        "path_length_min": [],
        "path_length_max": [],
        "delay_avg": [],
        "delay_min": [],
        "delay_max": [],
        # "delay_expected": [],
        # "delay_expected_isl": [],
        "drop_prob": []
    }

    for idx in indices:
        filename = os.path.join(directory, f"{base_name}{idx}.csv")
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue

        df = pd.read_csv(filename)

        # required_columns = {'Path Length', 'e2e delay', 'Status', "delay_expected", "delay_expected_isl"}
        required_columns = {'Path Length', 'e2e delay', 'Status'}
        if not required_columns.issubset(df.columns):
            print(f"File {filename} is missing required columns. Skipping.")
            continue

        # Drop probability는 전체 기준, path/delay는 success만
        success_df = df[df['Status'] == 'success']
        success_ratio = len(success_df) / len(df) if len(df) > 0 else 0

        stats["index"].append(idx)
        if not success_df.empty:
            path_length = success_df['Path Length'] - 1
            total_delay = success_df['e2e delay']

            stats["path_length_avg"].append(path_length.mean())
            stats["path_length_min"].append(path_length.min())
            stats["path_length_max"].append(path_length.max())
            stats["delay_avg"].append(total_delay.mean())
            stats["delay_min"].append(total_delay.min())
            stats["delay_max"].append(total_delay.max())
            # stats["delay_expected"].append(success_df['expected delay(result)'].mean())
            # stats["delay_expected_isl"].append(success_df['expected delay(isl)'].mean())
        else:
            # success가 하나도 없을 경우 NaN 처리
            stats["path_length_avg"].append(float('nan'))
            stats["path_length_min"].append(float('nan'))
            stats["path_length_max"].append(float('nan'))
            stats["delay_avg"].append(float('nan'))
            stats["delay_min"].append(float('nan'))
            stats["delay_max"].append(float('nan'))

        stats["drop_prob"].append(1 - success_ratio)

    # X축 값 고정
    x_ticks = stats["index"]

    # Plot Path Length
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

    # Plot End-to-End Delay
    plt.figure()
    plt.plot(x_ticks, stats["delay_avg"], marker='o', color='orange', label='Average')
    plt.plot(x_ticks, stats["delay_min"], marker='^', linestyle='--', color='green', label='Min')
    plt.plot(x_ticks, stats["delay_max"], marker='v', linestyle='--', color='red', label='Max')
    # plt.plot(x_ticks, stats["delay_expected"], marker='*', linestyle=':', color='grey', label='expected')
    # plt.plot(x_ticks, stats["delay_expected_isl"], marker='*', linestyle=':', color='grey', label='expected_isl')
    plt.xlabel(r"Packet Arrival Rate (Mbps)")
    plt.ylabel("End-to-End Delay (ms)")
    plt.xticks(x_ticks)
    plt.xlim(min(x_ticks), max(x_ticks))
    plt.legend(loc='center right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Drop Probability
    plt.figure()
    plt.plot(x_ticks, stats["drop_prob"], marker='o', color='red')
    plt.xlabel(r"Packet Arrival Rate (Mbps)")
    plt.ylabel("Drop Probability")
    plt.xticks(x_ticks)
    plt.xlim(min(x_ticks), max(x_ticks))
    plt.ylim(bottom=0)  # y축 0부터 시작
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_delay_components_bar(base_name, indices, directory='.',
                                 agg='mean',        # 'mean' 또는 'median'
                                 reserve_top=0.85,  # 범례를 위 바깥으로 빼기 위한 여백
                                 bar_width=0.25):
    """
    각 Np(파일 인덱스)별로 success 행만 집계하여
    Queuing / Propagation / Transmission 지연의 집계값(평균/중앙값)을
    '3개 막대'로 묶은 그룹드 바 차트로 그립니다.
    """
    required = {'Queuing Delay', 'Propagation Delay', 'Transmission Delay', 'Status'}

    x_labels = []      # 실제로 그려질 Np들
    q_vals, p_vals, t_vals = [], [], []

    for idx in indices:
        filename = os.path.join(directory, f"{base_name}{idx}.csv")
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue

        df = pd.read_csv(filename)
        if not required.issubset(df.columns):
            print(f"File {filename} is missing required columns {required - set(df.columns)}. Skipping.")
            continue

        df_ok = df[df['Status'] == 'success']
        if df_ok.empty:
            print(f"[{idx}] no success rows. Skipping.")
            continue

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

    # --- 그리기 ---
    x = np.arange(len(x_labels))
    fig, ax = plt.subplots()

    ax.bar(x - bar_width, q_vals, width=bar_width, label='Queuing')
    ax.bar(x,             p_vals, width=bar_width, label='Propagation')
    ax.bar(x + bar_width, t_vals, width=bar_width, label='Transmission')

    ax.set_xlabel(r"Packet Arrival Rate (Mbps)")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_labels])
    ax.set_ylim(bottom=0)   # y축 0부터 시작
    ax.margins(y=0)         # 여백 제거해서 바닥에 딱 붙게
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 범례를 그래프 바깥 위쪽으로(겹침 방지)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, reserve_top])  # 위쪽 공간 예약
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=True)
    plt.show()
# 사용 예시
if __name__ == '__main__':
    indices = (40, 80, 120, 160, 200, 240, 280, 320, 360)
    indices_cdf = (40, 120, 200, 280, 360)
    # indices = (40,80,160,240,320)
    # indices_cdf = (40, 160, 320)
    # legend 원하는 순서 예시: 평균 먼저, 그 다음 min/max
    legend_order = [
        "High (max)", "High (avg)", "High (min)",
        "Middle (max)", "Middle (avg)", "Middle (min)",
        "Low (max)", "Low (avg)", "Low (min)"
    ]

    # analyze_files_cdf(
    #     base_name='limited_Q_with_GSL_',
    #     indices=indices_cdf,
    #     qos=None,  # 전체 QoS 합산
    #     reserve_top=0.95  # legend 공간. 필요시 0.78~0.90 조정
    # )

    plot_arrival_rate_avg_over_start_at('.')
    # analyze_delay_components_bar(
    #     base_name='limited_Q_with_GSL_',
    #     indices=indices,
    #     agg='mean',  # 'median' 으로 바꾸면 중앙값
    #     reserve_top=0.84,
    #     directory='.'
    # )
    # analyze_files_overall(
    #     base_name='limited_Q_with_GSL_',
    #     indices=indices,
    #     directory='.'
    # )


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
