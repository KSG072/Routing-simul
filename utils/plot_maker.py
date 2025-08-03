import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
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


# 테스트 실행 예시
if __name__ == "__main__":
    N, M = 72, 22
    data = np.random.rand(N, M, 4)
    load_heatmap(N, M, data)
