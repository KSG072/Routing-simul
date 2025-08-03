import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import os


def analyze_files(base_name, indices, directory='.'):
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

    for idx in indices:
        filename = os.path.join(directory, f"{base_name}{idx}.csv")
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue

        df = pd.read_csv(filename)

        required_columns = {'Path Length', 'Queuing Delay', 'Propagation Delay', 'Status'}
        if not required_columns.issubset(df.columns):
            print(f"File {filename} is missing required columns. Skipping.")
            continue

        # Drop probability는 전체 기준, path/delay는 success만
        success_df = df[df['Status'] == 'success']
        success_ratio = len(success_df) / len(df) if len(df) > 0 else 0

        stats["index"].append(idx)
        if not success_df.empty:
            path_length = success_df['Path Length'] - 1
            total_delay = success_df['Queuing Delay'] + success_df['Propagation Delay']

            stats["path_length_avg"].append(path_length.mean())
            stats["path_length_min"].append(path_length.min())
            stats["path_length_max"].append(path_length.max())
            stats["delay_avg"].append(total_delay.mean())
            stats["delay_min"].append(total_delay.min())
            stats["delay_max"].append(total_delay.max())
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
    plt.plot(x_ticks, stats["path_length_max"], marker='v', linestyle='--', color='red', label='Max')
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
    plt.title("End-to-End Delay")
    plt.xlabel(r"$N_{p}$")
    plt.ylabel("Delay")
    plt.xticks(x_ticks)
    plt.xlim(min(x_ticks), max(x_ticks))
    plt.legend(loc='center right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Drop Probability
    plt.figure()
    plt.plot(x_ticks, stats["drop_prob"], marker='o', color='red')
    plt.title("Drop Probability")
    plt.xlabel(r"$N_{p}$")
    plt.ylabel("Probability")
    plt.xticks(x_ticks)
    plt.xlim(min(x_ticks), max(x_ticks))
    plt.ylim(bottom=0)  # y축 0부터 시작
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 사용 예시
if __name__ == '__main__':
    indices = (2, 6, 10, 14, 18, 22, 26, 30)
    analyze_files(base_name='seogwon_results_with_GSL_', indices=indices)