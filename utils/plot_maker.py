import csv
import matplotlib.pyplot as plt

# --- 설정: CSV 파일 경로 ---
file_path = '../results/path length_10000iterations.csv'

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
