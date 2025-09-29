import pandas as pd
import matplotlib.pyplot as plt

# 1) CSV 로드
rates = [360]
for rate in rates:
    path = f"events_{rate}Mbps.csv"  # 파일 경로
    df = pd.read_csv(path)

    # 2) 필요한 컬럼 확인 및 숫자형 변환
    assert {"time", "src_id", "dst_id", "generated_pkt_num"}.issubset(df.columns)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["generated_pkt_num"] = pd.to_numeric(df["generated_pkt_num"], errors="coerce")
    df = df.dropna(subset=["time", "generated_pkt_num"])

    # 3) time별 합계 집계
    agg = (
        df.groupby("time", as_index=False)["generated_pkt_num"]
          .sum()
          .rename(columns={"generated_pkt_num": "total_generated_pkt_num"})
          .sort_values("time")
    )

    # 4) 분포 그래프 (time vs total)
    plt.figure()
    plt.plot(agg["time"], agg["total_generated_pkt_num"])
    plt.xlabel("time (ms)")
    plt.ylabel("total_generated_pkt_num")
    plt.title(f"generated_pkt_num by time (arrival rate: {rate}")
    plt.tight_layout()
    plt.show()

    # # (선택) 롤링 평균으로 추세 보기
    # if len(agg) >= 500:
    #     plt.figure()
    #     roll = agg["total_generated_pkt_num"].rolling(window=500, min_periods=1).mean()
    #     plt.plot(agg["time"], roll)
    #     plt.xlabel("time (ms)")
    #     plt.ylabel("rolling mean (window=500 ms)")
    #     plt.title("Rolling mean of total_generated_pkt_num")
    #     plt.tight_layout()
    #     plt.show()
