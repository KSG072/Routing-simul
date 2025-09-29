import os, csv
from typing import Sequence, Any, Optional
import numpy as np

from parameters.PARAMS import inclination_deg, altitude_km, N, M, F, TRAFFIC_DENSITY, TRAFFIC_DENSITY_UNIFORM
from utils.walker_constellation import WalkerConstellation


def generate_onoff_event_csvs(
    src_nodes: Sequence[Any],
    dst_nodes: Sequence[Any],
    traffic_duration_ms: int = 20_000,
    num_flows: int = 3000,
    rates_mbps: Optional[Sequence[float]] = None,
    rate_min_mbps: float = 80.0,
    rate_max_mbps: float = 350.0,
    rate_step_mbps: float = 40.0,
    pkt_bits: int = 64_000,
    pareto_alpha: float = 1.5,
    mean_on_ms: float = 500.0,
    mean_off_ms: float = 500.0,
    out_dir: str = "../parameters/traffic",
    seed: Optional[int] = 42,

    # --- 옵션 ---
    steady_state_start: bool = True,
    start_on: bool = True,
    warmup_ms: int = 0,

    # 🔽 가중 샘플링 입력 (임의 모양 지원: rows×cols)
    sats_group_by_grid=None,  # rows×cols, 각 셀은 위성 ID 리스트
    traffic_map=None          # rows×cols, 각 셀 가중치 (음수 금지)
):
    """
    CSV 헤더: time, src_id, dst_id, generated_pkt_num
    """
    if traffic_duration_ms <= 0:
        raise ValueError("traffic_duration_ms는 양의 정수여야 합니다.")
    if pareto_alpha <= 1.0:
        raise ValueError("pareto_alpha는 1보다 커야 평균이 존재합니다.")
    if warmup_ms < 0:
        raise ValueError("warmup_ms는 0 이상이어야 합니다.")

    # 실제 시뮬레이션 길이(워밍업 포함)
    sim_duration_ms = traffic_duration_ms + warmup_ms
    sim_max_ms = sim_duration_ms - 1

    rng = np.random.default_rng(seed)

    # ---------- (A) grid 가중 방식 준비 (모양 자동 인식) ----------
    use_grid_weighted = (sats_group_by_grid is not None) and (traffic_map is not None)

    if use_grid_weighted:
        w = np.array(traffic_map, dtype=np.float64)
        rows, cols = w.shape  # ✅ 어떤 모양이든 지원
        if np.any(w < 0):
            raise ValueError("traffic_map에는 음수 가중치를 넣을 수 없습니다.")

        # 빈 셀(위성 0개)은 선택 불가 → 가중치 0
        nonempty = np.zeros((rows, cols), dtype=bool)
        if len(sats_group_by_grid) != rows or len(sats_group_by_grid[0]) != cols:
            raise ValueError(f"sats_group_by_grid 모양 {len(sats_group_by_grid)}×{len(sats_group_by_grid[0])} "
                             f"와 traffic_map 모양 {rows}×{cols} 이 일치해야 합니다.")
        for i in range(rows):
            for j in range(cols):
                nonempty[i, j] = len(sats_group_by_grid[i][j]) > 0

        w = w * nonempty  # 빈 셀은 자동 0

        if np.all(w == 0):
            # 모든 셀이 비었거나 가중치가 0이면 fallback: non-empty 균등
            w = nonempty.astype(np.float64)

        # 정규화 확률벡터
        p = w.ravel()
        p_sum = p.sum()
        if p_sum == 0:
            raise ValueError("선택 가능한 grid가 없습니다. (모든 셀 비어있음)")
        p = p / p_sum

        def sample_two_distinct_grids():
            # rows*cols 중 2개 비복원 추출 (가중 확률)
            idx2 = rng.choice(rows * cols, size=2, replace=False, p=p)
            i1, j1 = divmod(int(idx2[0]), cols)
            i2, j2 = divmod(int(idx2[1]), cols)
            return (i1, j1), (i2, j2)

        def sample_sat_from_cell(i, j):
            cell = sats_group_by_grid[i][j]
            return rng.choice(cell)

        def choose_pair_via_grid():
            (si, sj), (di, dj) = sample_two_distinct_grids()
            s = sample_sat_from_cell(si, sj)
            d = sample_sat_from_cell(di, dj)
            return s, d

    # ---------- (B) 기존 균등 방식 ----------
    else:
        pairs = [(s, d) for s in src_nodes for d in dst_nodes if s != d]
        if not pairs:
            raise ValueError("유효한 (src,dst) 쌍이 없습니다. (src != dst)")
        pairs = np.array(pairs, dtype=object)
        pair_probs = np.ones(len(pairs), dtype=np.float64) / len(pairs)

    # Pareto 관련
    xm_on_ms  = mean_on_ms  * (pareto_alpha - 1.0) / pareto_alpha
    xm_off_ms = mean_off_ms * (pareto_alpha - 1.0) / pareto_alpha

    def pareto_duration_ms(xm_ms: float) -> float:
        return float(xm_ms * (1.0 + rng.pareto(pareto_alpha)))

    def length_biased_pareto_ms(xm_ms: float) -> float:
        a_lb = pareto_alpha - 1.0
        if a_lb <= 0:
            return pareto_duration_ms(xm_ms)
        return float(xm_ms * (1.0 + rng.pareto(a_lb)))

    def sample_initial_state_and_remaining():
        p_on = mean_on_ms / (mean_on_ms + mean_off_ms)
        on0 = bool(rng.random() < p_on)
        xm0 = xm_on_ms if on0 else xm_off_ms
        L0 = length_biased_pareto_ms(xm0)
        age0 = rng.uniform(0.0, L0)
        remain0 = L0 - age0
        return on0, remain0

    # --- 송신률 목록 구성 (중복 제거) ---
    if rates_mbps is None:
        grid = np.arange(rate_min_mbps, rate_max_mbps + 1e-9, rate_step_mbps, dtype=float)
        rates = grid[grid <= rate_max_mbps].tolist()
        if len(rates) == 0 or abs(rates[-1] - rate_max_mbps) > 1e-9:
            rates.append(float(rate_max_mbps))
    else:
        rates = [float(r) for r in rates_mbps]
        if any(r <= 0 for r in rates):
            raise ValueError("rates_mbps에는 양수만 허용됩니다.")

    os.makedirs(out_dir, exist_ok=True)

    # --- 플로우별 (src,dst) 미리 뽑기 ---
    print(f"{num_flows}개의 플로우에 대한 (src, dst) 쌍을 미리 선택합니다...")
    if use_grid_weighted:
        flow_pairs = [choose_pair_via_grid() for _ in range(num_flows)]
    else:
        flow_pair_indices = rng.choice(len(pairs), size=num_flows, replace=True, p=pair_probs)
        flow_pairs = [pairs[i] for i in flow_pair_indices]

    # --- 시뮬레이션 ---
    from tqdm import tqdm
    for rate_mbps in rates:
        rate_bps = rate_mbps * 1e6
        dt_ms = (pkt_bits / rate_bps) * 1000.0

        bucket: dict[tuple[int, any, any], int] = {}

        # 플로우별 (src,dst)는 루프 밖에서 이미 선택됨
        for i in tqdm(range(num_flows), desc=f"Rate {rate_mbps} Mbps"):
            s_id, d_id = flow_pairs[i]

            # 초기화
            if steady_state_start:
                on, first_remain_ms = sample_initial_state_and_remaining()
                first = True
            else:
                on = bool(start_on)
                first = False

            t_ms = 0.0
            while t_ms < sim_duration_ms:
                dur_ms = (first_remain_ms if first else
                          pareto_duration_ms(xm_on_ms if on else xm_off_ms))
                first = False
                t_end_ms = min(t_ms + dur_ms, float(sim_duration_ms))

                if on and (t_end_ms - t_ms) >= dt_ms:
                    n_pkts = int((t_end_ms - t_ms) // dt_ms)
                    if n_pkts > 0:
                        ts_ms = t_ms + np.arange(n_pkts, dtype=np.float64) * dt_ms
                        ms_idx = np.floor(ts_ms + 1e-9).astype(np.int64)
                        np.minimum(ms_idx, sim_max_ms, out=ms_idx)
                        unique_ms, counts = np.unique(ms_idx, return_counts=True)
                        for m, c in zip(unique_ms.tolist(), counts.tolist()):
                            if m >= warmup_ms:
                                key = (int(m - warmup_ms), s_id, d_id)
                                bucket[key] = bucket.get(key, 0) + int(c)

                t_ms = t_end_ms
                on = not on

        out_path = os.path.join(out_dir, f"events_{int(rate_mbps)}Mbps.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["time", "src_id", "dst_id", "generated_pkt_num"])
            for (t_int, s_id, d_id), cnt in sorted(bucket.items(), key=lambda kv: kv[0][0]):
                if 0 <= t_int < traffic_duration_ms:
                    writer.writerow([t_int, s_id, d_id, cnt])

        print(f"[OK] {out_path}  (dt≈{dt_ms:.6f} ms, flows={num_flows}, "
              f"steady_start={steady_state_start}, warmup={warmup_ms} ms)")


def grouping_satellite_to_lat_lon_grid(
        satellites,
        sats_group_by_grid,
        lat_max_abs: float = 60.0  # |위도|가 넘어가면 가장자리 bin으로 클램프
):
    """
    sats_group_by_grid 모양(rows×cols)에 자동 적응.
    rows: 위도 bins, cols: 경도 bins.
    경도는 균등 bins(폭 = 360/cols), 위도는 [-lat_max_abs, +lat_max_abs]를 rows 등분.
    """
    rows = len(sats_group_by_grid)
    cols = len(sats_group_by_grid[0])

    # 위도 bin 너비
    lat_bin_width = 2 * lat_max_abs / rows

    def lat_to_idx(lat_deg: float) -> int:
        # [-lat_max_abs, lat_max_abs] 범위 밖의 값은 가장자리 bin으로 처리
        if lat_deg >= lat_max_abs:
            return 0
        if lat_deg <= -lat_max_abs:
            return rows - 1

        # 위쪽(북쪽)부터 0번 인덱스. (lat_max_abs - lat)으로 좌표 변환 후 너비로 나눔
        # 부동소수점 오차로 경계값에서 인덱스가 벗어나는 것을 방지하기 위해 min 사용
        idx = int((lat_max_abs - lat_deg) / lat_bin_width)
        return min(idx, rows - 1)

    def lon_to_idx(lon_deg_0_360: float) -> int:
        # 0..360 입력을 -180..180으로 이동 후 균등 bin
        if lon_deg_0_360 < 180:
            lon = lon_deg_0_360
        else:
            lon = lon_deg_0_360 - 360.0
        # lon = ((lon_deg_0_360 % 360.0) + 180.0) % 360.0 - 180.0  # [-180,180)
        width = 360.0 / cols
        idx = int((np.floor(lon) + 180.0) // width)
        # 부동소수 방어
        if idx < 0: idx = 0
        if idx >= cols: idx = cols - 1
        return idx

    grid = sats_group_by_grid
    for sat in satellites:
        # 위성 객체에 필요한 속성이 없으면 건너뜀
        lat = float(sat.real_latitude_deg)
        lon0_360 = float(sat.real_longitude_deg)

        li = lat_to_idx(lat)
        lj = lon_to_idx(lon0_360)
        grid[li][lj].append(sat.node_id)

    return grid


if __name__ == "__main__":
    constellation = WalkerConstellation(N=N, M=M, F=F,
                                        altitude_km=altitude_km,
                                        inclination_deg=inclination_deg)
    constellation.generate_constellation()
    satellites_dict = constellation.get_all_satellites()
    satellites = satellites_dict.values()

    # ✅ 트래픽 맵의 모양으로 그리드 크기 자동 설정
    tm = np.array(TRAFFIC_DENSITY, dtype=float)
    rows, cols = tm.shape
    sats_group_by_grid = [[[] for _ in range(cols)] for _ in range(rows)]

    sats_group_by_grid = grouping_satellite_to_lat_lon_grid(
        satellites, sats_group_by_grid,
        lat_max_abs=60.0  # 필요시 75, 90 등으로 조정
    )

    generate_onoff_event_csvs(
        src_nodes=satellites_dict.keys(),
        dst_nodes=satellites_dict.keys(),
        traffic_duration_ms=10_000,
        num_flows=3000,
        rates_mbps=[1, 40, 80, 120, 160, 200, 240, 280, 320, 360],          # 예시
        rate_min_mbps=80,
        rate_max_mbps=360,
        rate_step_mbps=40,
        pkt_bits=64_000,
        pareto_alpha=1.5,
        mean_on_ms=500.0,
        mean_off_ms=500.0,
        seed=42,
        steady_state_start=True,
        start_on=True,
        warmup_ms=2000,
        out_dir="../parameters/uneven traffic(latest)",
        sats_group_by_grid=sats_group_by_grid,
        traffic_map=TRAFFIC_DENSITY   # 🔥 rows×cols 어떤 크기든 OK
    )
