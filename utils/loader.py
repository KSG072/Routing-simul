import csv
import numpy as np
from typing import List, Tuple, Dict

from utils.ground_relay_node import GroundRelayNode


def compute_key_node_search_range_from_indices(P_GRk, R_GRk, N, M, latitude_deg, inclination_deg, altitude_km, min_elevation=10):
    """
    Implements the key node search range from Equations (22) and (23) of the paper,
    using already known RTPG indices (P_GRk^A, R_GRk^A) and (P_GRk^D, R_GRk^D).

    Returns:
        - P_range_asc: (P_min_asc, P_max_asc) for ascending satellite search
        - P_range_desc: (P_min_desc, P_max_desc) for descending satellite search
        - R_range_A: (R_min_A, R_max_A) for ascending
        - R_range_D: (R_min_D, R_max_D) for descending
    """
    # Constants
    re = 6371  # Earth radius in km
    alpha = np.deg2rad(inclination_deg)
    h = altitude_km

    # Step 1: compute rs (search radius)
    theta = np.deg2rad(min_elevation)  # minimum elevation angle
    gamma = np.arcsin(re * np.sin(theta + np.pi / 2) / (h + re))
    beta = np.pi / 2 - theta - gamma
    rs = re * beta

    # Step 2: Δh_min from Eq (21)
    h_min = alpha - np.arcsin(np.sin(alpha) * np.sin(np.pi / 2 - (2 * np.pi / M) * (M - 1)))
    # h_min = inclination_deg - np.arcsin(np.sin(alpha) * np.sin(np.pi / 2 - (2 * np.pi / M) * (M - 1)))

    # Step 3: ΔP and ΔR from Eq (16) and (17)
    lat_rad = np.deg2rad(latitude_deg)
    delta_omega_deg = 360 / N  # ΔΩ in degrees
    delta_P = 2 * int(np.ceil((180 * rs) / (np.pi * re * np.cos(lat_rad) * delta_omega_deg)))
    delta_R = 2 * int(np.ceil((180 * rs) / (np.pi * re * h_min)))

    # delta_R = min(1.5, delta_R)
    # delta_P = min(1.5 * (N / M), delta_R)

    # delta_R = 2 * int(np.ceil((180 * rs) / (np.pi * re * np.rad2deg(h_min))))

    # Step 4: Apply Eq (22), (23) to get search windows
    P_min = (P_GRk - delta_P / 2 + N) % N
    P_max = (P_GRk + delta_P / 2) % N
    R_min = (R_GRk - delta_R / 2 + M) % M
    R_max = (R_GRk + delta_R / 2) % M

    # return P_min, P_max, R_max, R_min
    return P_min, P_max, R_min, R_max

def batch_map_nodes(N, M, inclination_deg, altitude_km, nodes, region_indices_asc, region_indices_desc):
    batch_search_region_asc, batch_search_region_desc = [], []
    for idx, node in enumerate(nodes):
        (P_asc, R_asc) = region_indices_asc[idx]
        (P_desc, R_desc) = region_indices_desc[idx]

        latitude_deg = node.latitude_deg

        P_min_asc, P_max_asc, R_min_asc, R_max_asc = compute_key_node_search_range_from_indices(P_asc, R_asc, N, M, latitude_deg, inclination_deg, altitude_km)
        P_min_desc, P_max_desc, R_min_desc, R_max_desc = compute_key_node_search_range_from_indices(P_desc, R_desc, N, M, latitude_deg, inclination_deg, altitude_km)

        batch_search_region_asc.append((P_min_asc, P_max_asc, R_min_asc, R_max_asc))
        batch_search_region_desc.append((P_min_desc, P_max_desc, R_min_desc, R_max_desc))

    return batch_search_region_asc, batch_search_region_desc

def normalize_wrapped_regions(
    N: int, M: int,
    region_asc: List[Tuple[float, float]],
    region_desc: List[Tuple[float, float]],
    batch_search_region_asc: List[Tuple[float, float, float, float]],
    batch_search_region_desc: List[Tuple[float, float, float, float]]
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Given wrapped region ranges (P_min, P_max, R_min, R_max) for each relay,
    returns a list of lists, where each sublist contains non-wrapping region rectangles
    (p_start, r_start, p_end, r_end) for one relay.

    Output format:
        [
            [(p1_start, r1_start, p1_end, r1_end), (p2_start, r2_start, p2_end, r2_end), ...],
            ...
        ]
    """

    def unwrap(P, R, P_min, P_max, R_min, R_max, phase, rounding=True):
        # Normalize P direction (no full wrap split)
        if rounding:
            if P_min <= P_max:
                p_start = max(int(np.floor(P_min)), 0)
                p_end = min(int(np.ceil(P_max)), N-1)
            else:  # wrap-around
                if P > N // 2:
                    p_start = int(np.floor(P_min))
                    p_end = N - 1
                else:
                    p_start = 0
                    p_end = int(np.ceil(P_max))

            # Normalize R direction with phase-aware bounds
            if phase == 'asc':
                if R_min <= R_max:
                    r_start = max(int(np.floor(R_min)), M // 2)
                    r_end = min(int(np.ceil(R_max)), M-1)
                else:
                    r_start = max(int(np.floor(R_min)), M // 2)
                    r_end = M - 1
            elif phase == 'desc':
                if R_min <= R_max:
                    r_start = int(np.floor(R_min))
                    r_end = min(int(np.ceil(R_max)), M // 2 - 1)
                else:
                    r_start = 0
                    r_end = min(int(np.ceil(R_max)), M-1)
            else:
                raise ValueError(f"Unknown phase: {phase}")
        else:
            if P_min <= P_max:
                p_start = max(P_min, 0)
                p_end = min(P_max, N-1)
            else:  # wrap-around
                if P > N // 2:
                    p_start = P_min
                    p_end = N - 1
                else:
                    p_start = 0
                    p_end = P_max

            # Normalize R direction with phase-aware bounds
            if phase == 'asc':
                if R_min <= R_max:
                    r_start = max(R_min, M // 2)
                    r_end = min(R_max, M-1)
                else:
                    r_start = max(R_min, M // 2)
                    r_end = M - 1
            elif phase == 'desc':
                if R_min <= R_max:
                    r_start = R_min
                    r_end = min(R_max, M // 2 - 1)
                else:
                    r_start = 0
                    r_end = min(R_max, M-1)
            else:
                raise ValueError(f"Unknown phase: {phase}")

        return (p_start, r_start, p_end, r_end)

    result_asc_r = [unwrap(P, R, P_min, P_max, R_min, R_max, 'asc') for ((P, R), (P_min, P_max, R_min, R_max)) in zip(region_asc, batch_search_region_asc)]
    result_asc_nr = [unwrap(P, R, P_min, P_max, R_min, R_max, 'asc', rounding=False) for ((P, R), (P_min, P_max, R_min, R_max)) in zip(region_asc, batch_search_region_asc)]
    result_desc_r = [unwrap(P, R, P_min, P_max, R_min, R_max, 'desc') for ((P, R), (P_min, P_max, R_min, R_max)) in zip(region_desc, batch_search_region_desc)]
    result_desc_nr = [unwrap(P, R, P_min, P_max, R_min, R_max, 'desc', rounding=False) for ((P, R), (P_min, P_max, R_min, R_max)) in zip(region_desc, batch_search_region_desc)]

    return result_asc_r, result_desc_r, result_asc_nr, result_desc_nr


def load_ground_relays_from_csv(csv_path, idx):
    relays = {}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        node_id = idx
        for row in reader:
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            continent = row['continent']
            shape = row['shape']
            relay = GroundRelayNode(f"{continent}-{node_id}", latitude, longitude, continent, shape)
            relays[relay.node_id] = relay

            node_id += 1

    return relays

def prepare_node_routing_metadata(node, mapper, altitude_km):
    """
    하나의 노드 (user 또는 relay)에 대해 region 설정 및 탐색 영역 생성 + 정규화
    → 해당 노드 객체의 속성으로 직접 저장됨
    """
    # 1. region index 추출 (asc, desc)
    region_asc, region_desc = mapper.get_region_index_from_nodes(node)

    # 2. 탐색 polygon 계산
    # 노드의 ascending, descending 일때의 (p,r)값 도출
    search_asc, search_desc = batch_map_nodes(
        N=mapper.N, M=mapper.M,
        inclination_deg=np.rad2deg(mapper.inclination_rad),
        altitude_km=altitude_km,
        nodes=[node],
        region_indices_asc=[region_asc],
        region_indices_desc=[region_desc]
    )

    # 3. 정규화
    # 위 노드의 위치 (p,r)값에 의거한 search area 정의
    norm_asc_r, norm_desc_r, norm_asc_nr, norm_desc_nr = normalize_wrapped_regions(
        N=mapper.N, M=mapper.M,
        region_asc=[region_asc],
        region_desc=[region_desc],
        batch_search_region_asc=search_asc,
        batch_search_region_desc=search_desc
    )

    # 4. 노드에 저장
    node.region_asc = region_asc
    node.region_desc = region_desc
    node.search_regions_asc = norm_asc_r[0]
    node.search_regions_desc = norm_desc_r[0]

    return node

def load_event_schedule(csv_path: str, t_max_ms: int) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    이벤트 CSV를 메모리로 로드하여 time -> [(src, dst, count), ...] 딕셔너리로 반환.
    0ms부터 t_max_ms까지만 로드합니다.
    같은 (time, src, dst)가 여러 줄이면 count 합산.
    time은 float/int가 섞여 있어도 int(float(..))로 정규화.
    src == dst 는 무시.
    """
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(int))  # time -> {(src,dst): count}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"time", "src_id", "dst_id", "generated_pkt_num"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"이벤트 CSV 헤더가 잘못되었습니다. 필요한 컬럼: {required}")

        for row in reader:
            t = int(float(row["time"]))
            if t > t_max_ms:
                # CSV가 시간순으로 정렬되어 있다고 가정하고 조기 종료
                break

            s = int(row["src_id"])
            d = int(row["dst_id"])
            if s == d:
                continue

            c = int(row["generated_pkt_num"])
            agg[t][(s, d)] += c

    # dict[int, list[(src,dst,count)]]
    schedule = {t: [(s, d, c) for (s, d), c in pairs.items()] for t, pairs in agg.items()}
    return schedule