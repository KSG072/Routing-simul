import numpy as np
from numpy import pi

from typing import Tuple

TWO_PI = 2.0 * pi
_EPS = 1e-8  # increased EPS to reduce boundary crossing errors


# ------------------------------------------------------------------
# 최소 홉 추정 (공개 API)  ← 함수명/인자 절대 변경하지 말 것!
# ------------------------------------------------------------------

def min_hop_distance(src, dst, N, M, F):
    orb_s, sat_s = src.orbit_idx, src.sat_idx_in_orbit
    orb_d, sat_d = dst.orbit_idx, dst.sat_idx_in_orbit

    sat_after_r, sat_after_l = sat_s, sat_s

    right = (orb_d - orb_s) % N
    left = (orb_s - orb_d) % N

    if orb_s < orb_d:
        sat_after_l = (sat_s - F) % M
    elif orb_s > orb_d:
        sat_after_r = (sat_s + F) % M
    else:
        pass

    up_r = (sat_d - sat_after_r) % M
    up_l = (sat_d - sat_after_l) % M
    down_r = (sat_after_r - sat_d) % M
    down_l = (sat_after_l - sat_d) % M

    cand = [
        (right + up_r, +right, +up_r),  # right + up
        (right + down_r, +right, -down_r),  # right + down
        (left + up_l, -left, +up_l),  # left  + up
        (left + down_l, -left, -down_l),  # left  + down
    ]
    total, horiz, vert = min(cand, key=lambda x: x[0])

    return horiz, vert


def minimum_hop_estimation(src,
                           dst,
                           N: int,
                           M: int,
                           delta_f: float,
                           delta_phi: float,
                           *,
                           return_details: bool = False) -> Tuple[int, int]:
    """최소 홉 경로의 (horiz, vert) 부호 정수 반환.

    입력은 항상 올바르다고 가정; 방어적 검사하지 않음.
    src.position / dst.position → (P,R)
    src.phase_rad               → 출발 위상각(rad)
    """
    # --- 입력 해체 ---
    P1, R1 = src.region  # (오른쪽 증가, mod N) / (위쪽 증가, mod M)
    P2, R2 = dst.region
    eps = 1e-7
    u = src.phase_rad  # 위상각(rad)
    # print(f"{src.node_id}: ({P1},{R1}), {dst.node_id}: ({P2},{R2})")

    # wrap & check
    u_src = u if u >= pi / 2 else u + 2 * pi
    R_1 = np.floor((u_src - pi/2)/delta_phi - eps)
    if R1 != R_1:
        print("!!!!!!!!!! src: R is not matching !!!!!!!!!!!")
        print(f"R1: {R1}, R_1: {R_1}")
    elif P1 != src.orbit_idx:
        print("!!!!!!!!!! dst: P is not matching !!!!!!!!!!!")
        print(f"P1: {P1}, P_1: {src.orbit_idx}")
    u_dst = dst.phase_rad if dst.phase_rad >= pi / 2 else dst.phase_rad + 2 * pi
    R_2 = np.floor((u_dst - pi / 2) / delta_phi - eps)
    if R2 != R_2:
        print("!!!!!!!!!! dst: R is not matching !!!!!!!!!!!")
        print(f"R2: {R2}, R_2: {R_2}")
    elif P2 != dst.orbit_idx:
        print("!!!!!!!!!! dst: P is not matching !!!!!!!!!!!")
        print(f"P2: {P2}, P_2: {dst.orbit_idx}")

    # ------------------------------------------------------------------
    # 1) 수평 홉 후보 Hr, Hl  (식 11a, 11b)
    # ------------------------------------------------------------------
    Hr = (P2 - P1) % N  # 오른쪽으로 몇 홉?
    Hl = (P1 - P2) % N  # 왼쪽으로 몇 홉?

    # ------------------------------------------------------------------
    # 2) 오른쪽으로 Hr 이동했을 때 R 변화 (식 12)
    # ------------------------------------------------------------------
    # 출발 region 하한 경계 Ul = R1 * ΔΦ (wrap 2π)
    Ul = (R1 * delta_phi) % TWO_PI
    # 현재 region 내 오프셋: u - Ul (CCW) ∈ [0,ΔΦ)
    off = (u_src - pi/2) % delta_phi
    # 상한까지 남은 거리
    dist_upper = delta_phi - off  # ∈ (0,ΔΦ]
    dist_lower = off

    travel_r = Hr * delta_f
    if travel_r < dist_upper:
        crossings_r = 0
    else:
        remaining = travel_r - dist_upper
        crossings_r = int(np.ceil(remaining / delta_phi))
    R_after_r = (R1 + crossings_r) % M


    # ------------------------------------------------------------------
    # 3) 왼쪽으로 Hl 이동했을 때 R 변화 (식 13)
    # ------------------------------------------------------------------
    # 하한까지 거리 = off
    travel_l = Hl * delta_f
    if travel_l <= dist_lower:
        crossings_l = 0
    else:
        remaining = travel_l - dist_lower
        crossings_l = int(np.ceil(remaining / delta_phi))
    R_after_l = (R1 - crossings_l) % M

    # print(f"left delta r: {R_after_l}, right delta r: {R_after_r}")

    # ------------------------------------------------------------------
    # 4) 목적지 R2까지 수직 홉 후보 (식 14)
    # ------------------------------------------------------------------
    up_r   = (R2 - R_after_r) % M
    down_r = (R_after_r - R2) % M
    up_l   = (R2 - R_after_l) % M
    down_l = (R_after_l - R2) % M

    # ------------------------------------------------------------------
    # 5) 네 후보 총 홉수 비교 → 최소 선택 → 부호 부여
    # ------------------------------------------------------------------
    cand = [
        (Hr + up_r,   +Hr, +up_r),    # right + up
        (Hr + down_r, +Hr, -down_r),  # right + down
        (Hl + up_l,   -Hl, +up_l),    # left  + up
        (Hl + down_l, -Hl, -down_l),  # left  + down
    ]
    total, horiz, vert = min(cand, key=lambda x: x[0])

    if return_details:
        details = {
            'total': total,
            'Hr': Hr,
            'Hl': Hl,
            'R_after_r': R_after_r,
            'R_after_l': R_after_l,
            'up_r': up_r,
            'down_r': down_r,
            'up_l': up_l,
            'down_l': down_l,
            'u_src': u_src,
            'Ul': Ul,
            'dist_upper': dist_upper,
            'dist_lower': dist_lower,
        }
        return (horiz, vert), details
    else:
        return horiz, vert