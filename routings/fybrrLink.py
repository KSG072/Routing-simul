# fybrrlink_4isl.py
# ---------------------------------------------------------
# fybrrLink (4-ISL 버전: 상/하/좌/우만 연결)
# - 입력: src, dst (필수), flow_constraints (선택)
# - 링크 상태: 그래프 엣지의 측정값만 사용 (load/plr/jitter 등)
# - 초기 경로: 4-연결 브레젠험(Manhattan)
# - 검색창: 직사각 윈도우 (토러스 최소 Δn, Δm)
# - 비용: Cost = 1 / Score  (CD/AB/lat/PLR[+jitter/stability])
# - CD 완화: max_CD += delta_CD 반복
# - QoS: min AB, end-to-end delay, path PLR, mean jitter
# ---------------------------------------------------------

import math
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from parameters.PARAMS import *


SPEED_OF_LIGHT = C * 1000  # m/s
PEN_MAX = 1e12                  # 창 밖/차단용 큰 가중치


def _latency_ms(p_xyz, q_xyz):
    d = float(norm(q_xyz - p_xyz))
    return (d / SPEED_OF_LIGHT) * 1000.0


def _wrap(x, mod):  # torus wrap for negative steps
    return x % mod


class FybrrLink4ISL:
    """
    필요한 그래프 엣지 속성(측정치):
      - 'type'            : 'isl' (ISL만 비용 계산 대상으로 권장)
      - 'bandwidth_mbps'  : Bij
      - 'load_mbps'       : 현재 Load
      - 'plr'             : 링크 PLR (0~1)
      - 'jitter_ms'       : 지터(옵션, 없으면 0)
      - 'stability'       : 안정 플래그(옵션, 0/1)
    필요한 노드 속성:
      satellites[id].orbit_idx, satellites[id].sat_idx_in_orbit, satellites[id].cartesian_coords
      ground_relays[id].cartesian_coords
    """

    # -----------------------------------------------------
    # ctor & parameters
    # -----------------------------------------------------
    def __init__(self, rtpg, satellites, ground_relays):
        self.rtpg = rtpg
        self.satellites = satellites
        self.ground_relays = ground_relays

        # torus grid size
        self.N, self.M = int(rtpg.N), int(rtpg.M)

        # grid ↔ sat_id map
        self._coord_to_sat = [[None for _ in range(self.M)] for _ in range(self.N)]
        for sid, s in self.satellites.items():
            self._coord_to_sat[s.orbit_idx][s.sat_idx_in_orbit] = sid

        # score weights (기본: PLR까지만 반영; 지터/안정은 0)
        self.k1 = 1.0  # AB
        self.k2 = 1.0  # 1/latency
        self.k3 = 1.0  # (1 - PLR)
        self.k4 = 0.0  # 1/jitter (옵션)
        self.k5 = 0.0  # stability (옵션)

        # CD relaxation
        self.max_CD_init = 0.80
        self.delta_CD = 0.05

        # logs
        self.time = 0
        self.flow_log = defaultdict(dict)  # {(src,dst): {time: path}}

    # -----------------------------------------------------
    # basic helpers
    # -----------------------------------------------------
    def _is_sat(self, nid): return isinstance(nid, int) and (nid in self.satellites)

    def _node_xyz(self, nid):
        return (self.satellites[nid].cartesian_coords
                if self._is_sat(nid)
                else self.ground_relays[nid].cartesian_coords)

    def _nearest_satellite(self, ground_id):
        gx = self._node_xyz(ground_id)
        best, best_d2 = None, float('inf')
        for sid, s in self.satellites.items():
            d2 = float(np.sum((gx - s.cartesian_coords) ** 2))
            if d2 < best_d2:
                best, best_d2 = sid, d2
        return best

    def _edge_data(self, u, v): return self.rtpg.G.get_edge_data(u, v) or {}
    def _edge_B(self, u, v):    return float(self._edge_data(u, v).get('bandwidth_mbps', 0.0))
    def _edge_L(self, u, v):    return float(self._edge_data(u, v).get('load_mbps', 0.0))
    def _edge_plr(self, u, v):  return max(0.0, min(1.0, float(self._edge_data(u, v).get('plr', 0.0))))
    def _edge_jit(self, u, v):  return max(0.0, float(self._edge_data(u, v).get('jitter_ms', 0.0)))
    def _edge_stab(self, u, v):
        data = self._edge_data(u, v)
        if 'stability' in data:
            return 1.0 if data['stability'] else 0.0
        # 기본 추정: 같은 궤도면 1, 아니면 0
        if self._is_sat(u) and self._is_sat(v):
            su, sv = self.satellites[u], self.satellites[v]
            return 1.0 if (su.orbit_idx == sv.orbit_idx) else 0.0
        return 0.0

    def _is_isl(self, u, v): return self._edge_data(u, v).get('type', 'isl') == 'isl'

    def _grid_of(self, sid):
        s = self.satellites[sid]
        return (s.orbit_idx, s.sat_idx_in_orbit)

    def _sat_at(self, n, m):
        return self._coord_to_sat[_wrap(n, self.N)][ _wrap(m, self.M) ]

    # -----------------------------------------------------
    # 4-connected Bresenham (Manhattan straight path)
    # -----------------------------------------------------
    def _manhattan_bresenham_path(self, s_src, s_dst):
        """토러스 최소 Δn, Δm를 따라 4-연결 직선 경로(최단 L1) 생성."""
        n1, m1 = self._grid_of(s_src)
        n2, m2 = self._grid_of(s_dst)

        # torus-min deltas
        dn_raw = n2 - n1
        if abs(dn_raw) > self.N // 2: dn_raw = dn_raw - np.sign(dn_raw) * self.N
        dm_raw = m2 - m1
        if abs(dm_raw) > self.M // 2: dm_raw = dm_raw - np.sign(dm_raw) * self.M

        an, am = abs(int(dn_raw)), abs(int(dm_raw))
        sn, sm = (1 if dn_raw >= 0 else -1), (1 if dm_raw >= 0 else -1)

        # major/minor axis
        path = [s_src]
        n, m = n1, m1

        if an >= am:
            # vertical-major
            err = an // 2
            for _ in range(an):
                n += sn
                path.append(self._sat_at(n, m))
                err += am
                if err >= an and am > 0:
                    m += sm
                    path.append(self._sat_at(n, m))
                    err -= an
        else:
            # horizontal-major
            err = am // 2
            for _ in range(am):
                m += sm
                path.append(self._sat_at(n, m))
                err += an
                if err >= am and an > 0:
                    n += sn
                    path.append(self._sat_at(n, m))
                    err -= am

        # 인접 검증(상하좌우만): 그래프가 4-ISL로 정의되어 있어야 함
        for u, v in zip(path, path[1:]):
            if not self.rtpg.G.has_edge(u, v):
                raise RuntimeError("4-ISL 그래프에 인접 ISL이 누락되었습니다.")
        return path

    # -----------------------------------------------------
    # Rectangular search window (all shortest L1 combinations)
    # -----------------------------------------------------
    def build_rect_window(self, s_src, s_dst):
        """최단 L1 조합 전체를 포함하는 직사각 윈도우 노드 집합."""
        n1, m1 = self._grid_of(s_src)
        n2, m2 = self._grid_of(s_dst)

        # torus-min deltas
        dn_raw = n2 - n1
        if abs(dn_raw) > self.N // 2: dn_raw = dn_raw - np.sign(dn_raw) * self.N
        dm_raw = m2 - m1
        if abs(dm_raw) > self.M // 2: dm_raw = dm_raw - np.sign(dm_raw) * self.M

        an, am = abs(int(dn_raw)), abs(int(dm_raw))
        sn, sm = (1 if dn_raw >= 0 else -1), (1 if dm_raw >= 0 else -1)

        nodes = set()
        for a in range(0, an + 1):
            for b in range(0, am + 1):
                n = n1 + a * sn
                m = m1 + b * sm
                sid = self._sat_at(n, m)
                if sid is not None:
                    nodes.add(sid)
        return nodes, an, am  # an: vertical steps, am: horizontal steps

    # -----------------------------------------------------
    # Score → Cost, with CD rules
    # -----------------------------------------------------
    def _edge_cost(self, u, v, max_CD):
        # ISL만 비용 계산 대상으로(지상 링크는 보조 경로면 허용하되 높은 코스트 권장)
        if not self._is_isl(u, v):
            return PEN_MAX

        B = self._edge_B(u, v)
        L = self._edge_L(u, v)
        if B <= 0.0:
            return PEN_MAX

        CD = L / B
        if CD > 1.0:      # 과혼잡: 엄격 배제
            return PEN_MAX
        if CD > max_CD:   # 완화 전까지 차단
            return PEN_MAX

        # AB with 0.8 barrier
        AB = 0.0 if CD > 0.8 else max(B - L, 0.0)

        # components
        lat_ms = _latency_ms(self._node_xyz(u), self._node_xyz(v))
        plr = self._edge_plr(u, v)
        jit = self._edge_jit(u, v)
        stab = self._edge_stab(u, v)

        lat_term = self.k2 / max(lat_ms, 1e-12)
        jit_term = (self.k4 / max(jit, 1e-12)) if self.k4 > 0.0 else 0.0

        score = (self.k1 * AB) + lat_term + (self.k3 * (1.0 - plr)) + jit_term + (self.k5 * stab)
        if score <= 0.0:
            return PEN_MAX
        return 1.0 / score

    def _assign_costs_in_window(self, nodes_in_window, max_CD):
        G = self.rtpg.G
        for u, v in G.edges():
            # 창 밖은 차단, 단 지상↔창-위성 연결은 허용하려면 조건을 완화할 수 있음
            u_in = (u in nodes_in_window) or (not self._is_sat(u))
            v_in = (v in nodes_in_window) or (not self._is_sat(v))
            if not (u_in and v_in):
                G[u][v]['pen_w'] = PEN_MAX
            else:
                G[u][v]['pen_w'] = self._edge_cost(u, v, max_CD)

    # -----------------------------------------------------
    # QoS check
    # -----------------------------------------------------
    def _path_meets_qos(self, path, cons):
        if not cons:
            return True
        B_min   = cons.get('B_min', None)
        D_max   = cons.get('D_max_ms', None)
        J_max   = cons.get('J_max_ms', None)
        PLR_max = cons.get('PLR_max', None)

        # 1) min AB >= B_min
        if B_min is not None:
            for u, v in zip(path, path[1:]):
                B, L = self._edge_B(u, v), self._edge_L(u, v)
                if B <= 0.0:
                    return False
                CD = L / B
                AB = 0.0 if CD > 0.8 else max(B - L, 0.0)
                if AB < float(B_min):
                    return False

        # 2) delay (전파지연 합)
        if D_max is not None:
            total = 0.0
            for u, v in zip(path, path[1:]):
                total += _latency_ms(self._node_xyz(u), self._node_xyz(v))
            if total > float(D_max):
                return False

        # 3) path PLR
        if PLR_max is not None:
            survive = 1.0
            for u, v in zip(path, path[1:]):
                survive *= (1.0 - self._edge_plr(u, v))
            if (1.0 - survive) > float(PLR_max):
                return False

        # 4) mean jitter
        if J_max is not None:
            vals = [self._edge_jit(u, v) for u, v in zip(path, path[1:])]
            avgj = (sum(vals) / max(1, len(vals))) if vals else 0.0
            if avgj > float(J_max):
                return False

        return True

    # -----------------------------------------------------
    # Dijkstra with 'pen_w'
    # -----------------------------------------------------
    def _shortest_by_penalty(self, src, dst):
        path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='pen_w')
        return path

    # -----------------------------------------------------
    # Public API: ROUTE (src, dst, constraints?)
    # -----------------------------------------------------
    def route(self, src, dst, flow_constraints=None, qos_enabled=True):
        # 지상 노드는 최근접 위성으로 투영
        s_src = src if self._is_sat(src) else self._nearest_satellite(src)
        s_dst = dst if self._is_sat(dst) else self._nearest_satellite(dst)

        # 1) 초기 4-연결 브레젠험 경로
        init_path = self._manhattan_bresenham_path(s_src, s_dst)

        # 2) QoS 미사용 또는 만족 → 채택
        if (not qos_enabled) or self._path_meets_qos(init_path, flow_constraints):
            self._log_path((src, dst), init_path)
            return list(init_path)

        # 3) 직사각 창 생성
        nodes_in_window, an, am = self.build_rect_window(s_src, s_dst)

        # 4) CD 완화 반복
        max_CD = float(self.max_CD_init)
        no_of_iter = int(math.ceil((1.0 - max_CD) / max(1e-12, float(self.delta_CD))))
        last = None
        for _ in range(max(1, no_of_iter)):
            self._assign_costs_in_window(nodes_in_window, max_CD)
            try:
                cand = self._shortest_by_penalty(s_src, s_dst)
            except Exception:
                cand = None

            if cand and self._path_meets_qos(cand, flow_constraints):
                self._log_path((src, dst), cand)
                return list(cand)

            last = cand
            max_CD = min(1.0, max_CD + float(self.delta_CD))

        # 완화 끝까지 실패 → 마지막 후보 또는 초기 경로 반환
        self._log_path((src, dst), last if last else init_path)
        return list(last if last else init_path)

    # -----------------------------------------------------
    # tuning & logs
    # -----------------------------------------------------
    def set_score_weights(self, k1=1.0, k2=1.0, k3=1.0, k4=0.0, k5=0.0):
        self.k1, self.k2, self.k3, self.k4, self.k5 = map(float, (k1, k2, k3, k4, k5))

    def set_cd_relaxation(self, max_cd_init=0.80, delta_cd=0.05):
        self.max_CD_init = float(max_cd_init)
        self.delta_CD = float(delta_cd)

    def _log_path(self, fkey, path):
        self.flow_log[fkey][self.time] = list(path)

    def get_flow_log(self):
        return {str(k): v for k, v in self.flow_log.items()}
