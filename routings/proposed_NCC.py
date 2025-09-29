import math
from collections import defaultdict
# routings/proposed.py
# ---------------------------------------------------------
# "트래픽 맵 재가중 → 다익스트라" 기반 라우팅 테이블 생성기 (간단 구현)
# ---------------------------------------------------------
import csv, json
from collections import defaultdict
from math import floor

from mkl import peak_mem_usage
from tqdm import tqdm

from utils.RMSE import compute_rmse_between_totals_and_real_totals
from utils.plot_maker import load_heatmap
from numpy import pi
from numpy.linalg import norm
from parameters.PARAMS import C, PACKET_SIZE_BITS
import os
import pickle


def save_flow_log_pkl(flow_log, filepath):
    """
    flow_log를 그대로 피클로 저장.
    flow_log 형식 예:
      {
        (src, dst): {time_int: [path_node_ids...]},
        ...
      }
    """
    with open(filepath, 'wb') as f:
        pickle.dump(flow_log, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_flow_log_pkl(filepath):
    """
    피클에서 flow_log를 그대로 불러와 반환.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# ---------------------------------------------------------

def calculate_prop_delay(src, dst):
    distance = norm(dst-src)
    prop_delay_sec = distance/C
    return prop_delay_sec*1000

def prop_delay_estimation(path, satellites, ground_relays):
    expected_prop_delay = 0
    cur = satellites[path[0]]
    for hop in path[1:]:
        is_satellite = isinstance(hop, int)
        if is_satellite:
            togo = satellites[hop]
        else:
            togo = ground_relays[hop]

        cur_coords, togo_coords = cur.cartesian_coords, togo.cartesian_coords
        expected_prop_delay += calculate_prop_delay(cur_coords, togo_coords)

        cur = togo

    return expected_prop_delay

def _mm1k_queue_delay_ms(mu_pps: float, lam_pps: float, K: int) -> float:
    """
    M/M/1/K 평균 큐잉 지연(ms) 반환.
    - mu_pps: 서비스율 [packets/sec]
    - lam_pps: 도착율   [packets/sec]
    - K: 시스템 수용 한도(서비스 중 1 + 대기열 K-1)
    반환: Wq_ms (큐잉 지연, ms). 수치 안정성을 위해 ρ≈1, ρ→0 모두 안전 처리.
    """
    EPS = 1e-12
    if K <= 0 or mu_pps <= 0:
        return 0.0

    rho = lam_pps / max(mu_pps, EPS)

    # 정상상태 확률과 L, P_block 계산
    if abs(rho - 1.0) < 1e-9:
        # ρ == 1 특수해
        P0 = 1.0 / (K + 1.0)
        P_block = P0  # P_K
        L = K / 2.0
    else:
        rho_pow_K = rho**K
        rho_pow_K1 = rho**(K + 1)
        denom = (1.0 - rho_pow_K1)
        if abs(denom) < EPS:
            # ρ가 매우 1에 가까워 발생하는 수치 불안정 방지
            # 근사적으로 ρ=1 케이스로 처리
            P0 = 1.0 / (K + 1.0)
            P_block = P0
            L = K / 2.0
        else:
            P0 = (1.0 - rho) / denom
            P_block = P0 * rho_pow_K
            # L 공식
            num_L = rho * (1.0 - (K + 1.0) * rho_pow_K + K * rho_pow_K1)
            den_L = (1.0 - rho) * denom
            L = num_L / max(den_L, EPS)

    lam_eff = lam_pps * (1.0 - P_block)  # 성공적으로 들어오는 도착률
    if lam_eff <= EPS:
        return 0.0  # 거의 모두 차단 → 대기 X

    W_sec = L / lam_eff                      # 시스템 체류시간
    Wq_sec = max(0.0, W_sec - 1.0 / mu_pps)  # 큐잉 지연
    return Wq_sec * 1e3                      # ms
# ---------------------------------------------------------
# 핵심 클래스
# ---------------------------------------------------------
class FlowController:
    def __init__(self, rtpg, satellites, ground_relays, isl_capacity, gsl_up_capacity, gsl_down_capacity):
        self.rtpg = rtpg
        self.isl_capacity = isl_capacity
        self.gsl_up_capacity = gsl_up_capacity
        self.gsl_down_capacity = gsl_down_capacity
        self.satellites = satellites
        self.ground_relays = ground_relays
        self.time = 0
        self.flow_log = defaultdict(dict)
        self.hold_until = defaultdict(int)
        self.hold_steps = 100

        # flows[(src,dst)] = [demand, path, detours]
        self.flows = {}
        # edge_loads[(u,v)] = 누적 패킷 수 (유방향: (u,v) 그대로 사용)
        self.edge_loads = defaultdict(float)
        self.new_flow = set()

    # -------- util --------
    def _edge_key(self, u, v):
        # 유방향 그래프이므로 (u, v) 순서를 그대로 유지합니다.
        return (u, v)

    def _path_edges(self, path):
        for a, b in zip(path, path[1:]):
            yield self._edge_key(a, b)

    def _edge_type(self, u, v):
        data = self.rtpg.G.get_edge_data(u, v)
        try:
            return data.get('type', 'isl')  # 'isl' 또는 'gsl' 가정
        except AttributeError:
            print(data)

    def get_path(self, fkey):
        return self.flows[fkey][1]

    def _cap_of_edge(self, u, v):
        data = self.rtpg.G.get_edge_data(u, v) or {}
        t = data.get('type', 'isl')
        if t == 'isl':
            return float(self.isl_capacity)
        elif t == 'gsl_up':
            return float(self.gsl_up_capacity)
        elif t == 'gsl_down':
            return float(self.gsl_down_capacity)
        return 0.0

    def _edges_set(self, path):
        return set(self._path_edges(path))

    def _residual_overcnt(self, old_path, new_path, demand, overloaded_edges):
        """ 후보 경로 적용 '후' 과부하(edge load > cap)인 엣지 개수 """
        old_set = self._edges_set(old_path)
        new_set = self._edges_set(new_path)
        affected = set(overloaded_edges) | old_set | new_set

        over_cnt = 0
        for (u, v) in affected:
            load = self.edge_loads.get((u, v), 0.0)
            if (u, v) in old_set: load -= float(demand)
            if (u, v) in new_set: load += float(demand)
            if load > self._cap_of_edge(u, v):
                over_cnt += 1
        return over_cnt

    def _log_path(self, fkey, path):
        self.flow_log[fkey][self.time] = list(path)

    def _detour_penalty_now(self, fkey, detours):
        # 신규 플로우는 0, 그 외는 (1 + detours)
        return 0 if (fkey in self.new_flow) else (1 + int(detours))

    def _current_base_for_lex(self):
        # 현 시점 최대 detour_penalty + 1  (파라미터 없이 lex 보장)
        max_pen = 0
        for fkey, (_, _, detours) in self.flows.items():
            pen = 0 if (fkey in self.new_flow) else (1 + int(detours))
            if pen > max_pen:
                max_pen = pen
        return (max_pen + 1) or 1

    def get_delay_score_of_path(self, path, demand):
        """
            경로 end-to-end 지연 근사 [ms] 반환.
            - potential_prop_delay: 별도 제공된 prop_delay_estimation() 결과 [ms]
            - potential_queue_delay: M/M/1 큐잉 지연 합 [ms]
            전제: demand / edge_loads / link capacity 모두 Mbps 단위.
            """
        # 1) 전파지연 (이미 구현되어 있다고 했으니 그대로 호출)
        potential_prop_delay = prop_delay_estimation(path, self.satellites, self.ground_relays)

        # 2) 큐잉 지연 (M/M/1/K)
        # 패킷 크기: 클래스에 있으면 사용, 없으면 기본값
        packet_bits = PACKET_SIZE_BITS  # 필요하면 PACKET_SIZE_BITS로 교체
        potential_queue_delay = 0.0
        for u, v in self._path_edges(path):
            # 링크 용량 [Mbps] & 버퍼(bit) → K(패킷)
            et = self._edge_type(u, v)
            if et == 'isl':
                cap_mbps = float(self.isl_capacity)
            elif et == 'gsl_up':
                cap_mbps = float(self.gsl_up_capacity)
            elif et == 'gsl_down':
                cap_mbps = float(self.gsl_down_capacity)
            else:
                # 알 수 없는 타입은 경로 불가로 간주(단, 무한대는 피하고자 큰 수 리턴이 싫다면 그냥 continue도 가능)
                return math.inf

            if cap_mbps <= 0.0:
                return math.inf

            # 현재 부하 + 이 플로우 수요 [Mbps]
            lam_mbps = float(self.edge_loads.get((u, v), 0.0)) + float(demand)

            # pps 변환
            C_bps = cap_mbps * 1e6
            mu_pps = C_bps / max(packet_bits, 1)
            lam_pps = (lam_mbps * 1e6) / max(packet_bits, 1)

            # K 계산(버퍼를 패킷 개수로 환산)
            K = 500

            # 엣지 큐잉 지연(ms)
            Wq_ms = _mm1k_queue_delay_ms(mu_pps, lam_pps, K)
            potential_queue_delay += Wq_ms

        delay_score = potential_prop_delay + potential_queue_delay
        return delay_score

    def build_flows(self, src, dst, demand):
        path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='weight')
        detour_counts = 0
        self.flows[(src, dst)] = [float(demand), list(path), detour_counts]
        self.new_flow.add((src, dst))
        self._log_path((src,dst), path)

    def build_load_from_totals(self):
        """
        현재 flow[0] (스케일/게이팅 반영된 값)을 이용하여 edge_loads를 새로 누적한다.
        """
        self.edge_loads.clear()
        for demands, path, _ in self.flows.values():
            for e in self._path_edges(path):
                self.edge_loads[e] += float(demands)

    # -------- 4. 용량 초과 엣지 목록 산출 --------
    def check_map(self, excepted_edge = None):
        if excepted_edge is None:
            excepted_edge = []
        update_list = []
        for u, v, data in self.rtpg.G.edges(data=True):
            k = self._edge_key(u, v)
            if k in excepted_edge:
                continue
            w = self.edge_loads.get(k, 0.0)
            link_type = data.get('type')
            if link_type == 'isl':
                cap = self.isl_capacity
            elif link_type == 'gsl_up':
                cap = self.gsl_up_capacity
            elif link_type == 'gsl_down':
                cap = self.gsl_down_capacity
            else:
                continue
            if w > cap:
                update_list.append((k, w, cap))
        update_list.sort(key=lambda x: (x[1] / (x[2] if x[2] else 1)), reverse=True)
        return update_list

    # -------- 가중치 맵을 그래프에 반영(다익스트라에서 사용할 'pen_w') --------
    def _apply_penalty_weights(self):
        # 간단히: pen_w = base(1) + load
        for u, v in self.rtpg.G.edges():
            k = self._edge_key(u, v)
            self.rtpg.G[u][v]['pen_w'] = self.edge_loads.get(k, 0) + 1

    # -------- 6~8. 문제 엣지를 사용하는 flow를 정렬 --------
    def _flows_using_edge(self, edge_k):
        flows = []
        for fkey, (demands, path, detours) in self.flows.items():
            # _path_edges가 (u,v) 순서를 보장하므로 그대로 비교
            if edge_k in self._path_edges(path):
                flows.append((fkey, demands, path, detours))

        return flows

    def _remove_flow_contribution(self, fkey):
        demands, path, detours = self.flows[fkey]
        for e in self._path_edges(path):
            self.edge_loads[e] -= demands

    def _add_flow_contribution(self, fkey):
        demands, path, detours = self.flows[fkey]
        for e in self._path_edges(path):
            self.edge_loads[e] += demands

    def _get_secondary_path_and_cost(self, fkey, overloaded_edges):
        src, dst = fkey
        demand, old_path, detours = self.flows[(src, dst)]

        # (1) 기존 기여 잠시 제거
        self._remove_flow_contribution((src, dst))
        old_score = self.get_delay_score_of_path(old_path, demand)
        self._apply_penalty_weights()

        # (2) 과부하 엣지 차단
        original_weights = {}
        for u, v in overloaded_edges:
            if self.rtpg.G.has_edge(u, v):
                original_weights[(u, v)] = self.rtpg.G[u][v].get('pen_w', 0.0)
                self.rtpg.G[u][v]['pen_w'] = float('inf')

        # (3) 새 경로 탐색  ← 반환값 언팩 필요!
        new_path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='pen_w')
        new_score = self.get_delay_score_of_path(new_path, demand)

        # (4) 원복
        for (u, v), w in original_weights.items():
            if self.rtpg.G.has_edge(u, v):
                self.rtpg.G[u][v]['pen_w'] = w

        # (5) !!! 기존 기여 복구(평가만 했으니 원상 복구)
        self._add_flow_contribution((src, dst))

        # 영향을 받는 엣지 집합: 차단엣지 ∪ old_path ∪ new_path
        old_set = set(self._path_edges(old_path))
        new_set = set(self._path_edges(new_path))
        affected = set(overloaded_edges) | old_set | new_set

        # 후보 적용 '후' 과부하 엣지 개수
        over_cnt_after = 0
        for (u, v) in affected:
            load = self.edge_loads.get((u, v), 0.0)
            if (u, v) in old_set:
                load -= float(demand)
            if (u, v) in new_set:
                load += float(demand)
            if load > self._cap_of_edge(u, v):
                over_cnt_after += 1

        # detour_penalty: 신규 0, 그 외 (1 + detours)
        detour_penalty = 0 if (fkey in self.new_flow) else (1 + int(detours))

        # base = 현재 시점 모든 flow의 detour_penalty 최대값 + 1  (파라미터 불필요)
        max_pen = 0
        for gkey, (_, _, g_detours) in self.flows.items():
            pen = 0 if (gkey in self.new_flow) else (1 + int(g_detours))
            if pen > max_pen:
                max_pen = pen
        base = (max_pen + 1) or 1

        system_cost = base * over_cnt_after + detour_penalty
        delay_cost = new_score - old_score

        return new_path, delay_cost, system_cost

    # -------- 9. 반복 재라우팅 --------
    def solve(self, max_iter = 2000):
        """과부하 엣지가 없어질 때까지 반복적으로 재라우팅을 수행합니다."""
        update_list = self.check_map()
        iteration_count = 0
        updated_flow = set()
        used_paths = {}
        excepted_edge = []

        while update_list:
            edge0_tuple = update_list[0]
            edge0, load0, cap0 = edge0_tuple

            iteration_count += 1
            if iteration_count % 50 == 0:
                print(f"[Solve iter {iteration_count}] Overloaded edges: {len(update_list)}. Fixing edge: {edge0} (load: {load0:.2f}, cap: {cap0:.2f})")
            if iteration_count > max_iter:
                print("Max iterations reached, stopping.")
                break

            flows = self._flows_using_edge(edge0)
            if iteration_count % 50 == 0:
                print(f"  - Found {len(flows)} flows using edge {edge0}.")
                # print(f"Top 10 flows by (detours, demand): {flows[:10]}")
                # print(f"Bottom 10 flows by (detours, demand): {flows[-10:]}")

            # 1. 재라우팅 후보 찾기
            best_candidate = None
            best_key = (float('inf'), float('inf'))
            overloaded_edges = [item[0] for item in update_list]  # 현재 과부하인 모든 엣지

            for fkey, demand, path, detours in flows:
                if self.time < self.hold_until[fkey]:
                    continue

                # 과부하 엣지들을 회피하는 대체 경로 탐색
                block_edges = overloaded_edges + excepted_edge
                new_path, delay_cost, system_cost = self._get_secondary_path_and_cost(fkey, block_edges)
                if new_path == path or (fkey in used_paths and new_path in used_paths[fkey]):
                    continue

                cand_key = (system_cost, delay_cost)
                if cand_key < best_key:
                    best_key = cand_key
                    best_candidate = {"fkey": fkey, "new_path": new_path,
                                      "delay_cost": delay_cost, "system_cost": system_cost}

            # 2. 최적 후보로 재라우팅 수행
            if best_candidate:
                fkey = best_candidate["fkey"]
                new_path = best_candidate["new_path"]
                # if fkey in self.new_flow:
                #     print(f"  - Rerouting new flow {best_candidate['fkey']} with new path {best_candidate['new_path']}. Delay cost: {best_candidate['delay_cost']:.2f}, System cost: {best_candidate['system_cost']:.2f}, detour count: { self.flows[fkey][2]}")
                # else:
                #     print(f"  - Rerouting flow {best_candidate['fkey']} with new path {best_candidate['new_path']}. Delay cost: {best_candidate['delay_cost']:.2f}, System cost: {best_candidate['system_cost']:.2f}, detour count: { self.flows[fkey][2]}")


                # 기존 경로의 부하 제거
                self._remove_flow_contribution(fkey)
                # flow 정보 업데이트 (경로, 우회 횟수)
                old_path = list(self.flows[fkey][1])
                self.flows[fkey][1] = new_path
                if fkey not in self.new_flow and fkey not in updated_flow:
                    self.flows[fkey][2] += 1
                updated_flow.add(fkey)
                if fkey in used_paths.keys():
                    used_paths[fkey].append(old_path)
                else:
                    used_paths[fkey] = [old_path]
                # 새 경로의 부하 추가
                self._add_flow_contribution(fkey)
                self._log_path(fkey, new_path)
            else:
                # 이 엣지를 사용하는 어떤 flow도 재라우팅할 수 없는 경우
                # print(f"Warning: Could not find a viable reroute for any flow using edge {edge0}. Skipping this edge.")
                # 무한 루프를 피하기 위해 다음 반복에서 이 엣지를 제외
                excepted_edge.append(edge0)
                update_list.pop(0)
                continue

            # 상태가 변경되었으므로 다시 전체 초과 엣지 목록 계산
            update_list = self.check_map(excepted_edge)
        return updated_flow

    def hold_flows(self, time, updated_flows):
        for key in updated_flows:
            self.hold_until[key] = time + self.hold_steps

    def fix_flow(self, fkey, cur_id, new_next_hop):
        # 전제: fkey ∈ self.flows, cur_id ∈ 경로
        demand, old_path, detours = self.flows[fkey]
        base = list(old_path)

        # 2) cur_id 위치 찾기
        idx_s = base.index(cur_id)

        # 3) src→s_id 구간에서 next_hop이 이미 존재하면, 해당 지점~s_id 직전 삭제
        #    (루프/중복 제거, s_id는 남겨둠)
        if new_next_hop in base:
            k = base.index(new_next_hop)  # [0, idx_s) 범위에서 탐색
            if k < idx_s:
                del base[k + 1:idx_s + 1]
            else:  # k > idx_s
                del base[idx_s + 1:k]
        else:
            # 4) s_id 다음에 next_hop 삽입 (이미 그 다음이면 생략)
            base.insert(idx_s + 1, new_next_hop)

        # === 실제 반영 ===
        self._remove_flow_contribution(fkey)
        self.flows[fkey][1] = base
        self._add_flow_contribution(fkey)
        self._log_path(fkey, base)

    # -------- 결과 꺼내기 (필요 시) --------
    def get_flows(self):
        return self.flows

    def export_routing_tables_to_csv_json(self, filepath):
        """
        각 노드별 라우팅 테이블을 json 문자열로 csv에 저장합니다.
        헤더: ["node_id", "table"]
        table: { (src, dst): v }
        """
        node_tables = defaultdict(dict)
        for (src, dst), (_, path, _) in self.flows.items():
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key_str = f"{src},{dst}"  # 튜플을 문자열로 변환
                node_tables[u][key_str] = v

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "table"])
            for node_id, table in node_tables.items():
                table_str = json.dumps(table)
                writer.writerow([node_id, table_str])

    def export_flow_log_json(self, filepath):
        # JSON 직렬화를 위해 키를 문자열로 변환
        serializable = {}
        for (src, dst), tmap in self.flow_log.items():
            k = f"{src},{dst}"
            serializable[k] = {str(t): path for t, path in tmap.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False)

    def export_flow_log_csv(self, filepath):
        # columns: src,dst,time,path_json
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["src","dst","time","path"])
            for (src,dst), tmap in self.flow_log.items():
                for t, path in sorted(tmap.items()):
                    w.writerow([src, dst, t, json.dumps(path)])


    def get_edge_loads(self):
        return dict(self.edge_loads)

    def visualize_load(self, satellites, gen_rate: float, t: int):
        """
        현재 네트워크의 링크 부하를 히트맵으로 시각화합니다.
        이 메서드는 solve()가 완료된 후 호출되어야 합니다.
        """
        N = self.rtpg.N
        M = self.rtpg.M

        # 데이터 구조 초기화: data[sat_idx][orbit_idx] -> [top, bottom, left, right] load
        data = [[[0.0, 0.0, 0.0, 0.0] for _ in range(M)] for _ in range(N)]

        # node_id to (orbit_idx, sat_idx) 매핑 생성
        node_to_coord = {sat.node_id: (sat.orbit_idx, sat.sat_idx_in_orbit) for sat in satellites.values()}

        # 모든 엣지를 순회하며 부하 계산 및 데이터 구조 채우기
        for (u, v), used in self.edge_loads.items():
            edge_type = self._edge_type(u, v)
            if edge_type != 'isl':
                continue
            total = self.isl_capacity # ISL 용량 사용
            load = min(used / total, 1.0)

            sat = satellites[u]
            n, m = node_to_coord[u]
            if sat.isl_up == v:
                data[n][m][0] = load  # top link
            elif sat.isl_down == v:
                data[n][m][1] = load   # bottom link
            elif sat.isl_left == v:
                data[n][m][2] = load   # left link
            else: # sat.isl_right == v:
                data[n][m][3] = load  # right link

        # 시각화 함수 호출
        load_heatmap(gen_rate=gen_rate, t=t, N=N, M=M, data=data)

    def visualize_flow_load_distribution(self, gen_rate: float, t: int):
        """
        현재 self.flows에 있는 flow들의 부하(load) 분포를 히스토그램으로 시각화합니다.
        flow 개수, demand 총합, demand 평균 기준으로 세 개의 서브플롯을 생성합니다.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not self.flows:
            print("No flows to visualize.")
            return

        loads = [flow[0] for flow in self.flows.values()]
        num_bins = 15

        # 3개의 서브플롯 생성
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle(f'Flow Expected Demand Distribution at {t}ms (Rate: {gen_rate} Mbps)', fontsize=16)

        # --- 서브플롯 1: Flow 개수 기준 히스토그램 ---
        counts, bin_edges, _ = ax1.hist(loads, bins=num_bins, alpha=0.7, color='blue')
        ax1.set_title('Distribution by Flow Count')
        ax1.set_xlabel('Expected Demand')
        ax1.set_ylabel('Number of Flows (Count)')
        ax1.grid(True)

        # 통계 정보 추가 (첫 번째 플롯에만)
        mean_load = np.mean(loads)
        median_load = np.median(loads)
        max_load = np.max(loads)
        min_load = np.min(loads)
        std_load = np.std(loads)

        stats_text = (
            f"Mean: {mean_load:,.2f}\n"
            f"Median: {median_load:,.2f}\n"
            f"Max: {max_load:,.2f}\n"
            f"Min: {min_load:,.2f}\n"
            f"Std Dev: {std_load:,.2f}"
        )
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        # --- 서브플롯 2: Demand 총합 기준 히스토그램 ---
        # 동일한 bin_edges를 사용하여 구간을 일치시킵니다.
        sums, _, _ = ax2.hist(loads, bins=bin_edges, weights=loads, alpha=0.7, color='green')
        ax2.set_title('Distribution by Sum of Demand')
        ax2.set_xlabel('Expected Demand')
        ax2.set_ylabel('Sum of Expected Demand')
        ax2.grid(True)

        # --- 서브플롯 3: Demand 평균 기준 막대 그래프 ---
        # 0으로 나누는 것을 방지하며 평균 계산
        averages = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts != 0)

        # 막대 그래프의 x축 위치 계산 (각 bin의 중앙)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax3.bar(bin_centers, averages, width=np.diff(bin_edges), alpha=0.7, color='purple')
        ax3.set_title('Distribution by Average of Demand')
        ax3.set_xlabel('Expected Demand Bins')
        ax3.set_ylabel('Average Expected Demand')
        ax3.grid(True)

        # 레이아웃 조정 및 출력
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])  # suptitle과의 간격 조정
        plt.show()
# ---------------------------------------------------------
# (선택) 모듈 자체 실행 시 데모: 1~9 순서대로 수행
#  - 프로젝트의 기존 로더/파라미터를 그대로 사용
#  - 메인 시뮬레이터와 충돌 없도록 독립 실행 전용
# ---------------------------------------------------------
# if __name__ == "__main__":
#     # 기존 main이 쓰는 유틸 가져와 동일한 방식으로 RTPG 구성
#     from parameters.PARAMS import N, M, F, inclination_deg, ISL_RATE_LASER, SGL_KA_UPLINK, \
#     SGL_KA_DOWNLINK, TAU, TOTAL_TIME, PACKET_SIZE_BITS, TRAFFIC_DENSITY
#     from utils.walker_constellation import WalkerConstellation
#     from utils.rtpg_mapper import RTPGMapper
#     from utils.loader import load_ground_relays_from_csv, load_event_schedule, prepare_node_routing_metadata
#     from utils.RTPGGraph import RTPGGraph
#
#     directory = "flow_log(10seconds)"
#     MAX_ITERATIONS = 2000
#     # GENERATION_RATE_LIST_TEST = [360]  # Mbps
#     # GENERATION_RATE_LIST_TEST = [320, 80, 120]  # Mbps
#     GENERATION_RATE_LIST_TEST = [200,240,280,160,40]  # Mbps
#     update_duration = 1
#     T_ms = 95.4 * 60 * 1000  # 궤도 주기 (밀리초)
#     omega_s = 2 * pi / T_ms  # delta phase (deg)
#     # total_time = TOTAL_TIME  # 시뮬레이션 총 시간 (밀리초)
#     start_time = 0
#     total_time = 10000 # 시뮬레이션 총 시간 (밀리초)
#
#     # (참고) 기존 main의 초기화 흐름을 그대로 따름: Walker → Mapper → RTPG 구성 후
#     # connect_isl_links(), connect_ground_links() 호출 구조. :contentReference[oaicite:6]{index=6}
#
#     for generation_rate in GENERATION_RATE_LIST_TEST:
#         # 2) 파라미터/객체 초기화
#         mapper = RTPGMapper(N, M, F, inclination_deg)
#         constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=550, inclination_deg=inclination_deg)
#         constellation.generate_constellation()
#         satellites = constellation.get_all_satellites()  # dict
#
#         relay_csv_path = '../parameters/Ground_Relay_Coordinates.csv'
#         traffic_schedule_path = f'../parameters/uneven traffic(latest)/events_{generation_rate}Mbps.csv'
#         # table_csv_path = f'./routing table/routing_tables_{generation_rate}Mbps.csv'
#         ground_relays = load_ground_relays_from_csv(relay_csv_path, N * M)
#         print(f"Loading traffic schedule from {traffic_schedule_path} ...")
#         traffic_schedule = load_event_schedule(traffic_schedule_path, total_time)
#         print(f"Loaded {len(traffic_schedule)} traffic events.")
#
#         for gr in ground_relays.values():
#             prepare_node_routing_metadata(gr, mapper, 550)
#
#         # RTPG 구성 (기존 update_rtpg와 동일 논리)
#         rtpg = RTPGGraph(N=N, M=M, F=F)
#         sat_region_indices = mapper.batch_map(satellites.values())
#         rtpg.update_rtpg(satellites.values(), ground_relays.values(), sat_region_indices)
#
#
#         # (간단 용량 값) edge['type']이 'isl'/'gsl'로 들어옴. :contentReference[oaicite:8]{index=8}
#         # 여기서는 단순히 "한 플래닝에서 허용 가능한 패킷 수"로 임의 상수 지정
#         ISL_CAP = floor((ISL_RATE_LASER)/1000000)# bps -> Mbps
#         GSL_UP_CAP = floor((SGL_KA_UPLINK)/1000000)
#         GSL_DOWN_CAP = floor((SGL_KA_DOWNLINK)/1000000)
#         print(f"ISL_CAP={ISL_CAP}, GSL_UP_CAP={GSL_UP_CAP}, GSL_DOWN_CAP={GSL_DOWN_CAP}  (Mbps)")
#         totals = set()
#         gen = FlowController(rtpg, satellites, ground_relays, ISL_CAP, GSL_UP_CAP, GSL_DOWN_CAP)
#
#         # 구간별로 반복
#         for time in tqdm(range(start_time, total_time, update_duration)):
#             gen.new_flow.clear()
#             gen.time = time
#             if time != 0:
#                 for s in satellites.values():
#                     s.update_position(omega_s, update_duration)
#                 if time % 600 == 0:
#                     for s in satellites.values():
#                         s.update_lat_lon_for_RTPG()
#                     rtpg.reset_graph()
#                     sat_region_indices = mapper.batch_map(satellites.values())
#                     rtpg.update_rtpg(satellites.values(), ground_relays.values(), sat_region_indices)
#
#             generated_packets = traffic_schedule.pop(time, [])
#             for packet_data in generated_packets:
#                 src, dst, num_pkts = packet_data
#                 if (src, dst) not in totals:
#                     totals.add((src, dst))
#                     gen.build_flows(src, dst, generation_rate)
#                     gen.build_load_from_totals()
#                     updated_flows = gen.solve(max_iter=MAX_ITERATIONS) | {(src, dst)}
#                     for key in updated_flows:
#                         gen.hold_until[key] = gen.time + gen.hold_steps
#                 else:
#                     continue
#
#         # 결과 요약 출력
#         flows = gen.get_flows()
#         total_detours = sum(v[2] for v in flows.values())
#         print(f"[{generation_rate} Mbps], duration={start_time}ms to {total_time}ms, flows={len(flows)}, total_detours={total_detours}")
#         # 결과 시각화
#         # gen.visualize_load(satellites, gen_rate=generation_rate, t=total_time)
#         # 라우팅 테이블 csv로 저장
#         flow_path = f'./{directory}/flows_{generation_rate}.pkl'
#         json_path = f'./{directory}/flows_{generation_rate}.json'
#         save_flow_log_pkl(gen.flow_log, flow_path)
#         gen.export_flow_log_csv(json_path)
#         # cache_path = f'./{directory}/cache/{generation_rate}-{time}.pkl'
#         # gen.export_routing_tables_to_csv_json(table_csv_path)
#         # gen.export_cache(cache_path)



