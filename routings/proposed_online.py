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
import os
import pickle


def compute_segment_lambdas(
    traffic_schedule: dict,
    total_time: int,
    update_duration: int
):

    real_totals = {}
    segment_lambdas = {}
    for t_start in range(0, total_time, update_duration):
        t_end = min(t_start + update_duration, total_time)
        lambda_hist = 0
        real_total = {}

        for t, events in traffic_schedule.items():
            if t_start <= t < t_end:
                for src, dst, num_pkts in events:
                    if (src, dst) not in real_total:
                        real_total[(src, dst)] = 0
                    real_total[(src, dst)] += int(num_pkts) * PACKET_SIZE_BITS
                    lambda_hist += int(num_pkts)

        segment_lambdas[t_start] = lambda_hist
        real_totals[t_start] = real_total

    return real_totals, segment_lambdas



def build_grid_and_weights(satellites, traffic_density, lat_max_abs=60.0):
    import numpy as np
    tm = np.array(traffic_density, dtype=float)
    total_tm = np.sum(tm)
    rows, cols = tm.shape
    sats_group_by_grid = [[[] for _ in range(cols)] for _ in range(rows)]

    # grid에 위성 배치
    from routings.packet_generator import grouping_satellite_to_lat_lon_grid
    sats_group_by_grid = grouping_satellite_to_lat_lon_grid(
        satellites.values(), sats_group_by_grid, lat_max_abs=lat_max_abs
    )

    # 위성별 cell 매핑 및 가중치 계산
    sat_id_to_cell = {}
    W_sat = {}
    for i in range(rows):
        for j in range(cols):
            sat_ids = sats_group_by_grid[i][j]
            n = len(sat_ids)
            if n == 0:
                continue
            w = tm[i][j] / (n*total_tm)
            for sid in sat_ids:
                sat_id_to_cell[sid] = (i, j)
                W_sat[sid] = w
    return W_sat, sat_id_to_cell, sats_group_by_grid

def build_totals_from_map_grid_filtered(sat_ids, W_sat, sat_id_to_cell):
    sumW = sum(W_sat.get(s, 0.0) for s in sat_ids) or 1.0
    pi = {s: W_sat.get(s, 0.0) / sumW for s in sat_ids}
    Pi = {}
    for s in sat_ids:
        for d in sat_ids:
            if s == d:
                continue
            if sat_id_to_cell[s] == sat_id_to_cell[d]:
                continue
            Pi[(s, d)] = pi[s] * pi[d]
    # Z = sum(Pi.values()) or 1.0
    # print( sum(Pi.values())) # 0.9563540349191524
    # Pi = {k: v / Z for k, v in Pi.items()}
    return {k: v for k, v in Pi.items()}

def get_or_build_totals(sat_ids, W_sat, sat_id_to_cell):
    totals = build_totals_from_map_grid_filtered(
        sat_ids=sat_ids,
        W_sat=W_sat,
        sat_id_to_cell=sat_id_to_cell,
    )
    return totals

# ---------------------------------------------------------
# 핵심 클래스
# ---------------------------------------------------------
class RoutingTableGenerator:
    def __init__(self, rtpg, isl_capacity, gsl_up_capacity, gsl_down_capacity):
        self.rtpg = rtpg
        self.isl_capacity = isl_capacity
        self.gsl_up_capacity = gsl_up_capacity
        self.gsl_down_capacity = gsl_down_capacity
        self.time = 0

        # flows[(src,dst)] = [total_pkts, path(list of node_ids), detour_count]
        self.flows = {}
        # edge_loads[(u,v)] = 누적 패킷 수 (유방향: (u,v) 그대로 사용)
        self.edge_loads = defaultdict(float)

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

    # -------- 2. 초기화 + 3. flows 집계/최단경로 --------
    def load_flows(self, totals, cache_path):
        self.flows.clear()
        cache_needs_update = False

        # 1. 캐시 파일이 존재하면 불러오기
        if os.path.exists(cache_path):
            print(f"Loading flows from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                self.flows = pickle.load(f)

            # 2. totals와 캐시의 키 목록 비교
            totals_keys = set(totals.keys())
            flows_keys = set(self.flows.keys())

            missing_keys = totals_keys - flows_keys
            stale_keys = flows_keys - totals_keys

            # 3. 캐시에만 존재하는 오래된 flow 삭제
            if stale_keys:
                print(f"Removing {len(stale_keys)} stale flows from cache.")
                for key in stale_keys:
                    del self.flows[key]
                cache_needs_update = True

            # 4. 누락된 flow 계산 및 추가
            if missing_keys:
                if not self.flows and not stale_keys:  # 캐시가 아예 없었던 경우
                    print(f"Cache not found. Calculating {len(missing_keys)} flows...")
                else:  # 캐시가 불완전하거나 오래된 경우
                    print(f"Cache is outdated. Calculating {len(missing_keys)} missing flows...")

                for src, dst in tqdm(missing_keys, desc="Calculating missing flows"):
                    w_p = totals[(src, dst)]
                    path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='weight')
                    if path:
                        self.flows[(src, dst)] = [float(w_p), list(path), 0]
                cache_needs_update = True

            # 5. 변경 사항이 있으면 캐시 업데이트
            if cache_needs_update:
                print(f"Updating cache file: {cache_path}")
                cache_dir = os.path.dirname(cache_path)
                if cache_dir and not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                with open(cache_path, "wb") as f:
                    pickle.dump(self.flows, f)
                print(f"Saved updated flows to cache.")
            else:
                print("Flow cache is up to date.")
        else:
            print(f"Cache file not found. Calculating all {len(totals)} flows...")
            self.build_flows_from_totals(totals)
            print(f"Saving flows to cache: {cache_path}")
            cache_dir = os.path.dirname(cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cache_path, "wb") as f:
                pickle.dump(self.flows, f)
            print(f"Saved flows to cache.")

    def build_flows_from_totals(self, totals):
        for (src, dst), w_p in tqdm(totals.items(), desc="Building initial flows"):
            path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='weight')
            self.flows[(src, dst)] = [float(w_p), list(path), 0]

    # -------- 엣지 가중치 누적 --------
    def apply_lambda_hist_to_totals(self, lambda_hist: any):
        """
        totals 비율(flow[0])을 세그먼트 패킷 수 스케일(bits)로 변환해 flow[0]에 반영한다.
        edge_loads는 건드리지 않는다. (다음 단계인 build_load_from_totals에서 누적)

        Parameters
        ----------
        lambda_hist : int
            세그먼트 내 전체 패킷 수 합.
        beta : float
            외부 스케일 팩터 (기본 1.0). PMR 등 추가 스케일 필요 시 사용.
        """
        for fkey, flow in self.flows.items():
            path_activation_prop, path, detours = flow
            w_bits = float(path_activation_prop) * lambda_hist * PACKET_SIZE_BITS
            self.flows[fkey][0] = w_bits  # flow[0]에만 반영 (누적은 다음 단계에서)

    def build_load_from_totals(self):
        """
        현재 flow[0] (스케일/게이팅 반영된 값)을 이용하여 edge_loads를 새로 누적한다.
        """
        self.edge_loads.clear()
        for demands, path, _ in self.flows.values():
            w = float(demands)
            for e in self._path_edges(path):
                self.edge_loads[e] += w

    # -------- 4. 용량 초과 엣지 목록 산출 --------
    def check_map(self, delta=0.0):
        update_list = []
        for u, v, data in self.rtpg.G.edges(data=True):
            k = self._edge_key(u, v)
            w = self.edge_loads.get(k, 0.0)
            link_type = data.get('type')
            if link_type == 'isl':
                cap = self.isl_capacity * (1+delta)
            elif link_type == 'gsl_up':
                cap = self.gsl_up_capacity * (1+delta)
            elif link_type == 'gsl_down':
                cap = self.gsl_down_capacity * (1+delta)
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

    # -------- 6~8. 문제 엣지를 사용하는 flow를 정렬해 한 개씩 재라우팅 --------
    def _flows_using_edge(self, edge_k):
        users = []
        for fkey, (demands, path, detours) in self.flows.items():
            # _path_edges가 (u,v) 순서를 보장하므로 그대로 비교
            if edge_k in self._path_edges(path):
                users.append((fkey, demands, path, detours))
        # 1순위: 우회 횟수 오름차순, 2순위: 가중치 오름차순
        users.sort(key=lambda x: (x[3],x[1])) # (detours,total_pkts)
        return users

    def _remove_flow_contribution(self, fkey):
        demands, path, detours = self.flows[fkey]
        for e in self._path_edges(path):
            self.edge_loads[e] -= demands

    def _add_flow_contribution(self, fkey):
        demands, path, detours = self.flows[fkey]
        for e in self._path_edges(path):
            self.edge_loads[e] += demands

    def _reroute_one_flow(self, fkey, overloaded_edges):
        """한 개의 flow를 재라우팅합니다. 과부하 엣지를 회피하며, 성공 여부를 반환합니다."""
        total_pkts, old_path, detours = self.flows[fkey]
        self._remove_flow_contribution(fkey)
        self._apply_penalty_weights()

        # 현재 과부하 상태인 엣지 목록을 가져와서, 해당 엣지들의 가중치를 무한대로 설정
        original_weights = {}
        for u, v in overloaded_edges:
            if self.rtpg.G.has_edge(u, v):
                original_weights[(u, v)] = self.rtpg.G[u][v].get('pen_w', 0.0)
                self.rtpg.G[u][v]['pen_w'] = float('inf')

        src, dst = fkey
        # 가중치가 변경된 그래프에서 새로운 최단 경로 탐색
        new_path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='pen_w')

        # 탐색이 끝나면 가중치를 원래대로 복원
        for (u, v), w in original_weights.items():
            if self.rtpg.G.has_edge(u, v):
                self.rtpg.G[u][v]['pen_w'] = w

        # 새 경로를 찾지 못한 경우 (모든 경로에 과부하 엣지가 포함된 경우)
        if not new_path:
            # 원래 경로로 복구하고 실패 반환
            self._add_flow_contribution(fkey)
            return False

        # 새 경로를 찾은 경우, flow 정보 업데이트
        self.flows[fkey][1] = list(new_path)
        self.flows[fkey][2] = detours + 1
        self._add_flow_contribution(fkey)
        return True

    def _reroute_batch(self, fkeys, overloaded_edges):
        """여러 개의 flow를 한 번에 재라우팅합니다."""
        if not fkeys:
            return 0

        original_flows = {fkey: self.flows[fkey][:] for fkey in fkeys}
        for fkey in fkeys:
            self._remove_flow_contribution(fkey)

        self._apply_penalty_weights()
        original_weights = {}
        overloaded_keys = [e for e, _, _ in overloaded_edges]
        for u, v in overloaded_keys:
            if self.rtpg.G.has_edge(u, v):
                original_weights[(u, v)] = self.rtpg.G[u][v].get('pen_w', 1.0)
                self.rtpg.G[u][v]['pen_w'] = float('inf')

        successful_reroutes = 0
        for fkey in fkeys:
            src, dst = fkey
            new_path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='pen_w')

            if new_path:
                self.flows[fkey][1] = list(new_path)
                self.flows[fkey][2] += 1
                self._add_flow_contribution(fkey)
                successful_reroutes += 1
            else:
                self.flows[fkey] = original_flows[fkey]
                self._add_flow_contribution(fkey)

        for (u, v), w in original_weights.items():
            if self.rtpg.G.has_edge(u, v):
                self.rtpg.G[u][v]['pen_w'] = w

        return successful_reroutes

    # -------- 9. 반복 재라우팅 --------
    def solve(self):
        """과부하 엣지가 없어질 때까지 반복적으로 재라우팅을 수행합니다."""
        update_list = self.check_map()
        iteration_count = 0
        while update_list:
            edge0_tuple = update_list[0]
            edge0, load0, cap0 = edge0_tuple

            iteration_count += 1
            if iteration_count % 50 == 0:
                print(
                    f"[Solve iter {iteration_count}] Overloaded edges: {len(update_list)}. Fixing edge: {edge0} (load: {load0:.2f}, cap: {cap0:.2f})")
            if iteration_count > 2000:
                print("Max iterations reached, stopping.")
                break

            users = self._flows_using_edge(edge0)
            if not users:
                print(f"Warning: Overloaded edge {edge0} has no flows using it. Skipping.")
                update_list = [item for item in self.check_map() if item[0] != edge0]
                continue
            if iteration_count % 50 == 0:
                print(f"  - Found {len(users)} flows using edge {edge0}.")
                print(f"Top 10 flows by (detours, total_pkts): {users[:10]}")
                print(f"Bottom 10 flows by (detours, total_pkts): {users[-10:]}")

            # 초과분 만큼의 flow만 선택하여 재라우팅
            excess_load = load0 - cap0
            fkeys_to_reroute = []
            rerouted_load_sum = 0.0
            for fkey, load, _, _ in users:
                fkeys_to_reroute.append(fkey)
                rerouted_load_sum += load
                if rerouted_load_sum >= excess_load:
                    break

            if iteration_count % 50 == 0:
                print(
                    f"  - Excess load: {excess_load:.2f}. Rerouting {len(fkeys_to_reroute)} flows with total load {rerouted_load_sum:.2f}.")

            # 선택된 flow들을 배치 재라우팅
            rerouted_count = self._reroute_batch(fkeys_to_reroute, update_list)

            if rerouted_count == 0:
                print(f"Warning: Could not resolve overload on edge {edge0}. All selected flows are stuck. Stopping.")
                break

            # 상태가 변경되었으므로 다시 전체 초과 엣지 목록 계산
            update_list = self.check_map()

    def _reroute_batch_overall(self, overloaded_edges, k=5):
        """
        여러 개의 과부하 엣지를 전역적으로 고려하여 flow를 재라우팅합니다.
        1. 가장 심각한 k개의 엣지를 선택합니다.
        2. 해당 엣지를 사용하는 모든 flow를 수집합니다.
        3. 수집된 flow 중 일정 비율을 재라우팅합니다.
        """
        if not overloaded_edges:
            return 0

        # 1. 가장 심각한 k개의 엣지 선택
        top_k_edges_tuples = overloaded_edges[:k]
        top_k_edge_keys = {edge_tuple[0] for edge_tuple in top_k_edges_tuples}
        print(f'  - Considering top {len(top_k_edge_keys)} overloaded edges for global rerouting. (total overloaded: {len(overloaded_edges)})')


        # 2. 해당 엣지들을 사용하는 모든 flow를 중복 없이 수집 및 정렬
        all_users = {}  # fkey를 키로 사용하여 중복 방지
        for edge_k in top_k_edge_keys:
            users_of_one_edge = self._flows_using_edge(edge_k)
            for user_tuple in users_of_one_edge:
                fkey, demand = user_tuple[0], user_tuple[1]
                if fkey not in all_users:
                    all_users[fkey] = user_tuple

        if not all_users:
            return 0

        sorted_users = sorted(list(all_users.values()), key=lambda x: (x[3], x[1]))  # (detours, total_pkts)
        # print(f'  - Top 3 flows by (detours, total_pkts):')
        # for entry in sorted_users[:3]:
        #     print(f'    {entry}')
        # print(f'  - Bottom 3 flows by (detours, total_pkts): {sorted_users[-10:]}')
        # for entry in sorted_users[-3:]:
        #     print(f'    {entry}')

        # 3. 재라우팅할 flow의 개수 결정 및 선택
        num_candidates = len(sorted_users)
        num_total_flows = len(self.flows)
        #
        # 후보군의 100/k % 또는 전체 flow의 1% 중 큰 값
        num_to_reroute_by_k = math.ceil(num_candidates * 0.5)
        num_to_reroute_by_total = math.ceil(num_total_flows * 0.001)
        num_to_reroute = int(min(num_candidates, max(num_to_reroute_by_k, num_to_reroute_by_total)))

        # 실제 후보군 수를 넘지 않도록 조정
        num_to_reroute = min(num_to_reroute, num_candidates)

        fkeys_to_reroute = [fkey for fkey, _, _, _ in sorted_users[:num_to_reroute]]

        print(
            f"  - Rerouting {len(fkeys_to_reroute)} flows out of {num_candidates} candidates (total flows: {num_total_flows}).")

        # 4. 선택된 flow들을 일괄 재라우팅 (기존 _reroute_batch 로직과 동일)
        return self._reroute_batch(fkeys_to_reroute, overloaded_edges)

    def solve_overall(self, k=5, delta=0.0):
        """제안된 전역적 재라우팅 전략을 사용하여 과부하를 해결합니다."""
        update_list = self.check_map(delta=delta)
        iteration_count = 0
        while update_list:
            if len(update_list) == 1:
                self.solve()  # 기존 로컬 전략 사용
            else:
                iteration_count += 1
                # if iteration_count % 50 == 0:
                #     print(
                #         f"[Solve iter {iteration_count}] Overloaded edges: {len(update_list)}. Applying global strategy with k={k}.")
                if iteration_count > 2000:
                    print("Max iterations reached, stopping.")
                    break

                rerouted_count = self._reroute_batch_overall(update_list, k=k)

                # 재라우팅할 flow가 없거나, 시도했지만 모두 실패한 경우
                if rerouted_count == 0 and len(update_list) > 0:
                    # _reroute_batch_overall 내부에서 후보가 없는 경우는 0을 반환하므로, 교착 상태인지 확인 필요
                    # 간단하게는, reroute 시도가 있었는지 여부를 반환값으로 구분해야 하지만, 우선 경고 메시지로 처리
                    print(f"Warning: Could not reroute any flows for the current set of overloaded edges. Stopping.")
                    break

            # 상태가 변경되었으므로 다시 전체 초과 엣지 목록 계산
            update_list = self.check_map(delta=delta)

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

    def export_cache(self, filepath):
        # CSV 파일 저장 디렉토리 확인 및 생성
        csv_dir = os.path.dirname(filepath)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        # --- 관련 데이터 캐시 저장 ---
        cache_path = os.path.splitext(filepath)[0] + '.pkl'
        cache_data = {
            'rtpg': self.rtpg,
            'flows': self.flows,
            'edge_loads': self.edge_loads
        }

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Successfully saved cache data to {cache_path}")
        except Exception as e:
            print(f"Error saving cache to {cache_path}: {e}")

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
if __name__ == "__main__":
    # 기존 main이 쓰는 유틸 가져와 동일한 방식으로 RTPG 구성
    from parameters.PARAMS import N, M, F, inclination_deg, ISL_RATE_LASER, SGL_KA_UPLINK, \
    SGL_KA_DOWNLINK, TAU, TOTAL_TIME, PACKET_SIZE_BITS, TRAFFIC_DENSITY
    from utils.walker_constellation import WalkerConstellation
    from utils.rtpg_mapper import RTPGMapper
    from utils.loader import load_ground_relays_from_csv, load_event_schedule, prepare_node_routing_metadata
    from utils.RTPGGraph import RTPGGraph

    directory = "routing table(ver1.7.2)_nosig_avghist"
    # GENERATION_RATE_LIST_TEST = [360, 40, 240]  # Mbps
    # GENERATION_RATE_LIST_TEST = [320, 80, 200]  # Mbps
    GENERATION_RATE_LIST_TEST = [40]  # Mbps
    update_duration = 600
    T_ms = 95.4 * 60 * 1000  # 궤도 주기 (밀리초)
    omega_s = 2 * pi / T_ms  # delta phase (deg)
    # total_time = TOTAL_TIME  # 시뮬레이션 총 시간 (밀리초)
    start_time = 0
    total_time = 4200 # 시뮬레이션 총 시간 (밀리초)

    # (참고) 기존 main의 초기화 흐름을 그대로 따름: Walker → Mapper → RTPG 구성 후
    # connect_isl_links(), connect_ground_links() 호출 구조. :contentReference[oaicite:6]{index=6}

    for generation_rate in GENERATION_RATE_LIST_TEST:
        # 2) 파라미터/객체 초기화
        mapper = RTPGMapper(N, M, F, inclination_deg)
        constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=550, inclination_deg=inclination_deg)
        constellation.generate_constellation()
        satellites = constellation.get_all_satellites()  # dict

        # 1. future_constellation 및 future_rtpg 생성
        future_constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=550, inclination_deg=inclination_deg)
        future_constellation.generate_constellation()
        for s in future_constellation.get_all_satellites().values():
            s.update_position(omega_s, update_duration*2)
            s.update_lat_lon_for_RTPG()
        future_satellites = future_constellation.get_all_satellites()

        relay_csv_path = '../parameters/Ground_Relay_Coordinates.csv'
        traffic_schedule_path = f'../parameters/uneven traffic(latest)/events_{generation_rate}Mbps.csv'
        # table_csv_path = f'./routing table/routing_tables_{generation_rate}Mbps.csv'
        ground_relays = load_ground_relays_from_csv(relay_csv_path, N * M)
        print(f"Loading traffic schedule from {traffic_schedule_path} ...")
        traffic_schedule = load_event_schedule(traffic_schedule_path, total_time)
        print(f"Loaded {len(traffic_schedule)} traffic events.")

        for gr in ground_relays.values():
            prepare_node_routing_metadata(gr, mapper, 550)

        # RTPG 구성 (기존 update_rtpg와 동일 논리)
        rtpg = RTPGGraph(N=N, M=M, F=F)
        sat_region_indices = mapper.batch_map(satellites.values())
        rtpg.update_rtpg(satellites.values(), ground_relays.values(), sat_region_indices)

        future_rtpg = RTPGGraph(N=N, M=M, F=F)
        future_sat_region_indices = mapper.batch_map(future_satellites.values())
        future_rtpg.update_rtpg(future_satellites.values(), ground_relays.values(), future_sat_region_indices)

        # (간단 용량 값) edge['type']이 'isl'/'gsl'로 들어옴. :contentReference[oaicite:8]{index=8}
        # 여기서는 단순히 "한 플래닝에서 허용 가능한 패킷 수"로 임의 상수 지정
        ISL_CAP = floor((ISL_RATE_LASER*TAU) * update_duration)
        GSL_UP_CAP = floor((SGL_KA_UPLINK*TAU) * update_duration)
        GSL_DOWN_CAP = floor((SGL_KA_DOWNLINK*TAU) * update_duration)
        print(f"ISL_CAP={ISL_CAP}, GSL_UP_CAP={GSL_UP_CAP}, GSL_DOWN_CAP={GSL_DOWN_CAP}  (packets per planning)")

        # (추가) 위성별 지도 가중치
        W_sat, sat_id_to_cell, sats_group_by_grid = build_grid_and_weights(satellites, TRAFFIC_DENSITY)  # lon/lat -> (i,j) -> TRAFFIC_DENSITY[i][j]

        totals = get_or_build_totals(
            sat_ids=[sat_id for sat_id in satellites.keys()],
            W_sat=W_sat,
            sat_id_to_cell=sat_id_to_cell,
        ) # totals[(src,dst)] = w_p

        real_totals, lambda_hists = compute_segment_lambdas(
            traffic_schedule=traffic_schedule,
            total_time=total_time,
            update_duration=update_duration,
        )

        d_rmse = {}
        mean_error_ratios = {}

        # 구간별로 반복
        for time in range(start_time, total_time, update_duration):
            if time != 0:
                for s in satellites.values():
                    s.update_position(omega_s, update_duration)
                    s.update_lat_lon_for_RTPG()
                for f_s in future_satellites.values():
                    f_s.update_position(omega_s, update_duration)
                    f_s.update_lat_lon_for_RTPG()
                rtpg.reset_graph()
                future_rtpg.reset_graph()
                sat_region_indices = mapper.batch_map(satellites.values())
                future_sat_region_indices = mapper.batch_map(future_satellites.values())
                rtpg.update_rtpg(satellites.values(), ground_relays.values(), sat_region_indices)
                future_rtpg.update_rtpg(future_satellites.values(), ground_relays.values(), future_sat_region_indices)

            end_time = min(time + update_duration, total_time)

            # 구간 내 총 생성량
            lambda_hist = lambda_hists[time]
            real_total = real_totals[time]
            avg_lambda_hist = sum(lambda_hists.values()) / len(lambda_hists)

            print(f"\n=== Time {time}ms to {end_time}ms: Generating routing table for {generation_rate} Mbps traffic ===")
            print(f"Total packets to generate in this segment: {lambda_hist}")
            print(f"Average segment packets (for reference): {avg_lambda_hist:.2f}")

            # 1~3) 초기 상태 빌드
            gen = RoutingTableGenerator(rtpg, ISL_CAP, GSL_UP_CAP, GSL_DOWN_CAP)
            # 존재하지 않는 GSL 엣지 비활성화
            future_gsl_edges = set(
                (u, v) for u, v, data in future_rtpg.G.edges(data=True)
                if data.get('type', '').startswith('gsl')
            )
            disable_edges = []
            for u, v, data in rtpg.G.edges(data=True):
                if data.get('type', '').startswith('gsl'):
                    if (u, v) not in future_gsl_edges:
                        print(f"Disabling GSL edge ({u}, {v}) as it no longer exists in future constellation.")
                        # rtpg.G[u][v]['pen_w'] = float('inf')
                        disable_edges.append((u, v))
            for u, v in disable_edges:
                print(f"Removing edge ({u}, {v}) from the graph.")
                rtpg.G.remove_edge(u, v)

            cache_path = f"./routing table(ver2)/initial_flows-{time}.pkl"
            gen.load_flows(totals, cache_path)
            gen.apply_lambda_hist_to_totals(lambda_hist=avg_lambda_hist)

            inference_totals = totals.copy()
            for fkey, w_p in inference_totals.items():
                inference_totals[fkey] = w_p * avg_lambda_hist * PACKET_SIZE_BITS
            d_rmse[time], mean_error_ratios[time] = compute_rmse_between_totals_and_real_totals(inference_totals, real_total)
            # print(f"s_w of activated flows")
            # for key in real_total.keys():
            #     print(f"{key}: {totals.get(key,0)}")

            gen.build_load_from_totals()

            # 4~9) 반복 재라우팅
            gen.solve_overall(k=10)

            # 결과 요약 출력
            flows = gen.get_flows()
            total_detours = sum(v[2] for v in flows.values())
            print(f"[{generation_rate} Mbps], duration={time}ms to {end_time}ms, flows={len(flows)}, total_detours={total_detours}")
            # 결과 시각화
            # gen.visualize_load(satellites, gen_rate=generation_rate, t=total_time)
            # 라우팅 테이블 csv로 저장
            table_csv_path = f'./{directory}/routing_tables_{generation_rate}-{time}.csv'
            cache_path = f'./{directory}/cache/{generation_rate}-{time}.pkl'
            gen.export_routing_tables_to_csv_json(table_csv_path)
            gen.export_cache(cache_path)

        # --- d_rmse와 lambda_hists를 CSV로 저장하는 코드 시작 ---
        info_csv_path = f'./{directory}/info_{generation_rate}.csv'
        info_dir = os.path.dirname(info_csv_path)
        with open(info_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'time', 'value'])

            # d_rmse 데이터 저장
            for time_key, value in d_rmse.items():
                writer.writerow(['d_rmse', time_key, value])

            # 평균 오차율 데이터 저장
            for time_key, value in mean_error_ratios.items():
                writer.writerow(['error_ratio', time_key, value])

            # lambda_hists 데이터 저장
            for time_key, value in lambda_hists.items():
                writer.writerow(['lambda', time_key, value])

        print(f"Saved d_rmse and lambda_hists to {info_csv_path}")

