# routings/proposed.py
# ---------------------------------------------------------
# "트래픽 맵 재가중 → 다익스트라" 기반 라우팅 테이블 생성기 (간단 구현)
# ---------------------------------------------------------
import csv, json
from collections import defaultdict
from math import floor

from utils.plot_maker import load_heatmap
from numpy import pi

# (참고) RTPGGraph는 networkx Graph를 self.G로 보유, dijkstra_shortest_path 사용
# - dijkstra_shortest_path(source_id, target_id, weight: str) → (path, length(hops))
#   (가중치 이름은 'weight' 등 문자열, 반환 length는 hop 수) [NetworkX wrapper]
#   -> 본 코드에선 'pen_w'를 전달해 혼잡-인지 최단경로를 구함
#   (RTPGGraph 구현 참조)  # :contentReference[oaicite:2]{index=2}

# ---------------------------------------------------------
# 핵심 클래스
# ---------------------------------------------------------
class RoutingTableGenerator:
    def __init__(self, rtpg, isl_capacity, gsl_up_capacity, gsl_down_capacity):
        self.rtpg = rtpg
        self.isl_capacity = isl_capacity
        self.gsl_up_capacity = gsl_up_capacity
        self.gsl_down_capacity = gsl_down_capacity

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
        return data.get('type', 'isl')  # 'isl' 또는 'gsl' 가정

    # -------- 2. 초기화 + 3. flows 집계/최단경로 + 3. 엣지 가중치 누적 --------
    def build_initial_state(self, traffic_schedule):
        # 2) (src,dst)별 총 생성 패킷 집계
        totals = defaultdict(int)
        for _, events in traffic_schedule.items():
            for (src, dst, num_pkts) in events:
                totals[(src, dst)] += int(num_pkts)

        # flows 구성 + 기본 최단경로(옵션1) 계산
        for (src, dst), total_pkts in totals.items():
            path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='pen_w')
            self.flows[(src, dst)] = [total_pkts, list(path), 0]

        # 3) 누적합: 각 flow의 path 사용 엣지에 total_pkts 누적
        self.edge_loads.clear()
        for (src, dst), (total_pkts, path, _) in self.flows.items():
            for e in self._path_edges(path):
                self.edge_loads[e] += total_pkts

    # -------- 4. 용량 초과 엣지 목록 산출 --------
    def check_map(self):
        update_list = []
        for u, v, data in self.rtpg.G.edges(data=True):
            k = self._edge_key(u, v)
            w = self.edge_loads.get(k, 0.0)
            link_type = data.get('type')
            if link_type == 'isl':
                cap = self.isl_capacity
            else:  # 'gsl'
                if link_type == 'gsl_up':
                    cap = self.gsl_up_capacity
                else:
                    cap = self.gsl_down_capacity
            if w > cap:
                update_list.append((k, w, cap))
        # 초과율 높은 순으로 정렬
        update_list.sort(key=lambda x: (x[1] / (x[2] if x[2] else 1)), reverse=True)
        return [e for (e, _, _) in update_list]

    # -------- 가중치 맵을 그래프에 반영(다익스트라에서 사용할 'pen_w') --------
    def _apply_penalty_weights(self):
        # 간단히: pen_w = base(1) + load
        for u, v in self.rtpg.G.edges():
            k = self._edge_key(u, v)
            self.rtpg.G[u][v]['pen_w'] = self.edge_loads.get(k, 0) + 1

    # -------- 6~8. 문제 엣지를 사용하는 flow를 정렬해 한 개씩 재라우팅 --------
    def _flows_using_edge(self, edge_k):
        users = []
        for fkey, (total_pkts, path, detours) in self.flows.items():
            # _path_edges가 (u,v) 순서를 보장하므로 그대로 비교
            if edge_k in self._path_edges(path):
                users.append((fkey, total_pkts, path, detours))
        # 1순위: 우회 횟수 오름차순, 2순위: 가중치 오름차순
        users.sort(key=lambda x: (x[3],x[1])) # (detours,total_pkts)
        return users

    def _remove_flow_contribution(self, fkey):
        total_pkts, path, detours = self.flows[fkey]
        for e in self._path_edges(path):
            self.edge_loads[e] -= total_pkts

    def _add_flow_contribution(self, fkey):
        total_pkts, path, detours = self.flows[fkey]
        for e in self._path_edges(path):
            self.edge_loads[e] += total_pkts

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

    def solve(self):
        """과부하 엣지가 없어질 때까지 반복적으로 재라우팅을 수행합니다."""
        update_list = self.check_map()
        iteration_count = 0
        while update_list:
            edge0 = update_list[0]
            iteration_count += 1
            if iteration_count % 100 == 0:
                print(f"[Solve iter {iteration_count}] Overloaded edges: {len(update_list)}. Fixing edge: {edge0}")
            if iteration_count > 2000:
                print("Max iterations reached, stopping.")
                break

            users = self._flows_using_edge(edge0)
            if not users:
                # 해당 엣지를 쓰는 플로우가 없으면 다음 엣지로 (드문 경우)
                print(f"Warning: Overloaded edge {edge0} has no flows using it. Skipping.")
                update_list.pop(0)
                continue

            # 가장 우선순위 높은 flow부터 하나씩 재라우팅 시도
            rerouted_successfully = False
            for fkey, _, _, _ in users:
                if self._reroute_one_flow(fkey, update_list):
                    rerouted_successfully = True
                    break  # 하나 성공하면 루프 중단하고 상태 다시 확인

            if not rerouted_successfully:
                # 이 엣지를 사용하는 어떤 flow도 재라우팅할 수 없음 -> 교착 상태
                print(f"Warning: Could not resolve overload on edge {edge0}. All flows are stuck. Stopping.")
                break

            # 9) 다시 초과 엣지 탐색
            update_list = self.check_map()

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

def import_routing_tables_from_csv_json(rate, time):
    """
    저장된 라우팅 테이블 csv를 불러와 {node_id: {(src, dst): v}} 형태로 반환합니다.
    """
    filepath = f'./routing table(non-dir edge)/routing_tables_{rate}-{time}.csv'
    node_tables = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row["node_id"]
            table_str = row["table"]
            table_dict = json.loads(table_str)
            # 키를 튜플로 복원
            table = {}
            for k, v in table_dict.items():
                src, dst = k.split(",")
                # src, dst가 int라면 int로 변환 (필요시)
                try:
                    src = int(src)
                    dst = int(dst)
                except ValueError:
                    pass
                table[(src, dst)] = v
            node_tables[node_id] = table
    return node_tables

# ---------------------------------------------------------
# (선택) 모듈 자체 실행 시 데모: 1~9 순서대로 수행
#  - 프로젝트의 기존 로더/파라미터를 그대로 사용
#  - 메인 시뮬레이터와 충돌 없도록 독립 실행 전용
# ---------------------------------------------------------
if __name__ == "__main__":
    # 기존 main이 쓰는 유틸 가져와 동일한 방식으로 RTPG 구성
    from parameters.PARAMS import N, M, F, inclination_deg, ISL_RATE_LASER, SGL_KA_UPLINK, \
    SGL_KA_DOWNLINK, TAU, TOTAL_TIME, PACKET_SIZE_BITS
    from utils.walker_constellation import WalkerConstellation
    from utils.rtpg_mapper import RTPGMapper
    from utils.loader import load_ground_relays_from_csv, load_event_schedule, prepare_node_routing_metadata
    from utils.RTPGGraph import RTPGGraph

    GENERATION_RATE_LIST_TEST = [360]
    update_duration = 600
    T_ms = 95.4 * 60 * 1000  # 궤도 주기 (밀리초)
    omega_s = 2 * pi / T_ms  # delta phase (deg)
    # total_time = TOTAL_TIME  # 시뮬레이션 총 시간 (밀리초)
    total_time = 4200  # 시뮬레이션 총 시간 (밀리초)

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
        traffic_schedule_path = f'../parameters/uneven traffic (3000flows)/events_{generation_rate}Mbps.csv'
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
        ISL_CAP = floor((ISL_RATE_LASER*TAU / PACKET_SIZE_BITS) * update_duration)
        GSL_UP_CAP = floor((SGL_KA_UPLINK*TAU / PACKET_SIZE_BITS) * update_duration)
        GSL_DOWN_CAP = floor((SGL_KA_DOWNLINK*TAU /PACKET_SIZE_BITS) * update_duration)
        print(f"ISL_CAP={ISL_CAP}, GSL_UP_CAP={GSL_UP_CAP}, GSL_DOWN_CAP={GSL_DOWN_CAP}  (packets per planning)")

        # 구간별로 반복
        for start_time in range(0, total_time, update_duration):
            if start_time != 0:
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

            end_time = min(start_time + update_duration, total_time)
            # 해당 구간의 트래픽만 추출
            traffic_schedule_segment = {
                t: events for t, events in traffic_schedule.items()
                if start_time <= t < end_time
            }
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

            gen.build_initial_state(traffic_schedule_segment)

            # 4~9) 반복 재라우팅
            gen.solve()

            # 결과 요약 출력
            flows = gen.get_flows()
            total_detours = sum(v[2] for v in flows.values())
            print(f"[{generation_rate} Mbps], duration={start_time}ms to {end_time}ms, flows={len(flows)}, total_detours={total_detours}")

            # 추가된 시각화 함수 호출
            # rtpg.visualize_flow_paths(flows)
            # 결과 시각화
            gen.visualize_load(satellites, gen_rate=generation_rate, t=total_time)
            # 라우팅 테이블 csv로 저장
            table_csv_path = f'./routing table(non-dir edge)/routing_tables_{generation_rate}-{start_time}.csv'
            gen.export_routing_tables_to_csv_json(table_csv_path)
