# routings/proposed.py
# ---------------------------------------------------------
# "트래픽 맵 재가중 → 다익스트라" 기반 라우팅 테이블 생성기 (간단 구현)
# ---------------------------------------------------------

from collections import defaultdict

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
            path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='weight')
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
            if data.get('type') == 'isl':
                cap = self.isl_capacity
            else:  # 'gsl'
                # 간단히 보수적으로: 업/다운 중 더 작은 값을 기준으로 초과 판정
                cap = min(self.gsl_up_capacity, self.gsl_down_capacity)
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
            self.rtpg.G[u][v]['pen_w'] = 1.0 + self.edge_loads.get(k, 0.0)

    # -------- 6~8. 문제 엣지를 사용하는 flow를 정렬해 한 개씩 재라우팅 --------
    def _flows_using_edge(self, edge_k):
        users = []
        for fkey, (total_pkts, path, detours) in self.flows.items():
            # _path_edges가 (u,v) 순서를 보장하므로 그대로 비교
            if edge_k in self._path_edges(path):
                users.append((fkey, total_pkts, path, detours))
        # detour_count 오름차순, path 길이 내림차순
        users.sort(key=lambda x: (x[3], -len(x[2])))
        return users

    def _remove_flow_contribution(self, fkey):
        total_pkts, path, detours = self.flows[fkey]
        for e in self._path_edges(path):
            self.edge_loads[e] -= total_pkts

    def _add_flow_contribution(self, fkey):
        total_pkts, path, detours = self.flows[fkey]
        for e in self._path_edges(path):
            self.edge_loads[e] += total_pkts

    def _reroute_one_flow(self, fkey):
        # 7) 기존 가중치 제거 → pen_w 적용 → pen_w 기반 다익스트라 → 8) 새 경로 반영
        total_pkts, old_path, detours = self.flows[fkey]
        self._remove_flow_contribution(fkey)
        self._apply_penalty_weights()
        src, dst = fkey
        new_path, _ = self.rtpg.dijkstra_shortest_path(source_id=src, target_id=dst, weight='pen_w')
        self.flows[fkey][1] = list(new_path)
        self.flows[fkey][2] = detours + 1
        self._add_flow_contribution(fkey)

    # -------- 5,9. 반복 루프 --------
    def solve(self):
        update_list = self.check_map()
        iteration_count = 0
        while update_list:
            edge0 = update_list[0]
            iteration_count += 1
            if iteration_count % 100 == 0:
                print(f"[Solve iter {iteration_count}] Overloaded edges: {len(update_list)}. Fixing edge: {edge0}")
            users = self._flows_using_edge(edge0)
            if not users:
                # 해당 엣지를 쓰는 플로우가 없으면 다음 엣지로
                update_list = update_list[1:]
                continue
            # 가장 우선순위 높은 flow 한 개만 재라우팅
            fkey = users[0][0]
            self._reroute_one_flow(fkey)
            # 9) 다시 초과 엣지 탐색
            update_list = self.check_map()

    # -------- 결과 꺼내기 (필요 시) --------
    def get_flows(self):
        return self.flows

    def get_edge_loads(self):
        return dict(self.edge_loads)


# ---------------------------------------------------------
# (선택) 모듈 자체 실행 시 데모: 1~9 순서대로 수행
#  - 프로젝트의 기존 로더/파라미터를 그대로 사용
#  - 메인 시뮬레이터와 충돌 없도록 독립 실행 전용
# ---------------------------------------------------------
if __name__ == "__main__":
    # 기존 main이 쓰는 유틸 가져와 동일한 방식으로 RTPG 구성
    from parameters.PARAMS import GENERATION_RATE_LIST, N, M, F, inclination_deg, ISL_RATE_LASER, SGL_KA_UPLINK, SGL_KA_DOWNLINK, TAU, TOTAL_TIME
    from utils.walker_constellation import WalkerConstellation
    from utils.rtpg_mapper import RTPGMapper
    from utils.loader import load_ground_relays_from_csv, load_event_schedule, prepare_node_routing_metadata
    from utils.RTPGGraph import RTPGGraph

    # (참고) 기존 main의 초기화 흐름을 그대로 따름: Walker → Mapper → RTPG 구성 후
    # connect_isl_links(), connect_ground_links() 호출 구조. :contentReference[oaicite:6]{index=6}

    for generation_rate in GENERATION_RATE_LIST:
        # 2) 파라미터/객체 초기화
        mapper = RTPGMapper(N, M, F, inclination_deg)
        constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=550, inclination_deg=inclination_deg)
        constellation.generate_constellation()
        satellites = constellation.get_all_satellites()  # dict

        relay_csv_path = '../parameters/Ground_Relay_Coordinates.csv'
        traffic_schedule_path = f'../parameters/uneven traffic (3000flows)/events_{generation_rate}Mbps.csv'
        ground_relays = load_ground_relays_from_csv(relay_csv_path, N * M)
        traffic_schedule = load_event_schedule(traffic_schedule_path)

        for gr in ground_relays.values():
            prepare_node_routing_metadata(gr, mapper, 550)

        # RTPG 구성 (기존 update_rtpg와 동일 논리)
        rtpg = RTPGGraph(N=N, M=M, F=F)
        sat_region_indices = mapper.batch_map(satellites.values())
        for sat, region in zip(satellites.values(), sat_region_indices):
            sat.connected_grounds = []
            rtpg.add_satellite(sat, region)

        for gr in ground_relays.values():
            gr.connected_sats = []
            rtpg.add_relay(gr, (gr.region_asc, gr.region_desc), (gr.search_regions_asc, gr.search_regions_desc))

        rtpg.connect_isl_links()
        rtpg.connect_ground_links()  # only_isl=False 경로  :contentReference[oaicite:7]{index=7}

        # (간단 용량 값) edge['type']이 'isl'/'gsl'로 들어옴. :contentReference[oaicite:8]{index=8}
        # 여기서는 단순히 "한 플래닝에서 허용 가능한 패킷 수"로 임의 상수 지정
        ISL_CAP = ISL_RATE_LASER * TAU * TOTAL_TIME
        GSL_UP_CAP = SGL_KA_UPLINK * TAU * TOTAL_TIME
        GSL_DOWN_CAP = SGL_KA_DOWNLINK * TAU * TOTAL_TIME

        # 1~3) 초기 상태 빌드
        gen = RoutingTableGenerator(rtpg, ISL_CAP, GSL_UP_CAP, GSL_DOWN_CAP)
        gen.build_initial_state(traffic_schedule)

        # 4~9) 반복 재라우팅
        gen.solve()

        # 결과 요약 출력
        flows = gen.get_flows()
        total_detours = sum(v[2] for v in flows.values())
        print(f"[{generation_rate} Mbps] flows={len(flows)}, total_detours={total_detours}")

        # 추가된 시각화 함수 호출
        rtpg.visualize_flow_paths(flows)
