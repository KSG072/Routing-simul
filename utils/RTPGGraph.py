from collections import defaultdict
import networkx as nx
from itertools import islice
import numpy as np
from copy import deepcopy

from matplotlib import pyplot as plt


class RTPGGraph:
    def __init__(self, N, M, F):
        self.N = N  # Number of orbits
        self.M = M  # Number of regions in R direction
        self.F = F  # Phasing factor
        self.G = nx.DiGraph()
        self.ground_relays_set = set()

    def add_satellite(self, satellite, region):
        """
        satellite: 객체 (위치 정보 포함)
        region: (P, R) 좌표
        """
        self.G.add_node(
            satellite.node_id,
            type='satellite',
            position=region,
            obj=satellite
        )

    def add_relay(self, relay, region, search_region):
        self.G.add_node(
            relay.node_id,
            type='relay',
            position=region,
            search_region=search_region,
            obj=relay
        )
        self.ground_relays_set.add(relay.node_id)

    def add_node(self, node, phase, region, search_region, node_type):
        node_id = f"{node.node_id}_{phase}"
        self.G.add_node(
            node_id,
            type=node_type,
            position=region,
            search_region=search_region,  # tuple: (P_min, P_max, R_min, R_max)
            obj=node
        )

    def add_user(self, node, region, search_region, city):
        self.G.add_node(
            node.node_id,
            type="city" if city else "person",
            position=region,
            search_region=search_region,  # tuple: (P_min, P_max, R_min, R_max)
            obj=node
        )

    def connect_isl_links(self):
        """
        위성 간 ISL 연결:
        - Intra-orbit: 같은 궤도에서 앞뒤 위성
        - Inter-orbit: 같은 위상(위성 번호)에서 인접 궤도
        """
        sats = {
            nid: data for nid, data in self.G.nodes(data=True)
            if data["type"] == "satellite"
        }

        # indexing: (orbit_idx, sat_idx_in_orbit) → node_id
        sat_lookup = {}
        for nid, data in sats.items():
            sat = data["obj"]
            sat_lookup[(sat.orbit_idx, sat.sat_idx_in_orbit)] = nid

        for sid, sdata in sats.items():
            sat = sdata["obj"]
            o = sat.orbit_idx
            s = sat.sat_idx_in_orbit

            # Intra-orbit: same orbit, sat ± 1
            for ds in [-1, 1]:
                s2 = (s + ds) % self.M
                neighbor = sat_lookup.get((o, s2))
                if neighbor is not None:
                    self.G.add_edge(sid, neighbor, type='isl')
                    if ds < 0:
                        sat.isl_down = neighbor
                    else:
                        sat.isl_up = neighbor
                else:
                    print(neighbor)
                    print("fail to intra-isl")
                    print(o, s2)
                    # print(sat_lookup)
                    print(s, o, ds)
                    print("-----")


            # Inter-orbit: same sat_idx, orbit ± 1

            # 외곽 isl 테스트용 (그 외 처리 X)
            # if 0 < o < self.N-1:
            #     continue
            # else:
            if o == 0:
                left_sat_idx, right_sat_idx = (s - self.F) % self.M, s
            elif o == self.N-1:
                left_sat_idx, right_sat_idx = s, (s + self.F) % self.M
            else:
                left_sat_idx, right_sat_idx = s, s

            for do in [-1, 1]:
                o2 = (o + do) % self.N

                if do == -1:
                    neighbor = sat_lookup.get((o2, left_sat_idx))
                else:
                    neighbor = sat_lookup.get((o2, right_sat_idx))

                if neighbor is not None:
                    self.G.add_edge(sid, neighbor, type='isl')
                    if do < 0:
                        sat.isl_left = neighbor
                    else:
                        sat.isl_right = neighbor
                else:
                    print("fail to inter-isl")
                    print(s, o, do)
                    print(o, s2)
                    print(left_sat_idx, right_sat_idx)

    def connect_ground_links(self):
        """
        GSL 링크 연결 (relay 기준, ASC/DESC 포함)
        - search_region은 이미 P_min < P_max, R_min < R_max 로 정규화되어 있음
        """
        satellites = {
            nid: data for nid, data in self.G.nodes(data=True)
            if data["type"] == "satellite"
        }

        for relay_id, relay_data in self.G.nodes(data=True):
            if relay_data["type"] != "relay":
                continue

            relay = relay_data["obj"]

            for sat_id, sat_data in satellites.items():
                sat = sat_data["obj"]
                if sat.is_visible(relay.latitude_deg, relay.longitude_deg, in_graph=True):
                    self.G.add_edge(relay_id, sat_id, type="gsl_up")
                    self.G.add_edge(sat_id, relay_id, type="gsl_down")
                    relay.link_to_sat(sat_id)
                    sat.link_to_ground(relay_id)

            # for search_region in relay_data["search_region"]:
            #     P_min, R_min ,P_max, R_max = search_region
            #
            #     for sat_id, sat_data in satellites.items():
            #         P_sat, R_sat = sat_data["position"]
            #         sat = sat_data["obj"]
            #
            #         if P_min < P_max:
            #             if R_min < R_max:
            #                 if P_min <= P_sat <= P_max and R_min <= R_sat <= R_max and sat.is_visible(relay.latitude_deg, relay.longitude_deg, in_graph=True):
            #                     self.G.add_edge(relay_id, sat_id, type="gsl_up")
            #                     self.G.add_edge(sat_id, relay_id, type="gsl_down")
            #                     relay.link_to_sat(sat_id)
            #                     sat.link_to_ground(relay_id)
            #             else:
            #                 if P_min <= P_sat <= P_max and (R_min <= R_sat or R_sat <= R_max) and sat.is_visible(relay.latitude_deg, relay.longitude_deg, in_graph=True):
            #                     self.G.add_edge(relay_id, sat_id, type="gsl_up")
            #                     self.G.add_edge(sat_id, relay_id, type="gsl_down")
            #                     relay.link_to_sat(sat_id)
            #                     sat.link_to_ground(relay_id)
            #         else:
            #             if R_min < R_max:
            #                 if (P_min <= P_sat or P_sat <= P_max) and R_min <= R_sat <= R_max and sat.is_visible(relay.latitude_deg, relay.longitude_deg, in_graph=True):
            #                     self.G.add_edge(relay_id, sat_id, type="gsl_up")
            #                     self.G.add_edge(sat_id, relay_id, type="gsl_down")
            #                     relay.link_to_sat(sat_id)
            #                     sat.link_to_ground(relay_id)
            #             else:
            #                 if (P_min <= P_sat or P_sat <= P_max) and (
            #                         R_min <= R_sat or R_sat <= R_max) and sat.is_visible(relay.latitude_deg, relay.longitude_deg, in_graph=True):
            #                     self.G.add_edge(relay_id, sat_id, type="gsl_up")
            #                     self.G.add_edge(sat_id, relay_id, type="gsl_down")
            #                     relay.link_to_sat(sat_id)
            #                     sat.link_to_ground(relay_id)


    def connect_ground_links_for_only_isl(self):
        """
        GSL 링크 연결 (relay 기준, ASC/DESC 포함)
        - search_region은 이미 P_min < P_max, R_min < R_max 로 정규화되어 있음
        """
        satellites = {
            nid: data for nid, data in self.G.nodes(data=True)
            if data["type"] == "satellite"
        }

        for relay_id, relay_data in self.G.nodes(data=True):
            if relay_data["type"] != "relay":
                continue

            relay = relay_data["obj"]

            for search_region in relay_data["search_region"]:
                P_min, R_min ,P_max, R_max = search_region

                for sat_id, sat_data in satellites.items():
                    P_sat, R_sat = sat_data["position"]
                    sat = sat_data["obj"]

                    if P_min < P_max:
                        if R_min < R_max:
                            if P_min <= P_sat <= P_max and R_min <= R_sat <= R_max and sat.is_visible(relay.latitude_deg, relay.longitude_deg, in_graph=True):
                                self.G.add_edge(relay_id, sat_id, type="gsl", weight=1000)
                                relay.link_to_sat(sat_id)
                                sat.link_to_ground(relay_id)
                        else:
                            if P_min <= P_sat <= P_max and (R_min <= R_sat or R_sat <= R_max) and sat.is_visible(relay.latitude_deg, relay.longitude_deg, in_graph=True):
                                self.G.add_edge(relay_id, sat_id, type="gsl", weight=1000)
                                relay.link_to_sat(sat_id)
                                sat.link_to_ground(relay_id)
                    else:
                        if R_min < R_max:
                            if (P_min <= P_sat or P_sat <= P_max) and R_min <= R_sat <= R_max and sat.is_visible(relay.latitude_deg, relay.longitude_deg, in_graph=True):
                                self.G.add_edge(relay_id, sat_id, type="gsl", weight=1000)
                                relay.link_to_sat(sat_id)
                                sat.link_to_ground(relay_id)
                        else:
                            if (P_min <= P_sat or P_sat <= P_max) and (
                                    R_min <= R_sat or R_sat <= R_max) and sat.is_visible(relay.latitude_deg, relay.longitude_deg, in_graph=True):
                                self.G.add_edge(relay_id, sat_id, type="gsl", weight=1000)
                                relay.link_to_sat(sat_id)
                                sat.link_to_ground(relay_id)


    def connect_user_links(self, user):
        """
        GSL 링크 연결 (relay 기준, ASC/DESC 포함)
        - search_region은 이미 P_min < P_max, R_min < R_max 로 정규화되어 있음
        """
        satellites = {
            nid: data for nid, data in self.G.nodes(data=True)
            if data["type"] == "satellite"
        }

        candidates_id_pool = []
        candidates_pool = {}
        # print(user_data["position"])
        # print(user_data["search_region"])

        for search_region in (user.search_regions_asc, user.search_regions_desc):
            P_min, R_min ,P_max, R_max = search_region

            for sat_id, sat_data in satellites.items():
                P_sat, R_sat = sat_data["position"]
                sat = sat_data["obj"]
                if P_min <= P_max:
                    if R_min <= R_max:
                        if P_min <= P_sat <= P_max and R_min <= R_sat <= R_max and sat.is_visible(user.latitude_deg, user.longitude_deg, in_graph=True):
                            candidates_id_pool.append(sat_id)
                            candidates_pool[sat_id] = sat
                    else:
                        if  P_min <= P_sat <= P_max and (R_min <= R_sat or R_sat <= R_max) and sat.is_visible(user.latitude_deg, user.longitude_deg, in_graph=True):
                            candidates_id_pool.append(sat_id)
                            candidates_pool[sat_id] = sat
                else:
                    if R_min <= R_max:
                        if (P_min <= P_sat or P_sat <= P_max) and R_min <= R_sat <= R_max and sat.is_visible(user.latitude_deg, user.longitude_deg, in_graph=True):
                            candidates_id_pool.append(sat_id)
                            candidates_pool[sat_id] = sat
                    else:
                        if  (P_min <= P_sat or P_sat <= P_max) and (R_min <= R_sat or R_sat <= R_max) and sat.is_visible(user.latitude_deg, user.longitude_deg, in_graph=True):
                            candidates_id_pool.append(sat_id)
                            candidates_pool[sat_id] = sat

        if user.is_in_city:
            for sat_id in candidates_id_pool:
                self.G.add_edge(user.node_id, sat_id, type="usl")
                user.link_to_sat(sat_id)
        else:
            nearest_sat = candidates_id_pool[0]
            highest_angle = 0
            for sat_id in candidates_id_pool:
                temp = candidates_pool[sat_id].get_elevation_angle(user.latitude_deg, user.longitude_deg)
                if highest_angle < temp:
                    nearest_sat = sat_id
                    highest_angle = temp

            self.G.add_edge(user.node_id, nearest_sat, type="usl")
            user.link_to_sat(nearest_sat)



    def connect_node_links(self, relay):
        """
                GSL 링크 연결 (relay 기준, ASC/DESC 포함)
                - search_region은 이미 P_min < P_max, R_min < R_max 로 정규화되어 있음
                """
        satellites = {
            nid: data for nid, data in self.G.nodes(data=True)
            if data["type"] == "satellite"
        }

        for search_region in (relay.search_regions_asc, relay.search_regions_desc):
            P_min, R_min, P_max, R_max = search_region

            for sat_id, sat_data in satellites.items():
                P_sat, R_sat = sat_data["position"]
                sat = sat_data["obj"]

                if P_min < P_max:
                    if R_min < R_max:
                        if P_min <= P_sat <= P_max and R_min <= R_sat <= R_max and sat.is_visible(
                                relay.latitude_deg, relay.longitude_deg, in_graph=True):
                            self.G.add_edge(relay.node_id, sat_id, type="gsl")
                            relay.link_to_sat(sat_id)
                            sat.link_to_ground(relay.node_id)
                    else:
                        if P_min <= P_sat <= P_max and (R_min <= R_sat or R_sat <= R_max) and sat.is_visible(
                                relay.latitude_deg, relay.longitude_deg, in_graph=True):
                            self.G.add_edge(relay.node_id, sat_id, type="gsl")
                            relay.link_to_sat(sat_id)
                            sat.link_to_ground(relay.node_id)
                else:
                    if R_min < R_max:
                        if (P_min <= P_sat or P_sat <= P_max) and R_min <= R_sat <= R_max and sat.is_visible(
                                relay.latitude_deg, relay.longitude_deg, in_graph=True):
                            self.G.add_edge(relay.node_id, sat_id, type="gsl")
                            relay.link_to_sat(sat_id)
                            sat.link_to_ground(relay.node_id)
                    else:
                        if (P_min <= P_sat or P_sat <= P_max) and (
                                R_min <= R_sat or R_sat <= R_max) and sat.is_visible(relay.latitude_deg,
                                                                                     relay.longitude_deg,
                                                                                     in_graph=True):
                            self.G.add_edge(relay.node_id, sat_id, type="gsl")
                            relay.link_to_sat(sat_id)
                            sat.link_to_ground(relay.node_id)


    def dijkstra_shortest_path(self, source_id, target_id, weight):
        """
        NetworkX 기반 Dijkstra shortest path wrapper
        모든 edge weight는 기본적으로 1로 간주
        """
        if source_id not in self.G or target_id not in self.G:
            raise ValueError(f"Source or target not found in graph: {source_id}, {target_id}")

        try:
            path = nx.shortest_path(self.G, source=source_id, target=target_id, weight=weight)
            length = len(path) - 1
            return path, length
        except nx.NetworkXNoPath:
            raise RuntimeError(f"No path found between {source_id} and {target_id}")

    def k_shortest_paths(self, source_id, target_id, k, weight):
        """
        NetworkX 기반 K-shortest paths wrapper
        모든 edge weight는 기본적으로 1로 간주
        """
        if source_id not in self.G or target_id not in self.G:
            raise ValueError(f"Source or target not found in graph: {source_id}, {target_id}")

        try:
            paths_generator = nx.shortest_simple_paths(self.G, source=source_id, target=target_id, weight=weight)
            paths = list(islice(paths_generator, k))
            return paths
        except nx.NetworkXNoPath:
            raise RuntimeError(f"No path found between {source_id} and {target_id}")

    def reset_graph(self):
        """
        그래프를 초기 상태로 리셋합니다.
        """
        self.G = nx.DiGraph()
        self.ground_relays_set = set()

    def update_rtpg(self, satellites, ground_relays, sat_region_indices, only_isl=False):
        # 위성 등록
        for sat, region in zip(satellites, sat_region_indices):
            sat.connected_grounds = []
            self.add_satellite(sat, region)
            sat.region = region

        # Ground Relay 등록
        for gr in ground_relays:
            gr.connected_sats = []
            self.add_relay(gr, (gr.region_asc, gr.region_desc), (gr.search_regions_asc, gr.search_regions_desc))

        self.connect_isl_links()
        if not only_isl:
            self.connect_ground_links()

    def relay_edge_counts(self):
        """
        각 relay 노드가 가진 총 엣지 수를 리스트 형태로 반환합니다.
        :return: relay 노드별 엣지 개수 리스트
        """
        counts = []
        for nid, data in self.G.nodes(data=True):
            if data.get('type') == 'relay':
                counts.append(self.G.degree(nid))
        return counts

    def count_satellites_connected_to_relays(self):
        """
        relay 노드와 연결된 고유 위성 노드의 수를 반환합니다.
        :return: 연결된 위성 노드의 총 개수
        """
        satellite_ids = set()
        for relay_id, data in self.G.nodes(data=True):
            if data.get('type') != 'relay':
                continue
            for neighbor in self.G.neighbors(relay_id):
                if self.G.nodes[neighbor].get('type') == 'satellite':
                    satellite_ids.add(neighbor)
        return len(satellite_ids)

    def _split_positions_for_draw(self, G):
        """
        G를 읽어 'draw용 확장 그래프' Gd, pos, labels, orig2splits 매핑을 만든다.
        - 다중 position 노드는 _1, _2 ... 로 복제.
        - 원본 노드 속성은 얕은 복사 후 position만 해당 분할 좌표로 덮어씀.
        - 모든 원본 엣지는 (u_i, v_j) 조합으로 복제.
        """
        Gd = nx.Graph()
        orig2splits = defaultdict(list)
        labels = {}

        for nid, data in G.nodes(data=True):
            if "position" not in data:
                # 위치 없는 노드는 그대로 복사
                Gd.add_node(nid, **data)
                orig2splits[nid].append(nid)
                labels[nid] = nid
                continue

            p = data["position"]
            # 다중 여부
            if isinstance(p[0], (list, tuple)):
                # 여러 좌표
                for i, pt in enumerate(p, start=1):
                    new_id = f"{nid}_{i}"
                    # 속성 복사
                    ndict = dict(data)
                    ndict["position"] = tuple(pt)
                    Gd.add_node(new_id, **ndict)
                    orig2splits[nid].append(new_id)
                    labels[new_id] = f"{nid}_{i}"
            else:
                # 단일 좌표
                ndict = dict(data)
                ndict["position"] = tuple(p)
                Gd.add_node(nid, **ndict)
                orig2splits[nid].append(nid)
                labels[nid] = nid

        # 엣지 복제
        for u, v, edata in G.edges(data=True):
            for us in orig2splits[u]:
                for vs in orig2splits[v]:
                    Gd.add_edge(us, vs, **edata)

        # pos dict
        pos = {
            nid: (d["position"][0], d["position"][1])
            for nid, d in Gd.nodes(data=True)
            if "position" in d
        }

        return Gd, pos, labels, orig2splits

    def _color_by_type(self, t):
        if t == "satellite":
            return "skyblue"
        elif t == "relay":
            return "orange"
        elif t == "src":
            return "red"
        elif t == "dest":
            return "limegreen"
        elif t == "city":
            return "purple"
        elif t == "person":
            return "gray"
        else:
            return "lightgray"

    def visualize(self, highlight_path=None, split_multi=True, label_splits=True):
        """
        그래프 시각화.
        split_multi=True 이면 다중 position 노드를 분할해 그립니다(원본 self.G 불변).
        highlight_path: 원본 노드 ID 경로 (split_multi=True 인 경우 첫 분할 노드만 강조).
        """
        if not split_multi:
            Gd = self.G
            pos = {
                nid: (data["position"][0], data["position"][1])
                for nid, data in Gd.nodes(data=True)
                if "position" in data and not isinstance(data["position"][0], (list, tuple))
            }
            # 다중 좌표 노드는 첫 좌표만 사용해도 되지만
            # NumPy 오류 방지를 위해 제외(필요시 수정)
            labels = {nid: nid for nid in Gd.nodes()}
            orig2splits = {nid: [nid] for nid in Gd.nodes()}
        else:
            Gd, pos, labels, orig2splits = self._split_positions_for_draw(self.G)
            if not label_splits:
                # 라벨을 원본 ID만으로 통일하고 싶다면:
                labels = {nid2: orig for orig, splits in orig2splits.items() for nid2 in splits}

        # 색상 맵
        color_map = [self._color_by_type(Gd.nodes[n].get("type")) for n in Gd.nodes()]

        from matplotlib import pyplot as plt
        plt.figure(figsize=(16, 8))
        nx.draw(
            Gd,
            pos,
            with_labels=True,
            labels=labels,
            node_color=color_map,
            node_size=100,
            font_size=4,
            alpha=0.85
        )

        # 최단경로 강조
        if highlight_path:
            # highlight_path 는 '원본' 노드 ID 기준이라고 가정.
            # split_multi=True 인 경우 각 원본의 첫 분할 노드를 사용.
            hp_nodes = []
            for nid in highlight_path:
                splits = orig2splits.get(nid)
                if not splits:
                    continue
                hp_nodes.append(splits[0])  # 첫 번째 분할 사용 (필요시 규칙 변경)
            if len(hp_nodes) >= 2:
                path_edges = list(zip(hp_nodes, hp_nodes[1:]))
                nx.draw_networkx_edges(
                    Gd, pos,
                    edgelist=path_edges,
                    edge_color='crimson',
                    width=2.5
                )

        plt.title("RTPG Graph" + (" (split multi-pos)" if split_multi else ""))
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    def show_neighbors(self, node_id, include_pos=True, return_rows=False, sort_rows=True, file=None):
        """
        주어진 node_id에 직접 연결된 이웃 노드를 간단한 표 형태로 출력.

        Parameters
        ----------
        node_id : 해시가능
            기준 노드 ID.
        include_pos : bool (default=True)
            이웃 노드의 position 필드를 함께 출력할지 여부.
        return_rows : bool (default=False)
            True면 출력 대신 rows(list of tuples)를 반환.
        sort_rows : bool (default=True)
            edge_type, neighbor_type, neighbor_id 순으로 정렬.
        file : file-like (default=None)
            print 출력 대상 (예: 파일 핸들). None이면 stdout.

        Returns
        -------
        rows : list of tuples | None
            return_rows=True일 때만 반환.
        """
        G = self.G
        if node_id not in G:
            raise ValueError(f"Node '{node_id}' not found in graph.")

        node_type = G.nodes[node_id].get("type", "?")
        deg = G.degree(node_id)

        rows = []
        for nbr in G.neighbors(node_id):
            nd = G.nodes[nbr]
            e_data = G.get_edge_data(node_id, nbr, default={})
            edge_t = e_data.get("type", "-")
            pos = nd.get("position") if include_pos else None
            rows.append((nbr, nd.get("type", "?"), edge_t, pos))

        if sort_rows:
            rows.sort(key=lambda r: (str(r[2]), str(r[1]), str(r[0])))

        if return_rows:
            return rows

        # ----- 출력 포맷 준비 -----
        # 컬럼 라벨
        headers = ["neighbor_id", "type", "edge"]
        if include_pos:
            headers.append("position")

        # 각 열 폭 계산
        def _len(x):
            return len(str(x))

        col_widths = {
            "neighbor_id": max([len("neighbor_id")] + [_len(r[0]) for r in rows]) if rows else len("neighbor_id"),
            "type": max([len("type")] + [_len(r[1]) for r in rows]) if rows else len("type"),
            "edge": max([len("edge")] + [_len(r[2]) for r in rows]) if rows else len("edge"),
        }
        if include_pos:
            col_widths["position"] = max(
                [len("position")] + [_len(r[3]) for r in rows]
            ) if rows else len("position")

        # 헤더 출력
        header_line = (
            "Neighbors of '{nid}' (type={t}, degree={d})"
            .format(nid=node_id, t=node_type, d=deg)
        )
        print(header_line, file=file)
        print("-" * len(header_line), file=file)

        # 컬럼 제목행
        def _fmt_header(h):
            return h.ljust(col_widths[h])

        header_row = " | ".join(_fmt_header(h) for h in headers)
        print(header_row, file=file)
        print("-" * len(header_row), file=file)

        # 데이터 행
        for r in rows:
            vals = {
                "neighbor_id": str(r[0]).ljust(col_widths["neighbor_id"]),
                "type": str(r[1]).ljust(col_widths["type"]),
                "edge": str(r[2]).ljust(col_widths["edge"]),
            }
            if include_pos:
                vals["position"] = str(r[3]).ljust(col_widths["position"])
            line = " | ".join(vals[h] for h in headers)
            print(line, file=file)

    def integrity_check(self):
        checker = [[0 for col in range(72)] for row in range(22)]

        satellites = {
            nid: data for nid, data in self.G.nodes(data=True)
            if data["type"] == "satellite"
        }

        for s_id, data in satellites.items():
            p, r = data["position"]
            checker[r][p] += 1
        for row in checker:
            for item in row:
                if item == 1:
                    continue
                else:
                    print("!!!RTPG conflict emerge!!!")
                    for row in checker:
                        for item in row:
                            print(item, end="")
                        print()
                    return
        print("RTPG is mapped succesfully")


    def get_copy(self):
        return deepcopy(self)

    def _invalidate_views(self):
        self._views.clear()
        self._view_epoch += 1

    def visualize_flow_paths(self, flows, split_multi=True, label_splits=True):
        """
        여러 flow 경로를 함께 시각화합니다.
        flows: {'(src,dst)': [pkts, path, detours]} 형태의 딕셔너리
        """
        if not flows:
            print("No flows to visualize.")
            return

        # visualize와 동일한 그래프 준비 로직
        if not split_multi:
            Gd = self.G
            pos = {
                nid: (data["position"][0], data["position"][1])
                for nid, data in Gd.nodes(data=True)
                if "position" in data and not isinstance(data["position"][0], (list, tuple))
            }
            labels = {nid: nid for nid in Gd.nodes()}
            orig2splits = {nid: [nid] for nid in Gd.nodes()}
        else:
            Gd, pos, labels, orig2splits = self._split_positions_for_draw(self.G)
            if not label_splits:
                labels = {nid2: orig for orig, splits in orig2splits.items() for nid2 in splits}

        color_map = [self._color_by_type(Gd.nodes[n].get("type")) for n in Gd.nodes()]

        from matplotlib import pyplot as plt
        plt.figure(figsize=(16, 8))
        nx.draw(
            Gd, pos, with_labels=True, labels=labels, node_color=color_map,
            node_size=100, font_size=4, alpha=0.85
        )

        # 경로 하이라이트
        path_colors = ['crimson', 'blue', 'green', 'purple', 'orange', 'brown']
        color_idx = 0
        for fkey, (total_pkts, path, detours) in flows.items():
            hp_nodes = []
            for nid in path:
                splits = orig2splits.get(nid)
                if not splits: continue
                hp_nodes.append(splits[0])

            if len(hp_nodes) >= 2:
                path_edges = list(zip(hp_nodes, hp_nodes[1:]))
                nx.draw_networkx_edges(
                    Gd, pos,
                    edgelist=path_edges,
                    edge_color=path_colors[color_idx % len(path_colors)],
                    width=2.0
                )
                color_idx += 1

        plt.title(f"RTPG Graph with {len(flows)} flows" + (" (split multi-pos)" if split_multi else ""))
        plt.grid(True)
        plt.axis("equal")
        plt.show()