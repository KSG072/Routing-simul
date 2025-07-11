from shapely.geometry import Point
import networkx as nx
import numpy as np

class RTPGGraph:
    def __init__(self, N, M, F):
        self.N = N  # Number of orbits
        self.M = M  # Number of regions in R direction
        self.F = F  # Phasing factor
        self.G = nx.Graph()

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

    def add_relay(self, relay, phase, region, search_region):
        node_id = f"{relay.node_id}_{phase}"
        self.G.add_node(
            node_id,
            type='relay',
            position=region,
            search_region=search_region,
            obj=relay
        )

    def add_node(self, node, phase, region, search_region, node_type):
        node_id = f"{node.node_id}_{phase}"
        self.G.add_node(
            node_id,
            type=node_type,
            position=region,
            search_region=search_region,  # tuple: (P_min, P_max, R_min, R_max)
            obj=node
        )

    def add_user(self, node,  region, search_region, node_type):
        node_id = f"user-{node.node_id}"
        self.G.add_node(
            node_id,
            type=node_type,
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
                if neighbor:
                    self.G.add_edge(sid, neighbor, type='isl')

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

                if neighbor:
                    self.G.add_edge(sid, neighbor, type='isl')

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

            P_min, R_min ,P_max, R_max = relay_data["search_region"]

            for sat_id, sat_data in satellites.items():
                P_sat, R_sat = sat_data["position"]
                sat = sat_data["obj"]

                if P_min <= P_sat <= P_max and R_min <= R_sat <= R_max and sat.is_visible(relay):
                    self.G.add_edge(relay_id, sat_id, type="gsl")

    def connect_node_links(self, node_id, type):
        node_data = self.G.nodes[node_id]
        node = node_data["obj"]
        P_min, R_min, P_max, R_max = node_data["search_region"]

        for sid, sdata in self.G.nodes(data=True):
            if sdata["type"] != "satellite":
                continue

            sat = sdata["obj"]

            P, R = sdata["position"]
            if P_min <= P <= P_max and R_min <= R <= R_max and sat.is_visible(node):
                self.G.add_edge(node_id, sid, type=type)

    def dijkstra_shortest_path(self, source_id, target_id):
        """
        NetworkX 기반 Dijkstra shortest path wrapper
        모든 edge weight는 기본적으로 1로 간주
        """
        if source_id not in self.G or target_id not in self.G:
            raise ValueError(f"Source or target not found in graph: {source_id}, {target_id}")

        try:
            path = nx.shortest_path(self.G, source=source_id, target=target_id)
            length = len(path) - 1
            return path, length
        except nx.NetworkXNoPath:
            raise RuntimeError(f"No path found between {source_id} and {target_id}")

    def reset_graph(self):
        """
        그래프를 초기 상태로 리셋합니다.
        """
        self.G = nx.Graph()

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
