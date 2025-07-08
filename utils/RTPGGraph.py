from shapely.geometry import Point
import networkx as nx
import numpy as np

class RTPGGraph:
    def __init__(self, N, M):
        self.N = N  # Number of orbits
        self.M = M  # Number of regions in R direction
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

    def add_relay(self, relay, suffix, region, search_region):
        node_id = f"{relay.node_id}_{suffix}"
        self.G.add_node(
            node_id,
            type='relay',
            position=region,
            search_region=search_region,
            obj=relay
        )

    def add_node(self, node, suffix, region, search_region, node_type):
        node_id = f"{node.node_id}_{suffix}"
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
            for do in [-1, 1]:
                o2 = (o + do) % self.N
                neighbor = sat_lookup.get((o2, s))
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