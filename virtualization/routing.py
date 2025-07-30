import random
import numpy as np
from tqdm import tqdm

from utils.csv_maker import csv_create, csv_write
from utils.user_node import UserNode
from utils.ground_relay_node import GroundRelayNode
from utils.loader import batch_map_nodes, normalize_wrapped_regions
from utils.RTPGGraph import RTPGGraph

def generate_random_latlon(inclination_deg=53):
    lat = random.uniform(-inclination_deg, inclination_deg)
    lon = random.uniform(0, 360)
    return lat, lon

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

def run_routing_simulation(world, rtp_renderer, isl_only=False):
    mapper = world.mapper

    # 3. 기본 RTPG 구성 (src/dest 제외)
    rtpg = RTPGGraph(mapper.N, mapper.M, mapper.F)

    # 위성 등록
    for sat in world.satellites:
        rtpg.add_satellite(sat, sat.region)
    rtpg.connect_isl_links()

    # 기본 relay (지상국들) 등록
    for gr in world.ground_relays:
        rtpg.add_relay(gr,  gr.region_asc, gr.search_regions_asc)
        rtpg.add_relay(gr,  gr.region_desc, gr.search_regions_desc)
    rtpg.connect_ground_links()

    iterations = 1
    data = []

    for i in tqdm(range(iterations)):
        # 1. 발신자(src) / 수신자(dest) 무작위 생성
        src_lat, src_lon = generate_random_latlon()
        dest_lat, dest_lon = generate_random_latlon()

        src = UserNode(node_id=9999, latitude=src_lat, longitude=src_lon)
        dest = GroundRelayNode(node_id=10000, latitude=dest_lat, longitude=dest_lon, continent="Test")

        # 2. RTPG region asc/desc 매핑
        src = prepare_node_routing_metadata(src, mapper, altitude_km=world.altitude_km)
        dest = prepare_node_routing_metadata(dest, mapper, altitude_km=world.altitude_km)

        # 4. src/dest에 대한 4가지 경로 시도
        best_path = None
        best_src_mode = None
        best_dst_mode = None
        worst_path = None
        worst_src_mode = None
        worst_dst_mode = None
        min_hops = float("inf")
        max_hops = -1

        for src_mode in ["asc", "desc"]:
            for dst_mode in ["asc", "desc"]:
                src_id = f"{src.node_id}_{src_mode}"
                dst_id = f"{dest.node_id}_{dst_mode}"

                # src 추가 및 연결
                region_s = src.region_asc if src_mode == "asc" else src.region_desc
                search_s = src.search_regions_asc if src_mode == "asc" else src.search_regions_desc
                rtpg.add_node(src, src_mode, region_s, search_s, 'src')
                rtpg.connect_node_links(src_id, 'usl')

                # dest 추가 및 연결
                region_d = dest.region_asc if dst_mode == "asc" else dest.region_desc
                search_d = dest.search_regions_asc if dst_mode == "asc" else dest.search_regions_desc
                rtpg.add_node(dest, dst_mode, region_d, search_d, 'dest')
                rtpg.connect_node_links(dst_id, 'gsl')

                # 라우팅 시도
                try:
                    path, _ = rtpg.dijkstra_shortest_path(src_id, dst_id)
                    if len(path) < min_hops:
                        min_hops = len(path)
                        best_path = path
                        best_src_mode = src_mode  # ✅ 추가
                        best_dst_mode = dst_mode  # ✅ 추가
                    # if len(path) > max_hops:
                    #     max_hops = len(path)
                    #     worst_path = path
                    #     worst_src_mode = src_mode  # ✅ 추가
                    #     worst_dst_mode = dst_mode  # ✅ 추가
                except:
                    pass

                # cleanup
                rtpg.G.remove_node(src_id)
                rtpg.G.remove_node(dst_id)

        data.append([i, min_hops, max_hops])

        # 5. 시각화
        # src/dest의 asc/desc 복원
        region_s = src.region_asc if best_src_mode == "asc" else src.region_desc
        search_s = src.search_regions_asc if best_src_mode == "asc" else src.search_regions_desc
        src_id = f"{src.node_id}_{best_src_mode}"
        rtpg.add_node(src, best_src_mode, region_s, search_s, 'src')
        rtpg.connect_node_links(src_id, 'usl')

        region_d = dest.region_asc if best_dst_mode == "asc" else dest.region_desc
        search_d = dest.search_regions_asc if best_dst_mode == "asc" else dest.search_regions_desc
        dst_id = f"{dest.node_id}_{best_dst_mode}"
        rtpg.add_node(dest, best_dst_mode, region_d, search_d, 'dest')
        rtpg.connect_node_links(dst_id, 'gsl')

        world.current_src = src
        world.current_dest = dest
        world.current_src_mode = best_src_mode  # 'asc' or 'desc'
        world.current_dest_mode = best_dst_mode

        rtp_renderer.current_src = src
        rtp_renderer.current_dest = dest
        rtp_renderer.current_src_mode = best_src_mode  # 'asc'
        rtp_renderer.current_dest_mode = best_dst_mode

        # 시각화
        visualize_rtpg_graph(rtpg, highlight_path=best_path)
        print(f"[Routing] 최단 경로({best_src_mode}-{best_dst_mode}): {min_hops} hops")
        print(f"[Routing] 최장 경로({worst_src_mode}-{worst_dst_mode}): {max_hops} hops")

        gsl_num_array = rtpg.relay_edge_counts()
        print(f"각 지상 노드에 대한 위성링크 수 (최소, 최대, 평균): {min(gsl_num_array)}, {max(gsl_num_array)}, {sum(gsl_num_array)/len(gsl_num_array)}")
        print(f"지상 링크 보유 위성 수 (전체 위성 수 1584개): {rtpg.count_satellites_connected_to_relays()} 개")


        world.add_node_marker(src, role="src")
        world.add_node_marker(dest, role="dest")

        rtp_renderer.add_node_marker(src, best_src_mode, color=(1, 0, 0, 1))  # 빨강
        rtp_renderer.add_node_marker(dest, best_dst_mode, color=(0, 1, 0, 1))  # 초록


        world.draw_routing_path(best_path)
        # world.draw_routing_path(worst_path)
        rtp_renderer.draw_routing_path(best_path)
        # rtp_renderer.draw_routing_path(worst_path)
    # csv_create(['index', 'min_length', 'max_length'], '../results', f"path length_{iterations}iterations.csv")
    # csv_write(data, '../results', f"path length_{iterations}iterations.csv")

def run_isl_only_routing(world, rtp_renderer):
    mapper = world.mapper
    src = world.current_src
    dest = world.current_dest
    src_mode = world.current_src_mode
    dest_mode = world.current_dest_mode

    if not src or not dest:
        print("[Routing] ERROR: src/dest not set.")
        return

    rtpg = RTPGGraph(mapper.N, mapper.M, mapper.F)

    for sat in world.satellites:
        rtpg.add_satellite(sat, sat.region)
    rtpg.connect_isl_links()

    region_s = src.region_asc if src_mode == "asc" else src.region_desc
    search_s = src.search_regions_asc if src_mode == "asc" else src.search_regions_desc
    src_id = f"{src.node_id}_{src_mode}"
    rtpg.add_node(src, src_mode, region_s, search_s, "src")
    rtpg.connect_node_links(src_id, "usl")

    region_d = dest.region_asc if dest_mode == "asc" else dest.region_desc
    search_d = dest.search_regions_asc if dest_mode == "asc" else dest.search_regions_desc
    dst_id = f"{dest.node_id}_{dest_mode}"
    rtpg.add_node(dest, dest_mode, region_d, search_d, "dest")
    rtpg.connect_node_links(dst_id, 'gsl')

    try:
        path, _ = rtpg.dijkstra_shortest_path(src_id, dst_id)
        print(f"[Routing] ISL-only path: {len(path)} hops")
    except:
        print("[Routing] No ISL-only path found.")
        return

    world.draw_routing_path(path, color=(1, 0, 0, 1), isl=True)
    rtp_renderer.draw_routing_path(path, color=(1, 0, 0, 1), isl=True)


def clear_routing_visual(world, rtp_renderer, clear_all=False, isl=False):
    # 1. 경로 선 제거
    if hasattr(world, "routing_path_node") and world.routing_path_node:
        world.routing_path_node.removeNode()
        world.routing_path_node = None

    if hasattr(world, "ISL_routing_path_node") and world.ISL_routing_path_node:
        world.ISL_routing_path_node.removeNode()
        world.ISL_routing_path_node = None

    if hasattr(rtp_renderer, "routing_path_node") and rtp_renderer.routing_path_node:
        rtp_renderer.routing_path_node.removeNode()
        rtp_renderer.routing_path_node = None

    if hasattr(rtp_renderer, "ISL_routing_path_node") and rtp_renderer.ISL_routing_path_node:
        rtp_renderer.ISL_routing_path_node.removeNode()
        rtp_renderer.ISL_routing_path_node = None

    # 2. 마커 제거 (src/dest 노드)
    if clear_all:
        # World 마커는 .marker 에 저장됨 → 그대로 제거
        if hasattr(world, "current_src") and world.current_src and hasattr(world.current_src, "marker"):
            world.current_src.marker.removeNode()

        if hasattr(world, "current_dest") and world.current_dest and hasattr(world.current_dest, "marker"):
            world.current_dest.marker.removeNode()

        # RTPGRenderer는 marker dict에 저장됨
        if rtp_renderer.current_src and rtp_renderer.current_src_mode:
            src_id = f"{rtp_renderer.current_src.node_id}_{rtp_renderer.current_src_mode}"
            if src_id in rtp_renderer.markers:
                rtp_renderer.markers[src_id].removeNode()
                del rtp_renderer.markers[src_id]

        if rtp_renderer.current_dest and rtp_renderer.current_dest_mode:
            dest_id = f"{rtp_renderer.current_dest.node_id}_{rtp_renderer.current_dest_mode}"
            if dest_id in rtp_renderer.markers:
                rtp_renderer.markers[dest_id].removeNode()
                del rtp_renderer.markers[dest_id]

        # 3. 상태 초기화
        world.current_src = None
        world.current_dest = None
        world.current_src_mode = None
        world.current_dest_mode = None

        rtp_renderer.current_src = None
        rtp_renderer.current_dest = None
        rtp_renderer.current_src_mode = None
        rtp_renderer.current_dest_mode = None


def visualize_rtpg_graph(rtpg, highlight_path=None):
    import matplotlib.pyplot as plt
    import networkx as nx

    G = rtpg.G

    # 1. 노드 위치: (P, R) → x, y
    pos = {
        nid: (data["position"][0], data["position"][1])
        for nid, data in G.nodes(data=True)
        if "position" in data
    }

    # 2. 노드 색상
    color_map = []
    for nid, data in G.nodes(data=True):
        if data["type"] == "satellite":
            color_map.append("skyblue")
        elif data["type"] == "relay":
            color_map.append("orange")
        elif data["type"] == "src":
            color_map.append("red")
        elif data["type"] == "dest":
            color_map.append("limegreen")
        else:
            color_map.append("gray")

    # 3. 시각화
    plt.figure(figsize=(16, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=color_map,
        node_size=100,  # 👈 작게 줄이기
        font_size=4,
        alpha=0.85
    )

    # 4. 최단경로 강조
    if highlight_path:
        path_edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_edges(
            G, pos,
            edgelist=path_edges,
            edge_color='crimson',
            width=2.5
        )

    plt.title("RTPG Graph with SRC (red) and DEST (green)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()