# routings/proposed.py
import networkx as nx
from parameters.PARAMS import FILTER


# -------------------------
# 필터 (원본 유지)
# -------------------------
def check_pair(src, dst, filter_mode, cut_rate):
    if filter_mode in ("DROP", "LENGTH"):
        return (src.node_id, dst.node_id) in FILTER[filter_mode][:cut_rate-1]
    else:
        return src.latitude_deg <= cut_rate or dst.latitude_deg <= cut_rate


# -------------------------
# 남반구 경량 GR 선택 (심플)
# -------------------------
def n_light_load_ground_node_in_south(ground_relays, n=1):
    cands = [gr for gr in ground_relays if gr.latitude_deg < 0]
    cands.sort(key=lambda gr: gr.total_buffer_load())
    picked = [gr.node_id for gr in cands[:n]]
    return picked[0] if n == 1 else picked


# -------------------------
# (옵션1) with GR: 그래프 그대로 최단경로 (GR 개수 제한 없음)
# -------------------------
def shortest_path_with_gr(rtpg, src, dst):
    path, length = rtpg.dijkstra_shortest_path(
        source_id=src, target_id=dst, weight="weight"
    )
    return list(path), float(length)


# -------------------------
# (옵션3) without GR: ISL-only 그래프에서 최단경로
# -------------------------
def shortest_path_without_gr(rtpg_isl, src, dst):
    path, length = rtpg_isl.dijkstra_shortest_path(
        source_id=src, target_id=dst, weight="weight"
    )
    return list(path), float(length)


# -------------------------
# (옵션4) reverse of opt3:
#  - src/dst는 그대로
#  - intra-plane ISL은 down만 사용
#  - inter-plane ISL은 좌/우 그대로 사용
# -------------------------
def reverse_direction_path_of_opt3(rtpg_isl, src, dst):
    DG = nx.DiGraph()

    # 위성 노드만 추가
    for nid, data in rtpg_isl.G.nodes(data=True):
        if data["type"] == "satellite":
            DG.add_node(nid)

    # 방향성 에지 구성
    for nid, data in rtpg_isl.G.nodes(data=True):
        if data["type"] != "satellite":
            continue
        sat = data["obj"]
        # intra-plane: down만
        if getattr(sat, "isl_down", None) is not None:
            DG.add_edge(nid, sat.isl_down, weight=1)
        # inter-plane: 좌/우 그대로
        if getattr(sat, "isl_left", None) is not None:
            DG.add_edge(nid, sat.isl_left, weight=1)
        if getattr(sat, "isl_right", None) is not None:
            DG.add_edge(nid, sat.isl_right, weight=1)

    path = nx.shortest_path(DG, source=src, target=dst, weight="weight")
    return path, float(len(path) - 1)


# -------------------------
# (옵션2) with GR beyond equator:
#  - rtpg_gr_south(남반구 GR만 포함된 특수 그래프)에서 그냥 최단경로
#  - GR 수 제한 없음
# -------------------------
def shortest_path_with_gr_beyond_equator(rtpg_gr_south, src, dst):
    path, length = rtpg_gr_south.dijkstra_shortest_path(
        source_id=src, target_id=dst, weight="weight"
    )
    return list(path), float(length)
