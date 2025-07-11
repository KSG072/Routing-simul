import random
import csv
import numpy as np

from tqdm import tqdm

from parameters.PARAMS import *

from utils.walker_constellation import WalkerConstellation
from utils.loader import load_ground_relays_from_csv, batch_map_nodes, normalize_wrapped_regions
from utils.rtpg_mapper import RTPGMapper
from utils.RTPGGraph import RTPGGraph
from utils.key_node_extractor import batch_map_ground_relays, batch_map_users
from utils.user_node_generator import generate_users
from utils.satellite import Satellite


from shapely.geometry import Point



def select_next_satellite(current_sat, next_sat_horizontal, next_sat_vertical, remaining_hops_h, remaining_hops_v):
    Bs = SENDING_BUFFER_QUEUE_LASER_PACKETS
    R_ISL = ISL_RATE_LASER
    η1 = SMOOTHING_FACTORS[0]

    Q_i_h = len(current_sat.queue_laser_horizontal)
    Q_i_v = len(current_sat.queue_laser_vertical)

    Q_next_h = (len(next_sat_horizontal.queue_laser_horizontal) + len(next_sat_horizontal.queue_laser_vertical)) / 2
    Q_next_v = (len(next_sat_vertical.queue_laser_horizontal) + len(next_sat_vertical.queue_laser_vertical)) / 2

    T_h = (Q_i_h + Q_next_h) / R_ISL
    T_v = (Q_i_v + Q_next_v) / R_ISL

    Gamma = η1 * ((2 * Bs / R_ISL) - max(T_h, T_v))

    if Q_i_h >= Bs or Q_next_h >= Bs:
        if Q_i_v < Bs and Q_next_v < Bs:
            return next_sat_vertical
    if Q_i_v >= Bs or Q_next_v >= Bs:
        if Q_i_h < Bs and Q_next_h < Bs:
            return next_sat_horizontal

    if remaining_hops_h < remaining_hops_v:
        if T_h - T_v <= Gamma:
            return next_sat_vertical
    elif remaining_hops_v < remaining_hops_h:
        if T_v - T_h <= Gamma:
            return next_sat_horizontal

    return next_sat_horizontal if T_h <= T_v else next_sat_vertical

def select_ground_forwarding(current_sat, relay, gateway_satellites, tau):
    Bs = SENDING_BUFFER_QUEUE_LASER_PACKETS
    R_ISL = ISL_RATE_LASER
    R_down = SGL_KA_DOWNLINK
    η2 = SMOOTHING_FACTORS[1]
    η3 = SMOOTHING_FACTORS[2]
    N0 = MAX_GATEWAY_HOP_DIFF  # 설정 필요

    Q_sgl_i = len(current_sat.queue_sgl)

    Psi_down = η2 * tau * R_ISL + η3 * tau * R_down

    if Q_sgl_i <= Psi_down:
        return relay  # 직접 릴레이로 전송

    min_load = float('inf')
    selected_gateway = None

    for gw_sat in gateway_satellites:
        if gw_sat.node_id == current_sat.node_id:
            continue

        dist = abs(current_sat.orbit_idx - gw_sat.orbit_idx)  # 예시 거리 계산
        if dist > N0:
            continue

        load = len(gw_sat.queue_sgl)
        if load < min_load:
            min_load = load
            selected_gateway = gw_sat

    return selected_gateway if selected_gateway else relay

def select_uplink_forwarding(current_relay, candidate_satellites, tau):
    Bs = SENDING_BUFFER_QUEUE_LASER_PACKETS
    R_ISL = ISL_RATE_LASER
    R_up = SGL_KA_UPLINK
    η4 = SMOOTHING_FACTORS[3]
    η5 = SMOOTHING_FACTORS[4]
    N0 = MAX_GATEWAY_HOP_DIFF  # 또는 거리(km)로 변환 가능

    Psi_up = η4 * tau * R_ISL + η5 * tau * R_up

    # 기본 게이트웨이
    default_sat = candidate_satellites[0]  # 실제 KNBG-MHCE 결과 기반으로 설정

    Q_up_default = len(default_sat.queue_sgl_uplink)

    if Q_up_default <= Psi_up:
        return default_sat  # 기본 위성으로 전송

    min_load = float('inf')
    selected_sat = None

    for sat in candidate_satellites:
        if sat.node_id == default_sat.node_id:
            continue

        dist = abs(default_sat.orbit_idx - sat.orbit_idx)  # 단순화, 위경도 거리로 교체 가능
        if dist > N0:
            continue

        load = len(sat.queue_sgl_uplink)
        if load < min_load:
            min_load = load
            selected_sat = sat

    return selected_sat if selected_sat else default_sat

def calculate_remaining_hops(rtpg_graph, start_node_id, target_node_id, direction, orbit_idx_map):
    """
    direction: 'horizontal' 또는 'vertical'
    orbit_idx_map: 위성 node_id → orbit_idx 딕셔너리
    """
    try:
        path, _ = rtpg_graph.dijkstra_shortest_path(start_node_id, target_node_id)
    except:
        return float('inf')

    count = 0
    for idx in range(len(path) - 1):
        curr_id = path[idx]
        next_id = path[idx + 1]

        if rtpg_graph.G.nodes[next_id]["type"] != "satellite":
            continue  # 위성만 고려

        curr_orbit = orbit_idx_map.get(curr_id)
        next_orbit = orbit_idx_map.get(next_id)

        if direction == "horizontal" and next_orbit != curr_orbit:
            break
        if direction == "vertical" and next_orbit == curr_orbit:
            break

        count += 1

    return count


def find_candidate_neighbors(rtpg_graph, current_sat, orbit_idx_map):
    """
    현재 위성의 수평/수직 이웃 탐색
    """
    horizontal = None
    vertical = None

    for neighbor_id in rtpg_graph.G.neighbors(current_sat.node_id):
        neighbor_data = rtpg_graph.G.nodes[neighbor_id]
        if neighbor_data["type"] != "satellite":
            continue

        neighbor_sat = neighbor_data["obj"]

        if orbit_idx_map[neighbor_id] == orbit_idx_map[current_sat.node_id]:
            vertical = neighbor_sat
        else:
            horizontal = neighbor_sat

    return horizontal, vertical

# 위치 기반 혼잡 지역 판별 함수
def get_congestion_region(lat, lon):
    for area in CONGESTION_AREAS:
        if area["lat_min"] <= lat <= area["lat_max"] and area["lon_min"] <= lon <= area["lon_max"]:
            return area["city"]
    return "Others"


def update_rtpg(rtpg, satellites, ground_relays, mapper):
    sat_region_indices = mapper.batch_map(satellites)
    relay_region_indices_asc, relay_region_indices_desc = mapper.batch_map_nodes(ground_relays)
    batch_search_region_asc, batch_search_region_desc = batch_map_nodes(
        N, M, inclination_deg, altitude_km, ground_relays, relay_region_indices_asc, relay_region_indices_desc)
    result_asc_r, result_desc_r, result_asc_nr, result_desc_nr = normalize_wrapped_regions(
        N, M, relay_region_indices_asc, relay_region_indices_desc, batch_search_region_asc,
        batch_search_region_desc)

    for relay, region_asc, region_desc, original_region_asc, original_region_desc, search_region_asc, search_region_desc in zip(
            ground_relays, relay_region_indices_asc, relay_region_indices_desc, result_asc_nr, result_desc_nr,
            result_asc_r, result_desc_r):
        relay.region_asc = region_asc
        relay.region_desc = region_desc
        relay.original_region_asc = original_region_asc
        relay.original_region_desc = original_region_desc
        relay.search_regions_asc = search_region_asc
        relay.search_regions_desc = search_region_desc

    # 위성 등록
    for sat, region in zip(satellites, sat_region_indices):
        rtpg.add_satellite(sat, region)

    # Ground Relay 등록
    for gr in ground_relays:
        rtpg.add_relay(gr, "asc", gr.region_asc, gr.search_regions_asc)
        rtpg.add_relay(gr, "desc", gr.region_desc, gr.search_regions_desc)

    rtpg.connect_isl_links()
    rtpg.connect_ground_links()

    return rtpg

if __name__ == '__main__':
    # GSL O
    for genertation_rate in GENERATION_RATE_LIST:
        relay_csv_path = './Ground_Relay_Coordinates.csv'
        image_dir = "./fig/"

        # 파라미터 설정
        N = 72
        M = 22
        F = 39
        altitude_km = 550
        inclination_deg = 53

        total_users = 5000  # 예시

        dt = 0.001  # 1 ms
        total_time = 1  # seconds
        # total_time = 20.51  # seconds
        steps = int(total_time / dt)

        T_s = 95.4 * 60  # 궤도 주기 (초)
        omega_s = 2 * np.pi / T_s  # delta phase (deg)

        # 0. 필요한 모듈 초기화
        mapper = RTPGMapper(N, M, F, inclination_deg)

        # 1. Constellation 생성
        constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=altitude_km, inclination_deg=inclination_deg)
        constellation.generate_constellation()
        satellites = constellation.get_all_satellites()
        ground_relays = load_ground_relays_from_csv(relay_csv_path, N * M)

        # user
        users = generate_users(start_idx=0, total_count=total_users)
        user_region_indices_asc, user_region_indices_desc = mapper.batch_map_nodes(users)
        batch_search_region_asc, batch_search_region_desc = batch_map_nodes(
            N, M, inclination_deg, altitude_km, users, user_region_indices_asc, user_region_indices_desc)
        result_asc_r, result_desc_r, result_asc_nr, result_desc_nr = normalize_wrapped_regions(
            N, M, user_region_indices_asc, user_region_indices_desc, batch_search_region_asc,
            batch_search_region_desc)

        # plot_constellation(constellation, altitude_km=altitude_km, save_path='test.png')  # visualization

        # 초기 RTPG 생성 (위성, ground relay만)
        rtpg = update_rtpg(RTPGGraph(N=N, M=M, F=F), satellites, ground_relays, mapper)
        pa
        # ==============================
        # Time loop 시작
        # ==============================
        # 2. 사용자별 패킷 생성 시간표 정의
        for user in users:
            t = 0
            while t < total_time:
                user.packet_generation_times.append(t)
                t += PACKET_GEN_INTERVAL

        log_results = []
        # 3. 시뮬레이션 타임루프
        for i in tqdm(range(steps)):
            t = i * dt
            if i % 600 and i != 0:
                rtpg.reset_graph()
                rtpg = update_rtpg(rtpg, satellites, ground_relays, mapper)

            num_of_generated_packets = genertation_rate
            selected_users = random.sample(users, num_of_generated_packets)


            for user in selected_users:
                # User 전용 RTPG 복사
                rtpg_user = RTPGGraph(N=N, M=M)
                rtpg_user.G = rtpg.G.copy(as_view=False)  # 깊은 복사 유사

                # User만 추가
                rtpg_user.add_user(user, user.region_asc[0], user.region_asc[1])
                rtpg_user.add_user(user, user_region_indices_desc[0], batch_search_region_desc_users[0])

                # User만을 위한 연결만 수행
                satellites_subset = {nid: data for nid, data in rtpg_user.G.nodes(data=True) if data["type"] == "satellite"}
                polygons = rtpg_user.G.nodes[user.node_id]["search_polygons"]

                for sat_id, sat_data in satellites_subset.items():
                    P_sat, R_sat = sat_data["position"]
                    pt = Point(P_sat, R_sat)
                    if any(poly.covers(pt) for poly in polygons):
                        rtpg_user.G.add_edge(user.node_id, sat_id, type='usl')

                # 최단 경로 찾기
                relay_id = user.destination
                path, distance = rtpg_user.dijkstra_shortest_path(source_id=user.node_id,
                                                                  target_id=ground_relays[relay_id].node_id)
                isl_links = [(u, v) for u, v, d in rtpg_user.G.edges(data=True) if d["type"] == "isl"]
                if path is None:
                    log_results.append([t * 1000, user.node_id, relay_id, 0, "경로없음", "N/A"])  # ms 단위 기록
                    continue

                # visualize_rtpg_with_wraparound(rtpg_user.G, N, M, path)  # visualization


                packet = {"timestamp": t * 1000, "origin": user.node_id}
                success = True
                drop_location = "N/A"

                for idx in range(len(path) - 1):
                    curr_data = rtpg_user.G.nodes[path[idx]]
                    next_data = rtpg_user.G.nodes[path[idx + 1]]

                    curr_node = curr_data["obj"]
                    next_node = next_data["obj"]

                    # Case 1: 위성 → 위성
                    if isinstance(curr_node, Satellite) and next_data["type"] == "satellite":

                        orbit_idx_map = {nid: data["obj"].orbit_idx for nid, data in rtpg_user.G.nodes(data=True) if
                                         data["type"] == "satellite"}

                        next_sat_horizontal, next_sat_vertical = find_candidate_neighbors(rtpg_user, curr_node,
                                                                                          orbit_idx_map)

                        remaining_hops_h = calculate_remaining_hops(rtpg_user, next_sat_horizontal.node_id, ground_relays[relay_id].node_id,
                                                                    "horizontal", orbit_idx_map)
                        remaining_hops_v = calculate_remaining_hops(rtpg_user, next_sat_vertical.node_id, ground_relays[relay_id].node_id,
                                                                    "vertical", orbit_idx_map)

                        alt_next_node = select_next_satellite(
                            current_sat=curr_node,
                            next_sat_horizontal=next_sat_horizontal,
                            next_sat_vertical=next_sat_vertical,
                            remaining_hops_h=remaining_hops_h,
                            remaining_hops_v=remaining_hops_v
                        )
                        #
                        # # 포화도 또는 대체 경로 필요 판단
                        # if next_node.is_laser_buffer_full():
                        #     alt_next_node = select_next_satellite(
                        #         current_sat=curr_node,
                        #         next_sat_horizontal=next_node,  # 실제 수평 후보 구분 필요
                        #         next_sat_vertical=next_node,  # 실제 수직 후보 구분 필요
                        #         remaining_hops_h=0,  # 남은 홉 수 계산 필요
                        #         remaining_hops_v=0
                        #     )
                        #     next_node = alt_next_node

                        # 최종 전송 시도
                        if next_node.can_enqueue_laser():
                            next_node.enqueue_laser_packet(packet)
                        else:
                            success = False
                            drop_location = f"Satellite {next_node.node_id}"
                            break

                    # Case 2: 위성 → 지상 릴레이
                    elif isinstance(curr_node, Satellite) and next_data["type"] == "relay":

                        relay_obj = next_data["obj"]

                        gateway_satellites = []
                        for neighbor_id in rtpg_user.G.neighbors(relay_obj.node_id):
                            neighbor_data = rtpg_user.G.nodes[neighbor_id]
                            if neighbor_data["type"] == "satellite":
                                gateway_satellites.append(neighbor_data["obj"])

                        alt_next_hop = select_ground_forwarding(
                            current_sat=curr_node,
                            relay=relay_obj,
                            gateway_satellites=gateway_satellites,
                            tau=dt
                        )
                        if isinstance(alt_next_hop, Satellite):
                            # 대체 게이트웨이로 우회 필요
                            if alt_next_hop.can_enqueue_laser():
                                alt_next_hop.enqueue_laser_packet(packet)
                            else:
                                success = False
                                drop_location = f"Satellite {alt_next_hop.node_id} (우회 실패)"
                                break
                        else:
                            # 직접 지상 릴레이로 전송 (큐 고려 가능)
                            next_data["obj"].receive_packet(packet)

                    # Case 3: 지상 → 위성
                    elif curr_data["type"] == "relay" and next_data["type"] == "satellite":

                        relay_obj = curr_data["obj"]

                        candidate_uplink_satellites = []
                        for neighbor_id in rtpg_user.G.neighbors(relay_obj.node_id):
                            neighbor_data = rtpg_user.G.nodes[neighbor_id]
                            if neighbor_data["type"] == "satellite":
                                candidate_uplink_satellites.append(neighbor_data["obj"])

                        # 필요 시 포화도 기반 대체 위성 선택
                        alt_next_node = next_node  # 기본 유지
                        if next_node.is_laser_buffer_full():
                            alt_next_node = select_uplink_forwarding(
                                current_relay=curr_data["obj"],
                                candidate_satellites=candidate_uplink_satellites,
                                tau=dt
                            )

                        if alt_next_node.can_enqueue_laser():
                            alt_next_node.enqueue_laser_packet(packet)
                        else:
                            success = False
                            drop_location = f"Satellite {alt_next_node.node_id} (업링크 실패)"
                            break

                if success:
                    log_results.append([t * 1000, user.node_id, relay_id, len(path) - 1, "성공", "N/A"])
                else:
                    log_results.append([t * 1000, user.node_id, relay_id, len(path) - 1, "드롭", drop_location])

            for sat in satellites:
                sat.update_position(omega_s, dt)

        # --- 루프 종료 후 CSV 저장 ---
        csv_path = "simulation_results_detailed_with_GSL_" + str(genertation_rate) + ".csv"

        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time (ms)", "User ID", "Destination Relay ID", "Path Length",
                "Status", "Drop Location", "Drop Latitude", "Drop Longitude", "Drop Region"
            ])

            for row in log_results:
                time_ms, user_id, relay_id, path_len, status, drop_loc = row

                if "Satellite" in drop_loc:
                    # 위성 ID 추출
                    sat_id = int(drop_loc.split()[1])
                    sat_obj = next(sat for sat in satellites if sat.node_id == sat_id)
                    lat, lon = sat_obj.get_position()
                    region = get_congestion_region(lat, lon)
                elif "(업링크 실패)" in drop_loc or "(우회 실패)" in drop_loc:
                    parts = drop_loc.split()
                    sat_id = int(parts[1])
                    sat_obj = next(sat for sat in satellites if sat.node_id == sat_id)
                    lat, lon = sat_obj.get_position()
                    region = get_congestion_region(lat, lon)
                else:
                    lat, lon, region = "N/A", "N/A", "N/A"

                writer.writerow([
                    time_ms, user_id, relay_id, path_len, status, drop_loc, lat, lon, region
                ])

        print(f"\n✅ 상세 결과가 {csv_path}로 저장되었습니다.")


    # import matplotlib.pyplot as plt
    # import cartopy.crs as ccrs
    # import cartopy.feature as cfeature
    # import numpy as np
    #
    # # 혼잡 지역 예시
    # congestion_areas = [
    #     {"city": "New York", "lat_min": 40.5, "lat_max": 41.0, "lon_min": -74.3, "lon_max": -73.7},
    #     {"city": "Los Angeles", "lat_min": 34.0, "lat_max": 34.2, "lon_min": -118.5, "lon_max": -118.2},
    #     {"city": "London", "lat_min": 51.2, "lat_max": 51.7, "lon_min": -0.5, "lon_max": 0.3},
    #     # 필요 시 전체 추가
    # ]
    #
    # # 시뮬레이션 패킷 생성 로그 예시
    # packet_logs = [{"lat": 40.7, "lon": -74.0}, {"lat": 34.1, "lon": -118.4}, {"lat": 51.4, "lon": 0.0}]
    #
    # fig = plt.figure(figsize=(12, 6))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_global()
    # ax.coastlines()
    # ax.add_feature(cfeature.BORDERS)
    # ax.set_title("혼잡 지역 내 패킷 생성 분포")
    #
    # # 혼잡 지역 사각형 표시
    # for area in congestion_areas:
    #     ax.add_patch(plt.Rectangle(
    #         (area["lon_min"], area["lat_min"]),
    #         area["lon_max"] - area["lon_min"],
    #         area["lat_max"] - area["lat_min"],
    #         edgecolor='red', facecolor='none', linewidth=1.5, transform=ccrs.PlateCarree(), label=area["city"]
    #     ))
    #
    # # 패킷 생성 위치 플롯
    # lats = [p["lat"] for p in packet_logs]
    # lons = [p["lon"] for p in packet_logs]
    # ax.scatter(lons, lats, color='blue', s=20, transform=ccrs.PlateCarree(), label="패킷 생성 위치")
    #
    # plt.show()

    # # --- 루프 종료 후 CSV 저장 ---
    # csv_path = "simulation_results.csv"
    # with open(csv_path, mode='w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Time (s)", "User ID", "Destination Relay ID", "Path Length", "Status", "Drop Location"])
    #     writer.writerows(log_results)
    #
    # print(f"\n✅ 결과가 {csv_path}로 저장되었습니다.")

    # # for i in range(1):
    # for i in tqdm(range(steps)):
    #     t = i * dt
    #
    #     sat_region_indices = mapper.batch_map(satellites)
    #     relay_region_indices_asc, relay_region_indices_desc = mapper.batch_map_ground_relays(ground_relays)
    #     user_region_indices_asc, user_region_indices_desc = mapper.batch_map_users(users)
    #
    #     # plot_constellation(constellation, ground_relays=ground_relays, altitude_km=altitude_km)  # visualization
    #     # check_RTPG_conflicts(sat_region_indices, satellites, M)  # debug
    #
    #     batch_search_region_asc, batch_search_region_desc = batch_map_ground_relays(N, M, inclination_deg, altitude_km, ground_relays, relay_region_indices_asc, relay_region_indices_desc)
    #     batch_search_region_asc_users, batch_search_region_desc_users = batch_map_users(N, M, inclination_deg, altitude_km, users, user_region_indices_asc, user_region_indices_desc)
    #
    #     # 3. RTPG 그래프 구성
    #     rtpg = RTPGGraph(N=N, M=M)
    #     for sat, region in zip(satellites, sat_region_indices):
    #         rtpg.add_satellite(sat, region)
    #
    #     for relay, region, area in zip(ground_relays, relay_region_indices_asc, batch_search_region_asc):
    #         rtpg.add_relay(relay, region, area)
    #
    #     for user, region, area in zip(users, user_region_indices_asc, batch_search_region_asc_users):
    #         rtpg.add_user(user, region, area)
    #
    #     rtpg.connect_links()
    #
    #     tmp_user_id = random.randint(0,len(users)-1)
    #     tmp_ground_id = users[tmp_user_id].destination_ground_relay_id
    #
    #     # 다익스트라 최단 경로 조회
    #     path, distance = rtpg.dijkstra_shortest_path(source_id=users[tmp_user_id].node_id,
    #                                                  target_id=ground_relays[tmp_ground_id].node_id)
    #     # print("경로:", path)
    #     # print("거리:", distance)
    #
    #
    #     # visualize_rtpg_with_wraparound(rtpg.G, N, M, path)  # visualization
    #
    #     # Update position
    #     for sat in satellites:
    #         sat.update_position(omega_s, dt)
    #
    #     # if i % 100 == 0:
    #     #     plot_constellation(constellation, altitude_km=altitude_km, save_path=f"{image_dir}/frame_{i // 100:03d}.png")
    #
    #     # if i % 100 == 0:
    #     #     print(i, 'timesteps')
    #     #     print("경로:", path)
    #     #     print("거리:", distance)
    #
    #
    #
    # # plot_RTPG(N, M, region_indices, satellites) # visualization



