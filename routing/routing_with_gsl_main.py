#python 3.11.13

from warnings import catch_warnings

import numpy as np
from numpy.linalg import norm
from numpy.f2py.rules import generationtime

from tqdm import tqdm
from random import shuffle
import os
os.environ["PYCHARM_DISPLAY"] = "none"

from parameters.PARAMS import *
from routing.packet import Packet
from routing.sota import sat_to_sat_forwarding, sat_to_ground_forwarding, ground_to_sat_forwarding
from routing.minimum_hop_estimator import min_hop_distance
from utils.plot_maker import load_heatmap
from utils.walker_constellation import WalkerConstellation
from utils.loader import load_ground_relays_from_csv, batch_map_nodes, normalize_wrapped_regions, prepare_node_routing_metadata
from utils.rtpg_mapper import RTPGMapper
from utils.RTPGGraph import RTPGGraph
from utils.user_node_generator import generate_users, generate_cities
from utils.csv_maker import csv_write, csv_create


# 위치 기반 혼잡 지역 판별 함수
def get_congestion_region(lat, lon):
    for area in CONGESTION_AREAS:
        if area["lat_min"] <= lat <= area["lat_max"] and area["lon_min"] <= lon <= area["lon_max"]:
            return area["city"]
    return "Others"

def update_rtpg(rtpg, satellites, ground_relays, mapper):
    sat_region_indices = mapper.batch_map(satellites)

    # 위성 등록
    for sat, region in zip(satellites, sat_region_indices):
        sat.connected_grounds = []
        rtpg.add_satellite(sat, region)
        sat.region = region

    # Ground Relay 등록
    for gr in ground_relays:
        gr.connected_sats = []
        rtpg.add_relay(gr, (gr.region_asc, gr.region_desc), (gr.search_regions_asc, gr.search_regions_desc))

    rtpg.connect_isl_links()
    rtpg.connect_ground_links()

    return rtpg

def get_route(rtpg, user, ground_relays, n=1):
    rtpg.add_user(user, (user.region_asc, user.region_desc), (user.search_regions_asc, user.search_regions_desc),
                  user.is_in_city)

    # User만을 위한 연결만 수행
    rtpg.connect_user_links(user)

    # 최단 경로 찾기
    routes = []
    for _ in range(n):
        dst_id = random.choice(list(ground_relays.keys()))
        user.destination = dst_id

        route = rtpg.dijkstra_shortest_path(source_id=user.node_id, target_id=ground_relays[user.destination].node_id)
        routes.append(route)

    rtpg.G.remove_node(user.node_id)

    return routes

def calculate_hop_distance(packet, satellites):
    try:
        src, semi_dst = satellites[packet.curr], satellites[packet.key_node]
        h, v = min_hop_distance(src, semi_dst, N, M, F)
        packet.set_remaining_hops(h, v)
    except KeyError:
        print("D")

def calculate_prop_delay(src, dst):
    distance = norm(dst-src)
    prop_delay_sec = distance/C
    return prop_delay_sec*1000

def transfer(sequences, next_hops, src_coords, disconnected=None):
    if disconnected is None:
        disconnected = []
    direction = 0 # 0:up, 1:down, 2:left, 3:right, 4:ground, 5:satellite
    failed = []
    for pkts in sequences: # sequences: [[pkt, pkt, ...],[],[],[],[{gr1: [], gr2: [], ...}],[sat1: [], sat2: [], ...]]
        if direction <= 3:
            for pkt in pkts:
                if direction == 0:
                    pkt.remaining_v_hops -= 1
                elif direction == 1:
                    pkt.remaining_v_hops += 1
                elif direction == 2:
                    pkt.remaining_h_hops += 1
                elif direction == 3:
                    pkt.remaining_h_hops -= 1
                else:
                    pass
                next_hop = next_hops[direction]

                next_coords = next_hop.cartesian_coords
                pkt.set_propagation_delay(calculate_prop_delay(src_coords, next_coords))
                pkt.transmission_delay += TRANS_DELAY_ISL

                next_hop.receive_packet(pkt)
                pkt.curr = next_hop.node_id
                pkt.result.append(next_hop.node_id)
        else:
            if pkts:
                # print(pkts)
                try:
                    for detail_direction, packets in pkts[0].items():
                        for p in packets:
                            if detail_direction not in disconnected and detail_direction in next_hops[direction]:
                                next_hop = next_hops[direction][detail_direction]

                                next_coords = next_hop.cartesian_coords
                                p.set_propagation_delay(calculate_prop_delay(src_coords, next_coords))

                                next_hop.receive_packet(p)
                                p.curr = next_hop.node_id
                                p.result.append(next_hop.node_id)
                                """
                                위성->지상: 지상에서 보내줄 다음 위성 노드를 key node로 변경
                                지상->위성: 위성에서 향할 다음 지상 노드를 ground node로 변경
                                """
                                if direction == 4:
                                    p.transmission_delay += TRANS_DELAY_DOWN
                                else:
                                    p.transmission_delay += TRANS_DELAY_UP
                            else:
                                """패킷 생성 당시엔 있다고 판단된 경로가, 와보니 끊어져있는 상황"""
                                failed.append(p)

                except AttributeError:
                    print("앙앙앙~")
                except IndexError:
                    print("엉엉엉~")
        direction += 1

    return failed

if __name__ == '__main__':
    header = [
        "Time (ms)", "User ID", "Destination Relay ID", "Path Length", "result", "Queuing delays", "Queuing Delay", "Propagation Delay", "Transmission Delay",
        "Status", "Drop Location", "Drop Latitude", "Drop Longitude"
    ]
    filepath = "../results"
    # GSL O
    for genertation_rate in GENERATION_RATE_LIST:
        filename = "seogwon_results_with_GSL_" + str(genertation_rate) + ".csv"
        csv_create(header, filepath, filename)

        relay_csv_path = '../parameters/Ground_Relay_Coordinates.csv'
        results = []
        failed = []
        dropped = []

        # total_users = 5000  # 예시
        num_of_generated_packets = genertation_rate

        dt = TIME_SLOT  # 1 ms
        total_time = TOTAL_TIME  # second
        # total_time = 20.51 # seconds
        steps = int(total_time / dt)
        packet_scope = None

        T_ms = 95.4 * 60 * 1000  # 궤도 주기 (밀리초)
        omega_s = 2 * np.pi / T_ms  # delta phase (deg)

        """필요한 모듈 초기화"""
        mapper = RTPGMapper(N, M, F, inclination_deg)

        """Constellation & 지상 노드 생성"""
        constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=altitude_km, inclination_deg=inclination_deg)
        constellation.generate_constellation()
        satellites = constellation.get_all_satellites() # dictionary

        ground_relays = load_ground_relays_from_csv(relay_csv_path, N * M)
        # users = generate_users(start_idx=0, total_count=total_users)
        users = generate_cities(start_idx=0)

        for gr in ground_relays.values():
            gr = prepare_node_routing_metadata(gr, mapper, 550)
        for user in users.values():
            user = prepare_node_routing_metadata(user, mapper, 550)

        """초기 RTPG 생성"""
        rtpg = update_rtpg(RTPGGraph(N=N, M=M, F=F), satellites.values(), ground_relays.values(), mapper)
        rtpg.integrity_check()

        """ =============================
                 Time loop 시작
        =============================="""
        ended = 0
        succeeded = 0
        drop_cnt = 0
        fail_cnt = 0

        """시뮬레이션 타임루프"""
        for i in tqdm(range(steps)):
            t = i * dt
            """RTPG 업데이트"""
            if t % 600 == 0 and t != 0:
                rtpg.reset_graph()
                for s in satellites.values():
                    s.update_lat_lon_for_RTPG()
                rtpg = update_rtpg(rtpg, satellites.values(), ground_relays.values(), mapper)

            """ 패킷 생성
            1. 해당 타임 슬롯에 대해서 패킷을 생성할 사용자를 선택
            2. 신규 패킷 생성
            3. 중계 노드 중 하나를 destination으로 하여 rtpg기반 다익스트라 알고리즘으로 경로 형성. 패킷에 경로 정보 입력
            4. 키노드 추출 및 키노드까지의 최단 홉 거리 (수직+수평)계산은 로직 상 안함
            """
            selected_users = random.choices(list(users.values()), k=N_k)
            for user in selected_users:
                paths = get_route(rtpg, user, ground_relays, n = num_of_generated_packets)
                for path in paths:
                    new_packet = Packet(t)
                    new_packet.set_path_info(path[0])
                    user.receive_packet(new_packet)

            """
            각 패킷은 아래의 속성을 가짐 
            1. remaining_v_hops (key_nodes의 0번째에 도달할 수 있는 최단 수직홉 거리)
            2. remaining_h_hops (최단 수평홉 거리, 목적지는 위와 동일)
            각 속성은 양수일 경우 우/상향, 음수일 경우 좌/하향 방향을 나타냄
            
            시뮬레이션은 아래 세 페이즈로 구성됨

            1. 전송 페이즈: 방향 큐에 있는 패킷을 일괄적으로 다음 홉으로 전송하는 과정 (라우팅 알고리즘은 이미 수행 된 후)
            2. 실패/드롭 페이즈: 전송 후 큐의 용량을 초과한 패킷에 대한 드롭처리
            3. 라우팅 페이즈: 각 방향으로부터 전송 받은 패킷에 대하여 라우팅 알고리즘 수행(방향결정) 후 각 방향 큐에 enqueue
            # 예외처리
            예외: 사용자-위성-지상노드 순으로 순차적으로 전송과 라우팅 페이즈를 진행함에 따라서 드롭되는 패킷에 우선순위가 발생
            처리: 전송 페이즈에서 수신한 패킷을 shuffle 후 라우팅 페이즈에 진입
            """

            # packet_scope.show_detailed()
            """전송 페이즈"""

            for u in users.values():
                if u.has_packets(): # (up=[], down=[], left=[], right=[], ground=[], satellite)로 보낼 패킷 리스트
                    bullets = u.get_packets(dt)
                    next_hops = [ # (up, down, left, right, ground, satellite) 방향 다음 홉
                        [], [],
                        [], [],
                        [], {node_id: satellites[node_id] for node_id in u.connected_sats}
                    ]
                    failed += transfer(bullets, next_hops, u.cartesian_coords)
                else:
                    continue

            for s in satellites.values():
                if s.has_packets():
                    bullets = s.get_packets(dt) # (up, down, left, right, ground, satellite=[])로 보낼 패킷 리스트
                    next_hops = [ # (up, down, left, right, ground, satellite) 방향 다음 홉
                        satellites[s.isl_up], satellites[s.isl_down],
                        satellites[s.isl_left], satellites[s.isl_right],
                        {node_id: ground_relays[node_id] for node_id in s.connected_grounds}, []
                    ]
                    failed += transfer(bullets, next_hops, s.cartesian_coords, s.disconnected)
                else:
                    continue

            for g in ground_relays.values():
                if g.has_packets():
                    bullets = g.get_packets(dt) # (up=[], down=[], left=[], right=[], ground=[], satellite)로 보낼 패킷 리스트
                    next_hops = [ # (up, down, left, right, ground, satellite) 방향 다음 홉
                        [], [],
                        [], [],
                        [], {node_id: satellites[node_id] for node_id in g.connected_sats}
                    ]
                    failed += transfer(bullets, next_hops, g.cartesian_coords, g.disconnected)
                else:
                    continue

            """드롭 페이즈"""
            """전송 후 각 버퍼 큐에 남은 패킷 중 capacity를 초과한 것에 대한 tail-drop"""

            for s in satellites.values():
                dropped += s.drop_packet()
            if dropped:
                drop_cnt += len(dropped)
            while dropped:
                p = dropped.pop(0)
                end_node = satellites[p.curr]
                p.end(t, 'drop', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
                # print(2)
                results.append(p)

            # if t % 100 == 0 and t != 0:
            #     total_load_data = []
            #     orbit_data = []
            #     for s in satellites.values():
            #         if s.orbit_idx != 0 and s.sat_idx_in_orbit % M == 0:
            #             total_load_data.append(orbit_data)
            #             # print(orbit_data)
            #             orbit_data = []
            #         orbit_data.append(s.get_load_status())
            #     total_load_data.append(orbit_data)
            #     load_heatmap(num_of_generated_packets, t, N, M, total_load_data)
            """라우팅 방향 결정 & 큐 삽입 페이즈"""
            """
            사용자:
            1. 첫 홉 결정은 이미 결정되어 있음 (로드 고려 X)
            위성 (TMC):
            1. isl 라우팅 (자신이 keynode가 아니거나, 잔여 홉수가 남아있을 경우): 많이 남은 홉 수 우선 라우팅 (다음 위성 로드 고려)
            2. gsl 라우팅 (자신이 keynode일 경우): 보내고자 하는 위성으로 라우팅
              - 자신의 하향링크 버퍼큐에 쌓인 패킷 개수가 threshold를 넘을 경우, 해당 지상노드와 연결된 인접(P +-4) 위성 중 lightest한 위성으로 재라우팅
                -> 키노드 재설정 & 잔여 홉거리 재계산 -> 방향 결정 후, isl 큐에 삽입
            지상 노드 (TMC):
            1. Keynode인 위성으로 전송
              - Keynode 위성으로 가는 큐에 쌓인 패킷의 개수가 threshold를 넘을 경우, 해당 지상 노드와 연결된 P거리 4 이내인 위성 중 가장 덜 쌓인 위성 방향 큐로 삽입
                -> 키노드 재설정 & 잔여 홉거리 재계산 
            """
            for u in users.values():
                while u.storage:
                    packet = u.storage.popleft()
                    packet.ttl -= 1
                    if packet.key_node not in u.connected_sats:
                        print('tq')
                    else:
                        packet.next_ground_node_id()
                        family = (satellites[node_id] for node_id in users[packet.curr].connected_sats)
                        """지상-위성 라우팅 알고리즘 적용 부분"""
                        packet, direction = ground_to_sat_forwarding(u, packet, family)
                        u.enqueue_packet(direction, packet)
                        # u.enqueue_packet(packet.key_node, packet) # no 로드밸런싱


            for s in satellites.values():
                while s.storage:
                    packet = s.storage.popleft()
                    if packet.ttl >= 0:
                        packet.ttl -= 1
                    else:
                        failed.append(packet)
                        continue
                    if packet.curr == packet.key_node:
                        packet.next_key_node_id()
                        if packet.was_on_ground: # 지상으로부터 받은 패킷 -> 키노드, 잔여 홉거리 설정
                            calculate_hop_distance(packet, satellites)
                            packet.was_on_ground = False
                            if packet.curr == packet.key_node:
                                packet.next_key_node_id()
                                family = (satellites[node_id] for node_id in ground_relays[packet.ground_node].connected_sats)
                                need_detour, packet, direction = sat_to_ground_forwarding(s, packet, family)
                                if need_detour:
                                    packet.key_nodes.appendleft(packet.key_node)  # 다시 잠시 넣어둡니다.,...
                                    packet.set_key_node(direction)  # 새로운 keynode 설정
                                    calculate_hop_distance(packet, satellites)
                                else:
                                    s.enqueue_packet(direction, packet)
                                    continue  # 지상 큐에 삽입 후 다음 패킷 처리
                        else: # 지상으로 가야할 패킷
                            """우회가 필요할 경우 재설정된 keynode을 따라 홉 수 재설정 함수 내에서 packet 수정 거치고 나옴"""
                            try:
                                family = (satellites[node_id] for node_id in ground_relays[packet.ground_node].connected_sats)
                            except KeyError:
                                packet.show_detailed()
                                exit()
                            need_detour, packet, direction = sat_to_ground_forwarding(s, packet, family)
                            if need_detour:
                                packet.key_nodes.appendleft(packet.key_node) # 다시 잠시 넣어둡니다.,...
                                packet.set_key_node(direction) # 새로운 keynode 설정
                                calculate_hop_distance(packet, satellites)
                            else:
                                s.enqueue_packet(direction, packet)
                                continue # 지상 큐에 삽입 후 다음 패킷 처리
                    # 단순 위성 포워딩 (잔여 홉 있음)
                    horizontal = satellites[s.isl_left if packet.remaining_h_hops < 0 else s.isl_right]
                    vertical = satellites[s.isl_down if packet.remaining_v_hops < 0 else s.isl_up]
                    """위성-위성 라우팅 알고리즘 적용 부분 (Queuing delay는 여기서 계산됨)"""
                    direction = sat_to_sat_forwarding(s, horizontal, vertical, packet) # 0:up, 1:down, 2:left, 3:right
                    s.enqueue_packet(direction, packet)

            for g in ground_relays.values():
                while g.storage:
                    packet = g.storage.popleft()
                    if packet.curr == packet.destination: # 도착
                        end_node = ground_relays[packet.curr]
                        packet.end(t, 'success', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
                        results.append(packet)
                        succeeded += 1
                    else: # 다시 위성으로
                        if packet.ttl >= 0:
                            packet.ttl -= 1
                        else:
                            failed.append(packet)
                            continue
                        packet.was_on_ground = True
                        packet.next_ground_node_id()
                        family = (satellites[node_id] for node_id in ground_relays[packet.ground_node].connected_sats)
                        """지상-위성 라우팅 알고리즘 적용 부분"""
                        packet, direction = ground_to_sat_forwarding(g, packet, family)
                        g.enqueue_packet(direction, packet)

            """실패 페이즈"""
            """경로 설정 과정에서 위성-지상 노드 링크 끊김으로 인한 전송 실패"""

            if failed:
                fail_cnt += len(failed)
            while failed:
                # print(failed)
                p = failed.pop(0)
                if p.was_on_ground:
                    if p.curr == p.source:
                        end_node = users[p.curr]
                    else:
                        end_node = ground_relays[p.curr]
                else:
                    end_node = satellites[p.curr]
                if p.ttl > 0:
                    p.end(t, 'inconsistency', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
                else:
                    p.end(t, 'expired', end_node.node_id, end_node.latitude_deg, end_node.longitude_deg)
                # print(end_node.connected_sats)
                # p.show_detailed()
                results.append(p)

            """위성 공전"""
            for s in satellites.values():
                s.update_position(omega_s, dt)
                s.time_tic(dt)

            """링크 여부 확인"""
            for g in ground_relays.values():
                g.time_tic(dt)
                sats = (satellites[node_id] for node_id in g.connected_sats if node_id not in g.disconnected)
                for s in sats:
                    if not s.is_visible(g.latitude_deg, g.longitude_deg):
                        # print(121212)
                        s.disconnected.add(g.node_id)
                        g.disconnected.add(s.node_id)

            for u in users.values():
                sats = (satellites[node_id] for node_id in u.connected_sats if node_id not in u.disconnected)
                for s in sats:
                    if not s.is_visible(u.latitude_deg, u.longitude_deg):
                        # print(2323223)
                        s.disconnected.add(u.node_id)
                        u.disconnected.add(s.node_id)

            if len(results) >= 100:
                rows = []
                ended += len(results)
                while results:
                    packet = results.pop(0)
                    common_data = [packet.start_at, packet.source, packet.destination, len(packet.result)]
                    if packet.success:
                        drop_data = [packet.result , packet.queuing_delays,
                                     sum(packet.queuing_delays), packet.propagation_delays, packet.transmission_delay,
                                     'success',
                                     None, None, None]
                    else:
                        drop_data = [packet.result , packet.queuing_delays,
                            sum(packet.queuing_delays[:-1]), packet.propagation_delays, packet.transmission_delay,
                            'inconsistency' if packet.inconsistency else 'drop',
                            f"at {packet.result[-1]}",
                            packet.ended_lat, packet.ended_lon
                        ]
                    row = common_data + drop_data
                    rows.append(row)
                csv_write(rows, filepath, filename)
            # generation rate에 대한 for문 종료
        #나머지 데이터 입력
        rows = []
        ended += len(results)
        while results:
            packet = results.pop(0)
            common_data = [packet.start_at, packet.source, packet.destination, len(packet.result)]
            if packet.success:
                drop_data = [packet.result, packet.queuing_delays,
                             sum(packet.queuing_delays), packet.propagation_delays, packet.transmission_delay,
                             'success',
                             None, None, None]
            else:
                drop_data = [packet.result, packet.queuing_delays,
                             sum(packet.queuing_delays[:-1]), packet.propagation_delays, packet.transmission_delay,
                             'inconsistency' if packet.inconsistency else 'drop',
                             f"at {packet.result[-1]}",
                             packet.ended_lat, packet.ended_lon
                             ]
            row = common_data + drop_data
            rows.append(row)
        csv_write(rows, filepath, filename)
        print("\n--- Simulation Summary ---")
        print(f"Generated: {steps*N_k*genertation_rate}")
        print(f"Ended:     {ended}")
        print(f"Succeeded: {succeeded}")
        print(f"Failed:    {fail_cnt}")
        print(f"Dropped:   {drop_cnt}")
        print("--------------------------")