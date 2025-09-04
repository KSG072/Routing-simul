import math

from parameters.PARAMS import PACKET_SIZE_BITS, ISL_RATE_LASER, TAU


def get_next_hop_DBPR(cur_sat, hor, ver, dst_sat, packet):
    if ver.node_id == dst_sat.node_id:
        if cur_sat.isl_up == ver.node_id:
            direction = 0
        else:
            direction = 1
    elif hor.node_id == dst_sat.node_id:
        if cur_sat.isl_left == hor.node_id:
            direction = 2
        else:
            direction = 3
    else:
        Q_hor = get_virtual_DDD(hor, dst_sat)
        Q_ver = get_virtual_DDD(ver, dst_sat)

        if Q_ver < Q_hor:
            if cur_sat.isl_up == ver.node_id:
                direction = 0
            else:
                direction = 1
        else:
            if cur_sat.isl_left == hor.node_id:
                direction = 2
            else:
                direction = 3

    if direction == 0:
        packet.queuing_delays.append((cur_sat.isl_up_buffer.size*PACKET_SIZE_BITS)/(TAU*ISL_RATE_LASER))
    elif direction == 1:
        packet.queuing_delays.append((cur_sat.isl_down_buffer.size * PACKET_SIZE_BITS) / (TAU * ISL_RATE_LASER))
    elif direction == 2:
        packet.queuing_delays.append((cur_sat.isl_left_buffer.size * PACKET_SIZE_BITS) / (TAU * ISL_RATE_LASER))
    else:
        packet.queuing_delays.append((cur_sat.isl_right_buffer.size * PACKET_SIZE_BITS) / (TAU * ISL_RATE_LASER))

    return direction

# YDB
def get_virtual_DDD(cur_sat, target_sat):
    PARAM_C = 300 # 300 km/ms
    num_of_pkt_in_q = cur_sat.get_packet_number_to_target(target_sat.node_id)  # queue에 있는 pkt 수

    distance = DISTANCE_CALCULATION(cur_sat, target_sat)

    virtual_DDD = (distance / PARAM_C) * (num_of_pkt_in_q + 1)  # queue에 있는 pkt 수 + 현재 pkt

    return virtual_DDD


def DISTANCE_CALCULATION(src, dst):
    # if src.id == -1:
    #     print("@@@@@@@@@@@@@@@@@@@")
    #     return 99999999999999
    lat_src = math.radians(src.real_latitude_deg)
    lon_src = math.radians(src.real_longitude_deg)
    lat_dst = math.radians(dst.real_latitude_deg)
    lon_dst = math.radians(dst.real_longitude_deg)

    dlat = lat_dst - lat_src
    dlon = lon_dst - lon_src

    a = math.sin(dlat / 2) ** 2 + math.cos(lat_src) * math.cos(lat_dst) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    R_earth = 6371  # km
    r = R_earth + 550

    distance = r * c
    return distance