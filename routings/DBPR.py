import math


def get_next_hop_DBPR(cur_sat, hor, ver, dst_sat):
    if ver.node_id == dst_sat.node_id:
        if cur_sat.isl_up == ver.node_id:
            return 0
        else:
            return 1
    elif hor.node_id == dst_sat.node_id:
        if cur_sat.isl_left == hor.node_id:
            return 2
        else:
            return 3
    else:
        Q_hor = get_virtual_DDD(hor, dst_sat)
        Q_ver = get_virtual_DDD(ver, dst_sat)

        if Q_ver < Q_hor:
            if cur_sat.isl_up == ver.node_id:
                return 0
            else:
                return 1
        else:
            if cur_sat.isl_left == hor.node_id:
                return 2
            else:
                return 3

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