import numpy as np


def compute_key_node_search_range_from_indices(P_GRk, R_GRk, N, M, latitude_deg, inclination_deg, altitude_km):
    """
    Implements the key node search range from Equations (22) and (23) of the paper,
    using already known RTPG indices (P_GRk^A, R_GRk^A) and (P_GRk^D, R_GRk^D).

    Returns:
        - P_range_asc: (P_min_asc, P_max_asc) for ascending satellite search
        - P_range_desc: (P_min_desc, P_max_desc) for descending satellite search
        - R_range_A: (R_min_A, R_max_A) for ascending
        - R_range_D: (R_min_D, R_max_D) for descending
    """
    # Constants
    re = 6371  # Earth radius in km
    alpha = np.deg2rad(inclination_deg)
    h = altitude_km

    # Step 1: compute rs (search radius)
    theta = np.deg2rad(15)  # minimum elevation angle
    gamma = np.arcsin(re * np.sin(theta + np.pi / 2) / (h + re))
    beta = np.pi / 2 - theta - gamma
    rs = re * beta

    # Step 2: Δh_min from Eq (21)
    h_min = alpha - np.arcsin(np.sin(alpha) * np.sin(np.pi / 2 - (2 * np.pi / M) * (M - 1)))

    # Step 3: ΔP and ΔR from Eq (16) and (17)
    lat_rad = np.deg2rad(latitude_deg)
    delta_omega_deg = 360 / N  # ΔΩ in degrees
    delta_P = 2 * int(np.ceil((180 * rs) / (np.pi * re * np.cos(lat_rad) * delta_omega_deg)))
    delta_R = 2 * int(np.ceil((180 * rs) / (np.pi * re * h_min)))

    # Step 4: Apply Eq (22), (23) to get search windows
    P_min = (P_GRk - delta_P / 2 + N) % N
    P_max = (P_GRk + delta_P / 2) % N
    R_min = (R_GRk - delta_R / 2 + M) % M
    R_max = (R_GRk + delta_R / 2) % M

    return P_min, P_max, R_max, R_min

def batch_map_ground_relays(N, M, inclination_deg, altitude_km, ground_relays, relay_region_indices_asc, relay_region_indices_desc):
    batch_search_region_asc, batch_search_region_desc = [], []
    for idx, relay in enumerate(ground_relays):
        (P_asc, R_asc) = relay_region_indices_asc[idx]
        (P_desc, R_desc) = relay_region_indices_desc[idx]

        latitude_deg = relay.latitude_deg

        P_min_asc, P_max_asc, R_min_asc, R_max_asc = compute_key_node_search_range_from_indices(P_asc, R_asc, N, M, latitude_deg, inclination_deg, altitude_km)
        P_min_desc, P_max_desc, R_min_desc, R_max_desc = compute_key_node_search_range_from_indices(P_desc, R_desc, N, M, latitude_deg, inclination_deg, altitude_km)

        batch_search_region_asc.append((P_min_asc, P_max_asc, R_min_asc, R_max_asc))
        batch_search_region_desc.append((P_min_desc, P_max_desc, R_min_desc, R_max_desc))

    return batch_search_region_asc, batch_search_region_desc

def batch_map_users(N, M, inclination_deg, altitude_km, users, user_region_indices_asc, user_region_indices_desc):
    batch_search_region_asc, batch_search_region_desc = [], []
    for idx, user in enumerate(users):
        (P_asc, R_asc) = user_region_indices_asc[idx]
        (P_desc, R_desc) = user_region_indices_desc[idx]

        latitude_deg = user.latitude_deg

        P_min_asc, P_max_asc, R_min_asc, R_max_asc = compute_key_node_search_range_from_indices(P_asc, R_asc, N, M, latitude_deg, inclination_deg, altitude_km)
        P_min_desc, P_max_desc, R_min_desc, R_max_desc = compute_key_node_search_range_from_indices(P_desc, R_desc, N, M, latitude_deg, inclination_deg, altitude_km)

        batch_search_region_asc.append((P_min_asc, P_max_asc, R_min_asc, R_max_asc))
        batch_search_region_desc.append((P_min_desc, P_max_desc, R_min_desc, R_max_desc))

    return batch_search_region_asc, batch_search_region_desc