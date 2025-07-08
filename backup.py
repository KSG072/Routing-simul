from collections import defaultdict

from utils.walker_constellation import WalkerConstellation
from utils.ground_relay_loader import load_ground_relays_from_csv
from utils.rtpg_mapper import RTPGMapper
from utils.loader import batch_map_ground_relays, normalize_wrapped_regions

from visualization import World
from direct.showbase.ShowBase import ShowBase

import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import numpy as np


# Visualization
def plot_constellation(constellation, altitude_km=550, ground_relays=None, save_path=None):
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    r_earth = 6371
    r_orbit = r_earth + altitude_km
    colors = plt.cm.jet(np.linspace(0, 1, constellation.N))

    # 🌍 육지(대륙) 시각화
    land_path = shpreader.natural_earth(resolution='110m', category='physical', name='land')
    for record in shpreader.Reader(land_path).records():
        geometries = [record.geometry] if record.geometry.geom_type == 'Polygon' else record.geometry.geoms
        for geom in geometries:
            lon, lat = geom.exterior.xy
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)
            x = r_earth * np.cos(lat_rad) * np.cos(lon_rad)
            y = r_earth * np.cos(lat_rad) * np.sin(lon_rad)
            z = r_earth * np.sin(lat_rad)
            ax.plot_trisurf(x, y, z, color='lightgray', alpha=0.3, linewidth=0)

    # 🌍 해안선 시각화
    coast_path = shpreader.natural_earth(resolution='110m', category='physical', name='coastline')
    for record in shpreader.Reader(coast_path).records():
        lines = [record.geometry] if record.geometry.geom_type == 'LineString' else record.geometry.geoms
        for line in lines:
            lon, lat = line.xy
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)
            x = r_earth * np.cos(lat_rad) * np.cos(lon_rad)
            y = r_earth * np.cos(lat_rad) * np.sin(lon_rad)
            z = r_earth * np.sin(lat_rad)
            ax.plot(x, y, z, color='black', linewidth=0.5)

    # 🛰️ 위성 시각화
    positions = []
    color_list = []
    for orbit in constellation.orbits:
        color_i = colors[orbit.orbit_idx]
        orbit_pos = []
        for sat in orbit.satellites:
            lat, lon = sat.get_position()
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)
            x = r_orbit * np.cos(lat_rad) * np.cos(lon_rad)
            y = r_orbit * np.cos(lat_rad) * np.sin(lon_rad)
            z = r_orbit * np.sin(lat_rad)
            orbit_pos.append([x, y, z])
            positions.append([x, y, z])
            color_list.append(color_i)
        orbit_pos = np.array(orbit_pos)
        ax.plot(orbit_pos[:, 0], orbit_pos[:, 1], orbit_pos[:, 2], color=color_i, linewidth=0.5, alpha=0.6)

    positions = np.array(positions)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color=color_list, s=10)

    # 📡 Ground Relay 표시
    if ground_relays:
        for relay in ground_relays:
            lat_rad = np.deg2rad(relay.latitude)
            lon_rad = np.deg2rad(relay.longitude)
            x = r_earth * np.cos(lat_rad) * np.cos(lon_rad)
            y = r_earth * np.cos(lat_rad) * np.sin(lon_rad)
            z = r_earth * np.sin(lat_rad)
            ax.scatter(x, y, z, color='black', s=20, edgecolors='white', linewidths=0.8, zorder=5)

    # 🌐 기준선 (본초자오선, 적도, Z축)
    num_segments = 36
    lat_vals = np.linspace(-np.pi / 2, np.pi / 2, num_segments)
    lon_vals = np.linspace(0, 2 * np.pi, num_segments)

    # 본초자오선
    x = r_earth * np.cos(lat_vals)
    y = np.zeros_like(lat_vals)
    z = r_earth * np.sin(lat_vals)
    ax.plot(x, y, z, color='red', linewidth=1)

    # 적도
    x = r_earth * np.cos(lon_vals)
    y = r_earth * np.sin(lon_vals)
    z = np.zeros_like(lon_vals)
    ax.plot(x, y, z, color='orange', linewidth=1)

    # Z축 (극축)
    ax.plot([0, 0], [0, 0], [-r_earth * 1.5, r_earth * 1.5], color='blue', linewidth=1.5)

    # 기준 점
    ax.scatter(r_earth, 0, 0, color='red', s=30)
    ax.scatter(0, 0, r_earth, color='blue', s=30)

    # 설정
    ax.set_xlim([-r_orbit, r_orbit])
    ax.set_ylim([-r_orbit, r_orbit])
    ax.set_zlim([-r_orbit, r_orbit])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_alpha(0)
    ax.margins(0)
    ax.set_position([0, 0, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, transparent=True)
        plt.close(fig)
    else:
        plt.show()


def plot_RTPG(N, M, region_indices, satellites):
    """
    Parameters:
    - N: number of orbits (P)
    - M: number of phase bins (R)
    - region_indices: list of (P, R) pairs
    - satellites: list of Satellite objects corresponding to region_indices
    """
    import matplotlib.pyplot as plt
    import numpy as np

    region_indices = np.array(region_indices)
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.jet(np.linspace(0, 1, N))  # N개의 색상

    for (P, R), sat in zip(region_indices, satellites):
        orbit_idx = sat.orbit_idx
        ax.scatter(P, R, color=colors[orbit_idx], s=40, edgecolors='k', linewidths=0.5, zorder=3)

        # 텍스트는 점 아래에 표시
        text_label = str(sat.node_id % M)
        ax.text(P, R - 0.3, text_label, color='black', fontsize=8, ha='center', va='top', zorder=4)

    ax.set_xlabel('P index (Longitude region)')
    ax.set_ylabel('R index (Phase region)')
    ax.set_title('Satellite Mapping to RTPG Grid (P, R) by Orbit Color and Satellite ID mod M')

    ax.set_xticks(np.arange(0, N))
    ax.set_yticks(np.arange(0, M))
    ax.set_xlim([-0.5, N - 0.5])
    ax.set_ylim([-1, M])

    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.show()


def visualize_rtpg_with_wraparound(graph, N, M):
    pos = {node: data['position'] for node, data in graph.nodes(data=True)}
    colors = plt.cm.jet(np.linspace(0, 1, N))
    fig, ax = plt.subplots(figsize=(16, 9))

    masking = [False for _ in range(N)]

    upward = 0
    downward = 0
    for u, v, data in graph.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        if x1 < x2 and y1 < y2:
            upward += 1
        elif x1 < x2 and y1 > y2:
            downward += 1

    # Draw edges
    for u, v, data in graph.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        is_wrap_x = abs(x1 - x2) > N / 2
        is_wrap_y = abs(y1 - y2) > M / 2
        style = '-' if data['type'] == 'intra' else ':'
        color = 'red' if is_wrap_x or is_wrap_y else ('black' if data['type'] == 'intra' else 'gray')
        lw = 1.5 if data['type'] == 'intra' else 1.0

        if is_wrap_x:
            # Wrap horizontally (inter-orbit), draw around edges
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            ax.plot([x1, -1], [y1, y1], linestyle=style, color=color, linewidth=lw, zorder=1)
            ax.plot([N, x2], [y2, y2], linestyle=style, color=color, linewidth=lw, zorder=1)


        elif is_wrap_y:
            # Wrap vertically (intra-orbit), draw top-bottom arcs
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1

            ax.plot([x1, x1], [y1, -1.0], linestyle=style, color=color, linewidth=lw, zorder=1)
            ax.plot([x2, x2], [M, y2], linestyle=style, color=color, linewidth=lw, zorder=1)

            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1

            if x1 + 1 != x2:
                continue

            if upward > downward:
                ax.plot([x1, x2], [y1, y2 + M], linestyle=style, color=color, linewidth=lw, zorder=1)
                ax.plot([x1, x2], [y1 - M, y2], linestyle=style, color=color, linewidth=lw, zorder=1)
            else:
                ax.plot([x1, x2], [y1, y2 - M], linestyle=style, color=color, linewidth=lw, zorder=1)
                ax.plot([x1, x2], [y1 + M, y2], linestyle=style, color=color, linewidth=lw, zorder=1)
            masking[x1] = True
        else:
            ax.plot([x1, x2], [y1, y2], linestyle=style, color=color, linewidth=lw, zorder=1)

    # Draw nodes
    for node_id, data in graph.nodes(data=True):
        if data['type'] != 'satellite':
            continue

        P, R = data['position']

        ax.scatter(P, R, color=colors[P], s=40, edgecolors='k', linewidths=0.5, zorder=3)
        ax.text(P, R - 0.3, str(sat.node_id % M), color='black', fontsize=7, ha='center', va='top', zorder=10, bbox=dict(facecolor='white', alpha=0.8, ec="none"))

        if P == 0:
            ax.text(P - 0.5, R + 0.1, str(sat.node_id % M), color='black', fontsize=7, ha='center', va='top', zorder=10, bbox=dict(facecolor='white', alpha=0.8, ec="none"))
        elif P == N - 1:
            ax.text(P + 0.5, R + 0.1, str(sat.node_id % M), color='black', fontsize=7, ha='center', va='top', zorder=10, bbox=dict(facecolor='white', alpha=0.8, ec="none"))

        if not masking[P]:
            continue

        if upward > downward:
            if R == M - 1:
                ax.text(P + 0.5, R + 0.5, str(sat.node_id % M), color='black', fontsize=7, ha='center', va='top', zorder=10, bbox=dict(facecolor='white', alpha=0.8, ec="none"))
        else:
            if R == 0:
                ax.text(P + 0.1, R - 0.3, str(sat.node_id % M), color='black', fontsize=7, ha='center', va='top', zorder=10, bbox=dict(facecolor='white', alpha=0.8, ec="none"))

    # # 각 ground relay가 커버하는 RTPG 범위 계산
    # for i, relay in enumerate(ground_relays):
    #     P_range_asc, P_range_desc, R_range = compute_key_node_search_range(
    #         latitude_deg=relay.latitude,
    #         longitude_deg=-relay.longitude,
    #         N=N, M=M,
    #         inclination_deg=inclination_deg,
    #         altitude_km=altitude_km,
    #     )
    #
    #     center_P = (P_range_asc[0] + P_range_asc[1]) / 2
    #     center_R = (R_range[0] + R_range[1]) / 2
    #
    #     width = P_range_asc[1] - P_range_asc[0]  # 가로 지름
    #     height = R_range[1] - R_range[0]  # 세로 지름
    #
    #     # 안쪽 원 대신 점
    #     ax.scatter(center_P, center_R, s=100, facecolors='none', edgecolors='blue', linewidths=2.5, zorder=5)
    #
    #     # 바깥쪽 타원
    #     from matplotlib.patches import Ellipse
    #     outer_ellipse = Ellipse((center_P, center_R), width=width, height=height,
    #                             angle=0, edgecolor='green', facecolor='none', linewidth=2, zorder=5)
    #     ax.add_patch(outer_ellipse)

    for node_id, data in graph.nodes(data=True):
        if data['type'] != 'relay':
            continue

        for poly in data['outer_rect']:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='green', linewidth=1, zorder=11)

        for poly in data['search_polygons']:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='orange', linewidth=1, zorder=11)

        P_r, R_r = data['position']
        ax.scatter(P_r, R_r, color='green', s=80, edgecolors='k', linewidths=0.5, zorder=11)

    ax.set_xlabel('P index (Longitude region)')
    ax.set_ylabel('R index (Phase region)')
    ax.set_title('RTPG Graph with ISL Connections and Wraparound Highlighted')
    ax.set_xticks(np.arange(0, N))
    ax.set_yticks(np.arange(0, M))
    ax.set_xlim([-1, N])  # Extend x margin
    ax.set_ylim([-0.9, M-0.1])  # Extend y margin
    # ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.show()


def check_RTPG_conflicts(region_indices, satellites, M):
    """
    region_indices: List of (P, R) tuples
    satellites: Corresponding list of Satellite objects
    M: Number of satellites per orbit
    """
    # region → list of satellites
    region_map = defaultdict(list)

    for (P, R), sat in zip(region_indices, satellites):
        region_map[(P, R)].append(sat)

    # 출력
    total_conflicts = 0
    for (P, R), sats in region_map.items():
        if len(sats) > 1:
            total_conflicts += 1
            print(f"\n⚠️ Conflict at Region (P={P}, R={R}) with {len(sats)} satellites:")
            for sat in sats:
                print(f"  - sat_id % M = {sat.node_id % M:2d}, orbit = {sat.orbit_idx:2d}, "
                      f"{'Ascend' if sat.is_ascending() else 'Descend':8s}, "
                      f"lat = {sat.latitude_deg:6.2f}, lon = {sat.longitude_deg:6.2f}")
    print(f"\n✅ Total conflicts found: {total_conflicts}")


if __name__ == '__main__':
    relay_csv_path = './Ground_Relay_Coordinates.csv'
    image_dir = "./fig/"

    # 파라미터 설정
    N = 72
    M = 22
    F = 39
    altitude_km = 550
    inclination_deg = 53

    dt = 0.001  # 1 ms
    total_time = 20.51  # seconds
    steps = int(total_time / dt)

    T_s = 95.4 * 60  # 궤도 주기
    omega_s = 2 * np.pi / T_s

    # 1. Constellation 생성
    constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=altitude_km, inclination_deg=inclination_deg)
    constellation.generate_constellation()
    satellites = constellation.get_all_satellites()
    ground_relays = load_ground_relays_from_csv(relay_csv_path, N * M)

    # 2. 필요한 모듈 초기화
    mapper = RTPGMapper(N, M, F, inclination_deg)

    relay_region_indices_asc, relay_region_indices_desc = mapper.batch_map_ground_relays(ground_relays)
    batch_search_region_asc, batch_search_region_desc = batch_map_ground_relays(N, M, inclination_deg, altitude_km, ground_relays, relay_region_indices_asc, relay_region_indices_desc)
    result_asc, result_desc = normalize_wrapped_regions(N, M, relay_region_indices_asc, relay_region_indices_desc, batch_search_region_asc, batch_search_region_desc)

    for relay, region_asc, region_desc, search_region_asc, search_region_desc in zip(ground_relays, relay_region_indices_asc, relay_region_indices_desc, result_asc, result_desc):
        relay.region_asc = region_asc
        relay.region_desc = region_desc
        relay.search_regions_asc = search_region_asc
        relay.search_regions_desc = search_region_desc

    base = ShowBase()
    w = World(
        base,
        satellites=satellites,
        ground_relays=ground_relays,
        mapper=mapper,
        N=N,
        M=M,
        F=F,
        altitude_km=altitude_km,
        earth_radius_km=6371,
        inclination_deg=inclination_deg,
        T_s=T_s,
    )
    base.run()