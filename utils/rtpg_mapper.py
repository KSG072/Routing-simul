import numpy as np

class RTPGMapper:
    def __init__(self, N, M, F, inclination_deg):
        self.N = N  # Number of orbits
        self.M = M  # Satellites per orbit
        self.F = F  # Phasing factor
        self.inclination_rad = np.deg2rad(inclination_deg)
        self.delta_omega = 2 * np.pi / N   # ΔΩ
        self.delta_phi = 2 * np.pi / M     # ΔΦ
        self.delta_f = 2 * np.pi * F / (N * M)  # Δf = phase diff between inter-orbit links

    def get_region_index(self, satellite, floor=True):
        """
        Compute (P, R) region index from satellite lat/lon using paper equations.
        Includes floating point error correction near region boundaries.
        """
        alpha = self.inclination_rad
        phi_rad = np.deg2rad(satellite.latitude_deg)

        # Eq. (3): u
        # # 위경도기반 페이즈 추정
        # sin_ratio = np.clip(np.sin(phi_rad) / np.sin(alpha), -1.0, 1.0)
        #
        # if satellite.is_ascending():
        #     u = np.arcsin(sin_ratio)
        # else:
        #     u = np.sign(phi_rad) * np.pi - np.arcsin(sin_ratio)

        # 시뮬레이터 기반 페이즈 추출
        u = satellite.phase_rad

        # Eq. (8): u_bar
        u_bar = u if u >= np.pi / 2 else u + 2 * np.pi

        # # 경도 기반 상승교점경도 추출 -1
        # # Eq. (4)-(6): xi
        # lambda_mt = np.mod(np.deg2rad(satellite.longitude_deg), 2 * np.pi)
        # xi = np.arctan(np.cos(alpha) * np.tan(u))
        # if not satellite.is_ascending():
        #     xi += np.pi
        #
        # # Eq. (7): L_nt
        # L_nt = np.mod(lambda_mt - xi, 2 * np.pi)


        # 💡 보정값 (작은 epsilon)
        eps = 1e-7

        # Eq. (7): R index
        R_raw = (u_bar - np.pi / 2) / self.delta_phi

        direction_eps = eps
        # direction_eps = -eps if satellite.is_ascending() else eps

        # # 경도 기반 상승교점경도 추출 -2
        # Eq. (4)-(6): P index
        # P_raw = L_nt / self.delta_omega
        # # 시뮬레이션 기반 P추출
        P_raw = satellite.orbit_idx

        if floor:
            P = int(np.floor(P_raw + eps))
            R = int(np.floor(R_raw - direction_eps))
        else:
            P = P_raw + eps
            R = R_raw - direction_eps

        return (P % self.N, R % self.M)

    def batch_map(self, satellites):
        return [self.get_region_index(sat) for sat in satellites]

    def get_region_index_from_nodes(self, node):
        """
        Compute two (P, R) index pairs for a ground relay:
        one assuming ascending phase, and one descending.
        Returns: (P_asc, R_asc), (P_desc, R_desc)
        """
        alpha = self.inclination_rad
        phi_rad = np.deg2rad(node.latitude_deg)
        lambda_mt = np.mod(np.deg2rad(node.longitude_deg), 2 * np.pi)
        sin_ratio = np.clip(np.sin(phi_rad) / np.sin(alpha), -1.0, 1.0)

        # --- Ascending phase ---
        u_asc = np.arcsin(sin_ratio)
        u_bar_asc = u_asc if u_asc >= np.pi / 2 else u_asc + 2 * np.pi
        R_asc = (u_bar_asc - np.pi / 2 + 1e-8) / self.delta_phi
        xi_asc = np.arctan(np.cos(alpha) * np.tan(u_asc))
        L_nt_asc = np.mod(lambda_mt - xi_asc, 2 * np.pi)
        P_asc = L_nt_asc / self.delta_omega

        # --- Descending phase ---
        u_desc = np.sign(phi_rad) * np.pi - np.arcsin(sin_ratio)
        u_bar_desc = u_desc if u_desc >= np.pi / 2 else u_desc + 2 * np.pi
        R_desc = (u_bar_desc - np.pi / 2 + 1e-8) / self.delta_phi
        xi_desc = np.arctan(np.cos(alpha) * np.tan(u_desc)) + np.pi
        L_nt_desc = np.mod(lambda_mt - xi_desc, 2 * np.pi)
        P_desc = L_nt_desc / self.delta_omega

        # Return both as integer index tuples
        return (P_asc % self.N, R_asc % self.M), (P_desc % self.N, R_desc % self.M)

    def batch_map_nodes(self, nodes):
        batch_map_asc, batch_map_desc = [], []
        for relay in nodes:
            (P_asc, R_asc), (P_desc, R_desc) = self.get_region_index_from_nodes(relay)
            batch_map_asc.append((P_asc, R_asc))
            batch_map_desc.append((P_desc, R_desc))
        return batch_map_asc, batch_map_desc
# 모듈 로컬 헬퍼: u, P, R 및 xi 반환 (copy&paste)
def compute_u_P_R_xi(mapper, node):
    alpha = mapper.inclination_rad
    phi_rad = np.deg2rad(node.latitude_deg)
    lambda_mt = np.mod(np.deg2rad(node.longitude_deg), 2 * np.pi)
    sin_ratio = np.clip(np.sin(phi_rad) / np.sin(alpha), -1.0, 1.0)

    # Ascending
    u_asc = np.arcsin(sin_ratio)
    u_bar_asc = u_asc if u_asc >= np.pi/2 else u_asc + 2*np.pi
    R_asc = (u_bar_asc - np.pi/2 + 1e-8) / mapper.delta_phi
    xi_asc = np.arctan(np.cos(alpha) * np.tan(u_asc))
    L_nt_asc = np.mod(lambda_mt - xi_asc, 2*np.pi)
    P_asc = L_nt_asc / mapper.delta_omega

    # Descending
    u_desc = np.sign(phi_rad)*np.pi - np.arcsin(sin_ratio)
    u_bar_desc = u_desc if u_desc >= np.pi/2 else u_desc + 2*np.pi
    R_desc = (u_bar_desc - np.pi/2 + 1e-8) / mapper.delta_phi
    xi_desc = np.arctan(np.cos(alpha) * np.tan(u_desc)) + np.pi
    L_nt_desc = np.mod(lambda_mt - xi_desc, 2*np.pi)
    P_desc = L_nt_desc / mapper.delta_omega

    return (u_asc, P_asc, R_asc, xi_asc), (u_desc, P_desc, R_desc, xi_desc)

# if __name__ == "__main__":
#     # 설정
#     N, M, F = 6, 11, 1
#     inc_deg = 53
#     mapper = RTPGMapper(N, M, F, inc_deg)
#
#     # Node 클래스 정의
#     class Node:
#         def __init__(self, latitude, longitude=0.0):
#             self.latitude = latitude
#             self.longitude = longitude
#             self.u_asc = None; self.P_asc = None; self.R_asc = None; self.xi_asc = None
#             self.lat_rec_asc = None; self.lon_rec_asc = None
#             self.u_desc = None; self.P_desc = None; self.R_desc = None; self.xi_desc = None
#             self.lat_rec_desc = None; self.lon_rec_desc = None
#
#     # 테스트 위도
#     latitudes = [0, 20, 30, 40, 50, 60, 70, 80]
#     nodes = [Node(lat) for lat in latitudes]
#
#     # 경도 0도 고정
#     for node in nodes:
#         node.longitude = 0.0
#
#     # 계산 및 역산
#     for node in nodes:
#         (u_asc, P_asc, R_asc, xi_asc), (u_desc, P_desc, R_desc, xi_desc) = compute_u_P_R_xi(mapper, node)
#         node.u_asc, node.P_asc, node.R_asc, node.xi_asc = u_asc, P_asc, R_asc, xi_asc
#         node.u_desc, node.P_desc, node.R_desc, node.xi_desc = u_desc, P_desc, R_desc, xi_desc
#
#         # Ascending
#         phi_rec_asc = np.arcsin(np.sin(mapper.inclination_rad) * np.sin(u_asc))
#         node.lat_rec_asc = np.rad2deg(phi_rec_asc)
#         lambda_rec_asc = xi_asc + P_asc * mapper.delta_omega
#         node.lon_rec_asc = np.rad2deg(np.mod(lambda_rec_asc, 2*np.pi))
#
#         # Descending
#         phi_rec_desc = np.arcsin(np.sin(mapper.inclination_rad) * np.sin(u_desc))
#         node.lat_rec_desc = np.rad2deg(phi_rec_desc)
#         lambda_rec_desc = xi_desc + P_desc * mapper.delta_omega
#         node.lon_rec_desc = np.rad2deg(np.mod(lambda_rec_desc, 2*np.pi))
#
#     # 결과 출력 (한국어 헤더)
#     # 컬럼명과 너비 정의: (라벨, 너비)
#     columns = [
#         ("위도", 6),
#         ("상승 u(°)", 9),
#         ("상승 P", 7),
#         ("상승 R", 7),
#         ("재구성 위도(상)", 15),
#         ("재구성 경도(상)", 15),
#         ("하강 u(°)", 9),
#         ("하강 P", 7),
#         ("하강 R", 7),
#         ("재구성 위도(하)", 15),
#         ("재구성 경도(하)", 15)
#     ]
#     # 헤더 생성
#     header = ' | '.join(label.rjust(width) for label, width in columns)
#     print(header)
#     print('-' * len(header))
#     # 데이터 출력
#     for node in nodes:
#         values = [
#             f"{node.latitude:6.1f}",
#             f"{np.rad2deg(node.u_asc):9.3f}",
#             f"{node.P_asc:7.3f}",
#             f"{node.R_asc:7.3f}",
#             f"{node.lat_rec_asc:15.3f}",
#             f"{node.lon_rec_asc:15.3f}",
#             f"{np.rad2deg(node.u_desc):9.3f}",
#             f"{node.P_desc:7.3f}",
#             f"{node.R_desc:7.3f}",
#             f"{node.lat_rec_desc:15.3f}",
#             f"{node.lon_rec_desc:15.3f}"
#         ]
#         print(' | '.join(values))
