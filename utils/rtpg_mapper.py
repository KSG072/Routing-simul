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
        sin_ratio = np.clip(np.sin(phi_rad) / np.sin(alpha), -1.0, 1.0)

        if satellite.is_ascending():
            u = np.arcsin(sin_ratio)
        else:
            u = np.sign(phi_rad) * np.pi - np.arcsin(sin_ratio)

        # Eq. (8): u_bar
        u_bar = u if u >= np.pi / 2 else u + 2 * np.pi

        # Eq. (4)-(6): xi
        lambda_mt = np.mod(np.deg2rad(satellite.longitude_deg), 2 * np.pi)
        xi = np.arctan(np.cos(alpha) * np.tan(u))
        if not satellite.is_ascending():
            xi += np.pi

        # Eq. (7): L_nt
        L_nt = np.mod(lambda_mt - xi, 2 * np.pi)

        # 💡 보정값 (작은 epsilon)
        eps = 1e-6

        # Eq. (7): R index
        R_raw = (u_bar - np.pi / 2) / self.delta_phi

        direction_eps = -eps if satellite.is_ascending() else eps

        # Eq. (4)-(6): P index
        P_raw = L_nt / self.delta_omega

        if floor:
            P = int(np.floor(P_raw + eps))
            R = int(np.floor(R_raw + direction_eps))
        else:
            P = P_raw + eps
            R = R_raw + direction_eps

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
        phi_rad = np.deg2rad(node.latitude)
        lambda_mt = np.mod(np.deg2rad(node.longitude), 2 * np.pi)
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