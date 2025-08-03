# === walker_constellation.py ===
import numpy as np

from utils.satellite import Satellite
from utils.orbit import Orbit

class WalkerConstellation:
    def __init__(self, N, M, F, inclination_deg=53, altitude_km=550, offset_deg=0):
        self.N = N  # Number of orbits
        self.M = M  # Satellites per orbit
        self.F = F  # Phasing factor

        self.inclination_deg = inclination_deg
        self.inclination_rad = np.deg2rad(inclination_deg)
        self.altitude_km = altitude_km

        self.delta_omega = 2 * np.pi / N
        self.delta_phi = 2 * np.pi / M
        self.delta_f = 2 * np.pi * F / (N * M)

        self.offset_deg = offset_deg
        self.offset_rad = np.deg2rad(self.offset_deg)

        self.orbits = []
        self.satellites = {}

    def generate_constellation(self):
        node_id = 0
        for n in range(self.N):
            if self.inclination_deg == 90:
                lon_asc_node = (n * self.delta_omega / 2 + self.offset_rad) % (2 * np.pi)
            else:
                lon_asc_node = (n * self.delta_omega + self.offset_rad) % (2 * np.pi)
            orbit = Orbit(n, lon_asc_node)

            for m in range(self.M):
                sat = Satellite(node_id, orbit_idx=n, sat_idx_in_orbit=m,
                                inclination_rad=self.inclination_rad,
                                altitude_km=self.altitude_km)
                phase = (2 * np.pi * m / self.M + self.delta_f * n) % (2 * np.pi)
                sat.set_position_from_phase(phase, lon_asc_node)
                sat.cartesian_coords = sat.get_cartesian_coords()
                sat.update_lat_lon_for_RTPG()
                orbit.add_satellite(sat)
                self.satellites[sat.node_id] = sat

                node_id += 1

            self.orbits.append(orbit)

    def get_all_satellites(self):
        return self.satellites
