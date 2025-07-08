# === ground_relay_node.py ===
import numpy as np

class GroundRelayNode:
    def __init__(self, node_id, latitude, longitude, continent, earth_radius_km=6371):
        self.node_id = node_id
        self.latitude = latitude
        self.longitude = longitude
        self.continent = continent      # color group
        self.earth_radius = earth_radius_km

        self.region_asc = None
        self.region_desc = None
        self.original_region_asc = None
        self.original_region_desc = None
        self.search_regions_asc = None
        self.search_regions_desc = None

        self.marker = None

    def get_position(self):
        return (self.latitude, self.longitude)

    def get_cartesian_coords(self):
        lat_rad = np.deg2rad(self.latitude)
        lon_rad = np.deg2rad(self.longitude)
        r = self.earth_radius
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        return np.array([x, y, z])

    def __repr__(self):
        return (f"GroundRelayNode(id={self.node_id}, lat={self.latitude:.2f}, "
                f"lon={self.longitude:.2f}, continent='{self.continent}')")