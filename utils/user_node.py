import numpy as np

class UserNode:
    def __init__(self, node_id, latitude, longitude, earth_radius_km=6371):
        """
        위도/경도를 기반으로 사용자 노드를 초기화
        """
        self.node_id = node_id
        self.latitude = latitude
        self.longitude = longitude
        self.earth_radius = earth_radius_km

        # RTPG region (asc, desc 각각 1개)
        self.region_asc = None
        self.region_desc = None

        # 주변 탐색 영역 (Polygon)
        self.search_regions_asc = None
        self.search_regions_desc = None

        # 시각화용 Panda3D marker (필요 시)
        self.marker = None

    def get_position(self):
        """
        위도, 경도를 튜플로 반환
        """
        return (self.latitude, self.longitude)

    def get_cartesian_coords(self):
        """
        구면 좌표계를 3D 카르테시안 좌표로 변환
        """
        lat_rad = np.deg2rad(self.latitude)
        lon_rad = np.deg2rad(self.longitude)
        r = self.earth_radius

        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        return np.array([x, y, z])

    def __repr__(self):
        return (f"UserNode(id={self.node_id}, lat={self.latitude:.2f}, "
                f"lon={self.longitude:.2f})")