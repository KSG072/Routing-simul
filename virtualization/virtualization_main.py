# main.py
from enum import Enum, auto
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-title Earth Window")
loadPrcFileData("", "win-origin 50 50")
loadPrcFileData("", "win-size 700 700")

from direct.showbase.ShowBase import ShowBase

from visualization import World
from rtp_renderer import RTPRenderer
from virtualization.routing import run_routing_simulation, run_isl_only_routing, clear_routing_visual

from utils.walker_constellation import WalkerConstellation
from utils.rtpg_mapper import RTPGMapper
from utils.loader import load_ground_relays_from_csv, batch_map_nodes, normalize_wrapped_regions

import numpy as np

class RoutingMode(Enum):
    OFF = auto()
    FULL = auto()
    ISL_ONLY = auto()


class EarthApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        relay_csv_path = 'parameters/Ground_Relay_Coordinates.csv'
        N, M, F = 72, 22, 39    # walker-delta
        altitude_km, inclination_deg = 550, 53  # walker-delta

        # N, M, F = 12, 49, 1     # walker-star
        # altitude_km, inclination_deg = 1200, 90  # walker-star

        T_s = 95.4 * 60

        constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=altitude_km, inclination_deg=inclination_deg)
        constellation.generate_constellation()
        satellites_dict = constellation.get_all_satellites()
        ground_relays_dict = load_ground_relays_from_csv(relay_csv_path, N * M)
        satellites = list(satellites_dict.values())
        ground_relays = list(ground_relays_dict.values())
        mapper = RTPGMapper(N, M, F, inclination_deg)

        relay_region_indices_asc, relay_region_indices_desc = mapper.batch_map_nodes(ground_relays)
        batch_search_region_asc, batch_search_region_desc = batch_map_nodes(
            N, M, inclination_deg, altitude_km, ground_relays, relay_region_indices_asc, relay_region_indices_desc)
        result_asc_r, result_desc_r, result_asc_nr, result_desc_nr = normalize_wrapped_regions(
            N, M, relay_region_indices_asc, relay_region_indices_desc, batch_search_region_asc, batch_search_region_desc)

        for sat in satellites:
            sat.region = mapper.get_region_index(sat)

        for relay, region_asc, region_desc, original_region_asc, original_region_desc, search_region_asc, search_region_desc in zip(
            ground_relays, relay_region_indices_asc, relay_region_indices_desc, result_asc_nr, result_desc_nr, result_asc_r, result_desc_r):
            relay.region_asc = region_asc
            relay.region_desc = region_desc
            relay.original_region_asc = original_region_asc
            relay.original_region_desc = original_region_desc
            relay.search_regions_asc = search_region_asc
            relay.search_regions_desc = search_region_desc

        # Earth View
        self.world = World(
            base=self,
            satellites=satellites,
            ground_relays=ground_relays,
            mapper=mapper,
            N=N, M=M, F=F,
            altitude_km=altitude_km,
            earth_radius_km=6371,
            inclination_deg=inclination_deg,
            T_s=T_s,
        )

        # RTPG Renderer
        self.rtp_renderer = RTPRenderer(
            win_mgr=self,
            loader=self.loader,
            satellites=satellites,
            ground_relays=ground_relays,
            mapper=mapper,
            N=N, M=M
        )

        self.routing_mode = RoutingMode.OFF

        # self.taskMgr.add(self.update_rtp_task, "updateRTP")
        self.accept("k", self.rtp_renderer.toggle_real_time_mode)
        self.accept("a", self.rtp_renderer.toggle_search_area)
        self.accept('s', self.rtp_renderer.toggle_single_relay_box)
        self.accept("r", self.on_r_key_pressed)
        self.accept("c", self.rtp_renderer.check_r_degree)
        self.accept("d", self.rtp_renderer.apply_drop_heatmap_blue_red)

    def on_r_key_pressed(self):
        self.world.is_paused = True

        if self.routing_mode == RoutingMode.OFF:
            run_routing_simulation(self.world, self.rtp_renderer)
            self.routing_mode = RoutingMode.FULL

        elif self.routing_mode == RoutingMode.FULL:
            # clear_routing_visual(self.world, self.rtp_renderer)
            run_isl_only_routing(self.world, self.rtp_renderer)
            self.routing_mode = RoutingMode.ISL_ONLY

        elif self.routing_mode == RoutingMode.ISL_ONLY:
            clear_routing_visual(self.world, self.rtp_renderer, clear_all=True)
            self.routing_mode = RoutingMode.OFF
            self.world.is_paused = False

    def update_rtp_task(self, task):
            self.rtp_renderer.update()
            return task.cont

if __name__ == '__main__':
    app = EarthApp()
    app.run()
