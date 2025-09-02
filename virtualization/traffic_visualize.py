from enum import Enum, auto
from panda3d.core import loadPrcFileData
loadPrcFileData("", "window-title Earth Window")
loadPrcFileData("", "win-origin 50 50")
loadPrcFileData("", "win-size 700 700")

import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv
from utils.walker_constellation import WalkerConstellation
from utils.rtpg_mapper import RTPGMapper
from utils.loader import load_ground_relays_from_csv, batch_map_nodes, normalize_wrapped_regions
from matplotlib.colors import  Normalize, LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd
from pathlib import Path

color_map = {
    'asia': (1, 0, 0, 1),
    'N.america': (0, 1, 0, 1),
    'S.america': (0, 0, 1, 1),
    'africa': (0, 1, 1, 1),
    'europe': (1, 0, 1, 1),
    'oceania': (1.0, 0.65, 0.0, 1.0),
}

class TrafficVisualization():

    def __init__(self):

        relay_csv_path = 'parameters/Ground_Relay_Coordinates.csv'
        N, M, F = 72, 22, 39  # walker-delta
        altitude_km, inclination_deg = 550, 53  # walker-delta

        self.constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=altitude_km, inclination_deg=inclination_deg)
        self.constellation.generate_constellation()
        self.satellites_dict = self.constellation.get_all_satellites()
        self.ground_relays_dict = load_ground_relays_from_csv(relay_csv_path, N * M)
        self.satellites = list(self.satellites_dict.values())
        self.ground_relays = list(self.ground_relays_dict.values())
        self.mapper = RTPGMapper(N, M, F, inclination_deg)

        relay_region_indices_asc, relay_region_indices_desc = self.mapper.batch_map_nodes(self.ground_relays)
        batch_search_region_asc, batch_search_region_desc = batch_map_nodes(
            N, M, inclination_deg, altitude_km, self.ground_relays, relay_region_indices_asc, relay_region_indices_desc)
        result_asc_r, result_desc_r, result_asc_nr, result_desc_nr = normalize_wrapped_regions(
            N, M, relay_region_indices_asc, relay_region_indices_desc, batch_search_region_asc,
            batch_search_region_desc)

        for sat in self.satellites:
            sat.region = self.mapper.get_region_index(sat)

        for relay, region_asc, region_desc, original_region_asc, original_region_desc, search_region_asc, search_region_desc in zip(
                self.ground_relays, relay_region_indices_asc, relay_region_indices_desc, result_asc_nr, result_desc_nr,
                result_asc_r, result_desc_r):
            relay.region_asc = region_asc
            relay.region_desc = region_desc
            relay.original_region_asc = original_region_asc
            relay.original_region_desc = original_region_desc
            relay.search_regions_asc = search_region_asc
            relay.search_regions_desc = search_region_desc

# 경도 보정
def wrap_lon(lon):
    lon = float(lon)
    if lon > 180.0: lon -= 360.0
    if lon < -180.0: lon += 360.0
    return lon

def visualize_relay(
    relays,
    figsize=(14, 7),
    markersize=50,
    textsize=8,
    label_offset_pts=(8, 3),
    edgecolor="black",
    alpha=0.95,
    save_path=None,
    show=True,
):
    """relays: 각 원소가 node_id, latitude_deg, longitude_deg, marker_shape, continent 속성을 가짐"""



    shape2mpl = {
        "triangle": "^",
        "square":   "s",
        "diamond":  "D",
        "circle":   "o",
        "star":     "*",
    }

    # 지도 생성
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f7f5f2")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#777777")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#555555")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbbbbb", alpha=0.7, linestyle="--")

    for r in relays:
        node_id  = r["node_id"] if isinstance(r, dict) else r.node_id
        lat      = float(r["latitude_deg"] if isinstance(r, dict) else r.latitude_deg)
        lon      = wrap_lon(r["longitude_deg"] if isinstance(r, dict) else r.longitude_deg)
        shape    = (r["marker_shape"] if isinstance(r, dict) else r.marker_shape).lower()
        cont     = r["continent"] if isinstance(r, dict) else r.continent

        # 위도 필터
        if not (-60 <= lat <= 60):
            continue

        marker = shape2mpl.get(shape, "o")
        color  = color_map.get(cont, (1, 1, 1, 1))  # 정확 매칭, 없으면 흰색 fallback

        ax.scatter(
            lon, lat,
            s=markersize,
            marker=marker,
            c=[color],
            transform=ccrs.PlateCarree(),
            edgecolor=edgecolor, linewidths=0.7, alpha=alpha,
            zorder=10
        )
        ax.annotate(
            str(node_id),
            xy=(lon, lat),
            xycoords=ccrs.PlateCarree(),
            textcoords="offset points",
            xytext=label_offset_pts,
            fontsize=textsize, ha="left", va="center",
            color="black", alpha=0.9, zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.6),
            annotation_clip=True,
        )

    ax.set_title("Ground Relay Nodes", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def overlay_sat_drop_counts(
    satellites,
    counts_csv_path: str,
    ax=None,                        # ← 기존 visualize_relay가 리턴한 ax를 넣어 겹쳐 그리기
    markersize=60,
    edgecolor="black",
    alpha=0.95,
    limit_lat_abs: float | None = 60.0,   # None이면 제한 없음. ax가 이미 있으면 보통 그대로 둬도 됨
    show_colorbar: bool = True,
    colorbar_kwargs: dict | None = None,  # {"pad":0.02,"fraction":0.03} 같은 옵션
    show: bool = True,                    # True면 plt.show()
):
    """
    counts_csv_path: 헤더가 정확히 'node_id','drop_count'인 CSV
    satellites: node_id, latitude_deg, longitude_deg 속성이 있는 객체 리스트
    ax: visualize_relay(...)가 만든 축 객체. 없으면 새 지도 축을 만든다(PlateCarree).
    """
    # 0) 축/도화지
    created_new_ax = False
    if ax is None:
        fig = plt.figure(figsize=(14, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        created_new_ax = True
    fig = ax.figure  # colorbar 달 때 필요

    # 1) 위성 인덱스: node_id -> (lat, lon)
    def _wrap_lon(lon):
        lon = float(lon)
        if lon > 180.0: lon -= 360.0
        if lon < -180.0: lon += 360.0
        return lon

    sat_map = {}
    for s in satellites:
        nid = getattr(s, "node_id")
        lat = float(getattr(s, "latitude_deg"))
        lon = _wrap_lon(getattr(s, "longitude_deg"))
        sat_map[str(nid)] = (lat, lon, s.is_ascending())
        #
        # # CSV가 "12"처럼 문자열일 수 있으니 정수키도 보조로 매핑
        # try:
        #     sat_map[str(int(nid))] = (lat, lon, s.is_ascending())
        # except Exception:
        #     pass

    # 2) CSV 읽기
    node_ids, counts = [], []
    with open(counts_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "node_id" not in reader.fieldnames or "drop_count" not in reader.fieldnames:
            raise ValueError("CSV must have columns: 'node_id', 'drop_count'")
        for row in reader:
            nid = str(row["node_id"])
            try:
                c = float(row["drop_count"])
            except Exception:
                continue
            if nid in sat_map:
                node_ids.append(nid)
                counts.append(c)

    if not node_ids:
        print("[overlay_sat_drop_counts] 매칭되는 데이터가 없습니다.")
        return ax

    # 3) 좌표/필터링
    lats_asc, lons_asc, counts_asc = [], [], []
    lats_desc, lons_desc, counts_desc = [], [], []
    counts_f = []
    for nid, c in zip(node_ids, counts):
        lat, lon, is_ascending = sat_map[nid]
        if (limit_lat_abs is not None) and (abs(lat) > float(limit_lat_abs)):
            continue
        counts_f.append(c)
        if is_ascending:
            lats_asc.append(lat)
            lons_asc.append(lon)
            counts_asc.append(c)
        else:
            lats_desc.append(lat)
            lons_desc.append(lon)
            counts_desc.append(c)

    if not lats_asc+lats_desc:
        print("[overlay_sat_drop_counts] 위도 범위 필터 후 남은 점이 없습니다.")
        return ax

    # 4) 파란색→빨간색 컬러맵 & 정규화
    cmap = LinearSegmentedColormap.from_list("blue_to_red", [(0,0,1), (1,0,0)], N=256)
    vmin, vmax = float(np.min(counts_f)), float(np.max(counts_f))
    if vmin == vmax:
        vmin, vmax = 0.0, vmax if vmax > 0 else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    # 5) 산점도(원형) — 배경을 건드리지 않고 겹쳐 그리기
    sc_asc = ax.scatter(
        lons_asc, lats_asc,
        s=markersize, marker="^",
        c=counts_asc, cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        edgecolor=edgecolor, linewidths=0.7, alpha=alpha,
        zorder=20,                      # ← 배경/리레이보다 위에
    )
    sc_desc = ax.scatter(
        lons_desc, lats_desc,
        s=markersize, marker="v",
        c=counts_desc, cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        edgecolor=edgecolor, linewidths=0.7, alpha=alpha,
        zorder=20,  # ← 배경/리레이보다 위에
    )

    # 새로 만든 축이라면 범위 설정(기존 축이면 건드리지 않음)
    if created_new_ax and (limit_lat_abs is not None):
        ax.set_extent([-180, 180, -float(limit_lat_abs), float(limit_lat_abs)], crs=ccrs.PlateCarree())
        ax.set_title("Satellite Drop Counts Overlay (blue → red)")

    # # 6) 컬러바 (원하면)
    # if show_colorbar:
    #     cb_kwargs = {"orientation": "horizontal", "pad": 0.02, "fraction": 0.03}
    #     if isinstance(colorbar_kwargs, dict):
    #         cb_kwargs.update(colorbar_kwargs)
    #     cb = fig.colorbar(sc_asc, ax=ax, pos="top", **cb_kwargs)
    #     cb.set_label("Drop count")

    if show:
        plt.show()
    return ax

def \
        build_node_lookup(satellites, relays):
    """
    satellites/relays: traffic_visualize의 객체 리스트
    return: dict[str(node_id)] -> (lat, lon)
    """
    lut = {}
    for s in satellites:
        lut[str(getattr(s, "node_id"))] = (
            float(getattr(s, "latitude_deg")),
            wrap_lon(getattr(s, "longitude_deg")),
        )
    for g in relays:
        lut[str(getattr(g, "node_id"))] = (
            float(getattr(g, "latitude_deg")),
            wrap_lon(getattr(g, "longitude_deg")),
        )
    return lut

def load_node_counts_csv(path, rate_filter=None):
    """
    path: node_counts CSV 파일 경로
    rate_filter: [1,80,160,...] 처럼 arrival_rate를 필터링하고 싶을 때 리스트/집합 전달 (없으면 전체)
    필요한 컬럼: node_id, total_counts, success_counts, drop_counts
    (있다면 arrival_rate도 함께 읽음)
    """
    usecols = ["node_id", "total_counts", "success_counts", "drop_counts", "arrival_rate"]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False, encoding="utf-8-sig")
    # 숫자형 정리
    for c in ("total_counts", "success_counts", "drop_counts", "arrival_rate"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if rate_filter is not None and "arrival_rate" in df.columns:
        df = df[df["arrival_rate"].isin(rate_filter)]
    return df

def plot_node_counts_heatmap(
    tv,
    title: str,
    node_counts_csv: str,
    value_col: str = "total_counts",
    rate_filter=None,
    lon_range=(-180, 180),
    lat_range=(-90, 90),
    grid_deg: float = 1.0,
    vmin=None, vmax=None,
    show_colorbar=True,
    figsize=(14, 7),
    save_path=None, show=True,
    mark_zero_with_presence=True,
    presence_scope="satellites",
    gray_rgba=(0.6, 0.6, 0.6, 0.85),

    # 🔽 추가: 히트맵/오버레이 투명도
    heatmap_alpha: float = 0.6,
    presence_alpha: float = 0.5,
):
    """
    node_id를 위경도로 매핑하여 정수 격자에 가중치(카운트)를 누적 후, 해안선만 있는 바탕에 히트맵으로 표시.
    추가: 위성 존재하지만 해당 격자 누적 count가 0인 셀을 회색으로 표시(옵션).
    """

    # 0) 내부 헬퍼: 노드 리스트에서 (lat, lon) 추출 (최소한의 견고성)
    def _extract_lat_lon(nodes):
        lats, lons = [], []
        for n in (nodes or []):
            # 속성 형태
            lat, lon = n.real_latitude_deg, n.real_longitude_deg
            lats.append(lat); lons.append(wrap_lon(lon))

        return np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)

    # 1) CSV 로드 & 선택 컬럼 확인
    df = load_node_counts_csv(node_counts_csv, rate_filter=rate_filter)
    need = {"node_id", value_col}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {miss}")

    # 2) 노드 위치 LUT
    lut = build_node_lookup(tv.satellites, tv.ground_relays)

    # 3) (lon, lat, weight) 목록 만들기
    lats, lons, weights = [], [], []
    for nid, val in zip(df["node_id"].astype(str), pd.to_numeric(df[value_col], errors="coerce")):
        if pd.isna(val):
            continue
        if nid not in lut:
            continue
        lat, lon = lut[nid]
        if not (lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]):
            continue
        lats.append(lat); lons.append(lon); weights.append(float(val))

    if not weights:
        print("[heatmap] 누적할 데이터가 없습니다.")
        return

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # 4) 2D 히스토그램(가중치 합산) → 격자 히트맵
    lon_edges = np.arange(lon_range[0], lon_range[1] + grid_deg, grid_deg)
    lat_edges = np.arange(lat_range[0], lat_range[1] + grid_deg, grid_deg)

    # x=경도(lon), y=위도(lat)
    H, xedges, yedges = np.histogram2d(
        lons, lats, bins=[lon_edges, lat_edges], weights=weights
    )
    # 주 히트맵 배열 (0은 투명 처리 -> 아래서 회색 오버레이로 구분)
    C = H.T.astype(float)
    C[C == 0] = np.nan  # 0은 회색 오버레이 대상이거나 완전 빈 영역이므로 일단 NaN으로

    # 5) (옵션) 존재-마스크 계산: 위성(또는 all) 존재하는 셀
    zero_mask_with_presence = None
    if mark_zero_with_presence:
        sat_lats, sat_lons = _extract_lat_lon(tv.satellites)
        if presence_scope == "all":
            gr_lats, gr_lons = _extract_lat_lon(tv.ground_relays)
            sat_lats = np.concatenate([sat_lats, gr_lats]) if sat_lats.size or gr_lats.size else sat_lats
            sat_lons = np.concatenate([sat_lons, gr_lons]) if sat_lons.size or gr_lons.size else sat_lons

        # 범위 필터
        keep = (
            (lat_range[0] <= sat_lats) & (sat_lats <= lat_range[1]) &
            (lon_range[0] <= sat_lons) & (sat_lons <= lon_range[1])
        )
        sat_lats = sat_lats[keep]; sat_lons = sat_lons[keep]

        if sat_lats.size and sat_lons.size:
            P, _, _ = np.histogram2d(sat_lons, sat_lats, bins=[lon_edges, lat_edges])  # 존재 카운트
            P = P.T  # (lat, lon)
            # "존재하지만 카운트는 0"인 셀
            zero_mask_with_presence = (H.T == 0) & (P > 0)
        else:
            zero_mask_with_presence = np.zeros_like(H.T, dtype=bool)

    # 6) 컬러 맵/정규화: 파란(낮음) → 빨강(높음)
    cmap = LinearSegmentedColormap.from_list("blue_to_red", [(0, 0, 1), (1, 0, 0)], N=256)
    if vmin is None: vmin = np.nanmin(C)
    if vmax is None: vmax = np.nanmax(C)
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    # 7) 지도(해안선만) + 히트맵
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#444444")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbbbbb", alpha=0.6, linestyle="--")

    # 주 히트맵
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, C,
        cmap=cmap, norm=norm, shading="auto",
        transform=ccrs.PlateCarree(), zorder=5,
        alpha=heatmap_alpha,          # ← 추가
    )

    # 회색 오버레이: 존재하지만 count=0
    if mark_zero_with_presence and zero_mask_with_presence is not None:
        G = np.full_like(C, np.nan, dtype=float)
        G[zero_mask_with_presence] = 1.0
        gray_cmap = ListedColormap([gray_rgba])
        ax.pcolormesh(
            lon_edges, lat_edges, G,
            cmap=gray_cmap, vmin=0.0, vmax=1.0, shading="auto",
            transform=ccrs.PlateCarree(), zorder=6,
            alpha=presence_alpha,      # ← 추가
        )

    ax.set_title(f"{title} • value={value_col}", fontsize=13)

    if show_colorbar:
        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05)
        cb.set_label(f"{value_col} (aggregated in {grid_deg}° grid)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax



if __name__ == '__main__':
    tv = TrafficVisualization()
    node_counts_csv = "utils/node counts/node_counts_rate_200.csv"
    src_dst_counts_csv = "utils/traffic counts/traffic_counts_200.csv"
    grid_degree = 30
    # fig, ax = visualize_relay(tv.ground_relays)
    #
    # # 2) 그 위에 드롭 히트맵을 ‘겹쳐’ 그림
    # overlay_sat_drop_counts(
    #     tv.satellites,
    #     counts_csv_path="results/limited_Q_with_GSL_320__drop_counts.csv",
    #     ax=ax,  # ← 기존 축 위에 그리기
    #     markersize=60,
    #     limit_lat_abs=60.0,  # relay 지도와 동일 범위면 깔끔
    #     show_colorbar=True,
    #     show=True  # 이제 한 번에 표시
    # )
    # node_counts CSV 예: node_id,total_counts,success_counts,drop_counts[,arrival_rate]

    # 1) total_counts 히트맵 (전 지구, 1° 격자)
    # 위성 및 지상 노드의 패킷 수신 밀도
    plot_node_counts_heatmap(
        tv,
        "Traffic Heatmap by Rounded Lat/Lon",
        node_counts_csv=node_counts_csv,
        value_col="total_counts",  # success_counts / drop_counts 로 바꿔도 OK
        rate_filter=None,  # [80,160] 같이 arrival_rate 필터링하고 싶으면 지정
        lon_range=(-180, 180),
        lat_range=(-60, 60),
        grid_deg=grid_degree,
        show_colorbar=True,
        save_path=None,
        show=True
    )
    # # 출발, 도착 지점 결정 밀도
    plot_node_counts_heatmap(
        tv,
        "Src/Dst Heatmap by Rounded Lat/Lon",
        node_counts_csv=src_dst_counts_csv,
        value_col="total_counts",  # success_counts / drop_counts 로 바꿔도 OK
        rate_filter=None,  # [80,160] 같이 arrival_rate 필터링하고 싶으면 지정
        lon_range=(-180, 180),
        lat_range=(-60, 60),
        grid_deg=grid_degree,
        show_colorbar=True,
        save_path=None,
        show=True
    )
    # 출발, 도착 지점 결정 밀도 (드롭)
    # plot_node_counts_heatmap(
    #     tv,
    #     "Src/Dst Heatmap by Rounded Lat/Lon",
    #     node_counts_csv=node_counts_csv,
    #     value_col="drop_counts",  # success_counts / drop_counts 로 바꿔도 OK
    #     rate_filter=None,  # [80,160] 같이 arrival_rate 필터링하고 싶으면 지정
    #     lon_range=(-180, 180),
    #     lat_range=(-60, 60),
    #     grid_deg=grid_degree,
    #     show_colorbar=True,
    #     save_path=None,
    #     show=True
    # )
