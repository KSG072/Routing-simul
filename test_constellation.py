import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.walker_constellation import WalkerConstellation  # 반드시 구현되어 있어야 함

# 시뮬레이션 파라미터
N = 72
M = 22
F = 39
altitude_km = 550
inclination_deg = 53
earth_radius_km = 6371
T_s = 95.4 * 60
omega_s = 2 * np.pi / T_s

speed = 30

dt = 1  # seconds/frame (위상 update 속도)
r_orbit = earth_radius_km + altitude_km

# Walker 위성군 생성
constellation = WalkerConstellation(N=N, M=M, F=F, altitude_km=altitude_km, inclination_deg=inclination_deg)
constellation.generate_constellation()
satellites = constellation.get_all_satellites()
colors = plt.cm.jet(np.linspace(0, 1, N))
sat_colors = [colors[sat.orbit_idx] for sat in satellites]

# 3D 플롯 초기화
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-r_orbit, r_orbit])
ax.set_ylim([-r_orbit, r_orbit])
ax.set_zlim([-r_orbit, r_orbit])
ax.set_box_aspect([1, 1, 1])
ax.axis('off')

# 본초자오선
lat_vals = np.linspace(-np.pi / 2, np.pi / 2, 100)
x = earth_radius_km * np.cos(lat_vals)
y = np.zeros_like(lat_vals)
z = earth_radius_km * np.sin(lat_vals)
ax.plot(x, y, z, color='red', linewidth=1.2)

# 적도
lon_vals = np.linspace(0, 2 * np.pi, 100)
x = earth_radius_km * np.cos(lon_vals)
y = earth_radius_km * np.sin(lon_vals)
z = np.zeros_like(lon_vals)
ax.plot(x, y, z, color='orange', linewidth=1)

# (0°, 0°) 점 표시
ax.scatter([earth_radius_km], [0], [0], color='red', s=50)

# 위성 스캐터 초기화
scatter = None

target_id = M*4 + 7

def update(frame):
    global scatter
    sim_time = frame * dt
    xs, ys, zs = [], [], []
    sizes = []

    for sat in satellites:
        sat.update_position(omega_s, dt)

        lat, lon = sat.get_position()
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        x = r_orbit * np.cos(lat_rad) * np.cos(lon_rad)
        y = r_orbit * np.cos(lat_rad) * np.sin(lon_rad)
        z = r_orbit * np.sin(lat_rad)

        xs.append(x)
        ys.append(y)
        zs.append(z)

        if sat.node_id == target_id:
            phase_deg = np.rad2deg(sat.phase_rad) % 360
            print(f"[t={sim_time:.0f}s] node_id={sat.node_id:d} → lat={lat:.3f}, lon={lon:.3f}, phase={phase_deg:.3f}")
            sizes.append(100)  # 크기 강조
        else:
            sizes.append(10)

    # 기존 scatter 제거
    if scatter:
        scatter.remove()

    # 새로운 scatter 생성
    scatter = ax.scatter(xs, ys, zs, s=sizes, color=sat_colors)

# 애니메이션 실행
ani = FuncAnimation(fig, update, frames=int(T_s), interval=100, blit=False)
plt.show()