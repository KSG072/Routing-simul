# PARAM.py
#TODO
# 논문은 위성과 지상 릴레이 모두 laser와 KA 동시에 사용. 또한, 그에 맞게 각각의 전용 큐가 있음 
# 현재 구현은 위성은 laser, 지상 릴레이는 KA 사용한다고 가정. 큐 또한 그에 맞게 사용
# PARAM.py

import random

# Configuration of constellations
NUM_SATELLITES = 1584
N = 72
M = 22
F = 39
altitude_km = 550
inclination_deg = 53
MIN_ELEVATION = 10 # degrees

C = 3 * 100000

TAU = 0.001
TIME_SLOT = 1
TOTAL_TIME = 600
KILO, MEGA, GIGA = 1000, 1000000, 1000000000

# Packet size (bits)
PACKET_SIZE_BITS = 64 * KILO
# Buffer sizes (위성 - Laser만, 지상 - Ka만)
# 위성 전용 Laser 버퍼s
# SENDING_BUFFER_QUEUE_LASER = 10 * 10**6  # bits
# SENDING_BUFFER_QUEUE_LASER_PACKETS = SENDING_BUFFER_QUEUE_LASER // (1500 * 8)
SENDING_BUFFER_QUEUE_LASER = 10 * MEGA  # bits
SENDING_BUFFER_QUEUE_LASER_PACKETS = SENDING_BUFFER_QUEUE_LASER // PACKET_SIZE_BITS

# 지상 전용 Ka 버퍼
SENDING_BUFFER_QUEUE_KA = 5 * MEGA  # bits
SENDING_BUFFER_QUEUE_KA_PACKETS = SENDING_BUFFER_QUEUE_KA // PACKET_SIZE_BITS

# 대기 버퍼
PUBLIC_WAITING_BUFFER_QUEUE_LASER = 1 * GIGA  # bits
PUBLIC_WAITING_BUFFER_QUEUE_LASER_PACKETS = PUBLIC_WAITING_BUFFER_QUEUE_LASER // PACKET_SIZE_BITS

PUBLIC_WAITING_BUFFER_QUEUE_KA = 40 * MEGA  # bits
PUBLIC_WAITING_BUFFER_QUEUE_KA_PACKETS = PUBLIC_WAITING_BUFFER_QUEUE_KA // PACKET_SIZE_BITS

# ISL (Inter-Satellite Link) rates - Laser only
ISL_RATE_LASER = 2.5 * GIGA  # bps

# SGL (Satellite-Ground Link) rates - Ka band only
SGL_KA_DOWNLINK = 1.5 * GIGA  # bps
SGL_KA_UPLINK = 2 * GIGA  # bps

# Packet arrival rates
PACKET_ARRIVAL_RATE_KA = (1 * MEGA, 9 * MEGA)  # bps range (SGL)
PACKET_ARRIVAL_RATE_LASER = (80 * MEGA, 350 * MEGA)  # bps range (ISL)

# Smoothing factors
SMOOTHING_FACTORS = [0.5, 2, 1, 2, 1]

PACKET_PER_MS_ON_ISL = ISL_RATE_LASER*TAU // PACKET_SIZE_BITS
PACKET_PER_MS_ON_UPLINK = SGL_KA_UPLINK*TAU // PACKET_SIZE_BITS
PACKET_PER_MS_ON_DOWNLINK = SGL_KA_DOWNLINK*TAU // PACKET_SIZE_BITS

# Generation rate for Laser ISL (packets per 1 ms, randomly sampled within range)
GENERATION_RATE_LASER_MIN = int((PACKET_ARRIVAL_RATE_LASER[0] / PACKET_SIZE_BITS) * TAU)
GENERATION_RATE_LASER_MAX = int((PACKET_ARRIVAL_RATE_LASER[1] / PACKET_SIZE_BITS) * TAU)

GENERATION_RATE = random.randint(GENERATION_RATE_LASER_MAX, GENERATION_RATE_LASER_MAX)

# GENERATION_RATE_LIST = [27 ,30, 33]
# GENERATION_RATE_LIST = [6,10,13,16,20,23,26,30,33,36,40]
GENERATION_RATE_LIST = [2,3,4,5,6]


PACKET_GEN_INTERVAL = 1

MAX_SAT_P_DIFF = 4


# PARAMS.py 내 혼잡 지역 설정 (대표 혼잡 도시 30개)
# CONGESTION_AREAS = [
#         {"lat_min": 40.5, "lat_max": 41.0, "lon_min": -74.3, "lon_max": -73.7, "lat": 40.75, "lon": -74.0,
#          "city": "New york"},
#         {"lat_min": 34.0, "lat_max": 34.2, "lon_min": -118.5, "lon_max": -118.2, "lat": 34.1, "lon": -118.35,
#          "city": "LA"},
#         {"lat_min": 51.2, "lat_max": 51.7, "lon_min": -0.5, "lon_max": 0.3, "lat": 51.45, "lon": -0.1, "city": "런던"},
#         {"lat_min": 48.8, "lat_max": 49.0, "lon_min": 2.2, "lon_max": 2.5, "lat": 48.9, "lon": 2.35, "city": "파리"},
#         {"lat_min": 35.6, "lat_max": 35.8, "lon_min": 139.6, "lon_max": 139.9, "lat": 35.7, "lon": 139.75,
#          "city": "Tokyo"},
#         {"lat_min": 37.4, "lat_max": 37.7, "lon_min": 126.8, "lon_max": 127.2, "lat": 37.55, "lon": 127.0,
#          "city": "Seoul"},
#         {"lat_min": 39.8, "lat_max": 40.0, "lon_min": 116.2, "lon_max": 116.6, "lat": 39.9, "lon": 116.4,
#          "city": "Beijing"},
#         {"lat_min": 19.3, "lat_max": 19.6, "lon_min": -99.3, "lon_max": -99.0, "lat": 19.45, "lon": -99.15,
#          "city": "Mexico city"},
#         {"lat_min": 55.6, "lat_max": 56.0, "lon_min": 37.4, "lon_max": 37.8, "lat": 55.8, "lon": 37.6, "city": "모스크바"},
#         {"lat_min": -23.7, "lat_max": -23.4, "lon_min": -46.8, "lon_max": -46.4, "lat": -23.55, "lon": -46.6,
#          "city": "Sao paulo"},
#         {"lat_min": 28.4, "lat_max": 28.8, "lon_min": 76.8, "lon_max": 77.4, "lat": 28.6, "lon": 77.1, "city": "뉴델리"},
#         {"lat_min": 41.7, "lat_max": 42.0, "lon_min": 12.3, "lon_max": 12.7, "lat": 41.85, "lon": 12.5, "city": "로마"},
#         {"lat_min": 31.1, "lat_max": 31.3, "lon_min": 121.3, "lon_max": 121.7, "lat": 31.2, "lon": 121.5,
#          "city": "Shanghai"},
#         {"lat_min": -34.7, "lat_max": -34.4, "lon_min": -58.5, "lon_max": -58.0, "lat": -34.55, "lon": -58.25,
#          "city": "Buenos aires"},
#         {"lat_min": 52.3, "lat_max": 52.7, "lon_min": 4.7, "lon_max": 5.1, "lat": 52.5, "lon": 4.9, "city": "암스테르담"},
#         {"lat_min": 22.2, "lat_max": 22.5, "lon_min": 113.9, "lon_max": 114.3, "lat": 22.35, "lon": 114.1,
#          "city": "Hongkong"},
#         {"lat_min": 59.2, "lat_max": 59.5, "lon_min": 17.8, "lon_max": 18.2, "lat": 59.35, "lon": 18.0, "city": "스톡홀름"},
#         {"lat_min": 43.6, "lat_max": 44.0, "lon_min": -79.7, "lon_max": -79.2, "lat": 43.8, "lon": -79.45,
#          "city": "Toronto"},
#         {"lat_min": 1.2, "lat_max": 1.5, "lon_min": 103.6, "lon_max": 104.0, "lat": 1.35, "lon": 103.8, "city": "싱가포르"},
#         {"lat_min": 50.0, "lat_max": 50.2, "lon_min": 14.3, "lon_max": 14.6, "lat": 50.1, "lon": 14.45, "city": "프라하"},
#         {"lat_min": -33.9, "lat_max": -33.7, "lon_min": 151.1, "lon_max": 151.3, "lat": -33.8, "lon": 151.2,
#          "city": "Sydney"},
#         {"lat_min": -26.3, "lat_max": -26.0, "lon_min": 27.9, "lon_max": 28.2, "lat": -26.15, "lon": 28.05,
#          "city": "Johannesburg"},
#         {"lat_min": 35.0, "lat_max": 35.2, "lon_min": 135.6, "lon_max": 135.8, "lat": 35.1, "lon": 135.7,
#          "city": "Osaka"},
#         {"lat_min": 30.0, "lat_max": 30.2, "lon_min": 31.1, "lon_max": 31.4, "lat": 30.1, "lon": 31.25, "city": "카이로"},
#         {"lat_min": 53.3, "lat_max": 53.6, "lon_min": -6.4, "lon_max": -6.0, "lat": 53.45, "lon": -6.2, "city": "더블린"},
#         {"lat_min": 13.6, "lat_max": 13.9, "lon_min": 100.4, "lon_max": 100.8, "lat": 13.75, "lon": 100.6,
#          "city": "Bangkok"},
#         {"lat_min": 59.8, "lat_max": 60.0, "lon_min": 30.2, "lon_max": 30.5, "lat": 59.9, "lon": 30.35,
#          "city": "St. Petersburg"},
#         {"lat_min": 25.2, "lat_max": 25.4, "lon_min": 55.2, "lon_max": 55.4, "lat": 25.3, "lon": 55.3, "city": "두바이"},
#         {"lat_min": 35.6, "lat_max": 35.8, "lon_min": 51.2, "lon_max": 51.5, "lat": 35.7, "lon": 51.35, "city": "테헤란"},
#         {"lat_min": 41.0, "lat_max": 41.3, "lon_min": 28.8, "lon_max": 29.2, "lat": 41.15, "lon": 29.0, "city": "이스탄불"}
# ]

CONGESTION_AREAS = [
    {"lat_min": 40.5, "lat_max": 41.0, "lon_min": -74.3, "lon_max": -73.7, "lat": 40.75, "lon": -74.0, "city": "New york"},

    {"lat_min": 34.0, "lat_max": 34.2, "lon_min": -118.5, "lon_max": -118.2, "lat": 34.1, "lon": -118.35, "city": "La"},
    {"lat_min": 51.2, "lat_max": 51.7, "lon_min": -0.5,  "lon_max": 0.3,   "lat": 51.45, "lon": -0.1,  "city": "London"},
    {"lat_min": 48.8, "lat_max": 49.0, "lon_min": 2.2,   "lon_max": 2.5,   "lat": 48.9,  "lon": 2.35,  "city": "Paris"},
    {"lat_min": 35.6, "lat_max": 35.8, "lon_min": 139.6, "lon_max": 139.9, "lat": 35.7,  "lon": 139.75,"city": "Tokyo"},
    {"lat_min": 37.4, "lat_max": 37.7, "lon_min": 126.8, "lon_max": 127.2, "lat": 37.55, "lon": 127.0, "city": "Seoul"},

    {"lat_min": 39.8, "lat_max": 40.0, "lon_min": 116.2, "lon_max": 116.6, "lat": 39.9,  "lon": 116.4, "city": "Beijing"},
    {"lat_min": 19.3, "lat_max": 19.6, "lon_min": -99.3, "lon_max": -99.0, "lat": 19.45, "lon": -99.15,"city": "Mexico city"},
    # {"lat_min": 55.6, "lat_max": 56.0, "lon_min": 37.4,  "lon_max": 37.8,  "lat": 55.8,  "lon": 37.6,  "city": "Moscow"},
    {"lat_min": -23.7,"lat_max": -23.4,"lon_min": -46.8, "lon_max": -46.4, "lat": -23.55,"lon": -46.6, "city": "Sao paulo"},
    {"lat_min": 28.4, "lat_max": 28.8, "lon_min": 76.8,  "lon_max": 77.4,  "lat": 28.6,  "lon": 77.1,  "city": "New delhi"},
    {"lat_min": 41.7, "lat_max": 42.0, "lon_min": 12.3,  "lon_max": 12.7,  "lat": 41.85, "lon": 12.5,  "city": "Rome"},

    {"lat_min": 31.1, "lat_max": 31.3, "lon_min": 121.3, "lon_max": 121.7, "lat": 31.2,  "lon": 121.5, "city": "Shanghai"},
    {"lat_min": -34.7,"lat_max": -34.4,"lon_min": -58.5, "lon_max": -58.0, "lat": -34.55,"lon": -58.25,"city": "Buenos aires"},
    {"lat_min": 52.3, "lat_max": 52.7, "lon_min": 4.7,   "lon_max": 5.1,   "lat": 52.5,  "lon": 4.9,   "city": "Amsterdam"},
    {"lat_min": 22.2, "lat_max": 22.5, "lon_min": 113.9, "lon_max": 114.3, "lat": 22.35, "lon": 114.1, "city": "Hongkong"},
    # {"lat_min": 59.2, "lat_max": 59.5, "lon_min": 17.8,  "lon_max": 18.2,  "lat": 59.35, "lon": 18.0,  "city": "Stockholm"},
    {"lat_min": 43.6, "lat_max": 44.0, "lon_min": -79.7, "lon_max": -79.2, "lat": 43.8,  "lon": -79.45,"city": "Toronto"},

    {"lat_min": 1.2,  "lat_max": 1.5,  "lon_min": 103.6, "lon_max": 104.0, "lat": 1.35,  "lon": 103.8, "city": "Singapore"},
    {"lat_min": 50.0, "lat_max": 50.2, "lon_min": 14.3,  "lon_max": 14.6,  "lat": 50.1,  "lon": 14.45, "city": "Praha"},
    {"lat_min": -33.9,"lat_max": -33.7,"lon_min": 151.1, "lon_max": 151.3, "lat": -33.8, "lon": 151.2, "city": "Sydney"},
    {"lat_min": -26.3,"lat_max": -26.0,"lon_min": 27.9,  "lon_max": 28.2,  "lat": -26.15,"lon": 28.05, "city": "Johannesburg"},
    {"lat_min": 35.0, "lat_max": 35.2, "lon_min": 135.6, "lon_max": 135.8, "lat": 35.1,  "lon": 135.7, "city": "Osaka"},

    {"lat_min": 30.0, "lat_max": 30.2, "lon_min": 31.1,  "lon_max": 31.4,  "lat": 30.1,  "lon": 31.25, "city": "Cairo"},
    # {"lat_min": 53.3, "lat_max": 53.6, "lon_min": -6.4,  "lon_max": -6.0,  "lat": 53.45, "lon": -6.2,  "city": "Dublin"},
    {"lat_min": 13.6, "lat_max": 13.9, "lon_min": 100.4, "lon_max": 100.8, "lat": 13.75, "lon": 100.6, "city": "Bangkok"},
    # {"lat_min": 59.8, "lat_max": 60.0, "lon_min": 30.2,  "lon_max": 30.5,  "lat": 59.9,  "lon": 30.35, "city": "St. petersburg"},
    {"lat_min": 25.2, "lat_max": 25.4, "lon_min": 55.2,  "lon_max": 55.4,  "lat": 25.3,  "lon": 55.3,  "city": "Dubai"},
    {"lat_min": 35.6, "lat_max": 35.8, "lon_min": 51.2,  "lon_max": 51.5,  "lat": 35.7,  "lon": 51.35, "city": "Tehran"},
    {"lat_min": 41.0, "lat_max": 41.3, "lon_min": 28.8,  "lon_max": 29.2,  "lat": 41.15, "lon": 29.0,  "city": "Istanbul"},
]


CONGESTION_PROBABILITY = 1 # 혼잡 지역에서 생성될 확률