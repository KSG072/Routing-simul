import random
from user_node import UserNode
from parameters.PARAMS import CONGESTION_PROBABILITY, CONGESTION_AREAS

def generate_users(start_idx, total_count):
    users = []
    node_id = start_idx

    for _ in range(total_count):
        if random.random() < CONGESTION_PROBABILITY:
            # 혼잡 지역 중 하나 랜덤 선택
            area = random.choice(CONGESTION_AREAS)
            location = area["city"]
            is_in_city = True
            latitude = area["lat"]
            longitude = area["lon"]
        else:
            # 전역 랜덤 생성
            location = "remote"
            is_in_city = True
            latitude = random.uniform(-90, 90)
            longitude = random.uniform(-180, 180)

        user = UserNode(node_id, latitude, longitude)
        user.set_userinfo(is_in_city, location)
        users.append(user)
        node_id += 1

    return users
