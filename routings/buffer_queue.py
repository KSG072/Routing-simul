from collections import deque
from parameters.PARAMS import SENDING_BUFFER_QUEUE_LASER_PACKETS, SENDING_BUFFER_QUEUE_KA_PACKETS, PACKET_PER_MS_ON_ISL, PACKET_PER_MS_ON_UPLINK, PACKET_PER_MS_ON_DOWNLINK

class Buffer:
    def __init__(self, type):
        capacity = SENDING_BUFFER_QUEUE_KA_PACKETS if type == 'down' else SENDING_BUFFER_QUEUE_LASER_PACKETS
        if type == 'isl':
            self.rate = PACKET_PER_MS_ON_ISL
        elif type == 'up':
            self.rate = PACKET_PER_MS_ON_UPLINK
        else:
            self.rate = PACKET_PER_MS_ON_DOWNLINK
        self.buffer = deque()
        self.size = 0
        self.capture = 0
        self.past_size = 0
        self.capacity = capacity if type != 'up' else 100000000

    def enqueue(self, item):
        self.buffer.append(item)
        self.size += 1

    def dequeue(self):
        self.size -= 1
        return self.buffer.popleft()

    def dequeue_sequences(self, dt):
        """
        td_ms(ms) 동안 처리 가능한 아이템을 FIFO로 꺼내 리스트로 반환.
        self.rate: ms당 처리 가능한 아이템 수
        """
        max_can_dequeue = int(self.rate * dt)
        n = min(max_can_dequeue, self.size)
        # print(n)

        seq = []
        for _ in range(n):
            seq.append(self.dequeue())
        return seq

    def is_empty(self):
        return self.size == 0

    def __len__(self):
        return self.size

    def get_capacity(self):
        return self.capacity

    def get_load_status(self):
        return self.past_size

    def drop(self):
        dropped = []
        while self.size > self.capacity:
            dropped.append(self.buffer.pop())
            self.size -= 1
        self.past_size = self.capture
        self.capture = self.size
        return dropped[::-1]



# 커스텀 예외 클래스
class QueueFullError(Exception):
    pass