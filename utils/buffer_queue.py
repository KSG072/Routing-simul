class buffer:
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.capacity = capacity

    def enqueue(self, item):
        if self.size == self.capacity:
            raise QueueFullError("Queue is full - item dropped")

        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1

    def dequeue(self):
        item = self.buffer[self.head]
        self.buffer[self.head] = None
        self.head = (self.head + 1) % self.capacity
        self.size -= 1

        return item

    def is_full(self):
        return self.size == self.capacity

    def is_empty(self):
        return self.size == 0

    def __len__(self):
        return self.size

    def get_capacity(self):
        return self.capacity


# 커스텀 예외 클래스
class QueueFullError(Exception):
    pass