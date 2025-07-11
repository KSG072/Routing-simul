class Packet:
    def __init__(self, idx, src, dst, path):
        self.idx = idx
        self.source = src
        self.destination = dst
        self.path = path

        self.remaining_vertical_hops = None
        self.remaining_horizontal_hops = None

