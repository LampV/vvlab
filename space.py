import numpy as np
class Discreate:
    def __init__(self, shape, low=0, high=1):
        self.shape = shape
        self.high = high
        self.low = low
        self.data = np.zeros(shape)

    def set_data(self, data):
        self.data = data

        