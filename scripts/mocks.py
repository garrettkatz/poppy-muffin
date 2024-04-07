import numpy as np

class Motor:
    # present_ ("position", "speed", "load", "voltage", "temperature")
    pass

class PoppyHumanoid:
    def __init__(self):
        self.motors = []
        self._controllers = []
    def goto_position(self, trajectory, duration, wait):
        return
    def close(self):
        return

class OpenCVCamera:
    def __init__(self, name, p1, fps):
        self.frame = np.empty((120,240,3), dtype=np.uint8)
    def close(self):
        return
    

