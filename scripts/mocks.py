import numpy as np

class Motor:
    # present_ ("position", "speed", "load", "voltage", "temperature")
    def __init__(self, name):
        self.name = name
        self.id = 0

class PoppyHumanoid:
    def __init__(self):
        self.motors = []
        self._controllers = []
        self.mock = True
        self.motors = [Motor(name) for name in ('abs_y', 'abs_x', 'abs_z', 'bust_y', 'bust_x', 'head_z', 'head_y', 'l_shoulder_y', 'l_shoulder_x', 'l_arm_z', 'l_elbow_y', 'r_shoulder_y', 'r_shoulder_x', 'r_arm_z', 'r_elbow_y', 'l_hip_x', 'l_hip_z', 'l_hip_y', 'l_knee_y', 'l_ankle_y', 'r_hip_x', 'r_hip_z', 'r_hip_y', 'r_knee_y', 'r_ankle_y')]

    def goto_position(self, trajectory, duration, wait):
        return
    def close(self):
        return

class OpenCVCamera:
    def __init__(self, name, p1, fps):
        self.frame = np.empty((120,240,3), dtype=np.uint8)
    def close(self):
        return
    

