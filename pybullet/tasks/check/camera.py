import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv, convert_angles

env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
N = len(env.joint_index)

with open("../../../scripts/stand.pkl", "rb") as f: stand_dict = pk.load(f)
stand = env.angle_array(stand_dict)

env.set_position(stand)

while True:
    
    cam = pb.getDebugVisualizerCamera()
    print(cam)
    print(len(cam))
    width, height, view, proj, camup, camfwd, horz, vert, yaw, pitch, dist, targ = cam
    print(dist, yaw, pitch, targ)
    
    dist = 1.200002670288086
    yaw = 7.600003242492676
    pitch = -28.200000762939453
    targ = (-0.035683758556842804, -0.004761232063174248, 0.26399990916252136)
    pb.resetDebugVisualizerCamera(dist, yaw, pitch, targ)

    input("ready...")



