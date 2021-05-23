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

cam = (1.200002670288086,
       56.799964904785156,
       -22.20000648498535,
       (-0.010284600779414177, -0.012256712652742863, 0.14000000059604645))
pb.resetDebugVisualizerCamera(*cam)

while True:
    
    cam = pb.getDebugVisualizerCamera()
    print(cam)
    print(len(cam))
    width, height, view, proj, camup, camfwd, horz, vert, yaw, pitch, dist, targ = cam
    print(dist, yaw, pitch, targ)

    _, _, rgb, depth, segment = pb.getCameraImage(width, height, view, proj)
    
    with open("getcam.pkl","wb") as f: pk.dump((rgb, depth, segment), f)
    
    input("ready...")



