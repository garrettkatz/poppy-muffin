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

# got from running camera.py
cam = (1.200002670288086,
       56.799964904785156,
       -22.20000648498535,
       (-0.010284600779414177, -0.012256712652742863, 0.14000000059604645))
pb.resetDebugVisualizerCamera(*cam)

lift_dict = dict(stand_dict)
lift_dict.update({'l_ankle_y': 30, 'l_knee_y': 80, 'l_hip_y': -50})
lift = env.angle_array(lift_dict)

step_dict = dict(stand_dict)
step_dict.update({
    'l_ankle_y': 20, 'l_knee_y': 0, 'l_hip_y': -20,
    'r_ankle_y': -20, 'r_knee_y': 0, 'r_hip_y': 20})
step = env.angle_array(step_dict)

push_dict = dict(stand_dict)
push_dict.update({
    'l_ankle_y': 0, 'l_knee_y': 0, 'l_hip_y': 0,
    'r_ankle_y': 50, 'r_knee_y': 90, 'r_hip_y': -40})
push = env.angle_array(push_dict)

settle_dict = dict(stand_dict)
settle_dict.update({
    'l_ankle_y': 10, 'l_knee_y': 0, 'l_hip_y': -10,
    'r_ankle_y': -20, 'r_knee_y': 40, 'r_hip_y': -20})
settle = env.angle_array(settle_dict)

# env.set_position(stand)
# env.set_position(lift)
# env.set_position(step)
# env.set_position(push)
# env.set_position(settle)
# input("...")

# angles = [stand, lift, step, push]
angles = [stand, lift, step, settle, stand]
a = 0
while True:

    env.set_position(angles[a])
    # if a > 1: angles[a] = env.mirror_position(angles[a])
    if 0 < a < 4: angles[a] = env.mirror_position(angles[a])
    a += 1
    # if a == len(angles): a = 2
    if a == len(angles): a = 1
    input("...")


