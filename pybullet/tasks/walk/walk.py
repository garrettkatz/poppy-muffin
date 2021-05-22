import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv, convert_angles

env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
N = len(env.joint_index)

# got from running camera.py
cam = (1.200002670288086,
    15.999960899353027,
    -31.799997329711914,
    (-0.010284600779414177, -0.012256712652742863, 0.14000000059604645))
pb.resetDebugVisualizerCamera(*cam)

with open("../../../scripts/stand.pkl", "rb") as f: stand_dict = pk.load(f)
stand_dict.update({'r_shoulder_x': -20, 'l_shoulder_x': 20})
stand = env.angle_array(stand_dict)

# started with ../check/strides.py
lift_dict = dict(stand_dict)
# lift_dict.update({ # works
#     'l_ankle_y': 10, 'l_knee_y': 75, 'l_hip_y': -60,
#     'abs_y': 20, 'abs_x': 20, 'r_shoulder_y': -20
#     })
lift_dict.update({
    'l_ankle_y': -25, 'l_knee_y': 60, 'l_hip_y': -30,
    'r_ankle_y': -10, 'r_hip_y': 10,
    'r_shoulder_y': -45, 'r_shoulder_x': -45,
    'abs_x': 10,
    # 'abs_x': 30, 'l_shoulder_x': 0, 'r_shoulder_x': -40,
    # 'r_hip_x': -20, 'l_hip_x': -20,
    })
lift = env.angle_array(lift_dict)
step_dict = dict(stand_dict)
step_dict.update({
    'r_shoulder_y': -40, 'r_shoulder_x': -30,
    'l_ankle_y': 15, 'l_knee_y': 0, 'l_hip_y': -15,
    'r_ankle_y': -15, 'r_knee_y': 0, 'r_hip_y': 15})
step = env.angle_array(step_dict)
push_dict = dict(stand_dict)
push_dict.update({
    'l_ankle_y': 0, 'l_knee_y': 0, 'l_hip_y': 0,
    'r_ankle_y': -25, 'r_knee_y': 60, 'r_hip_y': -30})
push = env.angle_array(push_dict)
settle_dict = dict(stand_dict)
settle_dict.update({
    'l_ankle_y': 10, 'l_knee_y': 0, 'l_hip_y': -10,
    'r_ankle_y': -10, 'r_knee_y': 80, 'r_hip_y': -30})
settle = env.angle_array(settle_dict)

trajectory = [ # (angles, duration)
    (stand, 1),
    # (lift, .5), # works
    # (step, .3), # works
    (lift, .1),
    (step, .05),
    (push, .2),
    (env.mirror_position(step), .15),
    (env.mirror_position(push), .2),
    (step, .15),
    (push, .2),
] + [
    (env.mirror_position(step), .15),
    (env.mirror_position(push), .2),
    (step, .15),
    (push, .2),
]*10

# initial angles/position
env.set_position(trajectory[0][0])

input("ready...")
t = 0
while True:

    target, duration = trajectory[t]
    env.goto_position(target, duration)
    
    t += 1
    if t == len(trajectory): t -= 1

    # input('...')
    


