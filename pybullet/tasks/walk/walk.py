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
lift_dict.update({ # works
    'l_ankle_y': 10, 'l_knee_y': 75, 'l_hip_y': -60,
    'abs_y': 20, 'abs_x': 20, 'r_shoulder_y': -20
    })
# lift_dict.update({
#     'l_ankle_y': -20, 'l_knee_y': 75, 'l_hip_y': -60,
#     })
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
    'r_ankle_y': -10, 'r_knee_y': 80, 'r_hip_y': -30})
settle = env.angle_array(settle_dict)

trajectory = [ # (angles, duration)
    (stand, 1),
    (lift, .5), # works
    (step, .3), # works
    # (lift, .2),
]

# initial angles/position
env.set_position(trajectory[0][0])

input("ready...")
t = 0
while True:

    target, duration = trajectory[t]
    env.goto_position(target, duration)
    
    t += 1
    if t == len(trajectory): t -= 1

    input('...')
    


