import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv, convert_angles

angles = [{}]

env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
N = len(env.joint_index)

with open("../../../scripts/crawl2.pkl","rb") as f:
    crawl_angles = pk.load(f)

# from scripts:
crawl_angles.update({
    'l_hip_y': -100., 'l_knee_y': 133., 'r_hip_y': -100., 'r_knee_y': 133.,
    'bust_x': 0., 'abs_z': 0., 'abs_y': 45., 'abs_x':0.,
    'r_elbow_y':-115., 'l_elbow_y':-115.,
    'r_shoulder_x':0., 'l_shoulder_x':0.,
    'r_ankle_y':45., 'l_ankle_y':45.,
    'r_hip_z':-10., 'l_hip_z':10.,
    'r_hip_x':0., 'l_hip_x':0.,})

# pybullet specific (not compliant):
crawl_angles.update({
    # 'r_shoulder_x': -90, 'l_shoulder_x': -90, # somehow wrong in scripts?
    'r_shoulder_y': -45, 'l_shoulder_y': -45,}) # compliant in scripts

angles = [crawl_angles]

angle_updates = [
    {'l_shoulder_x': -13., 'l_hip_z': 20., 'r_hip_z':-20., 'bust_x':-10.}, # lean left
    {'r_shoulder_x':12., 'abs_x': -3., 'abs_y': 37., 'abs_z': -15., 'r_elbow_y':-90.}, # rotate torso
    {'l_elbow_y':-90., 'abs_z': 0., 'l_shoulder_y': 0}, # left elbow perp
    {'l_shoulder_x':0., 'r_shoulder_x':0.},
    {'r_elbow_y': -115., 'l_elbow_y': -115.,'l_hip_y': -70., 'l_knee_y': 100.,'r_hip_y': -70., 'r_knee_y': 100., 'l_shoulder_y': -45}, # forward thighs vertical
    {'r_hip_z': 5., 'r_hip_y': -95., 'l_hip_y': -95., 'l_hip_z': 18., 'abs_z': 18., 'l_knee_y': 100., 'r_knee_y': 100.}, # hip raised
    {'r_hip_z': -2., 'r_hip_y': -100., 'l_hip_y': -70., 'l_hip_z': 2., 'abs_z': -15., 'l_knee_y': 100., 'r_knee_y': 125.}, # right leg forward
    {'r_hip_z': 0., 'r_hip_y': -100., 'l_hip_y': -100., 'l_hip_z': 0., 'abs_z': 0., 'l_knee_y': 125., 'r_knee_y': 125.}, # left leg forward
]

for changes in angle_updates:
    angles.append(dict(angles[-1]))
    angles[-1].update(changes)
    
waypoints = np.zeros((len(angles), N))
for a,angle in enumerate(angles):
    cleaned = convert_angles(angle)
    for m,p in cleaned.items():
        waypoints[a, env.joint_index[m]] = p

# initial angles/position
for p, position in enumerate(waypoints[0]):
    pb.resetJointState(env.robot_id, p, position)
pb.resetBasePositionAndOrientation(env.robot_id,
    (0, 0, .3),
    pb.getQuaternionFromEuler((.25*np.pi,0,0)))

# waypoints = np.array([
#     [1. if i == env.joint_index["r_shoulder_x"] else 0. for i in range(N)],
#     ])


input("ready...")
w = 0
while True:

    target = waypoints[w]
    duration = .25
    env.goto_position(target, duration, hang=False)
    # env.set_position(target)
    # input('.')
    w = (w + 1) % len(waypoints)

# input('.')
# action = env.current_position()
# j = env.joint_index["l_elbow_y"]
# while True:
#     env.step(action)
#     # print(action[j])
#     # print(env.current_position()[j])
#     # input('.')



