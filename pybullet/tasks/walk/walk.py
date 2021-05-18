import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv, clean_angles

env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
N = len(env.joint_index)

angles = [{m:0 for m in env.joint_name.values()}]
for fn in ["preleft", "leftup", "leftswing", "leftstep"]:
    with open("../../../scripts/%s.pkl" % fn,"rb") as f:
        angles.append(pk.load(f))
    
waypoints = np.zeros((len(angles), N))
for a,angle in enumerate(angles):
    # cleaned = clean_angles(angle)
    cleaned = {m: angle[m] * np.pi / 180 for m in angle}
    for m,p in cleaned.items():
        waypoints[a, env.joint_index[m]] = p

# initial angles/position
for p, position in enumerate(waypoints[0]):
    pb.resetJointState(env.robot_id, p, position)
pb.resetBasePositionAndOrientation(env.robot_id,
    (0, 0, .43),
    pb.getQuaternionFromEuler((0,0,0)))

input("ready...")
w = 0
while True:

    target = waypoints[w]
    duration = 1
    env.goto_position(target, duration)
    # w = (w + 1) % len(waypoints)
    w = min(w + 1, len(waypoints)-1)
    input('...')

