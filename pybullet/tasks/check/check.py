import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv, clean_angles

env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
N = len(env.joint_index)

with open("../../../scripts/pb_check.pkl","rb") as f:
    angles = [pk.load(f)]

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
    (0, 0, .41),
    pb.getQuaternionFromEuler((0,0,0)))

input("ready...")

