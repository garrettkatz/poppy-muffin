import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv, convert_angles

env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
N = len(env.joint_index)

# actually as found via env.mirror_position, all _y joints should have sign +1
same = ['shoulder_y', 'elbow_y', 'hip_y', 'knee_y', 'ankle_y']

signs = {"l_": +1, "r_": -1}
rot = np.pi/8
angles = np.zeros(N)
for i, name in env.joint_name.items():
    angles[i] = rot
    if name[2:] not in same: angles[i] *= signs.get(name[:2], 0)

# initial angles/position
env.set_position(angles)
pb.resetBasePositionAndOrientation(env.robot_id,
    (0, 0, .41),
    pb.getQuaternionFromEuler((0,0,0)))

input("ready...")

angles[:] = rot
while True:
    env.set_position(angles)
    angles = env.mirror_position(angles)
    input("ready...")
    

