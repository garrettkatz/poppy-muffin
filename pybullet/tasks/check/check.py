import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv, convert_angles

env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
N = len(env.joint_index)

with open("../../../scripts/pb_check.pkl","rb") as f:
    angles = [pk.load(f)]

with open("../../../scripts/pb_check_legs.pkl","rb") as f:
    legs = pk.load(f)

angles[0].update(legs)

waypoints = np.zeros((len(angles), N))
for a,angle in enumerate(angles):
    cleaned = convert_angles(angle)
    for m,p in cleaned.items():
        waypoints[a, env.joint_index[m]] = p

# initial angles/position
env.set_position(waypoints[0])
pb.resetBasePositionAndOrientation(env.robot_id,
    (0, 0, .41),
    pb.getQuaternionFromEuler((0,0,0)))

input("ready...")

target = waypoints[0]
target[env.joint_index['l_shoulder_y']] = -250 * np.pi / 180
# target[env.joint_index['l_shoulder_y']] = -4.5
duration = .25
env.goto_position(target, duration, hang=False)
input("...")
env.set_position(target)

input("...")
