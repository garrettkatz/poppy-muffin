import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import pybullet as pb
import sys, os
sys.path.append('../../envs')
from humanoid import PoppyHumanoidEnv, convert_angles

np.set_printoptions(linewidth=1000)

if __name__ == "__main__":

    with open(os.environ["HOME"] + "/poindexter/knead.pkl", "rb") as f: waypoints = pk.load(f)
    
    env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)

    env.place_on_ground(waypoints[1])
    state = pb.getLinkState(env.robot_id, env.joint_index["l_ankle_y"])
    current_ankle_z = state[0][2]
    print("current ankle z", current_ankle_z)

    input('.')

    env.track_trajectory([(.15, env.angle_dict(waypoints[0]))])
    state = pb.getLinkState(env.robot_id, env.joint_index["l_ankle_y"])
    current_ankle_z = state[0][2]
    print("current ankle z", current_ankle_z)

    input('.')

    env.track_trajectory([(2, env.angle_dict(waypoints[0]))])
    state = pb.getLinkState(env.robot_id, env.joint_index["l_ankle_y"])
    current_ankle_z = state[0][2]
    print("current ankle z", current_ankle_z)

    input('.')

