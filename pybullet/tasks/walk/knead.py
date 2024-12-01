import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import pybullet as pb
import sys
sys.path.append('../../envs')
from humanoid import PoppyHumanoidEnv, convert_angles

np.set_printoptions(linewidth=1000)

if __name__ == "__main__":
    
    env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)

    # get ankle height when foot on ground
    state = pb.getLinkState(env.robot_id, env.joint_index["l_ankle_y"])
    ground_ankle_z = state[0][2]

    # stride the legs
    init_dict = env.angle_dict(env.get_position())
    init_dict["l_hip_y"] = -10
    init_dict["r_hip_y"] = +10
    init_dict["l_ankle_y"] = +10
    init_dict["r_ankle_y"] = -10
    env.set_position(env.angle_array(init_dict))

    # get new ankle height
    state = pb.getLinkState(env.robot_id, env.joint_index["l_ankle_y"])
    delta_ankle_z = state[0][2] - ground_ankle_z

    # shift to floor
    init_base, init_quat = pb.getBasePositionAndOrientation(env.robot_id)
    init_base = init_base[:2] + (init_base[2] - delta_ankle_z + .003,) # drop from slightly higher to avoid ground collision
    pb.resetBasePositionAndOrientation(env.robot_id, init_base, init_quat)
    env.track_trajectory([(2, init_dict)])

    # lower base to bend knees
    env.inverse_kinematics

    input('.')

