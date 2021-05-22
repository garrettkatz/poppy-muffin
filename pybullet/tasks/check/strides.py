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

input("ready...")



