"""
Attempt to enumerate connected component of non-falls around zero gait
"""
import pickle as pk
import numpy as np
import pybullet as pb
import sys
sys.path.append('../../envs')
from humanoid import PoppyHumanoidEnv, convert_angles

### parameters
timestep = .25
duration = 2.
show = True

### init

num_timesteps = int(np.ceil(duration / timestep + 1))
duration = num_timesteps * timestep

env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)

zero_gait = [
    {name: 0. for name in env.joint_name.values()}
    for _ in range(num_timesteps)]

frontier = [zero_gait]
explored = set()

while len(frontier) > 0:

    env.reset()

    stand = env.angle_array(stand_dict)

    env.set_position(stand)
    env.goto_position(stand, 1)
    init_base, init_quat = pb.getBasePositionAndOrientation(env.robot_id)


env.close()

