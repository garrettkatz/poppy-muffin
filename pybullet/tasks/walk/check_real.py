"""
Check joint angles at edge of stability in real world
"""
import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv, convert_angles

env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)

# build the "textbook"
hei = .025
half_exts = (.1, .1, hei)
coll = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_exts)
visu = pb.createVisualShape(pb.GEOM_BOX, halfExtents=half_exts, rgbaColor=(0,0,1,1))
mass = 2
quat = pb.getQuaternionFromEuler((0,0,0))
textbook = pb.createMultiBody(mass, coll, visu, basePosition=(0,0,hei), baseOrientation=quat)

pos, ori = pb.getBasePositionAndOrientation(env.robot_id)
pb.resetBasePositionAndOrientation(env.robot_id, pos[:2] + (pos[2]+2*hei,), ori)

# got from running camera.py
cam = (1.200002670288086,
    15.999960899353027,
    -31.799997329711914,
    (-0.010284600779414177, -0.012256712652742863, 0.14000000059604645))
pb.resetDebugVisualizerCamera(*cam)



# start (l_hip_y is offset, -3 acts like 0 in real world)
# off = -3
off = 0.

start_dict = {'l_hip_y': off + 1., 'l_ankle_y': -1., 'r_hip_y': -1., 'r_ankle_y': 1.}
start = env.angle_array(start_dict)

# env.set_position(start)
# env.goto_position(start, 1)

# input('.')

r_ank_dict = dict(start_dict)
r_ank_dict.update({'abs_y': 9, 'r_shoulder_y': -9, 'l_shoulder_y': -9})
r_ank = env.angle_array(r_ank_dict)
# puts center of mass right above right ankle spoke in forward-backward direction

# env.set_position(r_ank)
# env.goto_position(r_ank, 1)

# input('.')

com_tri_dict = dict(r_ank_dict)
com_tri_dict.update({'abs_x': 26, 'r_shoulder_x': -26, 'bust_x':-26, 'l_shoulder_x':26})
com_tri = env.angle_array(com_tri_dict)
# puts CoM in the support polygon of right foot and left toe
# but very unstable; sometimes would tip right, sometimes would tip left
# eventually settled into balance

env.set_position(com_tri)
env.goto_position(com_tri, 1)

input('.')

