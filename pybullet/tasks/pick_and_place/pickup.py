import sys
sys.path.append('../../envs')

import pybullet as pb
from ergo_jr import PoppyErgoJrEnv

env = PoppyErgoJrEnv(pb.POSITION_CONTROL)

import os
fpath = os.path.dirname(os.path.abspath(__file__))
fpath += '/../../../urdfs/objects'
pb.setAdditionalSearchPath(fpath)
block_id = pb.loadURDF('tall_block.urdf',
    basePosition = (0, -.16, .1),
    baseOrientation = pb.getQuaternionFromEuler((0,0,0)),
    useFixedBase=False)

# from check/camera.py
pb.resetDebugVisualizerCamera(
    1.2000000476837158, 56.799964904785156, -22.20000648498535,
    (-0.6051651835441589, 0.26229506731033325, -0.24448847770690918))

action = [0.]*env.num_joints
action[env.joint_index['m6']] = .5
env.set_position(action)

action[env.joint_index['m6']] = -.4
env.goto_position(action, .75)

input('.')
action[env.joint_index['m5']] = -.5
# action[env.joint_index['m4']] = -.5
env.goto_position(action, 1, hang=False)

while True: env.step(action)
