from math import sin, cos
import itertools as it
import sys
sys.path.append('../../envs')

import pybullet as pb
from ergo_jr import PoppyErgoJrEnv

env = PoppyErgoJrEnv(pb.POSITION_CONTROL)

import os
fpath = os.path.dirname(os.path.abspath(__file__))
fpath += '/../../../urdfs/objects'
pb.setAdditionalSearchPath(fpath)

# all 0/1 colors except for white/black
colors = list(it.product([0,1], repeat=3))[:-1]

block_config = [] # xyz position, rot angle, rgb color
for c in range(len(colors)):
    radius = -.16
    theta = 3.14 * (c+1) / (len(colors)+1)
    block_config.append( ((radius*cos(theta), radius*sin(theta), .01), theta, colors[c]) )

block_id = []
for pos, theta, rgb in block_config:
    bid = pb.loadURDF('cube.urdf',
        basePosition = pos,
        baseOrientation = pb.getQuaternionFromEuler((0,0,theta)),
        useFixedBase=False)
    pb.changeVisualShape(bid, linkIndex=-1, rgbaColor=rgb+(1,))
    block_id.append(bid)

# from check/camera.py
pb.resetDebugVisualizerCamera(
    1.2000000476837158, 56.799964904785156, -22.20000648498535,
    (-0.6051651835441589, 0.26229506731033325, -0.24448847770690918))

def get_tip_targets(p, q, d):
    m = pb.getMatrixFromQuaternion(q)
    return (
        (p[0]-d*m[1], p[1]-d*m[4], p[2]-d*m[7]),
        (p[0]+d*m[1], p[1]+d*m[4], p[2]+d*m[7]),
    )

action = [0.]*env.num_joints
env.set_position(action)
# input('.')

def pick_up(blk_id):
    pos, quat = pb.getBasePositionAndOrientation(blk_id)
    stage = pos[:2] + (.1,)    
    for way, delta in [(stage, .02), (pos, .02), (pos, .005), (stage, .005)]: 
        targs = get_tip_targets(way, quat, delta)
        action = env.inverse_kinematics([5, 7], targs)
        env.goto_position(action, 1)
        # input('.')

def put_down_on(blk_id):
    pos, quat = pb.getBasePositionAndOrientation(blk_id)
    pos = pos[:2] + (pos[2] + .03,)
    stage = pos[:2] + (.1,)    
    for way, delta in [(stage, .005), (pos, .005), (pos, .02), (stage, .02)]: 
        targs = get_tip_targets(way, quat, delta)
        action = env.inverse_kinematics([5, 7], targs)
        env.goto_position(action, 1)
        # input('.')

pick_up(block_id[6])
put_down_on(block_id[3])

action = env.get_position()
while True: env.goto_position(action, 1)
