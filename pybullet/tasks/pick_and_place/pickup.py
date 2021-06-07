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

num_blocks = len(colors)
initial_locations = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
initial_locations["b3"] = "b4"
initial_locations["b4"] = "b5"
initial_locations["b2"] = "b1"

def base_of(block, block_locations):
    support = block_locations[block]
    if support[0] == "t": return support, .01
    base, support_height = base_of(support, block_locations)
    return base, support_height + .0205

def tower_position(t, num_positions):
    radius = -.16
    theta = 3.14 * (t+1) / (num_positions+1)
    pos = (radius*cos(theta), radius*sin(theta), 0)
    quat = pb.getQuaternionFromEuler((0,0,theta))
    return pos, quat

block_config = {} # xyz position, xyzw quat, rgb color
for b in range(num_blocks):
    block = "b%d" % b
    base, height = base_of(block, initial_locations)
    base_num = int(base[1:])
    pos, quat = tower_position(base_num, num_blocks)
    pos = pos[:2] + (height,)
    block_config[block] = (pos, quat, colors[b])

block_id = {}
for blk, (pos, quat, rgb) in block_config.items():
    b = pb.loadURDF('cube.urdf', basePosition = pos, baseOrientation = quat, useFixedBase=False)
    pb.changeVisualShape(b, linkIndex=-1, rgbaColor=rgb+(1,))
    block_id[blk] = b

env.step() # let blocks settle

# from check/camera.py
pb.resetDebugVisualizerCamera(
    1.2000000476837158, 56.799964904785156, -22.20000648498535,
    (-0.6051651835441589, 0.26229506731033325, -0.24448847770690918))

def get_tip_targets(p, q, d):
    m = pb.getMatrixFromQuaternion(q)
    t1 = p[0]-d*m[1], p[1]-d*m[4], p[2]-d*m[7]
    t2 = p[0]+d*m[1], p[1]+d*m[4], p[2]+d*m[7]
    return (t1, t2)

action = [0.]*env.num_joints
# env.set_position(action)
env.goto_position(action, 1)

def pick_up(block):
    blk_id = block_id[block]
    pos, quat = pb.getBasePositionAndOrientation(blk_id)
    stage = pos[:2] + (.1,)
    for way, delta in [(stage, .02), (pos, .02), (pos, .01), (stage, .01)]: 
        targs = get_tip_targets(way, quat, delta)
        action = env.inverse_kinematics([5, 7], targs)
        env.goto_position(action, 1)

def put_down_on(obj):
    if obj[0] == "t":
        t = int(obj[1:])
        pos, quat = tower_position(t, num_blocks)
    else:
        blk_id = block_id[obj]
        pos, quat = pb.getBasePositionAndOrientation(blk_id)
    pos = pos[:2] + (pos[2] + .03,)
    stage = pos[:2] + (.1,)    
    for way, delta in [(stage, .005), (pos, .005), (pos, .02), (stage, .02)]: 
        targs = get_tip_targets(way, quat, delta)
        action = env.inverse_kinematics([5, 7], targs)
        env.goto_position(action, 1)
        # input('.')

goal_locations = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
goal_locations["b3"] = "b2"
goal_locations["b2"] = "b1"

blocks = ["b%d" % b for b in range(num_blocks)]
spots = ["t%d" % t for t in range(num_blocks)]

# inverse of block_locations
towers = {key: "none" for key in spots + blocks}
for block, location in initial_locations.items():
    towers[location] = block

def free_spot(towers):
    for spot in spots:
        if towers[spot] == 'none': return spot
    return False

def unstack(location, towers):
    above = towers[location]
    if above == "none": return True
    unstack(above, towers)
    pick_up(above)
    towers[location] = 'none'
    spot = free_spot(towers)
    if spot == False: return False
    put_down_on(spot)
    towers[spot] = above

def unstack_all(towers):
    for spot in spots:
        base = towers[spot]
        if base != "none": unstack(base, towers)

# bw alg:
# for each tower:
#  for each block (top-to-bottom) except lowest:
#    move it to a free spot on the table
# for each goal tower:
#  for each block (bottom-to-top) except lowest:
#    move it to goal support

# pick_up("b3")
# put_down_on("b6")
# unstack(towers["t5"], towers)
unstack_all(towers)

action = env.get_position()
env.goto_position(action, 1)
while True: env.step()
