from math import sin, cos
import itertools as it
import sys
sys.path.append('../../envs')

import pybullet as pb
from ergo_jr import PoppyErgoJrEnv

env = PoppyErgoJrEnv(pb.POSITION_CONTROL)

# from pickup.py
pb.resetDebugVisualizerCamera(
    1.2000000476837158, 56.799964904785156, -22.20000648498535,
    (-0.6051651835441589, 0.26229506731033325, -0.24448847770690918))
action = [0.23849625, 0.29942884, 0.2155047, 0.17707248, 0.35643884, 0., 0.13668322, 0.]

env.goto_position(action, 1)
# for t in range(240):
#     env.step(action)
print(env.get_tip_positions())
input('.')

env.set_position(action)
print(env.get_tip_positions())
input('.')

while True:
    env.step(action)
