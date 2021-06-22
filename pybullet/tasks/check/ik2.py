import sys
sys.path.append('../../envs')

import pybullet as pb
from ergo_jr import PoppyErgoJrEnv

env = PoppyErgoJrEnv(pb.POSITION_CONTROL)

# from check/camera.py
pb.resetDebugVisualizerCamera(
    1.2000000476837158, 56.799964904785156, -22.20000648498535,
    (-0.6051651835441589, 0.26229506731033325, -0.24448847770690918))

# fixed tip is joint 5, moving tip is joint 7
link_indices = [5, 7]
target_positions = [(.005, -.08, 0), (-.005, -.08, 0)]

angles = env.inverse_kinematics(link_indices, target_positions)

env.set_position(angles)

ans = pb.getLinkStates(env.robot_id, [5, 7])
print(len(ans))
for a in ans: print(a)

input('.')

while True: env.step(angles)

