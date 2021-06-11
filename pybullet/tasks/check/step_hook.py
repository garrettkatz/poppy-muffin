import sys
sys.path.append('../../envs')
import matplotlib.pyplot as pt

import pybullet as pb
from blocks_world import BlocksWorldEnv

step_log = []
def step_hook(env, action):
    if action is None: return
    position = env.get_position()
    delta = action - position
    rgb, _, _, coords_of = env.get_camera_image()
    step_log.append((position, delta, rgb, coords_of))

env = BlocksWorldEnv(pb.POSITION_CONTROL, step_hook=step_hook)
env.load_blocks(
    {"b0": "t0", "b1": "t1", "b2": "t2"})

action = [0.]*env.num_joints
env.goto_position([0.5]*env.num_joints, 20/240)
env.close()

position, delta, rgb, coords_of = zip(*step_log)
pt.ion()
for t in range(len(step_log)):
    print(t)
    print(position[t])
    print(delta[t])
    
    x, y = zip(*coords_of[t].values())
    pt.imshow(rgb[t])
    pt.plot(x, y, 'ro')
    pt.show()
    pt.pause(0.01)
    pt.cla()
