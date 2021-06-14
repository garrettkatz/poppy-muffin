import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from blocks_world import BlocksWorldEnv, random_thing_below

thing_below = random_thing_below(num_blocks=7, max_levels=3)
env = BlocksWorldEnv(pb.POSITION_CONTROL, show=False, control_period=12)
env.load_blocks(thing_below)
rgba, view, proj, coords_of = env.get_camera_image()
env.close()

np.save("tmp.npy", rgba)
rgba = np.load("tmp.npy")

import matplotlib.pyplot as pt
pt.imshow(rgba)
pt.show()


