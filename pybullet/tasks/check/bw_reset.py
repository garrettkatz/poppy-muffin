import sys
sys.path.append('../../envs')
import pybullet as pb
from blocks_world import BlocksWorldEnv, random_thing_below

env = BlocksWorldEnv()
thing_below = random_thing_below(num_blocks = 4, max_levels = 3)
env.load_blocks(thing_below)

input('.')

env.reset()

input('.')

thing_below = random_thing_below(num_blocks = 4, max_levels = 3)
env.load_blocks(thing_below)

input('.')
