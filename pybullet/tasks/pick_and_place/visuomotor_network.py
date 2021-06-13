import numpy as np
import torch as tr
from restack import DataDump, Restacker

class VisuoMotorNetwork(tr.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        position, rgb, coords_of = inputs
        x = tr.relu(self.conv1(rgb))
        x = tr.relu(self.conv2(x))

if __name__ == "__main__":
    
    import pybullet as pb
    import sys

    sys.path.append('../../envs')    
    from blocks_world import BlocksWorldEnv

    # generate pickup data
    num_blocks = 3
    thing_below = {"b%d"%b: "t%d"%b for b in range(num_blocks)}
    
    dump = DataDump()
    
    env = BlocksWorldEnv(pb.POSITION_CONTROL, control_period=20, show=True, step_hook = dump.step_hook)
    env.load_blocks(thing_below)

    goal_block_above = env.invert(thing_below)
    restacker = Restacker(env, goal_block_above, dump)
    restacker.pick_up("b1")
    env.close()

    _, (block,) = dump.data[0]["command"]
    position, action, rgb, coords_of = zip(*dump.data[0]["records"])
    
    # position = tr.stack(tuple(map(tr.tensor, position)))
    # action = tr.stack(tuple(map(tr.tensor, action)))
    # rgb = tr.stack(tuple(map(tr.tensor, rgb)))
    # coords_of = tr.stack(tuple(map(tr.tensor, [co[block] for co in coords_of])))

    position = tr.tensor(np.stack(position))
    action = tr.tensor(np.stack(action))
    rgb = tr.tensor(np.stack(rgb))
    coords_of = tr.tensor(np.stack([co[block] for co in coords_of]))

    inputs = (position[:-1], rgb[:-1], coords_of[:-1])
    outputs = (action[1:], coords_of[1:])
    
    print(coords_of)
