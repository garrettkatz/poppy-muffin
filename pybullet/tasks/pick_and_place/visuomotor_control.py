import sys, os
import pybullet as pb
import pickle as pk
import numpy as np
import torch as tr
from visuomotor_network import VisuoMotorNetwork, preprocess
sys.path.append('../../envs')    
from blocks_world import BlocksWorldEnv, random_thing_below

if __name__ == "__main__":

    # thing_below = random_thing_below(num_blocks, max_levels=3)
    # goal_thing_below = random_thing_below(num_blocks, max_levels=3)
    
    # num_blocks = 7
    # thing_below = {("b%d" % n): ("t%d" % n) for n in range(num_blocks)}
    # block, thing = "b3", "b4"

    with open("episodes/000/meta.pkl", "rb") as f:
        thing_below, _, _, _, commands = pk.load(f)
    
    _, (block, thing) = commands[0]
    env = BlocksWorldEnv(pb.POSITION_CONTROL, show=True, control_period=12)
    env.load_blocks(thing_below)

    _, _, _, coords_of = env.get_camera_image()
    block_coords = tr.tensor(np.stack([coords_of[block]])).float()
    thing_coords = tr.tensor(np.stack([coords_of[thing]])).float()
    
    # move to
    net = VisuoMotorNetwork()
    # net.load_state_dict(tr.load("net.pt"))
    net.load_state_dict(tr.load("net500.pt"))
    
    force_coords = False
    
    for t in range(100):
        position = env.get_position()
        if force_coords:
            rgba, _, _, coords_of = env.get_camera_image()
            block_coords = tr.tensor(np.stack([coords_of[block]])).float()
            thing_coords = tr.tensor(np.stack([coords_of[thing]])).float()
        else:
            rgba, _, _, _ = env.get_camera_image()
        
        position = tr.tensor(np.stack([position])).float()
        rgba = tr.tensor(np.stack([rgba]))
        if force_coords:
            rgb, block_coords, thing_coords = preprocess(rgba, block_coords, thing_coords)
        else:
            rgb, _, _ = preprocess(rgba, block_coords, thing_coords)

        inputs = position, rgb, block_coords, thing_coords
        outputs = net(inputs)
        action, block_coords, thing_coords = outputs
        print(block_coords)
        print(position)
        print(action)

        action = action.detach().numpy()[0]
        alpha = 1
        action = alpha*action + (1-alpha)*position.detach().numpy()[0]
        env.step(action, sleep=True)
        
        input('.')

