import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv

with open("../../../scripts/stand.pkl", "rb") as f: stand = pk.load(f)

def fitness(waypoints, durations):

    env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=False)
    env.set_position(stand)

    for w in range(len(waypoints)):
        target = waypoints[w]
        duration = durations[w]
        env.goto_position(target, duration)

    env.goto_position(stand, .1)
    env.goto_position(stand, 1)

    cart, quat = pb.getBasePositionAndOrientation(env.robot_id)
    
    # rewards:
    # same base height, orientation, and angles at end
    # base moved forward (fixed total duration, target forward motion)
    # balanced at end (base height, orientation, angles stay same for a second)
    # minimum path length in joint space (or numerical velocity/jerk)


if __name__ == "__main__":

    env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)
    
    for _ in range(2):
        input('.')
        env.reset()
    
        pos = env.current_position()
        tar = pos + .5
        print(pos)
        print(tar)
        base, _ = pb.getBasePositionAndOrientation(env.robot_id)
        print(base)
        
        env.goto_position(tar, 1)
        
        pos = env.current_position()
        print(pos)
        base, _ = pb.getBasePositionAndOrientation(env.robot_id)
        print(base)
        
        
        
