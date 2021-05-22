import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv, convert_angles

env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
N = len(env.joint_index)

szero = {env.joint_name[i]: 0 for i in range(N)}
with open("../../../scripts/stand.pkl", "rb") as f: stand = pk.load(f)
with open("../../../scripts/preleft.pkl", "rb") as f: preleft = pk.load(f)
with open("../../../scripts/leftup.pkl", "rb") as f: leftup = pk.load(f)
with open("../../../scripts/leftswing.pkl", "rb") as f: leftswing = pk.load(f)
with open("../../../scripts/leftstep.pkl", "rb") as f: leftstep = pk.load(f)
with open("../../../scripts/rightshift.pkl", "rb") as f: rightshift = pk.load(f)

# rightshift = dict(leftstep)
# # rightshift['bust_x'] = -leftswing['bust_x']
# rightshift['bust_x'] = 0
# rightshift['abs_x'] = -leftswing['abs_x']
# rightshift['abs_y'] = leftswing['abs_y'] + 30

# rightswing = dict(leftswing)
# rightswing['bust_x'] *= -1
# rightswing['abs_x'] *= -1
# for name in ['hip_x', 'hip_z', 'ankle_y']:
#     rightswing['l_'+name], rightswing['r_'+name] = -leftswing['r_'+name], -leftswing['l_'+name]
# for name in ['hip_y','knee_y']:
#     rightswing['l_'+name], rightswing['r_'+name] = leftswing['r_'+name], leftswing['l_'+name]

# angles = [leftswing, rightswing]

angles = [szero, preleft, leftup, leftswing, leftstep, rightshift, stand, stand, szero, szero]
durations = [
    .1, # to zeros
    .6, # to preleft
    1, # to left up
    1, # to left swing
    1, # to left step
    .25, # to right shift
    .5, # to stand
    .5, # stay stand
    .5, # to zero
    5, # stay zero
]
    
waypoints = np.zeros((len(angles), N))
for a,angle in enumerate(angles):
    cleaned = convert_angles(angle)
    for m,p in cleaned.items():
        waypoints[a, env.joint_index[m]] = p

# initial angles/position
env.set_position(waypoints[0])

input("ready...")
# waypoints = waypoints[:-1]
w = 0
while True:

    target = waypoints[w]
    duration = durations[w]
    env.goto_position(target, duration)

    # env.set_position(target)
    
    if w == len(waypoints)-1: durations[1] = .475
    if w == len(waypoints)-1: durations[2] = .8
    if w == len(waypoints)-1: durations[3] = .8
    
    # if w == len(waypoints) - 2: break    
    w = (w + 1) % len(waypoints)
    # w = min(w + 1, len(waypoints)-1)

    # input('...')
    


