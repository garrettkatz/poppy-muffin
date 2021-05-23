import pickle as pk
import numpy as np
import sys, time
sys.path.append('../../envs')

import pybullet as pb
from humanoid import PoppyHumanoidEnv, convert_angles

env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)
N = len(env.joint_index)

# got from running camera.py
cam = (1.200002670288086,
    15.999960899353027,
    -31.799997329711914,
    (-0.010284600779414177, -0.012256712652742863, 0.14000000059604645))
pb.resetDebugVisualizerCamera(*cam)

with open("../../../scripts/stand.pkl", "rb") as f: stand_dict = pk.load(f)
stand_dict.update({'r_shoulder_x': -20, 'l_shoulder_x': 20})
stand = env.angle_array(stand_dict)

# started with ../check/strides.py
lift_dict = dict(stand_dict)
lift_dict.update({
    'l_ankle_y': -25, 'l_knee_y': 60, 'l_hip_y': -30,
    'r_ankle_y': -10, 'r_hip_y': 10,
    'r_shoulder_y': -45, 'r_shoulder_x': -45,
    'abs_x': 10,
    })
lift = env.angle_array(lift_dict)
step_dict = dict(stand_dict)
step_dict.update({
    'r_shoulder_y': -40, 'r_shoulder_x': -30,
    'l_ankle_y': 15, 'l_knee_y': 0, 'l_hip_y': -15,
    'r_ankle_y': -15, 'r_knee_y': 0, 'r_hip_y': 15})
step = env.angle_array(step_dict)
push_dict = dict(stand_dict)
push_dict.update({
    'l_ankle_y': 0, 'l_knee_y': 0, 'l_hip_y': 0,
    'r_ankle_y': -25, 'r_knee_y': 60, 'r_hip_y': -30})
push = env.angle_array(push_dict)
settle_dict = dict(stand_dict)
settle_dict.update({
    'l_ankle_y': 10, 'l_knee_y': 0, 'l_hip_y': -10,
    'r_ankle_y': -20, 'r_knee_y': 40, 'r_hip_y': -20})
settle = env.angle_array(settle_dict)

def make_delta_policy(angle_delta, duration_delta):
    d_lift = lift + 0.02 * angle_delta[0]
    d_step = step + 0.02 * angle_delta[1]
    d_push = push + 0.02 * angle_delta[2]
    d_sett = settle + 0.02 * angle_delta[3]        
    d_durs = np.array([.1, .05, .2, .15, .05, .1]) + 0.005 * duration_delta

    def policy(num_steps):
    
        # [..., (angles, duration), ...]    
        # start walking
        trajectory = [
            (d_lift, d_durs[0]),
            (d_step, d_durs[1]),
        ]
        # continue walking
        for t in range(num_steps-1):
            if t % 2 == 0:
                trajectory += [
                    (d_push, d_durs[2]),
                    (env.mirror_position(d_step), d_durs[3])]
            else:
                trajectory += [
                    (env.mirror_position(d_push), d_durs[2]),
                    (d_step, d_durs[3])]
        
        # stop walking
        if (num_steps-1) % 2 == 0:
            trajectory += [(d_sett, d_durs[4])]
        else:
            trajectory += [(env.mirror_position(d_sett), d_durs[4])]
        trajectory += [(stand, d_durs[5])]
        
        return trajectory
    
    return policy

hand_coded_policy = make_delta_policy([0]*4, 0)

def run_policy(policy, num_steps):

    trajectory = policy(num_steps)
    
    # initial angles/position
    env.set_position(stand)
    env.goto_position(stand, 1)
    
    input("ready...")
    time.sleep(1)
    t = 0
    while True:
    
        target, duration = trajectory[t]
        env.goto_position(target, duration)
        
        t += 1
        if t == len(trajectory): t -= 1
    
        # input('...')


if __name__ == "__main__":

    # run_policy(policy, num_steps=5)
    
    from fitness import fitness

    # # result = fitness(env, hand_coded_policy)

    # angle_delta = np.random.randint(-5, 5, size=(4,env.num_joints))
    # duration_delta = np.random.randint(-5, 5, size=(6,))
    # result = fitness(env, make_delta_policy(angle_delta, duration_delta))
    
    # for num_steps in sorted(result.keys()):
    #     print("num steps", num_steps)
    #     print("distance: ", result[num_steps]["distance"])
    #     print("periodic: ", result[num_steps]["periodic"])


    # performance is  average distance + average periodic (over all steps)
    def measure_cost(angle_delta, duration_delta):
        result = fitness(env, make_delta_policy(angle_delta, duration_delta))
        distance = np.mean([res["distance"] for res in result.values()])
        periodic = np.mean([res["periodic"] for res in result.values()])
        return distance + periodic

    with open("optim.pkl", "rb") as f: deltas, costs = pk.load(f)    
    print("**** best ****")

    idx = np.argsort(costs)
    # angle_deltas, duration_deltas = deltas[idx[0]] # global best
    # angle_deltas, duration_deltas = deltas[idx[2]] # pareto optimal 
    angle_deltas, duration_deltas = deltas[idx[6]] # pareto optimal 

    # cost = measure_cost(angle_deltas, duration_deltas)
    # print(cost)

    policy = make_delta_policy(angle_deltas, duration_deltas)
    run_policy(policy, num_steps=9)
    
    
        


