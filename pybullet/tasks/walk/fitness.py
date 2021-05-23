import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb

with open("../../../scripts/stand.pkl", "rb") as f: stand_dict = pk.load(f)
stand_dict.update({'r_shoulder_x': -20, 'l_shoulder_x': 20})

def fitness(env, policy):

    stand = env.angle_array(stand_dict)

    result = {}
    for num_steps in range(1,8):
    # for num_steps in range(3,6):
        print("  num_steps = %d..." % num_steps)

        env.reset()
        env.set_position(stand)
        env.goto_position(stand, 1)
        init_base, init_quat = pb.getBasePositionAndOrientation(env.robot_id)

        logs = []
        trajectory = policy(num_steps)
        for target, duration in trajectory:
            log = env.goto_position(target, duration)
            logs.append(log)
        # should be stable and near stand position at end
        log = env.goto_position(stand, 4)
        logs.append(log)

        log = np.concatenate(logs, axis=0)
        base, quat = pb.getBasePositionAndOrientation(env.robot_id)

        result[num_steps] = {}
        
        # target forward distance
        target_step_distance = -.1
        target_distance = num_steps * target_step_distance
        actual_distance = base[1] - init_base[1]
        result[num_steps]["distance"] = abs(actual_distance - target_distance)
        
        # graceful stop: same base position and orientation (except for forward motion)
        quat_loss = np.sum((np.array(quat) - np.array(init_quat))**2)
        base_loss = (base[0]-init_base[0])**2 + (base[2]-init_base[2])**2
        result[num_steps]["periodic"] = quat_loss + base_loss

        # minimum path length in joint space (or numerical velocity/jerk)
        dp = (log[1:] - log[:-1]) / env.timestep
        dv = (dp[1:] - dp[:-1]) / env.timestep
        da = (dv[1:] - dv[:-1]) / env.timestep
        result[num_steps]["jerk"] = (da**2).sum() * env.timestep
        result[num_steps]["log"] = log
        
    return result

if __name__ == "__main__":

    from humanoid import PoppyHumanoidEnv
    
    env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=False)
    stand = env.angle_array(stand_dict)
    
    def policy(num_steps):
        target = stand + np.ones(env.num_joints) * .01
        weights = np.linspace(0, 1, num_steps).reshape(-1,1)
        waypoints = weights * target + (1 - weights) * stand
        durations = [.5] * num_steps        
        return waypoints, durations

    result = fitness(env, policy)
    # print(result)
    
    import matplotlib.pyplot as pt
    log = result[2]["log"]
    log = (log[1:]-log[:-1]) / env.timestep
    log = (log[1:]-log[:-1]) / env.timestep
    log = (log[1:]-log[:-1]) / env.timestep
    sq = (log**2).sum(axis=1)
    print(sq)
    print(sq.sum())
    pt.plot(log)
    pt.plot(sq)
    pt.show()

