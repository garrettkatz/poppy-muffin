"""
Attempt to enumerate connected component of non-falls around zero gait
Use RRT: https://en.wikipedia.org/wiki/Rapidly_exploring_random_tree
"""
import pickle as pk
import numpy as np
import pybullet as pb
import sys
sys.path.append('../../envs')
from humanoid import PoppyHumanoidEnv, convert_angles

### parameters
timestep = .2
duration = 2.
show = True

def make_cycle_traj(step_traj, num_cycles):
    traj = []
    for cycle in range(num_cycles):
        traj.extend(step_traj)
        traj.extend([
            # (dur, env.angle_dict(env.mirror_position(env.angle_array(pos))))
            (dur, env.mirror_dict(pos))
            for (dur, pos) in step_traj])
    return traj

def traj_from_array(arr, num_cycles):
    # assumes arr angles are in degrees
    step_traj = [(timestep, env.angle_dict(pos, convert=False)) for pos in arr]
    cycle_traj = make_cycle_traj(step_traj, num_cycles)
    init_dict = env.mirror_dict(step_traj[-1][1])
    return init_dict, cycle_traj

def run_traj(env, init_dict, traj):

    # start fresh
    env.reset()

    # check ankle positions in initial position
    init_pos = env.angle_array(init_dict)
    env.set_position(init_pos)
    init_base, init_quat = pb.getBasePositionAndOrientation(env.robot_id)
    ankle_z = []
    for lr in "lr":
        state = pb.getLinkState(env.robot_id, env.joint_index[lr + "_ankle_y"])
        ankle_z.append(state[0][2])
    ankle_z = min(ankle_z)

    # for stable init poses with feet flat on the ground, ankle_z should be ~.01265
    # so adjust base accordingly and stable poses will settle
    dz = .01265 - ankle_z
    init_base = init_base[:2] + (init_base[2] + dz,)
    
    # drop from slightly higher to avoid ground collision
    drop_base = init_base[:2] + (init_base[2] + .003,)
    pb.resetBasePositionAndOrientation(env.robot_id, drop_base, init_quat)
    env.track_trajectory([(2, init_dict)])

    # try performing trajectory with another settle at end
    positions = env.track_trajectory(cycle_traj + [(2, init_dict)])
    post_base, post_quat = pb.getBasePositionAndOrientation(env.robot_id)

    # return results
    return positions, init_base, init_quat, post_base, post_quat

def compute_jerk(positions, timestep):
    # minimum path length in joint space (or numerical velocity/jerk)
    dp = (positions[1:] - positions[:-1]) / env.timestep
    dv = (dp[1:] - dp[:-1]) / env.timestep
    da = (dv[1:] - dv[:-1]) / env.timestep
    return (da**2).sum() * timestep


if __name__ == "__main__":

    num_timesteps = int(np.ceil(duration / timestep))
    duration = num_timesteps * timestep
    # input(f"{num_timesteps} timesteps...")
    
    env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)
    
    zero_array = np.zeros((num_timesteps, env.num_joints))
    # init_dict, cycle_traj = traj_from_array(zero_array, num_cycles = 1)
    
    rand_array = np.random.randint(-1, 2, (num_timesteps, env.num_joints))
    init_dict, cycle_traj = traj_from_array(rand_array, num_cycles = 1)
    
    result = run_traj(env, init_dict, cycle_traj)
    positions, init_base, init_quat, post_base, post_quat = result
    
    # print("zero cycle")
    # for (dur, angs) in zero_cycle_traj: print(dur, angs)
    # input("..")
    
    env.close()
    
    print('init base vs post base')
    print(init_base, post_base)
    print('init quat vs post quat')
    print(init_quat, post_quat)
    print(f'jerk = {compute_jerk(positions, timestep)}')
    
    import matplotlib.pyplot as pt
    pt.plot(positions * 180. / np.pi)
    pt.show()
    
