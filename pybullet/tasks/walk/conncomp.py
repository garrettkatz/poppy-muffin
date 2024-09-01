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

def run_traj(env, step_array, num_cycles):
    # step_array[t]: angle array at timestep t (1-based) for a single footstep.
    # it is mirrored for the second foot of the cycle
    # the cycle is repeated the requested number of times
    # step_array[-1] is mirrored for the initial pose before the trajectory

    init_dict, cycle_traj = traj_from_array(step_array, num_cycles)

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

    # get state after settle
    post_base, post_quat = pb.getBasePositionAndOrientation(env.robot_id)
    post_linv, post_angv = pb.getBaseVelocity(env.robot_id)

    # return results
    return positions, init_base, init_quat, post_base, post_quat, post_linv, post_angv

def is_fall(init_base, post_base, relative_tolerance = .05):
    return (np.fabs(init_base[2] - post_base[2]) / init_base[2]) > relative_tolerance

# all objectives are minimizations

def compute_jerk(positions):
    # minimum path length in joint space (or numerical velocity/jerk)
    # modulo constant scale factor (mean instead of sum, timestep not included)
    dp = (positions[1:] - positions[:-1])
    dv = (dp[1:] - dp[:-1])
    da = (dv[1:] - dv[:-1])
    return (da**2).mean()

def forward_motion(init_base, post_base):
    # returns change in base y-coordinate
    # "forward" direction is negative y, lower is better
    return post_base[1] - init_base[1]

def lateral_motion(init_base, post_base):
    return np.fabs(init_base[0] - post_base[0])

def base_spin(init_quat, post_quat):
    diff_quat = pb.getDifferenceQuaternion(init_quat, post_quat)
    axis, ang = pb.getAxisAngleFromQuaternion(diff_quat)
    return ang

def ending_velocity(post_linv, post_angv):
    return np.sum(np.array(post_linv + post_angv)**2)

def get_objectives(result, relative_tolerance = .05):
    # returns is_fall, other objectives
    positions, init_base, init_quat, post_base, post_quat, post_linv, post_angv = result
    return is_fall(init_base, post_base, relative_tolerance), (
        forward_motion(init_base, post_base),
        lateral_motion(init_base, post_base),
        base_spin(init_quat, post_quat),
        ending_velocity(post_linv, post_angv),
        compute_jerk(positions),
    )

if __name__ == "__main__":

    # setup paramters
    num_timesteps = int(np.ceil(duration / timestep))
    duration = num_timesteps * timestep
    num_cycles = 1

    relative_fall_tolerance = 0.05

    do_rrt = False
    show_rrt = False
    show_best = True
    max_samples = 50
    
    # ### test one random traj

    # env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)

    # rand_array = np.random.randint(-1, 2, (num_timesteps, env.num_joints))    
    # result = run_traj(env, rand_array, num_cycles = 1)
    # positions, init_base, init_quat, post_base, post_quat, post_linv, post_angv = result
    
    # print('init base vs post base')
    # print(init_base, post_base)
    # print('init quat vs post quat')
    # print(init_quat, post_quat)
    # print(f'jerk = {compute_jerk(positions)}')
    # print(f'fall = {is_fall(init_base, post_base)}')
    # print(f'forward = {forward_motion(init_base, post_base)}')
    # print(f'lateral = {lateral_motion(init_base, post_base)}')
    # print(f'spin = {base_spin(init_quat, post_quat)}')
    # print(f'vel = {ending_velocity(post_linv, post_angv)}')

    # input('.')
    # env.close()
    
    # # import matplotlib.pyplot as pt
    # # pt.plot(positions * 180. / np.pi)
    # # pt.show()

    ### RRT

    if do_rrt:

        env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=show_rrt)
    
        # allocate samples
        all_samples = np.empty((max_samples, num_timesteps, env.num_joints))
        all_falls = np.empty(max_samples, dtype=bool)
        all_objectives = np.empty((max_samples, 5))
    
        # root tree at zero
        all_samples[0,:,:] = 0
        num_samples = 1
        result = run_traj(env, all_samples[0], num_cycles)
        all_falls[0], all_objectives[0] = get_objectives(result, relative_fall_tolerance)
    
        while num_samples < max_samples:
            print(f"{num_samples} of {max_samples} samples...")
    
            # sample random configuration
            rand_array = np.random.uniform(-360, 360, (num_timesteps, env.num_joints))
        
            # find closest configuration so far
            dists = np.sum((all_samples[:num_samples] - rand_array)**2, axis=(1,2))
            closest = np.argmin(dists)
            print(f"closest = {closest}")

            # if closest was a fall, reject sample
            if all_falls[closest]:
                print("[fall]")
                continue
            
            # normalize update direction to the double-unit cube and round to nearest degree
            direction = rand_array - all_samples[closest]
            direction = direction / np.fabs(direction).max()
            direction = direction.round()
        
            # evaluate new sample
            all_samples[num_samples] = all_samples[closest] + direction
            result = run_traj(env, all_samples[num_samples], num_cycles)
            all_falls[num_samples], all_objectives[num_samples] = get_objectives(result, relative_fall_tolerance)
            num_samples += 1
        
            print('objs: progress, lateral, spin, velocity, jerk')
            print(f'fall = {all_falls[num_samples-1]}, objs = {all_objectives[num_samples-1]}')
        
            # input('.')
    
        np.savez("rrt.npz", samples=all_samples, falls=all_falls, objectives=all_objectives)
    
        env.close()

    npz = np.load("rrt.npz")
    all_samples = npz["samples"]
    all_falls = npz["falls"]
    all_objectives = npz["objectives"]

    # sort criteria lexicographically
    criteria = np.concatenate((all_falls[:,np.newaxis], all_objectives), axis=1)
    lex = np.lexsort(criteria.T[::-1])

    # show "best" result lexicographically
    if show_best:
        env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)

        cam = (1.200002670288086,
            15.999960899353027,
            -31.799997329711914,
            (-0.010284600779414177, -0.012256712652742863, 0.14000000059604645))
        pb.resetDebugVisualizerCamera(*cam)
        input('.')

        result = run_traj(env, all_samples[lex[0]], num_cycles)
        positions = result[0]
        input('.')

        env.close()

        import matplotlib.pyplot as pt
        pt.plot(positions * 180. / np.pi)
        pt.show()

    # print results
    print('falls, objs: progress, lateral, spin, velocity, jerk')
    print(criteria.round(4))
    print("sorted:")
    print(criteria[lex].round(4))

