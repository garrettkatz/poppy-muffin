"""
Attempt to enumerate connected component of non-falls around zero gait
Use RRT: https://en.wikipedia.org/wiki/Rapidly_exploring_random_tree
"""
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import pybullet as pb
import sys
sys.path.append('../../envs')
from humanoid import PoppyHumanoidEnv, convert_angles

np.set_printoptions(linewidth=1000)

### parameters
timestep = .2
duration = 2.

num_timesteps = int(np.ceil(duration / timestep))
duration = num_timesteps * timestep
num_cycles = 1

relative_fall_tolerance = 0.05

do_rrt = False
show_rrt = False
pack_rrt = True # convert samples to trajectories and save
show_bests = 50 # number of best results to show, 0 for none
max_samples = 16000

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

def array_from_design(design):
    # design[0]: init pos
    # design[1:T-1]: finite difference acceleration
    init_pos = design[0]
    last_pos = env.mirror_position(init_pos)

    acc = design[1:]
    vel = acc.cumsum(axis=0) # initial velocity zero
    pos = init_pos + vel.cumsum(axis=0)

    return np.concatenate((pos, last_pos[np.newaxis]), axis=0)

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
        num_dominators = np.empty(max_samples, dtype=int)
    
        # root tree at zero
        all_samples[0,:,:] = 0
        num_samples = 1
        result = run_traj(env, all_samples[0], num_cycles)
        all_falls[0], all_objectives[0] = get_objectives(result, relative_fall_tolerance)
        num_dominators[0] = 0

        rejects = 0
        while num_samples < max_samples:
            print(f"{num_samples} of {max_samples} samples ({rejects} rejects)  ...")
    
            # sample random configuration
            # rand_array = np.random.uniform(-15, 15, (num_timesteps, env.num_joints)) # design = pos
            rand_array = np.random.uniform(-3, 3, (num_timesteps, env.num_joints)) # design = acc
        
            # find closest configuration so far
            dists = np.sum((all_samples[:num_samples] - rand_array)**2, axis=(1,2))
            closest = np.argmin(dists)
            print(f"closest = {closest}")

            # if closest was a fall, reject sample
            if all_falls[closest]:
                # print("[fall]")
                rejects += 1
                continue

            # also tend to reject samples with more dominators
            if np.random.rand() > 1 / (1 + num_dominators[closest]):
                # print(f"[reject {num_dominators[closest]} dominators]")
                rejects += 1
                continue

            # reset reject counter here once valid sample is obtained
            rejects = 0

            # normalize update direction to the double-unit cube and round to nearest degree
            direction = rand_array - all_samples[closest]
            direction = direction / np.fabs(direction).max()
            direction = direction.round()

            # smooth trajectories with only one joint change per time-step
            mx = np.fabs(direction + 0.01*np.random.randn(*direction.shape)).argmax(axis=1) # randomness breaks ties
            for t, i in enumerate(mx):
                direction[t,:i] = 0
                direction[t,i+1:] = 0

            # print(direction)
            # input('.')
        
            # evaluate new sample
            all_samples[num_samples] = all_samples[closest] + direction
            step_array = array_from_design(all_samples[num_samples])
            result = run_traj(env, step_array, num_cycles)
            # print(all_samples[num_samples])
            # pt.plot(result[0] * 180. / np.pi)
            # pt.show()
            all_falls[num_samples], all_objectives[num_samples] = get_objectives(result, relative_fall_tolerance)

            # count dominators among non-falls
            if not all_falls[num_samples]:
                stand_idx = np.flatnonzero(~all_falls[:num_samples])
                num_dominators[num_samples] = (all_objectives[stand_idx] < all_objectives[num_samples]).all(axis=1).sum()
                num_dominators[stand_idx] += (all_objectives[num_samples] < all_objectives[stand_idx]).all(axis=1)

            num_samples += 1

            criteria = np.concatenate((all_objectives[:num_samples], num_dominators[:num_samples,np.newaxis]), axis=1)
            criteria = criteria[~all_falls[:num_samples]]

            print('progress, lateral, spin, velocity, jerk, dominators')
            print(criteria.round(4))
            # print(f'fall = {all_falls[num_samples-1]}, objs = {all_objectives[num_samples-1]}')

            # input('.')
    
        np.savez("rrt.npz", samples=all_samples, falls=all_falls, objectives=all_objectives)
    
        env.close()

    if pack_rrt:

        env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=show_rrt)

        npz = np.load("rrt.npz")
        all_samples = npz["samples"]
        all_falls = npz["falls"]
        all_objectives = npz["objectives"]

        all_trajs = []
        for s, sample in enumerate(all_samples):
            print(f"{s} of {len(all_samples)}")
            step_array = array_from_design(sample)
            init_dict, cycle_traj = traj_from_array(step_array, num_cycles)
            traj_array = np.empty((1 + len(cycle_traj), env.num_joints))
            traj_array[0] = env.angle_array(init_dict)
            for t, (_, angle_dict) in enumerate(cycle_traj):
                traj_array[t+1] = env.angle_array(angle_dict)
            all_trajs.append(traj_array)

            pt.plot(traj_array.T)

        all_trajs = np.stack(all_trajs)

        joint_names = tuple(env.joint_name[i] for i in range(env.num_joints))
        np.savez("rrt_trajs.npz", trajs=all_trajs, falls=all_falls, objectives=all_objectives, joint_names=joint_names)

        env.close()

    npz = np.load("rrt.npz")
    all_samples = npz["samples"]
    all_falls = npz["falls"]
    all_objectives = npz["objectives"]

    # filter out the falls
    all_samples = all_samples[~all_falls]
    all_objectives = all_objectives[~all_falls]
    all_falls = all_falls[~all_falls]

    # filter out the dominated samples
    dominated = np.empty(len(all_samples), dtype=bool)
    for n in range(len(all_samples)):
        dominated[n] = (all_objectives < all_objectives[n]).all(axis=1).any()
    all_samples = all_samples[~dominated]
    all_objectives = all_objectives[~dominated]
    all_falls = all_falls[~dominated]

    # scale up jerk
    all_objectives[:,-1] *= .1 / all_objectives[:,-1].max()

    # # filter too jerky
    # smooth = (all_objectives[:,4] < .04)
    # all_samples = all_samples[smooth]
    # all_objectives = all_objectives[smooth]
    # all_falls = all_falls[smooth]

    # filter to acceptable objective values (based on histograms)
    filt = (all_objectives[:,1] < .005) & (all_objectives[:,2] < .01) & (all_objectives[:,3] < .002)
    all_samples = all_samples[filt]
    all_objectives = all_objectives[filt]
    all_falls = all_falls[filt]

    # sort by objectives lexicographically
    lex = np.lexsort(all_objectives.T[::-1])

    obj_names = ("progress", "lateral", "spin", "velocity", "jerk")
    fig, axs = pt.subplots(1, len(obj_names), figsize=(len(obj_names)*3, 3))
    for i, (name, ax) in enumerate(zip(obj_names, axs)):
        ax.hist(all_objectives[:,i])
        ax.set_xlabel(name)
    axs[0].set_ylabel("Frequency")
    pt.tight_layout()
    # pt.show()

    pt.figure()
    pt.plot(all_objectives[:,0], all_objectives[:,4], 'k.')
    pt.xlabel("progress")
    pt.ylabel("jerk")
    pt.show()

    # show "best" results lexicographically
    if show_bests > 0:

        env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)
        cam = (1.200002670288086,
            15.999960899353027,
            -31.799997329711914,
            (-0.010284600779414177, -0.012256712652742863, 0.14000000059604645))
        pb.resetDebugVisualizerCamera(*cam)

        print([env.joint_name[i] for i in range(len(env.joint_name))])
        input('...')

        for n in range(show_bests):
            print(f"design {n}: {all_objectives[lex[n]].round(4)}")

            input('.')    
            step_array = array_from_design(all_samples[lex[n]])
            result = run_traj(env, step_array, num_cycles)
            positions = result[0]
            input('.')
    
            pt.plot(positions * 180. / np.pi)
            pt.show()

        env.close()

    # print results
    print('progress, lateral, spin, velocity, jerk')
    # print(all_objectives.round(4))
    print("sorted:")
    print(all_objectives[lex].round(4))

