import os, sys, glob
import pickle as pk
import numpy as np
import poppy_wrapper as pw
from empirical_knead_forward import build_angle_dicts
try:
    from pypot.creatures import PoppyHumanoid as PH
    from pypot.sensor import OpenCVCamera
    import pypot.utils.pypot_time as time
except:
    from mocks import PoppyHumanoid as PH
    from mocks import OpenCVCamera
    import time

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

if __name__ == "__main__":

    # parameters
    resume = True
    num_samples = 3 # number of samples for each trajectory
    num_cycles = 3 # a cycle is a left then right step
    timestep = 0.25
    pausestep = 1.25
    left_params = (12, 2, 10, 4, 12, -7, 10, -7, 3, 0)
    right_params = (12, 2, 10, 4, 12, -3, 8, -5, 3, 0)
    l_hip_y_0 = -3
    noise_stdev = 0.125 # deg
    datapath = "empirical_knead_search/"

    # init robot
    poppy = pw.PoppyWrapper(PH())
    
    # PID tuning
    K_p, K_i, K_d = 8.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

    ### build nominal trajectory
    foot_angles = {
        "left": build_angle_dicts(poppy, l_hip_y_0, *left_params),
        "right": build_angle_dicts(poppy, l_hip_y_0, *right_params),
    }

    # init stance
    init_angs = foot_angles["left"][0]

    # left foot
    left_traj = [(timestep, angle_dict) for angle_dict in foot_angles["left"][1:5]]
    left_traj.append( (pausestep, foot_angles["left"][4]) ) # pause to stabilize

    # right foot with mirroring and asymmetrical hip offset
    right_traj = []
    for angle_dict in foot_angles["right"][1:]:
        # don't overwrite original
        angle_dict = dict(angle_dict)
        # undo asymmetrical hip offset
        angle_dict["l_hip_y"] -= l_hip_y_0
        # mirror angles
        angle_dict = poppy.get_mirror_angles(angle_dict)
        # redo asymmetrical hip offset
        angle_dict["l_hip_y"] += l_hip_y_0
        # save new waypoint
        right_traj.append((timestep, angle_dict))

    right_traj.append((pausestep, angle_dict)) # pause to stabilize

    # concate trajectories for one cycle
    cycle_traj = left_traj + right_traj

    if resume:

        with open("%shistory.pkl" % datapath, "rb") as f: history = pk.load(f)
        all_trajs, all_step_counts = history       
    
    else:

        if not os.path.exists(datapath):
            os.system("mkdir %s" % datapath)
        os.system("rm -fr %s*" % datapath)
    
        all_trajs = [cycle_traj]
        all_step_counts = [[]]
        history = all_trajs, all_step_counts

    # collect data
    input("[Enter] to enable torques")
    poppy.enable_torques()

    while True:

        # if latest trajectory has missing samples, redo it
        if len(all_step_counts[-1]) < num_samples:
            perturbed_cycle = all_trajs[-1]

        # otherwise, try a new perturbation of best one so far
        else:

            # identify best trajectory so far
            all_step_count_avgs = np.array([np.mean(sc) for sc in all_step_counts])
            all_step_count_stds = np.array([np.std(sc) for sc in all_step_counts])
            best_traj_idx = np.argmax(all_step_count_avgs - all_step_count_stds)
            best_traj = all_trajs[best_traj_idx]

            # build up perturbed trajectory with number of steps requested
            # copy before perturbations
            perturbed_cycle = [(dur, dict(ang)) for (dur, ang) in best_traj]
            # perturb all but final pose between cycles
            for (dur, ang) in perturbed_cycle[:-1]:
                for name in ang:
                    ang[name] = ang[name] + np.random.randn()*noise_stdev

            # add new trajectory to search history
            all_trajs.append(perturbed_cycle)
            all_step_counts.append([])

        # repeat multiple cycles
        multi_traj = []
        for cycle in range(num_cycles): multi_traj += perturbed_cycle

        # run the current trajectory
        print("%d trajectories, starting sample %d of %d" % (
            len(all_step_counts), len(all_step_counts[-1]), num_samples
        ))

        input("[Enter] to goto init (suspend with strap)")
        _ = poppy.track_trajectory([(1., init_angs)], overshoot=1.)
        
        input("[Enter] to walk (get ready with strap)")        
        buffers, elapsed, waypoint_timepoints = poppy.track_trajectory(multi_traj, overshoot=1., binsize=None)

        num_success = int(input("Enter number of steps without falling: "))

        # update and save search history
        all_step_counts[-1].append(num_success)
        history = all_trajs, all_step_counts
        with open("%shistory.pkl" % datapath, "wb") as f: pk.dump(history, f)

        # display progress
        all_step_count_avgs = np.array([np.mean(sc) for sc in all_step_counts])
        all_step_count_stds = np.array([np.std(sc) for sc in all_step_counts])
        best_idx = (all_step_count_avgs - all_step_count_stds).argmax()
        print("%d trajs, best is %.3f +/- %.3f, nominal is %.3f +/- %.3f" % (
            len(all_trajs), 
            all_step_count_avgs[best_idx],
            all_step_count_stds[best_idx],
            all_step_count_avgs[0],
            all_step_count_stds[0],
        ))

        # get next user command
        cmd = input("[n]ext cycle, [c]ompliant rest, [] will abort: ")
        if cmd == "c":

            input("[Enter] to go compliant (hold strap first)")
            poppy.disable_torques()
            input("[Enter] to enable torques")
            poppy.enable_torques()

        if cmd == "": break

    input("[Enter] to go compliant (hold strap first)")
    poppy.disable_torques()
        
    # reset PID
    K_p, K_i, K_d = 4.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)
    
    # print("don't forget poppy.close()")
    print("closing poppy...")
    poppy.close()
    print("closed.")




