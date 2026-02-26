import sys, glob
import pickle as pk
import numpy as np
import poppy_wrapper as pw
from empirical_knead_forward import build_angle_dicts
from stitch_preanalysis import get_run_filepaths
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
    num_cycles = 3 # a cycle is a left then right step
    timestep = 0.25
    pausestep = 1.25
    left_params = (12, 2, 10, 4, 12, -7, 10, -7, 3, 0)
    right_params = (12, 2, 10, 4, 12, -3, 8, -5, 3, 0)
    l_hip_y_0 = -3
    noise_stdev = 0.0 # 0.125 # deg
    datapath = "empirical_knead_lqr/"
    num_interp = 2 # for LQR controller observation

    # load fitted_lqr.py results
    with open(f"lqr_ni{num_interp}.pkl","rb") as f: (Kontrollers, _) = pk.load(f)

    # retrieve nominal trajectory, same as when fitting A, B in LQR
    run_filepaths = get_run_filepaths()
    with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)

    # extract number of successful footsteps
    _, _, stdevs, nfsteps = zip(*run_filepaths)
    stdevs = np.array(stdevs)
    nfsteps = np.array(nfsteps)
    #print(f"{((stdevs=="0.0") & (nfsteps == 6)).sum()} noiseless success episodes")

    # flatten data for linear fits
    X = obs.reshape(len(run_filepaths), 31, -1)
    A = cmd.reshape(len(run_filepaths), 30, -1)

    # pool across footstep cycle
    X_pool = np.concatenate([
        X[:,0:11],
        X[:,10:21],
        X[:,20:31],
    ], axis=0)
    A_pool = np.concatenate([
        A[:,0:10],
        A[:,10:20],
        A[:,20:30],
    ], axis=0)
    stdevs_pool = np.concatenate([stdevs]*3)
    success_pool = np.concatenate([
        (nfsteps >= 2),
        (nfsteps >= 4),
        (nfsteps == 6),
    ])

    # get target trajectory for one cycle (average within noiseless success)
    X0 = X_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)

    # duplicate for multiple cycles
    Kontrollers = [Kontrollers[n] for n in sorted(Kontrollers.keys())]
    Kontrollers = num_cycles * Kontrollers

    # get current number of samples for counter
    counter = len(glob.glob(datapath + "traj_*.pkl"))
    print("starting counter from %d" % counter)

    # init robot
    poppy = pw.PoppyWrapper(PH(), OpenCVCamera("poppy-cam", 0, 24))
    
    # PID tuning
    K_p, K_i, K_d = 8.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

    # build nominal trajectory
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

    # collect data
    input("[Enter] to enable torques")
    poppy.enable_torques()

    while True:

        # build up full trajectory with number of steps requested
        full_traj = cycle_traj * num_cycles

        input("[Enter] to goto init (suspend with strap)")
        _ = poppy.track_trajectory([(1., init_angs)], overshoot=1.)
        
        input("[Enter] to walk (get ready with strap)")        
        buffers, elapsed, waypoint_timepoints, control_adjustments = poppy.track_trajectory_lqr(
            full_traj, num_interp, Kontrollers, X0, overshoot=1., binsize=5)

        num_success = input("Enter number of steps without falling: ")

        # save data and update counter
        with open(datapath + "traj_%02d_%s.pkl" % (counter, num_success), "wb") as f:
            pk.dump(full_traj, f)
        with open(datapath + "bufs_%02d_%s.pkl" % (counter, num_success), "wb") as f:
            pk.dump((buffers, elapsed, waypoint_timepoints, poppy.motor_names), f)
        with open(datapath + "ctrl_%02d_%s.pkl" % (counter, num_success), "wb") as f:
            pk.dump(control_adjustments, f)
        counter += 1
        print("%d episodes recorded." % counter)

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


