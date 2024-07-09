import time
import sys, os
import pickle as pk
import random
import numpy as np
import poppy_wrapper as pw
try:
    from pypot.creatures import PoppyHumanoid
    from pypot.sensor import OpenCVCamera
    import pypot.utils.pypot_time as time
except:
    from mocks import PoppyHumanoid
    from mocks import OpenCVCamera
    import time
from inflect import get_inflection_trajectories

DO_CUR = True # whether to explore current boundary level
DO_INC = False # whether to explore incremented boundary level
assert DO_CUR or DO_INC

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

# filename for results
fname = "search_inflect.pkl"

# modified results in place
def execute(poppy, inflect_angles, results, boundary, params):

    print(("executing boundary %d, params: " % boundary) +  str(params))

    # get planned trajectory
    timepoints, trajs = get_inflection_trajectories(inflect_angles, poppy.motor_names)
    planned = np.array([trajs[name] for name in poppy.motor_names]).T

    # convert to poppy dict
    trajectory = []
    dur = timepoints[1] - timepoints[0]
    for t in range(len(timepoints)):
        angles = {joint_name: float(traj[t]) for (joint_name, traj) in trajs.items()}
        trajectory.append( (dur, angles) )

    # goto init position
    _, init_angles = trajectory[0]
    input("[Enter] to goto init (may want to suspend first)")
    _ = poppy.track_trajectory([(1., init_angles)], overshoot=1.)
    input("[Enter] to proceed")        

    # run on poppy
    buffers, time_elapsed = poppy.track_trajectory(trajectory, overshoot=1.)

    # print deviation in final angles
    deviation = np.mean(np.fabs(buffers["position"][-1] - planned[-1]))
    print("|final planned - actual| ~ %f" % deviation)

    # specifically for boundary thigh
    idx = poppy.motor_index["r_hip_y"]
    planned_thigh = planned[-1,idx]
    actual_thigh = buffers["position"][-1][idx]
    deviation = np.fabs(planned_thigh - actual_thigh)
    print("|final planned - actual thigh| = |%f - %f| = %f" % (planned_thigh, actual_thigh, deviation))

    # get user input
    print("Did poppy fall and how do you want to proceed?")
    while True:
        try:
            sf, cq = input("Enter [s|f][c|q] for [s]tand|[f]all and [c]ontinue|[q]uit: ")
            success = (sf == "s")
            break
        except:
            print("input error")
            pass

    # save results
    actual = np.array(buffers["position"])
    downsample = int(float(len(actual)) / len(planned)) # to reduce disk space
    results[boundary][params] = (success, actual[::downsample], time_elapsed[::downsample], planned, timepoints)
    with open(fname, "wb") as f: pk.dump((results, poppy.motor_names), f)

    # follow user command
    if cq == "q":
        input("[Enter] to go to zero and then compliant (hold strap first)")
        zero_angles = {name: 0. for name in init_angles}
        _ = poppy.track_trajectory([(1., zero_angles)], overshoot=1.)
        poppy.disable_torques()

    return success, cq

if __name__ == "__main__":

    # set up param names and ranges in degrees (excluding boundary thigh, set separately)
    param_names, param_lows, param_highs = zip(*[
        ("thigh_ext",   0., 6.),
        ("stance_knee", 0., 6.),
        ("swing_knee",  0., 10.),
        ("ankle_ext",   0., 6.),
        ("ankle_flex", -6., 0.),
        ("lean",        6., 10.),
        ("sway",        0., 10.),
    ])
    
    # param units (deg/level) * param level (int) = param value (deg)
    param_units = 2.0
    
    # results[boundary][params] = (success, position buffer, time elapsed, planned, timepoints)
    # success: whether these params succeeded (didn't fall)
    # buffer, time elapsed: as returned by track_trajectory
    
    # load the previous results
    if not os.path.exists(fname):
        results = {0: {}}
    else:
        with open(fname, "rb") as f: results, _ = pk.load(f) # second element is motor names
    
    # make sure requested boundary is valid
    boundary = int(sys.argv[1])
    if boundary not in results:
        print("Largest boundary is %d, not up to %d yet" % (max(results.keys()), boundary))
        assert False
    if boundary + 1 not in results: results[boundary+1] = {}
    
    # set up successful param sets
    good_params = []
    for params, data in results[boundary].items():
        success = data[0]
        if success: good_params.append(params)
    
    # start with zero param set if none searched yet
    if len(good_params) == 0:
        if boundary != 0:
            print("No good params, need more runs at boundary %d" % (boundary-1))
            assert False
        # initial lean range at least 2 deg
        good_params = [(0,0,0,0,0,int(round(float(param_lows[5])/param_units)),0)]

    # initialize hardware
    poppy = pw.PoppyWrapper(
        PoppyHumanoid(),
        # OpenCVCamera("poppy-cam", 0, 10),
    )
    
    # PID tuning
    K_p, K_i, K_d = 8.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)
    
    input("[Enter] to enable torques")
    poppy.enable_torques()

    # start sampling and executing
    while True:
    
        # sample a successful param set found so far (initially just all params=0)
        print(good_params)
        print("^^ %d good params so far, %d samples total" % (len(good_params), len(results[boundary])))
        print(param_names)
        params = random.choice(good_params)
    
        # sample one of its neighbors N with same boundary thigh angle
        p = random.randrange(len(params)) # which parameter to change
        options = [params[p]-1, params[p]+1] # what it can be changed to
    
        # don't change outside (low, high) range
        for option in list(options):
            if option * param_units < param_lows[p]: options.remove(option)
            if option * param_units > param_highs[p]: options.remove(option)
        v = random.choice(options)
    
        # construct neighbor
        neighbor = params[:p] + (v,) + params[p+1:]
        
        # skip neighbors already executed/recorded
        if neighbor in results[boundary]: continue
    
        # set up inflection input
        current_angles = {}
        for (name, level) in zip(param_names, neighbor):
            current_angles[name] = float(level * param_units)

        if DO_CUR:
    
            # execute neighbor at current boundary thigh angle
            current_angles["boundary_thigh"] = float(boundary * param_units)
            success, selection = execute(poppy, current_angles, results, boundary, neighbor)
    
            # save good params
            if success: good_params.append(neighbor)
    
            # quit if requested
            if selection == "q": break
    
            # skip increment if failed
            if not success: continue

        if DO_INC:

            # execute neighbor at incremented boundary thigh angle
            next_angles = dict(current_angles)
            next_angles["boundary_thigh"] = float((boundary+1) * param_units)
            success, selection = execute(poppy, next_angles, results, boundary+1, neighbor)
        
            # quit if requested
            if selection == "q": break

    # revert PID
    K_p, K_i, K_d = 4.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

    # close connection
    print("Closing poppy...")
    poppy.close()
    print("closed.")

#     solve one degree increment (.25) in boundary thigh angle before incrementing further
#     use monte-carlo sampling, all 80 at every node is too much [although should 7 parameter case be handled differently?]
#     inner loop (one boundary thigh angle increment):
#         sample a successful param set found so far (initially just all params=0)
#         sample one of its neighbors N with same boundary thigh angle
#         if N obeys inequalities and hasn't been executed/recorded yet:
#             execute N as well as N with boundary thigh angle incremented (N')
#             record success/stability of N and N'
#             If N' successful: you're done (or keep looping until timeout to search for more viable N')
#     outer loop (incrementing boundary thigh angle):
#         use N' (one or multiple) from previous iteration to initialize successful param sets found so far
#         if boundary thigh angle = large target:
#             you're done
#             from successful param sets at each boundary thigh angle, extract most stable one
#     properties: this algorithm explores a **connected component** of success

# implementation:
#     after each N check, prompt in case of fall for manual hardware reset, but if no fall it can proceed without reseting
#     save results after each N check and allow easy restarting where left off
#         this is already quite easy since you are just sampling; load the previously checked records and keep sampling
#         maybe one record file per boundary thigh angle


