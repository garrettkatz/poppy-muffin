import sys
import pickle as pk
import numpy as np
import poppy_wrapper as pw
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

def build_angle_dicts(poppy, l_hip_y_0, lean, forward, bend1, bend2, bend3, lift, tilt, stretch, sway, tweak):
    """
    poppy: PoppyWrapper instance
    l_hip_y_0: empirically found l_hip_y needs a correction offset
    lean: torso forward, whole time
    forward: additional hip rotation at init to achieve forward movement
    bend1: both legs bent at first
    bend2: lesser bend in left leg for pressure
    bend3: greater bend in left leg for lift
    lift: extra raise in left leg from hip so that toe doesn't land behind, should be negative
    tilt: r_hip_y rotation during bend2 to keep stance foot flat
    stretch: l_hip_y outward stretch during bend3 (but actually negative is better after adding tilt)
    sway: abs_x sway at end to reduce recoil when planting
    tweak: l_ankle_y slight toe raise at plant to avoid spin (will be negated)

    successful left swing: 12 10 4 15 0 10 -7 (12/28 3:08)
    successful right swing: 12 10 4 15 0 8 -7 (12/28 4:01)
    but in both cases, swing foot tends to land slightly behind its starting point
    The zeros are for new lift parameter to counteract this effect

    with lift:
    successful left swing: 12 10 4 12 -5 10 -7 (12/29 4:36)
    swing foot lands well, but there is some scary top-heavy wobbling
    new sway parameter to counteract this effect

    with sway and forward:
    left: 12 2 10 4 12 -7 10 -7 3
    right: 12 2 10 4 12 -3 8 -5 3
    can do >=12 steps with pausestep = 1.25 (1/6 6:08), but spins slightly left on left steps
    performance degraded a bit later into the evening.
    the larger left angles produce slightly more aggressive motions, maybe this is a problem.

    with tweak:
    Enter num_steps: 12
    Enter pausestep: 1.25
    left: 12 2 10 4 12 -7 10 -7 3 0
    right: 12 2 10 4 12 -3 8 -5 3 0
    final zeros are pre-tweak version, did not yet try left tweak slightly positive.
    successful and interesting result (1/8 3:32pm):
    12 steps with pausestep 1.25 works surpisingly well, but 1.5 does *not* work
    did not have to use any tweak, and it actually walked in a straight line (no spin).
    the crab-walk only happened with 1.5, maybe because of the frequency of slight sways in center of mass.
        waiting until 1.5 might let the COM sway back over the foot that is supposed to swing
        at this moment, straightening that swing leg makes the nominal stance leg lift off the ground
        then tilt/stretch makes the nominal stance leg, which is in fact lifted, land further to the side.
    this result was early in the session, just one 1.5 failed and then 1.25 worked.
    doesn't rule out performance degradation (spin, etc.) later in the session.
    next several attempted runs all aborted mid-way thru track traj due to OSError, must be loose wiring

    """

    zero_angles = {name: 0. for name in poppy.motor_names}
    zero_angles["l_hip_y"] = l_hip_y_0

    # initial knees bent
    bent1_angles = poppy.get_sway_angles(zero_angles, abs_x = 0., abs_y = lean)
    bent1_angles.update({
        "r_ankle_y": -(bend1 - forward), "l_ankle_y": -(bend1 + forward),
        "r_knee_y": 2*bend1, "l_knee_y": 2*bend1,
        "r_hip_y": -(bend1 + forward), "l_hip_y": -(bend1 - forward) + l_hip_y_0,
    })

    # straighten for pressure through swing foot
    # without changing forward angle, this should generate a forward rotation torque around stance ankle?
    bent2_angles = dict(bent1_angles)
    bent2_angles.update({
        "l_ankle_y": -(bend2 + forward),
        "l_knee_y": 2*bend2,
        "l_hip_y": -(bend2 - forward) + l_hip_y_0,
        "r_hip_x": -tilt, # keep stance foot flat
    })

    # flex to lift swing foot
    # no forward angle now, at the apex of swing
    bent3_angles = dict(bent2_angles)
    bent3_angles.update({
        "l_ankle_y": -bend3,
        "l_knee_y": 2*bend3,
        "l_hip_y": -bend3 + lift + l_hip_y_0, # counteract toe landing behind
        "l_hip_x": stretch, # counteract gravity pulling foot inwards
        "r_hip_x": -tilt,
    })

    # back to plant but with some sway to reduce recoil
    # forward angles are inverted
    bent4_angles = poppy.get_sway_angles(zero_angles, abs_x = sway, abs_y = lean)
    bent4_angles.update({
        "r_ankle_y": -(bend1 + forward), "l_ankle_y": -(bend1 - forward) - tweak,
        "r_knee_y": 2*bend1, "l_knee_y": 2*bend1,
        "r_hip_y": -(bend1 - forward), "l_hip_y": -(bend1 + forward) + l_hip_y_0,
    })

    # back to inverted plant fully, no sway or tweak
    bent5_angles = poppy.get_sway_angles(bent4_angles, abs_x = 0., abs_y = lean)
    bent5_angles["l_ankle_y"] += tweak

    return bent1_angles, bent2_angles, bent3_angles, bent4_angles, bent5_angles

if __name__ == "__main__":

    poppy = pw.PoppyWrapper(PH())
    
    # PID tuning
    K_p, K_i, K_d = 8.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

    # num_steps = 12
    timestep = .25
    # timestep = 1. # for visual inspection check when suspended off the ground
    # pausestep = 3.
    l_hip_y_0 = -3
    # do_foot = {"left": True, "right": True}
    
    zero_angles = {name: 0. for name in poppy.motor_names}
    zero_angles["l_hip_y"] = l_hip_y_0

    while True:

        num_steps = int(input("Enter num_steps: "))
        pausestep = float(input("Enter pausestep: ")) # >= 1 seems to work

        do_foot = {}
        do_foot["left"] = (input("Enter left (True/False): ")[0] == "T")
        do_foot["right"] = (input("Enter right (True/False): ")[0] == "T")
        print(do_foot)

        foot_angles = {}
        for foot in ("left", "right"):
            if not do_foot[foot]: continue
            while True:
                angs = input("Enter %s <lean> <forward> <bend1 both> <bend2 push> <bend3 lift> <lift hip> <tilt right> <stretch left> <sway abs> <tweak ankle>: " % foot)
                try:
                    lean, forward, bend1, bend2, bend3, lift, tilt, stretch, sway, tweak = tuple(map(float, angs.split()))
                    break
                except:
                    print("invalid format")
                    continue

            foot_angles[foot] = build_angle_dicts(poppy, l_hip_y_0, lean, forward, bend1, bend2, bend3, lift, tilt, stretch, sway, tweak)

        input("[Enter] to enable torques")
        poppy.enable_torques()

        input("[Enter] to goto init (suspend with strap)")
        if do_foot["left"]:
            init_bent = foot_angles["left"][0]
        else:
            init_bent = foot_angles["right"][0]
        _ = poppy.track_trajectory([(1., init_bent)], overshoot=1.)
        
        input("[Enter] to quickly step (get ready with strap)")

        left_traj = []
        if do_foot["left"]:
            left_traj = [
                (timestep, foot_angles["left"][1]),
                (timestep, foot_angles["left"][2]),
                (timestep, foot_angles["left"][3]),
                (timestep, foot_angles["left"][4]),
                (pausestep, foot_angles["left"][4]), # pause to stabilize
            ]

        # code to mirror trajectory
        right_traj = []
        if do_foot["right"]:
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

        # build up trajectory with number of left and/or right steps requested
        foot_trajs = {"left": left_traj, "right": right_traj}
        traj = []
        step = 0
        while True:
            for foot in ("left", "right"):
                if do_foot[foot]:
                    traj += foot_trajs[foot]
                    step += 1
                    if step == num_steps: break
            if step == num_steps: break
        
        buffers, elapsed = poppy.track_trajectory(traj, overshoot=1.)
    
        with open("empirical_forward_traj.pkl", "wb") as f: pk.dump(traj, f)
        with open("empirical_forward_bufs.pkl", "wb") as f: pk.dump((buffers, elapsed, poppy.motor_names), f)

        input("[Enter] to go compliant (hold strap first)")
        poppy.disable_torques()
    
        q = input("q to abort, otherwise will repeat")
        if q == "q": break
        
    # reset PID
    K_p, K_i, K_d = 4.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)
    
    # print("don't forget poppy.close()")
    print("closing poppy...")
    poppy.close()
    print("closed.")



