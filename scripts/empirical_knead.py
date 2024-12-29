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

def build_angle_dicts(poppy, l_hip_y_0, lean_angle, bend1, bend2, bend3, lift, tilt, stretch, sway):
    """
    poppy: PoppyWrapper instance
    l_hip_y_0: empirically found l_hip_y needs a correction offset
    lean angle: torso forward, whole time
    bend1: both legs bent at first
    bend2: lesser bend in left leg for pressure
    bend3: greater bend in left leg for lift
    lift: extra raise in left leg from hip so that toe doesn't land behind, should be negative
    tilt: r_hip_y rotation during bend2 to keep stance foot flat
    stretch: l_hip_y outward stretch during bend3 (but actually negative is better after adding tilt)
    sway: abs_x sway at end to reduce recoil when planting

    successful left swing: 12 10 4 15 0 10 -7 (12/28 3:08)
    successful right swing: 12 10 4 15 0 8 -7 (12/28 4:01)
    but in both cases, swing foot tends to land slightly behind its starting point
    The zeros are for new lift parameter to counteract this effect

    with lift:
    successful left swing: 12 10 4 12 -5 10 -7 (12/29 4:36)
    swing foot lands well, but there is some scary top-heavy wobbling
    new sway parameter to counteract this effect

    with sway:
    left: 12 10 4 12 -5 10 -9 3 (12/29 4:55) maybe best so far, only some forward/back wobble on ankles
    right: 12 10 4 12 -3 8 -5 3 (12/29 5:22) also pretty good but still landing slightly behind
    can do >=12 steps with pausestep = 1. (12/29 6:03)

    """

    zero_angles = {name: 0. for name in poppy.motor_names}
    zero_angles["l_hip_y"] = l_hip_y_0

    # initial knees bent
    bent1_angles = poppy.get_sway_angles(zero_angles, abs_x = 0., abs_y = lean_angle)
    bent1_angles.update({
        "r_ankle_y": -bend1, "l_ankle_y": -bend1,
        "r_knee_y": 2*bend1, "l_knee_y": 2*bend1,
        "r_hip_y": -bend1, "l_hip_y": -bend1 + l_hip_y_0,
    })

    # straighten for pressure through swing foot
    bent2_angles = dict(bent1_angles)
    bent2_angles.update({
        "l_ankle_y": -bend2,
        "l_knee_y": 2*bend2,
        "l_hip_y": -bend2 + l_hip_y_0,
        "r_hip_x": -tilt, # keep stance foot flat
    })

    # flex to lift swing foot
    bent3_angles = dict(bent2_angles)
    bent3_angles.update({
        "l_ankle_y": -bend3,
        "l_knee_y": 2*bend3,
        "l_hip_y": -bend3 + lift + l_hip_y_0, # counteract toe landing behind
        "l_hip_x": stretch, # counteract gravity pulling foot inwards
        "r_hip_x": -tilt,
    })

    # back to plant but with some sway to reduce recoil
    bent4_angles = poppy.get_sway_angles(bent1_angles, abs_x = sway, abs_y = lean_angle)

    # back to plant fully
    bent5_angles = dict(bent1_angles)

    return bent1_angles, bent2_angles, bent3_angles, bent4_angles, bent5_angles

if __name__ == "__main__":

    poppy = pw.PoppyWrapper(PH())
    
    # PID tuning
    K_p, K_i, K_d = 8.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)

    num_steps = 12
    timestep = .25
    # pausestep = 3.
    l_hip_y_0 = -3
    do_foot = {"left": True, "right": True}
    
    zero_angles = {name: 0. for name in poppy.motor_names}
    zero_angles["l_hip_y"] = l_hip_y_0
    
    input("[Enter] to enable torques")
    poppy.enable_torques()
    
    input("[Enter] to goto init (suspend first)")
    init_angles = poppy.get_sway_angles(zero_angles, abs_x = 0., abs_y = 8.)
    _ = poppy.track_trajectory([(1., init_angles)], overshoot=1.)
    
    while True:

        pausestep = float(input("Enter pausestep: "))

        foot_angles = {}
        for foot in ("left", "right"):
            if not do_foot[foot]: continue
            while True:
                angs = input("Enter %s <lean> <bend1 both> <bend2 push> <bend3 lift> <lift hip> <tilt right> <stretch left> <sway abs>: " % foot)
                try:
                    lean_angle, bend1, bend2, bend3, lift, tilt, stretch, sway = tuple(map(float, angs.split()))
                    break
                except:
                    print("invalid format")
                    continue

            foot_angles[foot] = build_angle_dicts(poppy, l_hip_y_0, lean_angle, bend1, bend2, bend3, lift, tilt, stretch, sway)
    
        input("[Enter] to bend knees")
        if do_foot["left"]:
            init_bent = foot_angles["left"][0]
        else:
            init_bent = foot_angles["right"][0]
        _ = poppy.track_trajectory([(2., init_bent)], overshoot=1.)
        
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
    
        with open("empirical_traj.pkl", "wb") as f: pk.dump(traj, f)
        with open("empirical_bufs.pkl", "wb") as f: pk.dump((buffers, elapsed, poppy.motor_names), f)
    
        qr = input("[Enter] to continue, r to rest, q to abort")
    
        if qr in "qr":
    
            input("[Enter] to go compliant (hold strap first)")
            poppy.disable_torques()
    
            if qr == "q": break
            # else qr == "r"
    
            input("[Enter] to enable torques")
            poppy.enable_torques()
            
            input("[Enter] to goto init (suspend first)")
            init_angles = poppy.get_sway_angles(zero_angles, abs_x = 0., abs_y = 8.)
            _ = poppy.track_trajectory([(1., init_angles)], overshoot=1.)
    
        
    # reset PID
    K_p, K_i, K_d = 4.0, 0.0, 0.0
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)
    
    # print("don't forget poppy.close()")
    print("closing poppy...")
    poppy.close()
    print("closed.")

