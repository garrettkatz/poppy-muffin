from pypot.creatures import PoppyHumanoid as PH
from pypot.sensor import OpenCVCamera
import pypot.utils.pypot_time as time
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import dances
from record_angles import hotkeys

p = PH()
c = OpenCVCamera("poppy-cam",0,10)

def gobuf(angs, duration, bufsize):

    # buffers
    bufkeys = ("position", "speed", "load", "voltage", "temperature")
    buffers = {key: np.empty((bufsize, len(angs)) for key in bufkeys}
    timepoints = np.linspace(0, duration, bufsize)
    motor_names = sorted(angs.keys())

    start = time.time()
    p.goto_position(angs, duration, wait=False)

    for t in range(bufsize):

        # update buffer
        for key in bufkeys:
            buffers['position'][t] = [
                getattr(getattr(p, motor_name), "present_" + key)
                for motor_name in motor_names]

        # sleep to next timepoint
        if t < bufsize - 1:
            time_elapsed = time.time() - start
            time.sleep(max(0, timepoints[t+1] - time_elapsed)

    return buffers, motor_names

def angs():
    return {m.name: m.present_position for m in p.motors}

def go(angs):
    p.goto_position(angs, 1, wait=True)

def save(angs, name):
    with open(name, "wb") as f: pk.dump(angs, f)

def load(name):
    with open(name, "rb") as f: return pk.load(f)

def toggle(motors):
    if type(motors) is not list: motors = [motors]
    for m in motors: m.compliant = not m.compliant

def do(traj):
    for m in p.motors: m.compliant = False
    inits = {m.name: m.present_position for m in p.motors}
    for (duration, angles) in traj:
        for m in p.motors:
            if m.name not in angles:
                angles[m.name] = inits[m.name]
        p.goto_position(angles, duration, wait=True)

def sit():
    with open("sit_angles.pkl","r") as f: sit_angles = pk.load(f)
    for m in p.motors: m.compliant = False
    p.goto_position(sit_angles, 5, wait=True)

def zero():
    zero_angles = {m.name: 0. for m in p.motors}
    for m in p.motors: m.compliant = False
    p.goto_position(zero_angles, 5, wait=True)

print("Created poppy humanoid p and opencvcamera c with 10 fps.  Don't forget to p.close() and c.close() before quit() when you are finished to clean up the motor state.")

