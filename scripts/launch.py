from pypot.creatures import PoppyHumanoid as PH
from pypot.sensor import OpenCVCamera
import pypot.utils.pypot_time as time
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from record_angles import hotkeys

p = PH()
c = OpenCVCamera("poppy-cam",0,10)

def gobuf(angs, duration, bufsize):

    # buffers
    bufkeys = ("position", "speed", "load", "voltage", "temperature")
    buffers = {key: np.empty((bufsize, len(angs))) for key in bufkeys}
    timepoints = np.linspace(0, duration, bufsize)
    motor_names = sorted(angs.keys())
    success = True
    t = 0

    try:

        start = time.time()
        p.goto_position(angs, duration, wait=False)

        for t in range(bufsize):

            # update buffer
            for key in bufkeys:
                buffers[key][t] = [
                    getattr(getattr(p, motor_name), "present_" + key)
                    for motor_name in motor_names]

            # abort if approaching dangerous voltage/temperature
            # https://emanual.robotis.com/docs/en/dxl/mx/mx-12w/#control-table-of-eeprom-area
            # https://emanual.robotis.com/docs/en/dxl/mx/mx-28/#control-table-of-eeprom-area
            # https://emanual.robotis.com/docs/en/dxl/mx/mx-64/#control-table-of-eeprom-area
            # pypot/dynamixel/conversion.py converts dynamixel voltage register to volts (divides by 10)
            for m in p.motors:
                safe_voltage = (7 < m.present_voltage < 15)
                safe_temperature = (m.present_temperature < 70)
                if not (safe_voltage and safe_temperature):
                    success = False
            if not success:
                for m in p.motors:
                    m.goal_position = m.present_position
                for key in bufkeys:
                    buffers[key] = buffers[key][:t+1]
                print("Unsafe motion!!!")
                break

            # sleep to next timepoint
            if t < bufsize - 1:
                time_elapsed = time.time() - start
                print(t, time_elapsed, timepoints[t])
                time.sleep(max(0, timepoints[t+1] - time_elapsed))

        return success, buffers, motor_names

    except:

        buffers = {key: buf[:t] for (key, buf) in buffers.items()}
        with open("gobuf.pkl","wb") as f: pk.dump((success, buffers, motor_names), f)

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

