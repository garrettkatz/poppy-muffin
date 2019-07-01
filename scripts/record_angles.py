from pypot.creatures import PoppyHumanoid as PH
from pypot.sensor import OpenCVCamera
import time
import numpy as np
import matplotlib.pyplot as pt
import pickle as pk
import signal
import sys

def record_angles(poppy, period=1., dump=None):

    angles = []

    def signal_handler(sig, frame):
        if dump is not None:
            print('Saving angles...')
            with open(dump,"w") as f:
                pk.dump(angles, f)
            print('Done.')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    while True:
        position = {
            m.name: m.present_position for m in p.motors}
        print(position.values())
        angles.append(position)
        time.sleep(period)

def hotkeys(p, motors, verbose=True):
    angles = []
    while True:
        if verbose:
            print("Present positions:")
            print({m.name: m.present_position for m in motors})
        prompt = "Options:\n[q]uit\n[c]lose p and quit\n[s]tore\n[t]oggle\n...? "
        key = raw_input(prompt)
        if key == "q": break
        if key == "c":
            p.close()
            print("Did p.close().")
            break
        if key == "s":
            angles.append({m.name: m.present_position for m in p.motors})
        if key == "t":
            for m in motors: m.compliant = not m.compliant

    return angles

def show_angles(angles):
    names = list(angles[0].keys())
    angles = np.array([
        [angles[t][n] for n in names]
        for t in range(len(angles))])
    pt.plot(angles)
    pt.show()

if __name__ == "__main__":

    period = .1
    dump = "recorded_angles.pkl"
    
    p = PH()
    angles = record_angles(p, period, dump)


