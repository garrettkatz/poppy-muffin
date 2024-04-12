import matplotlib.pyplot as pt
import numpy as np
import glob

fnames = glob.glob("hand_tune_buffers_*.pkl")

arrows = {
    "bl": np.array([-1, -1]),
    "b" : np.array([ 0, -1]),
    "br": np.array([+1, -1]),
    "l" : np.array([-1,  0]),
    "s" : np.array([ 0,  0]),
    "r" : np.array([+1,  0]),
    "fl": np.array([-1, +1]),
    "f" : np.array([ 0, +1]),
    "fr": np.array([+1, +1]),
}

for fname in fnames:
    params = fname[len("hand_tune_buffers_"):-len(".pkl")].split("_")
    if len(params) != 5: continue # old format
    init_abs_x, init_abs_y, swing_abs_x, swing_abs_y, fall = params
    pt.plot(int(swing_abs_x), int(swing_abs_y), 'ko')
    pt.quiver(int(swing_abs_x), int(swing_abs_y), *arrows[fall])
    # pt.plot(
    #     [int(swing_abs_x), int(swing_abs_x)+0.5*arrows[fall][0]],
    #     [int(swing_abs_y), int(swing_abs_y)+0.5*arrows[fall][1]],
    #     color='k', linestyle='-')

pt.xlabel("swing_abs_x")
pt.ylabel("swing_abs_y")
pt.title("forward")
pt.show()
