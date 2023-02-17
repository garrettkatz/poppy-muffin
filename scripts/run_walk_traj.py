import pickle as pk
import numpy as np
try:
    from pypot.creatures import PoppyHumanoid as PH
    import pypot.utils.pypot_time as time
except:
    from mocks import PoppyHumanoid as PH
    import time

poppy = PH()

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

with open('pypot_traj1.pkl', "rb") as f: trajs = pk.load(f)

hang = True
num_periods = 1

input('[Enter] to start (turns off compliance)')
for m in poppy.motors: m.compliant = False

# buffer motion
motor_names = tuple(m.name for m in poppy.motors)
bufkeys = ("position", "speed", "load", "voltage", "temperature")
buffers = {key: [] for key in bufkeys}
durations = []

cmd = ''
for period in range(num_periods):

    if hang:
        if cmd == 'q': break
        cmd = input('[Enter] for next step, [q] to abort: ')
    for traj in trajs:
        if hang:
            if cmd == 'q': break
            cmd = input('[Enter] for next trajectory, [q] to abort: ')
        for t, (duration, angles) in enumerate(traj):
            # if hang:
            #     if cmd == 'q': break
            #     cmd = input('[Enter] for next waypoint (%d of %d) [%.2f], [q] to abort: ' % (t, len(traj), duration))
            #     if cmd == 'q': break
            start = time.time()
            poppy.goto_position(angles, duration, wait=True)
            durations.append(time.time() - start)
            for key in bufkeys:
                buffers[key].append(tuple(
                    getattr(motor, "present_" + key)
                    for motor in poppy.motors))

with open('walk_buffers.pkl', "wb") as f: pk.dump((motor_names, buffers, durations), f)

input('[Enter] to stop (turns on compliance)')
for m in poppy.motors: m.compliant = True

poppy.close()
print("closed")

