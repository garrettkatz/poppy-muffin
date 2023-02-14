import pickle as pk
import numpy as np
from pypot.creatures import PoppyHumanoid as PH

poppy = PH()

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

with open('pypot_traj1.pkl', "rb") as f: trajs = pk.load(f)

hang = True
num_steps = 1
input('[Enter] to start (turns off compliance)')

for m in poppy.motors: m.compliant = False

for step in range(num_steps):

    if hang: input('[Enter] for next step')
    for (speed, traj) in trajs:
        # if hang: input('[Enter] for next trajectory')
        for t in range(len(traj)):
            # if hang: input("[Enter] for next waypoint")
            poppy.goto_position(traj[t], duration=1/speed, wait=True)

    # # mirror for next step
    # # env not available!
    # trajs = [(
    #     np.stack([
    #         env.mirror_position(angles)
    #         for angles in traj]),
    #     speed)
    #     for (speed, traj) in trajs]

poppy.close()

