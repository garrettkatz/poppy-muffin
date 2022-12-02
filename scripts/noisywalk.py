"""
take several walk cycles spaced by Enter, until Ctrl-C
each cycle is a left then right step, with angles mirrored
between cycles, small random noise is added to the angles and duration
last set of angles is returned
start just with joints: ankle, knee, hip, shoulder, elbow (all _y)
for _y joints, mirrored angles should be identical
like humans, opposite arm and leg swing forward
"""

# define initial stance/swing parameters
stance = {
    'l_ankle_y': 0, 'l_knee_y': 0, 'l_hip_y': 0,
    'l_elbow_y': 0, 'l_shoulder_y': 0,
    'r_ankle_y': 0, 'r_knee_y': 0, 'r_hip_y': 0,
    'r_elbow_y': 0, 'r_shoulder_y': 0,
}
swing = {
    'l_ankle_y': 0, 'l_knee_y': 0, 'l_hip_y': 0,
    'l_elbow_y': 0, 'l_shoulder_y': 0,
    'r_ankle_y': 0, 'r_knee_y': 0, 'r_hip_y': 0,
    'r_elbow_y': 0, 'r_shoulder_y': 0,
}
stance_to_swing = 1
swing_to_stance = 1


from pypot.creatures import PoppyHumanoid as PH
import signal

p = PH()

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

# for safer keyboard interruptible busy-loops
busy_interrupted = False
def busy_interrupt_handler(signum, frame):
    global busy_interrupted
    busy_interrupted = True
signal.signal(signal.SIGINT, busy_interrupt_handler)

# start in stance pose
input("Enter to go to stance pose...")

while not busy_interrupted:

    # wait for user
    input("Enter for a step cycle...")

    # swing pose
    p.goto_position(swing, 1, wait=True)

    # mirrored stance pose

    # mirrored swing pose

    # stance pose

p.close()

