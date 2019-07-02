from pypot.creatures import PoppyHumanoid as PH
import pickle as pk
from record_angles import hotkeys
import time

with open("crawl2.pkl","r") as f:
    crawl_angles = pk.load(f)

p = PH()

key = raw_input("reposition arms? [y/n]")

if key == "y":
    for arm in ["left","right"]:
        raw_input('Enter for %s arm compliance...' % arm)
        motors = [m for m in p.arms if m.name[0] == arm[0]]
        for m in motors: m.compliant=True
        while True:
            raw_input('Enter for %s arm noncompliant...' % arm)
            pos = getattr(p, arm[0]+"_shoulder_y").present_position
            if pos < -90 or pos > 90:
                print("shoulder_y pos=%f not in [-90..90]!" % pos)
            else:
                for m in motors: m.compliant = False
                break

key = raw_input("reposition legs? [y/n]")

if key == "y":
    for leg in ["left","right"]:
        raw_input('Enter for %s leg compliance...' % leg)
        motors = [m for m in p.legs if m.name[0] == leg[0]]
        for m in motors: m.compliant=True
        while True:
            raw_input('Enter for %s leg noncompliant...' % leg)
            pos = getattr(p, leg[0]+"_hip_z").present_position
            if leg == "left" and (pos < -180 or pos > 22):
                print("hip pos=%f not in [-180..22]!" % pos)
            elif leg == "right" and (pos < -22 or pos > 180):
                print("hip pos=%f not in [-22..180]!" % pos)
            else:
                for m in motors: m.compliant = False
                break

key = raw_input('Reset crawl pose and compliance? [y/n]')

# overwrite with optimized angles
crawl_angles.update({
    'l_hip_y': -100., 'l_knee_y': 133., 'r_hip_y': -100., 'r_knee_y': 133.,
    'bust_x': 0., 'abs_z': 0., 'abs_y': 45., 'abs_x':0.,
    'r_elbow_y':-115., 'l_elbow_y':-115.,
    'r_shoulder_x':0., 'l_shoulder_x':0.,
    'r_ankle_y':45., 'l_ankle_y':45.,
    'r_hip_z':-10., 'l_hip_z':10.,
    'r_hip_x':0., 'l_hip_x':0.,})

# # Initial position with knees and elbows close:
# crawl_angles.update(
#     {'r_elbow_y': -115., 'l_elbow_y': -115.,
#     'r_hip_y':-100., 'l_hip_y': -100.,
#     'r_knee_y':133., 'l_knee_y':133.})

if key == "y":
    for m in p.motors:
        if m in [p.r_shoulder_y, p.l_shoulder_y]: # misbehaving shoulder
            crawl_angles.pop(m.name)
            m.compliant = True
        else:
            m.compliant = False
    
    p.goto_position(crawl_angles, 3, wait=True)

wait_each = True
movement_time = 1.

while True:
    key = raw_input('Ready for one move? (q to quit)')
    if key == 'q': break

    # lift torso to swing right arm:
    if wait_each:
        key = raw_input('Ready for left arm in? (q to quit)')
        if key == 'q': break
    p.goto_position( # left arm in, hips out, bust turn for center of gravity
        {'l_shoulder_x': -13., 'l_hip_z': 20., 'r_hip_z':-20., 'bust_x':-10.}, movement_time, wait=True)
    if wait_each:
        key = raw_input('Ready for torso rotate, right elbow perp? (q to quit)')
        if key == 'q': break
    p.goto_position( # torso rotate
        {'r_shoulder_x':12., 'abs_x': -3., 'abs_y': 37., 'abs_z': -15., 'r_elbow_y':-90.}, movement_time, wait=True)
    if wait_each:
        key = raw_input('Ready for left elbow perp? (q to quit)')
        if key == 'q': break
    p.goto_position( # left elbow perp
        {'l_elbow_y':-90., 'abs_z': 0.}, movement_time, wait=True)
    if wait_each:
        key = raw_input('Ready for shoulders out? (q to quit)')
        if key == 'q': break
    p.goto_position( # left elbow perp
        {'l_shoulder_x':0., 'r_shoulder_x':0.}, movement_time, wait=True)


    ### get knees fully bent

    if wait_each:
        key = raw_input('Ready for all the way forward, thighs vertical? (q to quit)')
        if key == 'q': break
    p.goto_position(
        {'r_elbow_y': -115., 'l_elbow_y': -115.,
        'l_hip_y': -70., 'l_knee_y': 100.,
        'r_hip_y': -70., 'r_knee_y': 100.}, movement_time, wait=True)
    if wait_each:
        key = raw_input('Ready for right hip raised? (q to quit)')
        if key == 'q': break
    p.goto_position(
        {'r_hip_z': 5., 'r_hip_y': -95., 'l_hip_y': -95., 'l_hip_z': 18.,
        'abs_z': 18., 'l_knee_y': 100., 'r_knee_y': 100.}, movement_time, wait=True)
    if wait_each:
        key = raw_input('Ready for right leg forward? (q to quit)')
        if key == 'q': break
    p.goto_position(
        {'r_hip_z': -2., 'r_hip_y': -100., 'l_hip_y': -70., 'l_hip_z': 2.,
        'abs_z': -8., 'l_knee_y': 100., 'r_knee_y': 125.}, movement_time, wait=True)

    if wait_each:
        key = raw_input('Ready for left hip raised? (q to quit)')
        if key == 'q': break
    p.goto_position(
        {'r_hip_z': -18., 'r_hip_y': -100., 'l_hip_y': -70., 'l_hip_z': 5.,
        'abs_z': -18., 'l_knee_y': 100., 'r_knee_y': 125.}, movement_time, wait=True)
    if wait_each:
        key = raw_input('Ready for left leg forward? (q to quit)')
        if key == 'q': break
    p.goto_position(
        {'r_hip_z': 0., 'r_hip_y': -100., 'l_hip_y': -100., 'l_hip_z': 0.,
        'abs_z': 0., 'l_knee_y': 125., 'r_knee_y': 125.}, movement_time, wait=True)

    if wait_each:
        key = raw_input('Ready for reset to initial? (q to quit)')
        if key == 'q': break
    p.goto_position(crawl_angles,2, wait=True)


    print("finished loop.")

    
key = raw_input('Revert to initial crawl? [y/n]')
if key == "y":
    p.goto_position(crawl_angles, 3, wait=True)


p.close()


