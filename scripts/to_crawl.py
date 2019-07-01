from pypot.creatures import PoppyHumanoid as PH
import pickle as pk

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
    'l_hip_y': -80., 'l_knee_y': 105., 'r_hip_y': -80., 'r_knee_y': 105.,
    'bust_x': 0., 'abs_z': 0., 'abs_y': 45., 'abs_x':0.,
    'r_elbow_y':-90., 'l_elbow_y':-90.,
    'r_shoulder_x':0., 'l_shoulder_x':0.,
    'r_ankle_y':45., 'l_ankle_y':45.,
    'r_hip_z':-10., 'l_hip_z':10.,
    'r_hip_x':0., 'l_hip_x':0.,})

if key == "y":
    for m in p.motors:
        if m in [p.r_shoulder_y, p.l_shoulder_y]: # misbehaving shoulder
            crawl_angles.pop(m.name)
            m.compliant = True
        else:
            m.compliant = False
    
    p.goto_position(crawl_angles, 3, wait=True)

while True:
    key = raw_input('Ready for one step? (q to quit)')
    if key == 'q': break

    # move right leg with hip lift
    p.goto_position( # left thigh vertical with right hip raise
        {'l_hip_y': -70., 'l_knee_y': 90., 'r_elbow_y': -100., 'l_elbow_y': -100.,
        'abs_x': -6., 'abs_y': 45., 'abs_z': 12., 'r_hip_z': -5., 'l_hip_z': 5.},
        3, wait=True)
    p.goto_position( # left thigh back, right thigh forward
        {'l_hip_y': -62., 'l_knee_y': 82., 'r_elbow_y': -115., 'l_elbow_y': -115.,
        'abs_x': 0., 'abs_y': 45., 'abs_z': 0., 'r_hip_z': -10., 'l_hip_z': 10.},
        3, wait=True)

    # lift torso to swing right arm:
    # initial:
    # {u'abs_y': 47.78, u'abs_z': -2.77, u'r_elbow_y': -117.85, u'l_shoulder_x': -0.5100000000000051}
    p.goto_position( # lift torso
        {'abs_y': 38., 'abs_z': -20., 'r_elbow_y': -125., 'l_shoulder_x': 15.},
        3, wait=True)
    p.goto_position( # lower torso
        {'abs_y': 45., 'abs_z': 0., 'r_elbow_y': -90., 'l_shoulder_x': 0.},
        3, wait=True)

    # # key = raw_input('Ready for one half step? (q to quit)')
    # if key == "q": break
    # p.goto_position({'r_hip_y': -50.,'l_hip_y':-75.,'r_knee_y':70.,'l_knee_y':90.,'r_hip_z':-25.,'l_hip_z':0.}, 2, wait=True)
    # p.goto_position({'r_hip_y': -100.,'l_hip_y':-75.,'r_knee_y':130.,'l_knee_y':90.,'r_hip_z':-24.,'l_hip_z':0.}, 2, wait=True)
    # # p.goto_position({'r_hip_y': -100.,'l_hip_y':-24.,'r_knee_y':130.,'l_knee_y':53.,'r_hip_z':0.,'l_hip_z':0.}, 2, wait=True)

    # # key = raw_input('Ready for other half step? (q to quit)')
    # if key == "q": break
    # p.goto_position({'r_hip_y': -75.,'l_hip_y':-50.,'r_knee_y':90.,'l_knee_y':70.,'r_hip_z':0.,'l_hip_z':25.}, 2, wait=True)
    # p.goto_position({'r_hip_y': -75.,'l_hip_y':-100.,'r_knee_y':90.,'l_knee_y':130.,'r_hip_z':0.,'l_hip_z':25.}, 2, wait=True)
    # # p.goto_position({'r_hip_y': -24.,'l_hip_y':-100.,'r_knee_y':53.,'l_knee_y':130.,'r_hip_z':0.,'l_hip_z':0.}, 2, wait=True)
    
p.close()
