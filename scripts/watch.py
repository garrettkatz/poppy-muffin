from pypot.creatures import PoppyHumanoid as PH
from pypot.sensor import OpenCVCamera
import matplotlib.pyplot as plt
import time
import numpy as np
import sys

jlims = {
    "head_y": [-22., 20.], # backward/forward
    "head_z": [-58., 40.], # right/left
    "l_ankle_y": [-40, 40], # flex/point
    "r_ankle_y": [-40, 40], # flex/point
}
fps = 10
# moving_speed = 25

p = PH()
for m in jlims.keys():
    getattr(p,m).compliant=False
    # getattr(p,m).moving_speed = moving_speed

p.goto_position({"r_ankle_y":-40,"l_ankle_y":40}, 1, wait=True)

c = OpenCVCamera("poppy-cam",0,fps)

def handle_close(event):
    c.close()
    p.close()
    sys.exit(0)

plt.ion()
fig = plt.figure()
fig.canvas.mpl_connect("close_event", handle_close)

while True:

    if p.r_ankle_y.present_position > +35 or p.l_ankle_y.present_position < -35:
        p.r_ankle_y.goto_position(-40, .8, wait=False)
        p.l_ankle_y.goto_position(+40, .8, wait=False)
    if p.r_ankle_y.present_position < -35 or p.l_ankle_y.present_position > +35:
        p.r_ankle_y.goto_position(+40, .8, wait=False)
        p.l_ankle_y.goto_position(-40, .8, wait=False)

    plt.pause(.5)
    frame = c.frame.astype(float)/255.
    plt.pause(2./fps)
    
    new_frame = c.frame.astype(float)/255.
    diff = np.fabs(new_frame - frame)
    frame = new_frame
    
    i = np.arange(diff.shape[0])[:,np.newaxis,np.newaxis]
    j = np.arange(diff.shape[1])[np.newaxis,:,np.newaxis]
    ds = diff.sum()
    ci = (diff*i).sum() / ds
    cj = (diff*j).sum() / ds
    
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(frame[:,:,[2,1,0]])
    plt.subplot(1,2,2)
    plt.imshow(diff[:,:,[2,1,0]])
    plt.scatter([cj],[ci],c='r')
    plt.draw()

    ### constant velocity based on sign
    di = ci - diff.shape[0]/2
    dj = cj - diff.shape[1]/2
    
    dirs = {"head_y": +np.sign(di), "head_z": -np.sign(dj)}
        
    targets = {m: min(max(
        getattr(p,m).present_position + 15*dirs[m],
        jlims[m][0]), jlims[m][1])
        for m in dirs.keys()}

    ### absolute position based on centroid    
    # targets = {
    #     "head_y": jlims["head_y"][0] + ci/diff.shape[0]*(jlims["head_y"][1] - jlims["head_y"][0]),
    #     "head_z": jlims["head_z"][1] + cj/diff.shape[1]*(jlims["head_z"][0] - jlims["head_z"][1]),
    # }

    ### velocity based on centroid
    # di = (ci - diff.shape[0]/2) / (diff.shape[0]/2)
    # dj = (cj - diff.shape[1]/2) / (diff.shape[1]/2)
    
    # dirs = {"head_y": +di, "head_z": -dj}
        
    # targets = {m: min(max(
    #     getattr(p,m).present_position + 50*dirs[m],
    #     jlims[m][0]), jlims[m][1])
    #     for m in dirs.keys()}

    p.goto_position(targets, 2, wait=True)

c.close()
p.close()

