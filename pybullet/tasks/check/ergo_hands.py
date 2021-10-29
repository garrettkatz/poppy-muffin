"""
To manually choose a new camera viewpoint and get its parameters:
1. Set MANUALLY_GET_NEW_CAM to True
2. Run this script, manually modify the view, then press Enter in the console
3. Copy-paste the camera parameters that get printed and use as args to resetDebugVisualizerCamera
"""
import pickle as pk
import numpy as np
import sys
sys.path.append('../../envs')

import pybullet as pb
from ergo import PoppyErgoEnv, convert_angles

# set this to True if you want to choose a new camera view
MANUALLY_GET_NEW_CAM = False

# set it to False to double-check the camera view that you copy-pasted

# this launches the simulator
env = PoppyErgoEnv(pb.POSITION_CONTROL)

# this loops until you get the camera view you want
while MANUALLY_GET_NEW_CAM:
    
    # Get the current camera view parameters
    cam = pb.getDebugVisualizerCamera()
    width, height, view, proj, camup, camfwd, horz, vert, yaw, pitch, dist, targ = cam
    # print(cam)
    # print(len(cam))

    # Render the camera image in the simulator
    _, _, rgb, depth, segment = pb.getCameraImage(width, height, view, proj)

    # Save the rendered camera image to file
    with open("getcam.pkl","wb") as f: pk.dump((rgb, depth, segment), f)

    # Print the parameters to be copy-pasted
    # can use these as input to pb.resetDebugVisualizerCamera()
    print("Camera parameters for resetDebugVisualizerCamera:")
    print(str((dist, yaw, pitch, targ)))

    # If dissatisfied, tweak the view and press Enter to print tweaked parameters
    input("ready...")

# Copy-paste the camera parameters here to check that it worked:
cam = 1.2000000476837158, 3.199998140335083, -16.20000648498535, (-0.14883437752723694, 0.8789047002792358, 0.046143874526023865)

if not MANUALLY_GET_NEW_CAM:
    pb.resetDebugVisualizerCamera(*cam)
    input("waiting to close...")
