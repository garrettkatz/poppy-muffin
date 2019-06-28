from pypot.creatures import PoppyHumanoid as PH
from pypot.sensor import OpenCVCamera
import time
import numpy as np
import matplotlib.pyplot as plt
from record_angles import hotkeys

p = PH()
c = OpenCVCamera("poppy-cam",0,10)

print("Created poppy humanoid p and opencvcamera c with 10 fps")

