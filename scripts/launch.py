from pypot.creatures import PoppyHumanoid as PH
from pypot.sensor import OpenCVCamera
import time
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from record_angles import hotkeys

p = PH()
c = OpenCVCamera("poppy-cam",0,10)

def angs():
    return {m.name: m.present_position for m in p.motors}

def go(angs):
    p.goto_position(angs, 1, wait=True)

def save(angs, name):
    with open(name, "wb") as f: pk.dump(angs, f)

def load(name):
    with open(name, "rb") as f: return pk.load(f)

def toggle(motors):
    if type(motors) is not list: motors = [motors]
    for m in motors: m.compliant = not m.compliant

print("Created poppy humanoid p and opencvcamera c with 10 fps.  Don't forget to p.close() and c.close() before quit() when you are finished to clean up the motor state.")

