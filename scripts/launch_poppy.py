import pickle as pk
import numpy as np
import poppy_wrapper as pw
try:
    from pypot.creatures import PoppyHumanoid
    from pypot.sensor import OpenCVCamera
except:
    from mocks import PoppyHumanoid
    from mocks import OpenCVCamera

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

poppy = pw.PoppyWrapper(
    PoppyHumanoid(),
    OpenCVCamera("poppy-cam", 0, 10))

print("Created poppy with 10 fps camera.  Don't forget to poppy.close() before quit() when you are finished to clean up the motor state.")

