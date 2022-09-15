import sys
import itertools as it
import numpy as np
from pypot.creatures import PoppyHumanoid as PH

period = 10**6
ph = PH()
names = [m.name for m in ph.motors]

try:
    for i in it.count():
        for t in range(period): pass

        load = [m.present_load for m in ph.motors]
        voltage = [m.present_voltage for m in ph.motors]
        temperature = [m.present_temperature for m in ph.motors]

        print("\n%d" % i)
        print("load = %.2f-%.2f (%s-%s)" % (
            np.min(load), np.max(load), names[np.argmin(load)], names[np.argmax(load)]))
        print("volt = %.2f-%.2f (%s-%s)" % (
            np.min(voltage), np.max(voltage), names[np.argmin(voltage)], names[np.argmax(voltage)]))
        print("temp = %.2f-%.2f (%s-%s)" % (
            np.min(temperature), np.max(temperature), names[np.argmin(temperature)], names[np.argmax(temperature)]))

except:
    ph.close()
    sys.exit()

