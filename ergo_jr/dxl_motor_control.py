import time

# for pip install --user poppy-ergo-jr, from_json source is at:
# ~/.local/lib/python3.8/site-packages/pypot/robot/__init__.py
from pypot.robot import from_json

# custom configuration based on PoppyErgoJr config file
# for pip install --user poppy-ergo-jr, config file is at:
# ~/.local/lib/python3.8/site-packages/poppy_ergo_jr/configuration/poppy_ergo_jr.json
json_file = "dbg_config.json"
extra = {}
r = from_json(json_file, sync=True, strict=True, use_dummy_io=False, **extra)

# get the first motor from the config
m = r.m1
print(m)

# make it non-compliant to enable motion
m.compliant = False

# make a slight motion
m.goal_position = m.present_position + 15

# wait half a second for the motion to complete
# then print the position again to see the difference
time.sleep(.5)
print(m)

# make it compliant again so it can be moved by hand
m.compliant = False

