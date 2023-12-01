# moving_speed appears to be velocity limit: https://docs.poppy-project.org/en/software-libraries/pypot
# velocity limit info: https://emanual.robotis.com/docs/en/dxl/mx/mx-28-2/#velocity-limit
# values from 0 to 1023
# units are 0.229rpm
# e.g. 200 * 0.229rpm ~ 45rpm ~ .76 rps ~ 274 deg/s

import pypot.dynamixel
try:
    import pypot.utils.pypot_time as time
except:
    import time

start_pos = 132.9
end_pos = 70.
duration = 1. # seconds
motor_id = 14 # l_knee_y

lower_dxl_io = pypot.dynamixel.DxlIO('/dev/ttyACM1')
lower_dxl_io.enable_torque((motor_id,))

lower_dxl_io.set_moving_speed({motor_id: 100})
lower_dxl_io.set_goal_position({motor_id: start_pos})
time.sleep(1.)

# motion parameters
distance = start_pos - end_pos
avg_speed = distance / duration

# first test: set a single goal position and moving speed, just measure position at high frequency
# no sticking observed, except
#   at very beginning (velocity 0 for a few ms)
#   near end (position bounces between 69.49 and 69.41, limited position sensor accuracy
max_speed = 2. * avg_speed # degrees / second
moving_speed = int(max_speed * 60 * (1./360.) * (1./.229)) # .229 rotations / minute = 1 unit
print(moving_speed)
lower_dxl_io.set_moving_speed({motor_id: moving_speed})

timepoints = []
positions = []

start_time = time.time()
lower_dxl_io.set_goal_position({motor_id: end_pos})
itr = 0
while True:
    lower_dxl_io.set_goal_position({motor_id: end_pos}) # see if many send commands creates jitter
    pos = lower_dxl_io.get_present_position((motor_id,))[0]
    positions.append(pos)
    timepoints.append(time.time() - start_time)
    if itr % 10 == 0:
        if itr > 10:
            vel = positions[-1] - positions[-11]
            print("%d: time=%.3f s, delta=%.3f, position=%.3f vs %.3f deg" % (
                itr, timepoints[-1], vel, positions[-1], end_pos))
        else:
            print("%d: time=%.3f s, position=%.3f vs %.3f deg" % (itr, timepoints[-1], positions[-1], end_pos))
    itr += 1
    if timepoints[-1] >= duration: break

time.sleep(1.)
pos = lower_dxl_io.get_present_position((motor_id,))[0]
print("eventual position = %.3f deg" % pos)


# # high-frequency linearly interpolate between starting and target angle
# # set max speed at beginning, based on target duration and distance
# # save target **time** to reach goal
# # and goal position at each interpolated point
# # 

# # lower_dxl_io.get_present_position((motor_id,))
# # # (132.97,)
# # lower_dxl_io.get_moving_speed((motor_id,))
# # # (0.0,)
# # lower_dxl_io.set_moving_speed({motor_id: 100})
# # lower_dxl_io.get_moving_speed((motor_id,))
# # # (199.728,)
# # lower_dxl_io.set_goal_position({motor_id: 100.})

# return to start
lower_dxl_io.set_moving_speed({motor_id: 100})
lower_dxl_io.set_goal_position({motor_id: start_pos})
time.sleep(1.)

lower_dxl_io.disable_torque((motor_id,))
lower_dxl_io.close()

