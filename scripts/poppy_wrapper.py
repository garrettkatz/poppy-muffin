import os
import pickle as pk
import signal
import numpy as np
import json

try:
    import pypot.utils.pypot_time as time
except:
    import time


# for python 2.7 on poppy odroid board
if hasattr(__builtins__, 'raw_input'): input=raw_input

# for safer keyboard interruptible busy-loops
INTERRUPTED = False
def custom_interrupt_handler(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True

class PoppyWrapper:
    def __init__(self, robot, camera=None):
        """
        Inputs:
            poppy: pypot.creatures.PoppyHumanoid
            camera: pypot.sensor.OpenCVCamera (or None to ignore camera)
        """
        self.robot = robot
        self.camera = camera # image shape is (480, 640, 3)

        self.motors = self.robot.motors
        self.motor_names = tuple(str(m.name) for m in self.motors)
        self.motor_ids = tuple(m.id for m in self.motors)
        self.motor_index = {name: idx for (idx, name) in enumerate(self.motor_names)}

        self.num_joints = len(self.motors)

        # cache look-ups for low-level motor io
        self.name_of = dict(zip(self.motor_ids, self.motor_names)) # name of id
        self.low_level = ()
        for ctrl in self.robot._controllers:
            ctrl_ids = tuple(ctrl.io.scan())
            ctrl_idx = [self.motor_ids.index(cid) for cid in ctrl_ids] # list for numpy indexing
            self.low_level += ((ctrl.io, ctrl_ids, ctrl_idx),)

        # including motor remapping (unless mock object)
        self.motor_directions = np.ones(self.num_joints)
        self.motor_offsets = np.zeros(self.num_joints)
        if not hasattr(robot, "mock"):

            config_base = os.path.dirname(__import__('poppy_humanoid').__file__)
            config_file = os.path.join(config_base, 'configuration', 'poppy_humanoid.json')
            with open(config_file) as f:
                config = json.load(f)

            for motor_name, properties in config['motors'].items():
                motor_index = self.motor_index[motor_name]
                if properties['orientation'] == 'indirect':
                    self.motor_directions[motor_index] = -1
                self.motor_offsets[motor_index] = properties['offset']

    def close(self):
        self.robot.close()
        if self.camera is not None: self.camera.close()

    def remap_low_to_high(self, positions):
        # positions is array of angles, same order as self.motor_names
        
        # remap based on:
        # https://github.com/poppy-project/poppy-humanoid/blob/master/software/poppy_humanoid/configuration/poppy_humanoid.json
        # https://github.com/poppy-project/pypot/blob/master/pypot/dynamixel/motor.py#L56
        return positions * self.motor_directions - self.motor_offsets

    def remap_high_to_low(self, positions):
        # positions is one of:
        #  array of angles, same order as self.motor_names
        #  dict mapping names to angles
        # returns same type with angles remapped
        if type(positions) == dict:
            low = {}
            for name, angle in positions.items():
                idx = self.motor_index[name]
                low[name] = (angle + self.motor_offsets[idx]) * self.motor_directions[idx]
            return low
        else:
            return (positions + self.motor_offsets) * self.motor_directions # /+-1 == *+-1

    def dict_from(self, positions):
        # positions is array of angles, same order as self.motor_names
        return {name: positions[idx] for (name, idx) in self.motor_index.items()}

    # low-level control to side-step laggy sync-loop
    def get_present_positions(self):
        # lists positions in same order as self.motors
        positions = np.full(len(self.motor_names), np.nan)
        for (io, ids, idx) in self.low_level:
            response = io.get_present_position(ids)
            if len(response) == 0: raise OSError # happens when usbs are knocked?
            positions[idx] = response
        return positions

    def set_goal_positions(self, setpoints):
        # setpoints[name]: angle
        for (io, ids, _) in self.low_level:
            targets = {id: setpoints[self.name_of[id]] for id in ids if self.name_of[id] in setpoints}
            io.set_goal_position(targets)

    def set_moving_speeds(self, setpoints):
        # setpoints[name]: speed
        for (io, ids, _) in self.low_level:
            targets = {id: setpoints[self.name_of[id]] for id in ids if self.name_of[id] in setpoints}
            io.set_moving_speed(targets)

    def enable_torques(self):
        for (io, ids, _) in self.low_level: io.enable_torque(ids)

    def disable_torques(self):
        for (io, ids, _) in self.low_level: io.disable_torque(ids)

    def comply(self, mode=True):
        for m in self.motors: m.compliant = mode

    def is_safe(self):
        """
        Returns True iff motors are in safe voltage/temp ranges based on:
            https://emanual.robotis.com/docs/en/dxl/mx/mx-12w/#control-table-of-eeprom-area
            https://emanual.robotis.com/docs/en/dxl/mx/mx-28/#control-table-of-eeprom-area
            https://emanual.robotis.com/docs/en/dxl/mx/mx-64/#control-table-of-eeprom-area
        accounts for pypot.dynamixel.conversion.py converting dynamixel voltage register to volts (divides by 10)
        """
        for m in self.motors:
            safe_voltage = (7 < m.present_voltage < 15)
            safe_temperature = (m.present_temperature < 70)
            if not (safe_voltage and safe_temperature):
                return False
        return True

    def goto_position(self, angles, duration=1.0, bufsize=10, speed_ratio=.5, motion_window=.5, binsize=None, verbose=True):
        """
        Moves to target position, blocks until motion has finished
        Freezes motion early if overloading motors, motion is restricted, or user interrupts
        Returns buffers of joint positions, speeds, loads, voltages, temperatures, and intermediate targets
        Saves intermediate buffers to file gobuf.pkl on OSError due to loose wiring

        Inputs:
            angles[motor.name]: target angle for named motor
            duration: target time in seconds to reach target
            bufsize: number of timesteps to sample buffers
            speed_ratio: abort if actual speed less than this ratio of commanded speed (set to 0 to disable)
            motion_window: timespan in seconds used to measure actual speed
            binsize: bin factor for image buffer (if None, no images are buffered)
            verbose: whether to print elapsed times
        Outputs:
            success: True iff motion completed successfully
            buffers[register][t, j]: buffered info from joint j register at timestep t
            time_elapsed[t]: actual time elapsed in seconds at timestep t
        """

        # temporary custom handling of Ctrl-C for user-stopped motion
        default_interrupt_handler = signal.signal(signal.SIGINT, custom_interrupt_handler)

        # initialize buffers
        bufkeys = ("position", "speed", "load", "voltage", "temperature")
        buffers = {key: np.empty((bufsize, self.num_joints)) for key in bufkeys + ("target",)}
        if binsize is not None: buffers['images'] = []
        time_points = np.linspace(0, duration, bufsize)
        time_elapsed = np.empty(bufsize)

        # convert motion window to discrete number of buffer time_points
        motion_dt = max(1, int(bufsize * motion_window / duration))

        # track successful motion
        success = True

        # fail gracefully on OSErrors due to loose wiring
        try:

            # launch the motion (non-blocking)
            start = time.time()
            self.robot.goto_position(angles, duration, wait=False)

            # record effective commanded speeds
            speeds = np.array([m.moving_speed for m in self.motors])
    
            # populate buffers and abort when needed
            for t in range(bufsize):
    
                # update buffers
                for key in bufkeys:
                    buffers[key][t] = [getattr(motor, "present_" + key) for motor in self.motors]

                # including current target position
                buffers["target"][t] = [motor.goal_position for motor in self.motors]

                # and images if requested
                if binsize is not None:
                    frame = self.camera.frame[::binsize, ::binsize].copy()
                    buffers['images'].append(frame)

                # check user interrupt
                if INTERRUPTED:
                    print("Interrupted motion!")
                    print("Set poppy_wrapper.INTERRUPTED to False to re-enable motion")
                    success = False

                # # check safe voltage/temperature
                # if not self.is_safe():
                #     print("Unsafe motion!!!")
                #     success = False
    
                # check restricted motion
                mt = t - motion_dt
                if mt >= 0:
                    dp = buffers['position'][t] - buffers['position'][mt]
                    dt = time_points[t] - time_points[mt]
                    is_restricted = (np.fabs(dp / dt) < speed_ratio * speeds)
                    if is_restricted.any():
                        print('Restricted motion!!!')
                        print([self.motor_names[r] for r in np.flatnonzero(is_restricted)])
                        success = False

                # abort if any checks failed
                if not success:
                    # Freeze motion at present position
                    for m in self.motors:
                        m.goal_position = m.present_position
                    # Drop incomplete buffer time_points
                    for key in buffers.keys():
                        buffers[key] = buffers[key][:t+1]
                    # Exit loop
                    print("Aborting motion!!!")
                    break
    
                # sleep to next timepoint
                time_elapsed[t] = time.time() - start
                if t < bufsize - 1:
                    if verbose:
                        dangle = max([np.fabs(m.present_position - angles[m.name]) for m in self.motors])
                        print("%d: %.3fs elapsed, timepoint=%.3fs, dangle=%.3f" % (t, time_elapsed[t], time_points[t], dangle))
                    time.sleep(max(0, time_points[t+1] - time_elapsed[t]))

            if verbose:
                dangle = max([np.fabs(m.present_position - angles[m.name]) for m in self.motors])
                print("done: %.3fs elapsed, dangle=%.3f" % (time_elapsed[-1], dangle))
    
        # Dump partial buffers to file on error
        except OSError:
            buffers = {key: buf[:t] for (key, buf) in buffers.items()}
            with open("gobuf.pkl","wb") as f: pk.dump((success, buffers, time_elapsed, self.motor_names), f)
            success = False

        # restore default interrupt handler
        finally:
            signal.signal(signal.SIGINT, default_interrupt_handler)

        return success, buffers, time_elapsed

    def track_trajectory(self, trajectory, binsize=None, overshoot=None, ms_rpms = 0.165, fps=16):
        # !! works well with durations around 1/5 sec, but poorly with durations around 1/24 sec
        # trajectory = [..., (duration (sec), waypoint) ...]
        # waypoint[name] = angle (deg)
        # binsize: image binning (if None, does not save images)
        # overshoot: how many degrees to overshoot goal position (if None, does not overshoot)
        #    smooths trajectory with non-zero velocity at intermediate waypoints
        # rpm_unit: 1 moving speed unit = ms_rpms rotations per minute (docs say .114, but too fast)
        # fps: frames per second for images

        # temporary custom handling of Ctrl-C for user-stopped motion
        default_interrupt_handler = signal.signal(signal.SIGINT, custom_interrupt_handler)

        # initialize buffers
        bufkeys = ("position", "speed", "load", "voltage", "temperature")
        buffers = {key: [] for key in bufkeys + ("target",)}
        if binsize is not None: buffers['images'] = []
        time_elapsed = []
        waypoint_timepoints = []

        # preprocess trajectory
        durations, waypoints = zip(*trajectory)
        timepoints = np.array(durations).cumsum()

        # extract motor names involved in trajectory
        motor_names = list(waypoints[0].keys())

        # fail gracefully on OSErrors due to loose wiring
        try:

            num_frames = 0
            start_time = time.time()
            for n in range(len(trajectory)):

                # mark time passed for current waypoint
                time_passed = time.time() - start_time
                waypoint_timepoints.append(time_passed)

                # skip intermediate waypoints that are behind schedule
                if n + 1 != len(trajectory) and time_passed >= timepoints[n]: continue

                # otherwise, set goal positions for current waypoint
                positions = self.remap_low_to_high(self.get_present_positions()) # low-level
                targets = positions.copy() # low-level
                # positions = [m.present_position for m in self.motors] # high-level
                # targets = list(positions)
                duration = timepoints[n] - time_passed
                goal_positions, moving_speeds = {}, {}
                for name in motor_names:
                    motor = getattr(self.robot, name)
                    index = self.motor_index[name]

                    position = positions[index]
                    targets[index] = waypoints[n][name]
                    goal_positions[name] = waypoints[n][name]

                    # only do overshoot if requested and before last waypoint
                    if overshoot is None or n + 1 == len(timepoints):
                        do_overshoot = False
                
                    # don't overshoot when direction changes
                    else:
                        current_direction = np.sign(waypoints[n][name] - position)
                        next_direction = np.sign(waypoints[n+1][name] - waypoints[n][name])
                        do_overshoot = (current_direction == next_direction)

                    # limit speed based on target duration
                    distance = np.fabs(waypoints[n][name] - position)
                    max_speed = distance / duration # units = degs / sec
                    # max_speed = int(max_speed * 60. / 360. / .114) # units = .114 rotations / min
                    max_speed = int(max_speed * 60. / 360. / ms_rpms) # units = ms_rpms rotations / min
                    max_speed = min(max_speed, 500) # don't go too fast
                    max_speed = max(max_speed, 1) # 0 means as fast as possible, so avoid this too
                    moving_speeds[name] = max_speed

                    # apply overshoot
                    if do_overshoot: goal_positions[name] += overshoot * current_direction
                    targets[index] = goal_positions[name]

                    # # high-level
                    # motor.moving_speed = moving_speeds[name]
                    # motor.goal_position = goal_positions[name]

                # low-level
                self.set_moving_speeds(moving_speeds)
                self.set_goal_positions(self.remap_high_to_low(goal_positions))

                # busy buffer loop until current timepoint is reached
                while True:
                    busy_time = time.time() - start_time

                    # # avoid overloading syncloop if polling high-level
                    # if busy_time < timepoints[n]: continue

                    # update buffers
                    for key in bufkeys:
                        if key == 'position':
                            buffers[key].append(self.remap_low_to_high(self.get_present_positions())) # low-level, more accurate
                            # buffers[key].append([m.present_position for m in self.motors]) # high-level
                        else:
                            buffers[key].append([getattr(motor, "present_" + key) for motor in self.motors])
    
                    # including current goal position (includes overshoot)
                    buffers["target"].append(targets)
    
                    # and images if requested
                    if binsize is not None:
                        current_fps = num_frames / busy_time
                        if current_fps < fps:
                            frame = self.camera.frame[::binsize, ::binsize].copy()
                            buffers['images'].append(frame)
                            num_frames += 1
                        else:
                            buffers['images'].append(None)
    
                    # and finally timing
                    time_elapsed.append(busy_time)
    
                    # stop when current timepoint is reached
                    if time_elapsed[-1] > timepoints[n]: break

        # Don't crash on loose wiring errors
        except OSError as err:
            print("Aborting Trajectory due to OSError!")
            print(err) # prints blank?

        finally:

            # restore default interrupt handler
            signal.signal(signal.SIGINT, default_interrupt_handler)

            # save results
            buffers = {key: np.array(buf) for (key, buf) in buffers.items()}
            with open("traj_buf.pkl","wb") as f:
                pk.dump((buffers, time_elapsed, waypoint_timepoints, self.motor_names), f)

        # return results
        return buffers, time_elapsed, waypoint_timepoints

    def track_trajectory_lqr(self, trajectory, T, K, X0, clip=1.0, binsize=None, overshoot=None, ms_rpms = 0.165, fps=16):
        # closed-loop trajectory tracking based on logic and outputs from fitted_lqr.py
        # !! works well with durations around 1/5 sec, but poorly with durations around 1/24 sec
        # trajectory = [..., (duration (sec), waypoint) ...]
        # waypoint[name] = angle (deg)
        # T: number of observation interpolation timepoints per waypoint for controller
        # K[n]: Control matrix for issuing nth trajectory waypoint
        # X0[n]: Nominal target observations before issuing nth waypoint
        # clip: Control outputs always clipped to this magnitude (degrees)
        # binsize: image binning (if None, does not save images)
        # overshoot: how many degrees to overshoot goal position (if None, does not overshoot)
        #    smooths trajectory with non-zero velocity at intermediate waypoints
        # rpm_unit: 1 moving speed unit = ms_rpms rotations per minute (docs say .114, but too fast)
        # fps: frames per second for images

        # temporary custom handling of Ctrl-C for user-stopped motion
        default_interrupt_handler = signal.signal(signal.SIGINT, custom_interrupt_handler)

        # initialize buffers
        bufkeys = ("position", "speed", "load", "voltage", "temperature")
        buffers = {key: [] for key in bufkeys + ("target",)}
        if binsize is not None: buffers['images'] = []
        time_elapsed = []
        waypoint_timepoints = []
        control_adjustments = []

        # preprocess trajectory
        durations, waypoints = zip(*trajectory)
        timepoints = np.array(durations).cumsum()

        # extract motor names involved in trajectory
        motor_names = list(waypoints[0].keys())

        # initialize interpolated observations with current position
        positions = self.remap_low_to_high(self.get_present_positions()) # low-level
        observation = positions[:,None]*np.ones(T) # will get overwritten in-place

        # fail gracefully on OSErrors due to loose wiring
        try:

            num_frames = 0
            start_time = time.time()
            for n in range(len(trajectory)):

                # mark time passed for current waypoint
                time_passed = time.time() - start_time
                waypoint_timepoints.append(time_passed)

                # skip intermediate waypoints that are behind schedule
                if n + 1 != len(trajectory) and time_passed >= timepoints[n]: continue

                # otherwise, prepare new waypoint using controller and current observations
                dx = observation.flatten() - X0[n]
                du = (K[n] * dx).sum(axis=1) # raw linear control output
                control_adjustments.append(du) # save it
                du = np.clip(du, -clip, +clip) # clip for safety

                # set goal positions for current waypoint
                positions = self.remap_low_to_high(self.get_present_positions()) # low-level
                # positions = [m.present_position for m in self.motors] # high-level
                # apply control adjustment to waypoint targets
                targets = positions.copy()
                for name in motor_names:
                    motor = getattr(self.robot, name)
                    index = self.motor_index[name]
                    targets[index] = waypoints[n][name]
                targets = targets + du
                # apply overshoots
                duration = timepoints[n] - time_passed
                goal_positions, moving_speeds = {}, {}
                for name in motor_names:
                    motor = getattr(self.robot, name)
                    index = self.motor_index[name]
                    position = positions[index]
                    goal_positions[name] = targets[index]

                    # only do overshoot if requested and before last waypoint
                    if overshoot is None or n + 1 == len(timepoints):
                        do_overshoot = False
                
                    # don't overshoot when direction changes
                    else:
                        current_direction = np.sign(waypoints[n][name] - position)
                        next_direction = np.sign(waypoints[n+1][name] - waypoints[n][name])
                        do_overshoot = (current_direction == next_direction)

                    # limit speed based on target duration
                    distance = np.fabs(goal_positions[name] - position)
                    max_speed = distance / duration # units = degs / sec
                    # max_speed = int(max_speed * 60. / 360. / .114) # units = .114 rotations / min
                    max_speed = int(max_speed * 60. / 360. / ms_rpms) # units = ms_rpms rotations / min
                    max_speed = min(max_speed, 500) # don't go too fast
                    max_speed = max(max_speed, 1) # 0 means as fast as possible, so avoid this too
                    moving_speeds[name] = max_speed

                    # apply overshoot
                    if do_overshoot: goal_positions[name] += overshoot * current_direction
                    targets[index] = goal_positions[name]

                    # # high-level
                    # motor.moving_speed = moving_speeds[name]
                    # motor.goal_position = goal_positions[name]

                # low-level
                self.set_moving_speeds(moving_speeds)
                self.set_goal_positions(self.remap_high_to_low(goal_positions))

                # busy buffer loop until current timepoint is reached
                while True:
                    busy_time = time.time() - start_time

                    # # avoid overloading syncloop if polling high-level
                    # if busy_time < timepoints[n]: continue

                    # update buffers
                    for key in bufkeys:
                        if key == 'position':
                            buffers[key].append(self.remap_low_to_high(self.get_present_positions())) # low-level, more accurate
                            # buffers[key].append([m.present_position for m in self.motors]) # high-level
                        else:
                            buffers[key].append([getattr(motor, "present_" + key) for motor in self.motors])
    
                    # including current goal position (includes overshoot)
                    buffers["target"].append(targets)
    
                    # and images if requested
                    if binsize is not None:
                        current_fps = num_frames / busy_time
                        if current_fps < fps:
                            frame = self.camera.frame[::binsize, ::binsize].copy()
                            buffers['images'].append(frame)
                            num_frames += 1
                        else:
                            buffers['images'].append(None)
    
                    # and finally timing
                    time_elapsed.append(busy_time)
    
                    # stop when current timepoint is reached
                    if time_elapsed[-1] > timepoints[n]: break

                # next timepoint reached, update observation for next controller command
                elapsed = np.array(time_elapsed)
                if n == 0:
                    t_int = np.linspace(0, timepoints[n], T)
                    mask = elapsed <= timepoints[n]
                else:
                    t_int = np.linspace(timepoints[n-1], timepoints[n], T)
                    mask = (timepoints[n-1] <= elapsed) & (elapsed <= timepoints[n])
                for j in range(25):
                    obs_j = [buffers["position"][m][j] for m in np.flatnonzero(mask)]
                    observation[j] = np.interp(t_int, elapsed[mask], obs_j)

        # Don't crash on loose wiring errors
        except OSError as err:
            print("Aborting Trajectory due to OSError!")
            print(err) # prints blank?

        finally:

            # restore default interrupt handler
            signal.signal(signal.SIGINT, default_interrupt_handler)

            # save results
            buffers = {key: np.array(buf) for (key, buf) in buffers.items()}
            with open("traj_buf.pkl","wb") as f:
                pk.dump((buffers, time_elapsed, waypoint_timepoints, self.motor_names), f)

        # return results
        return buffers, time_elapsed, waypoint_timepoints, control_adjustments

    # small helper to set specific angles
    def goto_angles(self, angles):
        self.track_trajectory([(1., angles)])

    def get_sway_angles(self, angle_dict, abs_x, abs_y):
        # set upper body sway (from poindexter/hand.py without degree conversion)
        angle_dict = dict(angle_dict)
        angle_dict['abs_x'] = abs_x
        angle_dict['abs_y'] = abs_y
        angle_dict['bust_x'] = -abs_x
        angle_dict['bust_y'] = -abs_y
    
        # slightly wide arms to reduce self-collision
        angle_dict['r_shoulder_x'] = min(-5., -abs(abs_x))
        angle_dict['l_shoulder_x'] = max(+5., +abs(abs_x))
    
        return angle_dict

    # mirror joints laterally (from poindexter/hand.py)
    def get_mirror_angles(self, angle_dict):
        mirrored = dict(angle_dict)
        for name, angle in angle_dict.items():
    
            # don't negate y-axis rotations
            sign = 1 if name[-2:] == "_y" else -1
    
            # swap right and left
            mirror_name = name
            if name[:2] == "l_": mirror_name = "r_" + name[2:]
            if name[:2] == "r_": mirror_name = "l_" + name[2:]
    
            # assign mirrored angle
            mirrored[mirror_name] = angle * sign
    
        return mirrored


