import numpy as np
from pypot import dynamixel
from pypot.sensor import OpenCVCamera
from pypot.dynamixel.io import Dxl320IO

try:
    # if running on hardware
    import pypot.utils.pypot_time as time
except:
    # otherwise
    import time

# for python 2.7 on poppy odroid board
if hasattr(__builtins__, 'raw_input'): input=raw_input

class ErgoJrWrapper:

    def __init__(self, fps=None):

        self.camera = None
        if fps is not None:
            self.camera = OpenCVCamera("ergo-jr-cam", 0, fps)
        
        self.dxl_io = Dxl320IO('/dev/ttyAMA0')
        self.ids = self.dxl_io.scan([1,2,3,4,5,6])

        self.joint_name = tuple("m"+str(i) for i in self.ids)
        self.parent_index = []
        self.translation = []
        self.orientation = []
        self.axis = []

    def close(self):
        if self.camera is not None: self.camera.close()
        self.dxl_io.close()

    def get_joint_info(self):
        """
        TBD
        Return joint_info, a tuple of joint information
        joint_info[i] = (joint_name, parent_index, translation, orientation, axis) for ith joint
        joint_name: string identifier for joint
        parent_index: index in joint_info of parent (-1 is base)
        translation: (3,) translation vector in parent joint's local frame
        orientation: (4,) orientation quaternion in parent joint's local frame
        axis: (3,) rotation axis vector in ith joint's own local frame (None for fixed joints)
        """
        return tuple(zip(self.joint_name, self.parent_index, self.translation, self.orientation, self.axis))

    # PID tuning
    def get_pid_gains(self):
        return self.dxl_io.get_pid_gain(self.ids)

    def set_pid_gains(self, K_p, K_i, K_d):
        self.dxl_io.set_pid_gain({i: (K_p, K_i, K_d) for i in self.ids})

    def _get_position(self):
        return np.array(self.dxl_io.get_present_position(self.ids))

    def _array_to_dict(self, arr):
        return {n: a for (n,a) in zip(self.joint_name, arr)}

    def _dict_to_array(self, dct):
        return np.array([dct[n] for n in self.joint_name])

    def get_current_angles(self):
        """
        Get PyPot-style dictionary of current angles
        Returns angle_dict where angle_dict[joint_name] == angle (in degrees)
        """
        return self._array_to_dict(self._get_position())

    def set_compliance(self, comply):
        if comply:
            self.dxl_io.disable_torque(self.ids)
        else:
            self.dxl_io.enable_torque(self.ids)

    def goto_position(self, target, duration=1.):
        """
        PyPot-style method that commands the arm to given target joint angles
        target is a dictionary where target[joint_name] == angle (in degrees)
        duration: time in seconds for the motion to complete
        """

        # pypot does not work well with control period higher than 0.2 sec
        control_period = .2

        # get current/target angle arrays
        current = self._get_position()
        target = self._dict_to_array(target)

        # linearly interpolate trajectory
        num_steps = int(duration / control_period + 1)
        weights = np.linspace(0, 1, num_steps).reshape(-1,1)
        trajectory = weights * target + (1 - weights) * current

        # run trajectory
        for waypoint in trajectory:
            self.dxl_io.set_goal_position({i: a for (i,a) in zip(self.ids, waypoint)})
            time.sleep(control_period)

    def track_trajectory(self, trajectory, binsize=None, overshoot=None, ms_rpms = 0.165, fps=16):
        # !! works well with durations around 1/5 sec, but poorly with durations around 1/24 sec
        # trajectory = [..., (duration (sec), waypoint) ...]
        # waypoint[name] = angle (deg)
        # binsize: image binning (if None, does not save images)
        # overshoot: how many degrees to overshoot goal position (if None, does not overshoot)
        #    smooths trajectory with non-zero velocity at intermediate waypoints
        # rpm_unit: 1 moving speed unit = ms_rpms rotations per minute (docs say .114, but too fast)
        # fps: frames per second for images

        # sanity-check input
        for t, (dur, _) in enumerate(trajectory):
            assert dur >= .1, "duration of transition %d is %f, durations >= .2 work best" % (t, dur)

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
