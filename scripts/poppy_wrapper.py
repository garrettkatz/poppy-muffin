import pickle as pk
import signal
import numpy as np

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
        self.num_joints = len(self.motors)

    def close(self):
        self.robot.close()
        if self.camera is not None: self.camera.close()

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

    def goto_position(self, angles, duration=1.0, bufsize=10, speed_ratio=.5, motion_window=.5, binsize=None):
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

                # check safe voltage/temperature
                if not self.is_safe():
                    print("Unsafe motion!!!")
                    success = False
    
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
                    print("%d: %.3fs elapsed, timepoint=%.3fs" % (t, time_elapsed[t], time_points[t]))
                    time.sleep(max(0, time_points[t+1] - time_elapsed[t]))
    
        # Dump partial buffers to file on error
        except OSError:
            buffers = {key: buf[:t] for (key, buf) in buffers.items()}
            with open("gobuf.pkl","wb") as f: pk.dump((success, buffers, time_elapsed, motor_names), f)
            success = False

        # restore default interrupt handler
        finally:
            signal.signal(signal.SIGINT, default_interrupt_handler)

        return success, buffers, time_elapsed
        
    
