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
        Return joint_info, a tuple of joint information
        joint_info[i] = (joint_name, parent_index, translation, orientation, axis) for ith joint
        joint_name: string identifier for joint
        parent_index: index in joint_info of parent (-1 is base)
        translation: (3,) translation vector in parent joint's local frame
        orientation: (4,) orientation quaternion in parent joint's local frame
        axis: (3,) rotation axis vector in ith joint's own local frame (None for fixed joints)
        """
        return tuple(zip(self.joint_name, self.parent_index, self.translation, self.orientation, self.axis))

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


        
