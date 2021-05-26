import pybullet as pb
from pybullet_data import getDataPath
import time
import numpy as np

class PoppyEnv(object):

    # override this for urdf logic, should return robot pybullet id
    def load_urdf(self):
        return 0

    def __init__(self, control_mode, timestep=1/240, show=True):

        self.control_mode = control_mode
        self.timestep = timestep
        self.show = show

        self.client_id = pb.connect(pb.GUI if show else pb.DIRECT)
        pb.setTimeStep(timestep)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(getDataPath())
        pb.loadURDF("plane.urdf")
        
        # use overridden loading logic
        self.robot_id = self.load_urdf()

        self.num_joints = pb.getNumJoints(self.robot_id)        
        self.joint_name, self.joint_index = {}, {}
        for i in range(self.num_joints):
            name = pb.getJointInfo(self.robot_id, i)[1].decode('UTF-8')
            self.joint_name[i] = name
            self.joint_index[name] = i
        
        self.initial_state_id = pb.saveState(self.client_id)
    
    def reset(self):
        pb.restoreState(stateId = self.initial_state_id)
        
    def step(self, action, sleep=None):
        if sleep is None: sleep = self.show
        pb.setJointMotorControlArray(
            self.robot_id,
            jointIndices = range(len(self.joint_index)),
            controlMode = self.control_mode,
            targetPositions = action,
        )
        pb.stepSimulation()
        if sleep: time.sleep(self.timestep)
    
    # get/set joint angles as np.array
    def get_position(self):
        states = pb.getJointStates(self.robot_id, range(len(self.joint_index)))
        return np.array([state[0] for state in states])    
    def set_position(self, position):
        for p, angle in enumerate(position):
            pb.resetJointState(self.robot_id, p, angle)

    # convert a pypot style dictionary {... name:angle ...} to joint angle array
    # if convert == True, convert from degrees to radians
    def angle_array(self, angle_dict, convert=True):
        angle_array = np.zeros(self.num_joints)
        for name, angle in angle_dict.items():
            angle_array[self.joint_index[name]] = angle
        if convert: angle_array *= np.pi / 180
        return angle_array

    # pypot-style command, goes to position in give duration
    # target is a joint angle array
    # duration is desired duration of motion
    # if hang==True, wait for user enter at each timestep of motion
    def goto_position(self, target, duration, hang=False):
        current = self.get_position()
        num_steps = int(duration / self.timestep + 1)
        weights = np.linspace(0, 1, num_steps).reshape(-1,1)
        trajectory = weights * target + (1 - weights) * current

        positions = np.empty((num_steps, self.num_joints))
        for a, action in enumerate(trajectory):
            self.step(action)
            positions[a] = self.get_position()
            if hang: input('..')

        return positions
    

