import pybullet as pb
from pybullet_data import getDataPath
import pickle as pk
import time
import numpy as np

class PoppyHumanoidEnv(object):
    def __init__(self, control_mode, timestep=1/240):

        self.control_mode = control_mode
        self.timestep = timestep

        pb.connect(pb.GUI)
        pb.setTimeStep(timestep)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(getDataPath())
        pb.loadURDF("plane.urdf")

        import os
        os.system('pwd')
        print(os.path.dirname(os.path.abspath(__file__)))
        fpath = os.path.dirname(os.path.abspath(__file__))
        fpath += '/../../urdfs/humanoid'
        print(fpath)
        pb.setAdditionalSearchPath(fpath)
        self.robot_id = pb.loadURDF(
            'poppy_humanoid.pybullet.urdf',
            basePosition = (0, 0, .41),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)))
        
        self.joint_name, self.joint_index = {}, {}
        for i in range(pb.getNumJoints(self.robot_id)):
            name = pb.getJointInfo(self.robot_id, i)[1].decode('UTF-8')
            self.joint_name[i] = name
            self.joint_index[name] = i
        
    def step(self, action, sleep=True):
        pb.setJointMotorControlArray(
            self.robot_id,
            jointIndices = range(len(self.joint_index)),
            controlMode = self.control_mode,
            targetPositions = action,
        )
        pb.stepSimulation()
        if sleep: time.sleep(self.timestep)
    
    def current_position(self):
        states = pb.getJointStates(self.robot_id, range(len(self.joint_index)))
        return np.array([state[0] for state in states])
    
    def set_position(self, position):
        # position: np.array
        for p, angle in enumerate(position):
            pb.resetJointState(self.robot_id, p, angle)

    def goto_position(self, target, duration, hang=False):
        current = self.current_position()
        num_steps = int(duration / self.timestep + 1)
        weights = np.linspace(0, 1, num_steps).reshape(-1,1)
        trajectory = weights * target + (1 - weights) * current
        for action in trajectory:
            self.step(action)
            if hang: input('..')
            
# convert from physical robot angles to pybullet angles
# degrees are converted to radians
# other conversions are automatic from poppy_humanoid.pybullet.urdf
def convert_angles(angles):
    cleaned = {}
    for m,p in angles.items():
        cleaned[m] = p * np.pi / 180
    return cleaned

