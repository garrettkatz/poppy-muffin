import pybullet as pb
from pybullet_data import getDataPath
import pickle as pk
import time
import numpy as np

class PoppyHumanoidEnv(object):
    def __init__(self, control_mode, timestep=1/240, show=True):

        self.control_mode = control_mode
        self.timestep = timestep

        self.client_id = pb.connect(pb.GUI if show else pb.DIRECT)
        pb.setTimeStep(timestep)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(getDataPath())
        pb.loadURDF("plane.urdf")

        import os
        os.system('pwd')
        fpath = os.path.dirname(os.path.abspath(__file__))
        fpath += '/../../urdfs/humanoid'
        print(fpath)

        pb.setAdditionalSearchPath(fpath)
        self.robot_id = pb.loadURDF(
            'poppy_humanoid.pybullet.urdf',
            basePosition = (0, 0, .43),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)))
        
        self.joint_name, self.joint_index = {}, {}
        for i in range(pb.getNumJoints(self.robot_id)):
            name = pb.getJointInfo(self.robot_id, i)[1].decode('UTF-8')
            self.joint_name[i] = name
            self.joint_index[name] = i
        
        self.initial_state_id = pb.saveState(self.client_id)
    
    def reset(self):
        pb.restoreState(stateId = self.initial_state_id)
        
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
    
    def mirror_position(self, position):
        mirrored = np.empty(len(position))
        for i, name in self.joint_name.items():
            sign = 1 if name[-2:] == "_y" else -1 # don't negate y-axis rotations
            mirror_name = name # swap right and left
            if name[:2] == "l_": mirror_name = "r_" + name[2:]
            if name[:2] == "r_": mirror_name = "l_" + name[2:]
            mirrored[self.joint_index[mirror_name]] = position[i] * sign        
        return mirrored
            
# convert from physical robot angles to pybullet angles
# degrees are converted to radians
# other conversions are automatic from poppy_humanoid.pybullet.urdf
def convert_angles(angles):
    cleaned = {}
    for m,p in angles.items():
        cleaned[m] = p * np.pi / 180
    return cleaned

