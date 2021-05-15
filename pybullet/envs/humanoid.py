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
        # pb.setAdditionalSearchPath('../urdfs/humanoid')
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

    def goto_position(self, target, duration):
        current = self.current_position()
        num_steps = int(duration / self.timestep + 1)
        weights = np.linspace(0, 1, num_steps).reshape(-1,1)
        trajectory = weights * target + (1 - weights) * current
        for action in trajectory: self.step(action)
            
def clean_angles(angles):
    # transforms angles measured on real poppy for pybullet
    # angles should be {name: degrees} dictionary
    # shoulder_x off by 90 degrees
    # also, required lower/uppers switched and negated in urdf,
    # except for preserved joints below
    # all other joint angles need to be negated
    # all angles also converted to radians

    preserved = ["l_hip_y", "l_ankle_y", "l_shoulder_y", "l_elbow_y"]
    cleaned = {}
    for m,p in angles.items():
        if m[2:] == 'shoulder_x': p -= 90
        if m not in preserved: p = -p
        cleaned[m] = p * np.pi / 180
    return cleaned

