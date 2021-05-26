import pybullet as pb
from pybullet_data import getDataPath
import pickle as pk
import os, time
import numpy as np

class PoppyErgoJrEnv(object):
    def __init__(self, control_mode, timestep=1/240, show=True):

        self.control_mode = control_mode
        self.timestep = timestep
        self.show = show

        self.client_id = pb.connect(pb.GUI if show else pb.DIRECT)
        pb.setTimeStep(timestep)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(getDataPath())
        pb.loadURDF("plane.urdf")

        fpath = os.path.dirname(os.path.abspath(__file__))
        fpath += '/../../urdfs/ergo_jr'
        pb.setAdditionalSearchPath(fpath)
        self.robot_id = pb.loadURDF(
            'poppy_ergo_jr.pybullet.urdf',
            basePosition = (0, 0, 0),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)),
            # globalScaling=0.01,
            useFixedBase=True)
        self.num_joints = pb.getNumJoints(self.robot_id)
        
        self.joint_name, self.joint_index = {}, {}
        for i in range(pb.getNumJoints(self.robot_id)):
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


if __name__ == '__main__':

    env = PoppyErgoJrEnv(pb.POSITION_CONTROL)
    N = len(env.joint_index)
    action = [0.]*N
    input('...')
    while True:
        env.step(action)
        time.sleep(0.01)


