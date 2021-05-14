import pybullet as pb
from pybullet_data import getDataPath
import pickle as pk
import time
import numpy as np

class PoppyErgoJrEnv(object):
    def __init__(self, control_mode):

        self.control_mode = control_mode

        pb.connect(pb.GUI)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(getDataPath())
        pb.loadURDF("plane.urdf")

        pb.setAdditionalSearchPath('../../urdfs/ergo_jr')
        self.poppyid = pb.loadURDF(
            'poppy_ergo_jr.urdf',
            basePosition = (0, 0, 0),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)),
            globalScaling=0.01,
            useFixedBase=True)
        
        self.joint_name, self.joint_index = {}, {}
        for i in range(pb.getNumJoints(self.poppyid)):
            name = pb.getJointInfo(self.poppyid, i)[1].decode('UTF-8')
            self.joint_name[i] = name
            self.joint_index[name] = i

    def step(self, action):
        pb.setJointMotorControlArray(
            self.poppyid,
            jointIndices = range(pb.getNumJoints(self.poppyid)),
            controlMode = self.control_mode,
            targetPositions = action,
            targetVelocities = [0.001]*len(action),
        )
        pb.stepSimulation()

if __name__ == '__main__':

    env = PoppyErgoJrEnv(pb.POSITION_CONTROL)
    N = len(env.joint_index)
    action = [0.]*N
    input('...')
    while True:
        env.step(action)
        time.sleep(0.01)


