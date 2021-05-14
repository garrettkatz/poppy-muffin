import pybullet as pb
from pybullet_data import getDataPath
import time
import numpy as np

class PoppyHumanoidEnv(object):
    def __init__(self, control_mode):
        self.control_mode = control_mode

        pb.connect(pb.GUI)
        pb.setGravity(0, 0, -9.81)

        pb.setAdditionalSearchPath(getDataPath())
        pb.loadURDF("plane.urdf")

        pb.setAdditionalSearchPath('./urdf')
        self.poppyid = pb.loadURDF(
            'poppy_humanoid.urdf',
            basePosition = (0, 0, .41),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)))

    def step(self, action):
        pb.setJointMotorControlArray(
            self.poppyid,
            jointIndices = range(pb.getNumJoints(self.poppyid)),
            controlMode = self.control_mode,
            targetPositions = action,
        )
        pb.stepSimulation()
        time.sleep(0.01)

if __name__ == '__main__':
    env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
    action = [0] * pb.getNumJoints(env.poppyid)
    for _ in range(20000):
        env.step(action)
