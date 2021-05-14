import pybullet as pb
from pybullet_data import getDataPath
import pickle as pk
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

    angles = []
    for fn in ["preleft", "leftup", "leftswing", "leftstep"]:
        with open("../../scripts/%s.pkl" % fn,"rb") as f:
            angles.append(pk.load(f))

    env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
    N = len(env.joint_index)
    
    waypoints = []
    for a in angles:
        action = []
        for i in range(N):
            action.append(a[env.joint_name[i]] * np.pi / 360)
        waypoints.append(action)

    tol = .05
    counter, max_counter = 0, 100
    while True:
        joints = [tup[0] for tup in pb.getJointStates(env.poppyid, range(N))]
        action = waypoints[0]
        env.step(action)
        
        diff = np.fabs(np.array(joints) - np.array(action)).max()
        if diff < tol and counter == max_counter and len(waypoints) > 1:
            waypoints = waypoints[1:]
            counter = 0

        counter += 1        
        time.sleep(0.01)


