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

        pb.setAdditionalSearchPath('../../urdfs/humanoid')
        self.robot_id = pb.loadURDF(
            'poppy_humanoid.urdf',
            basePosition = (0, 0, .41),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)))
        
        self.joint_name, self.joint_index = {}, {}
        for i in range(pb.getNumJoints(self.robot_id)):
            name = pb.getJointInfo(self.robot_id, i)[1].decode('UTF-8')
            self.joint_name[i] = name
            self.joint_index[name] = i
        
    def step(self, action):
        pb.setJointMotorControlArray(
            self.robot_id,
            jointIndices = range(len(self.joint_index)),
            controlMode = self.control_mode,
            targetPositions = action,
        )
        pb.stepSimulation()

def next_position(current, target, speed, timestep):
    v = target - current
    v *= speed / np.sum(v**2)**.5
    return current + v * timestep

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
    waypoints = np.array(waypoints)
    
    # waypoints = np.array([
    #     [1. if i == env.joint_index["r_shoulder_x"] else 0. for i in range(N)],
    #     ])
    
    tol = .05
    speed = 10
    timestep = 1/240
    while True:
        current = np.array([tup[0] for tup in pb.getJointStates(env.robot_id, range(N))])
        target = waypoints[0]
        diff = np.fabs(target - current).max()
        if diff < tol:
            if len(waypoints) > 1:
                waypoints = waypoints[1:]
                target = waypoints[0]
            else:
                break

        action = next_position(current, target, speed, timestep)
        print(str(action))
        env.step(action)

        time.sleep(0.01)

