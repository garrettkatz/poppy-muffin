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
            

def next_position(current, target, speed, timestep):
    v = target - current
    v *= speed / np.sum(v**2)**.5
    return current + v * timestep

if __name__ == '__main__':

    # angles = []
    # for fn in ["preleft", "leftup", "leftswing", "leftstep"]:
    #     with open("../../scripts/%s.pkl" % fn,"rb") as f:
    #         angles.append(pk.load(f))
    
    angles = [{}]

    env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
    N = len(env.joint_index)

    with open("../../scripts/crawl2.pkl","rb") as f:
        crawl_angles = pk.load(f)
    
    # from scripts:
    crawl_angles.update({
        'l_hip_y': -100., 'l_knee_y': 133., 'r_hip_y': -100., 'r_knee_y': 133.,
        'bust_x': 0., 'abs_z': 0., 'abs_y': 45., 'abs_x':0.,
        'r_elbow_y':-115., 'l_elbow_y':-115.,
        'r_shoulder_x':0., 'l_shoulder_x':0.,
        'r_ankle_y':45., 'l_ankle_y':45.,
        'r_hip_z':-10., 'l_hip_z':10.,
        'r_hip_x':0., 'l_hip_x':0.,})
    
    # pybullet specific:
    crawl_angles.update({
        'r_shoulder_x': -90, 'l_shoulder_x': -90,
        'r_shoulder_y': 45, 'l_shoulder_y': 45,})
    
    flips = ["l_hip_y", "l_ankle_y", "l_shoulder_y", "l_elbow_y"]
    
    for m,p in crawl_angles.items():
        s = 1 if m in flips else -1
        pb.resetJointState(env.robot_id, env.joint_index[m], s * p * np.pi / 180)
    pb.resetBasePositionAndOrientation(env.robot_id,
        (0, 0, .3),
        pb.getQuaternionFromEuler((.25*np.pi,0,0)))
    
    waypoints = np.zeros((len(angles), N))
    for a,angle in enumerate(angles):
        for m,p in angle.items():
            waypoints[a, env.joint_index[m]] = p * np.pi / 180
    
    # waypoints = np.array([
    #     [1. if i == env.joint_index["r_shoulder_x"] else 0. for i in range(N)],
    #     ])
    
    # while True:

    #     target = waypoints[0]
    #     duration = 1
    #     env.goto_position(target, duration)
    #     input("done...")
    #     waypoints = waypoints[1:]
    #     if len(waypoints) == 0: break
    
    input('.')
    action = env.current_position()
    j = env.joint_index["r_hip_y"]
    while True:
        env.step(action)
        # print(action[j])
        # print(env.current_position()[j])
        # input('.')

