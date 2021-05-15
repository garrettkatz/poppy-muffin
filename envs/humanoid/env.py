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
    
    # pybullet specific (not compliant):
    crawl_angles.update({'r_shoulder_y': 45, 'l_shoulder_y': 45,})
    
    angles = [crawl_angles]
    
    angle_updates = [
        {'l_shoulder_x': -13., 'l_hip_z': 20., 'r_hip_z':-20., 'bust_x':-10.}, # lean left
        {'r_shoulder_x':12., 'abs_x': -3., 'abs_y': 37., 'abs_z': -15., 'r_elbow_y':-90.}, # rotate torso
        {'l_elbow_y':-90., 'abs_z': 0., 'l_shoulder_y': 0}, # left elbow perp
        {'l_shoulder_x':0., 'r_shoulder_x':0.},
        {'r_elbow_y': -115., 'l_elbow_y': -115.,'l_hip_y': -70., 'l_knee_y': 100.,'r_hip_y': -70., 'r_knee_y': 100., 'l_shoulder_y': 45}, # forward thighs vertical
        {'r_hip_z': 5., 'r_hip_y': -95., 'l_hip_y': -95., 'l_hip_z': 18., 'abs_z': 18., 'l_knee_y': 100., 'r_knee_y': 100.}, # hip raised
        {'r_hip_z': -2., 'r_hip_y': -100., 'l_hip_y': -70., 'l_hip_z': 2., 'abs_z': -15., 'l_knee_y': 100., 'r_knee_y': 125.}, # right leg forward
        {'r_hip_z': 0., 'r_hip_y': -100., 'l_hip_y': -100., 'l_hip_z': 0., 'abs_z': 0., 'l_knee_y': 125., 'r_knee_y': 125.}, # left leg forward
    ]
    
    for changes in angle_updates:
        angles.append(dict(angles[-1]))
        angles[-1].update(changes)
        
    waypoints = np.zeros((len(angles), N))
    for a,angle in enumerate(angles):
        cleaned = clean_angles(angle)
        for m,p in cleaned.items():
            waypoints[a, env.joint_index[m]] = p

    # initial angles/position
    for p, position in enumerate(waypoints[-1]):
        pb.resetJointState(env.robot_id, p, position)
    pb.resetBasePositionAndOrientation(env.robot_id,
        (0, 0, .3),
        pb.getQuaternionFromEuler((.25*np.pi,0,0)))
    
    # waypoints = np.array([
    #     [1. if i == env.joint_index["r_shoulder_x"] else 0. for i in range(N)],
    #     ])
    

    input("ready...")
    w = 0
    while True:

        target = waypoints[w]
        duration = .25
        env.goto_position(target, duration)
        w = (w + 1) % len(waypoints)
    
    # input('.')
    # action = env.current_position()
    # j = env.joint_index["l_elbow_y"]
    # while True:
    #     env.step(action)
    #     # print(action[j])
    #     # print(env.current_position()[j])
    #     # input('.')

