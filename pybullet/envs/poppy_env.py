import pybullet as pb
from pybullet_data import getDataPath
import time
import numpy as np

class PoppyEnv(object):

    # override this for urdf logic, should return robot pybullet id
    def load_urdf(self):
        return 0

    def __init__(self,
        control_mode=pb.POSITION_CONTROL,
        timestep=1/240,
        control_period=1,
        show=True,
        step_hook=None
    ):

        # step_hook(env, action) is called in each env.step(action)
        if step_hook is None: step_hook = lambda env, action: None

        self.control_mode = control_mode
        self.timestep = timestep
        self.control_period = control_period
        self.show = show
        self.step_hook = step_hook

        self.client_id = pb.connect(pb.GUI if show else pb.DIRECT)
        if show: pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
        pb.setTimeStep(timestep)
        pb.setGravity(0, 0, -9.81)
        pb.setAdditionalSearchPath(getDataPath())
        pb.loadURDF("plane.urdf")
        
        # use overridden loading logic
        self.robot_id = self.load_urdf()

        self.num_joints = pb.getNumJoints(self.robot_id)
        self.joint_name, self.joint_index, self.joint_fixed = {}, {}, {}
        for i in range(self.num_joints):
            info = pb.getJointInfo(self.robot_id, i)
            name = info[1].decode('UTF-8')
            self.joint_name[i] = name
            self.joint_index[name] = i
            self.joint_fixed[i] = (info[2] == pb.JOINT_FIXED)
        
        self.initial_state_id = pb.saveState(self.client_id)
    
    def reset(self):
        pb.restoreState(stateId = self.initial_state_id)
    
    def close(self):
        pb.disconnect()
        
    def step(self, action=None, sleep=None):
        
        self.step_hook(self, action)
    
        if action is not None:
            duration = self.control_period * self.timestep
            distance = np.fabs(action - self.get_position())
            pb.setJointMotorControlArray(
                self.robot_id,
                jointIndices = range(len(self.joint_index)),
                controlMode = self.control_mode,
                targetPositions = action,
                targetVelocities = [0]*len(action),
                positionGains = [.25]*len(action), # important for constant position accuracy
                # maxVelocities = distance / duration,
            )

        if sleep is None: sleep = self.show
        if sleep:
            for _ in range(self.control_period):
                start = time.perf_counter()
                pb.stepSimulation()
                duration = time.perf_counter() - start
                remainder = self.timestep - duration
                if remainder > 0: time.sleep(remainder)
        else:
            for _ in range(self.control_period):
                pb.stepSimulation()
    
    # get/set joint angles as np.array
    def get_position(self):
        states = pb.getJointStates(self.robot_id, range(len(self.joint_index)))
        return np.array([state[0] for state in states])    
    def set_position(self, position):
        for p, angle in enumerate(position):
            pb.resetJointState(self.robot_id, p, angle)

    # convert a pypot style dictionary {... name:angle ...} to joint angle array
    # if convert == True, convert from degrees to radians
    def angle_array(self, angle_dict, convert=True):
        angle_array = np.zeros(self.num_joints)
        for name, angle in angle_dict.items():
            angle_array[self.joint_index[name]] = angle
        if convert: angle_array *= np.pi / 180
        return angle_array
    # convert back from dict to array
    def angle_dict(self, angle_array, convert=True):
        return {
            name: angle_array[j] * 180/np.pi
            for j, name in enumerate(self.joint_index)}

    # pypot-style command, goes to position in give duration
    # target is a joint angle array
    # speed is desired joint speed
    # if hang==True, wait for user enter at each timestep of motion
    def goto_position(self, target, speed=1., hang=False):

        current = self.get_position()
        distance = np.sum((target - current)**2)**.5
        duration = distance / speed

        num_steps = int(duration / (self.timestep * self.control_period) + 1)
        weights = np.linspace(0, 1, num_steps).reshape(-1,1)
        trajectory = weights * target + (1 - weights) * current

        positions = np.empty((num_steps, self.num_joints))
        for a, action in enumerate(trajectory):
            self.step(action)
            positions[a] = self.get_position()
            if hang: input('..')

        return positions
    
    # Run IK, accounting for fixed joints
    def inverse_kinematics(self, link_indices, target_positions, num_iters=1000):

        angles = pb.calculateInverseKinematics2(
            self.robot_id,
            link_indices,
            target_positions,
            # residualThreshold=1e-4, # default 1e-4 not enough for ergo jr
            maxNumIterations=num_iters, # default 20 usually not enough
        )

        a = 0
        result = self.get_position()
        for r in range(self.num_joints):
            if not self.joint_fixed[r]:
                result[r] = angles[a]
                a += 1
        
        return result


