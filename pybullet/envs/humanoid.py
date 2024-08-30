import pybullet as pb
import os, time
import numpy as np
from poppy_env import PoppyEnv

class PoppyHumanoidEnv(PoppyEnv):
    
    # Humanoid-specific urdf loading logic
    def load_urdf(self):
        fpath = os.path.dirname(os.path.abspath(__file__))
        fpath += '/../../urdfs/humanoid'
        pb.setAdditionalSearchPath(fpath)
        robot_id = pb.loadURDF(
            'poppy_humanoid.pybullet.urdf',
            basePosition = (0, 0, .43),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)))
        return robot_id

    # Get mirrored version of position across left/right halves of body
    def mirror_position(self, position):
        mirrored = np.empty(len(position))
        for i, name in self.joint_name.items():
            sign = 1 if name[-2:] == "_y" else -1 # don't negate y-axis rotations
            mirror_name = name # swap right and left
            if name[:2] == "l_": mirror_name = "r_" + name[2:]
            if name[:2] == "r_": mirror_name = "l_" + name[2:]
            mirrored[self.joint_index[mirror_name]] = position[i] * sign        
        return mirrored

    # version of mirror_position for angle dict input and output
    def mirror_dict(self, position_dict):
        mirrored = {}
        for (name, angle) in position_dict.items():
            sign = 1 if name[-2:] == "_y" else -1 # don't negate y-axis rotations
            mirror_name = name # swap right and left
            if name[:2] == "l_": mirror_name = "r_" + name[2:]
            if name[:2] == "r_": mirror_name = "l_" + name[2:]
            mirrored[mirror_name] = angle * sign
        return mirrored
    
    # override step and goto for humanoid walking
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
    def goto_position(self, target, duration, hang=False):
        current = self.get_position()
        num_steps = int(duration / self.timestep + 1)
        weights = np.linspace(0, 1, num_steps).reshape(-1,1)
        trajectory = weights * target + (1 - weights) * current

        positions = np.empty((num_steps, self.num_joints))
        for a, action in enumerate(trajectory):
            self.step(action)
            positions[a] = self.get_position()
            if hang: input('..')

        return positions

# convert from physical robot angles to pybullet angles
# degrees are converted to radians
# other conversions are automatic from poppy_humanoid.pybullet.urdf
def convert_angles(angles):
    cleaned = {}
    for m,p in angles.items():
        cleaned[m] = p * np.pi / 180
    return cleaned

if __name__ == "__main__":
    
    env = PoppyHumanoidEnv(pb.POSITION_CONTROL)
    input('...')
