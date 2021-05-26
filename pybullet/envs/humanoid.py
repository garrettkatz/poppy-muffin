import pybullet as pb
import os
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
