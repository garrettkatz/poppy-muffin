import pybullet as pb
import os
import numpy as np
from poppy_env import PoppyEnv

class PoppyErgoJrEnv(PoppyEnv):
    def load_urdf(self):
        fpath = os.path.dirname(os.path.abspath(__file__))
        fpath += '/../../urdfs/ergo_jr'
        pb.setAdditionalSearchPath(fpath)
        robot_id = pb.loadURDF(
            'poppy_ergo_jr.pybullet.urdf',
            basePosition = (0, 0, 0),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)),
            useFixedBase=True)
        return robot_id

if __name__ == '__main__':

    env = PoppyErgoJrEnv(pb.POSITION_CONTROL)
    action = [0.]*env.num_joints
    input('...')
    while True:
        env.step(action)
        time.sleep(0.01)


