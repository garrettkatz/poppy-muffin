import pybullet as pb
import os, time
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
    
    def get_tip_positions(self):
        states = pb.getLinkStates(self.robot_id, [5, 7])
        return (states[0][0], states[1][0])
    
    # def get_camera_image(self):
    #     import pickle as pk
    #     while True:
    #         cam = pb.getDebugVisualizerCamera()
    #         width, height, view, proj, camup, camfwd, horz, vert, yaw, pitch, dist, targ = cam
    #         view = pb.computeViewMatrix(
    #             cameraEyePosition=(0,0,.01),
    #             cameraTargetPosition=(0,-.2,.01),
    #             cameraUpVector=(0,0,1)
    #         )
    #         proj = pb.computeProjectionMatrixFOV(
    #             fov=60,
    #             aspect = 1024/768,
    #             nearVal=-.01,
    #             farVal=-.2,
    #         )
    #         width, height = 1024, 768
    #         print(view)
    #         print(proj)
    #         print(width, height)
    #         _, _, rgb, depth, segment = pb.getCameraImage(width, height, view, proj)
    #         with open("ergo_jr_pov.pkl","wb") as f: pk.dump((width, height, view, proj), f)            
    #         input("ready...")

if __name__ == '__main__':

    env = PoppyErgoJrEnv(pb.POSITION_CONTROL)

    # from check/camera.py
    pb.resetDebugVisualizerCamera(
        1.2000000476837158, 56.799964904785156, -22.20000648498535,
        (-0.6051651835441589, 0.26229506731033325, -0.24448847770690918))

    action = [0.]*env.num_joints
    action[env.joint_index['m6']] = .5
    while True:
        env.step(action)


