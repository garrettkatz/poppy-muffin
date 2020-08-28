import pybullet as p
import time
import numpy as np
import pybullet_data

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("poppy_ergo_jr.urdf",cubeStartPos, cubeStartOrientation, globalScaling=0.01)
num_steps = 4000
num_joints = p.getNumJoints(boxId)
for i in range (num_steps):
    # p.setJointMotorControlArray(boxId, np.arange(6), p.POSITION_CONTROL,
    #     targetPositions=np.ones(6))
    p.setJointMotorControlArray(boxId, np.arange(num_joints), p.POSITION_CONTROL,
        targetPositions=np.ones(num_joints) * i / num_steps) # partial
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
# print("joints",p.getNumJoints(boxId)) # 6
p.disconnect()


