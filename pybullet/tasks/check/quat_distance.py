import numpy as np
import pybullet as pb

q1 = pb.getQuaternionFromEuler((0, 0, 0))
q2 = pb.getQuaternionFromEuler((0.001, 0.001, 0.001))

print(q1)
print(q2)

d = pb.getDifferenceQuaternion(q1, q2)
theta = 2*np.arccos(abs(sum([v1*v2 for (v1, v2) in zip(q1, q2)])))
print(theta)
