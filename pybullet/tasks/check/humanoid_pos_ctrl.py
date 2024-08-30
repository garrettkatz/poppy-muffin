from time import perf_counter
import numpy as np
import pybullet as pb
import sys
sys.path.append('../../envs')
from humanoid import PoppyHumanoidEnv, convert_angles

def goto_pos(env, angle_array, duration):
    action = angle_array
    num_steps = int(np.ceil(duration / (env.timestep * env.control_period) + 1))
    positions = np.empty((num_steps, env.num_joints))
    pb.setJointMotorControlArray(
        env.robot_id,
        jointIndices = range(len(env.joint_index)),
        controlMode = env.control_mode,
        targetPositions = action,
        targetVelocities = [0]*len(action),
        positionGains = [.25]*len(action), # important for constant position accuracy
    )
    for t in range(num_steps):
        for _ in range(env.control_period):
            pb.stepSimulation()
        positions[t] = env.get_position()
    return positions

if __name__ == "__main__":

    env = PoppyHumanoidEnv(pb.POSITION_CONTROL, control_period=10, show=True)

    buffers = []

    trajectory = [(5., {name: 0. for name in env.joint_name.values()})]
    positions = env.track_trajectory(trajectory)
    buffers.append(positions)

    # also check ankle z coordinates when flat on ground
    ankle_z = []
    for lr in "lr":
        state = pb.getLinkState(env.robot_id, env.joint_index[lr + "_ankle_y"])
        ankle_z.append(state[0][2])
    print('ankle_z')
    print(ankle_z)
    input('..')

    trajectory[0][1]["r_shoulder_y"] = 3.
    positions = env.track_trajectory(trajectory)
    buffers.append(positions)

    positions = np.concatenate(buffers, axis=0) * 180. / np.pi

    # target = np.zeros(len(env.joint_name))

    # positions = goto_pos(env, target, 5.)

    # target[env.joint_index["r_shoulder_y"]] = 1.
    # positions = goto_pos(env, target, 5.)
    
    # input('.')
    
    # start_time = perf_counter()
    # while True:
    #     env.step(target)
    #     if perf_counter() - start_time > duration: break
    
    # input('.')
    env.close()

    import matplotlib.pyplot as pt
    pt.plot(positions)
    pt.show()
    
