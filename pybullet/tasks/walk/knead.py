"""
Run this after generating knead_grid.pkl in poindexter and copying to this directory
"""
import itertools as it
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import pybullet as pb
from time import perf_counter
import sys, os
sys.path.append('../../envs')
from humanoid import PoppyHumanoidEnv, convert_angles

np.set_printoptions(linewidth=1000)

SETTLE = 4

def run_gait(env, waypoints, num_cycles, timestep):
    # each cycle is two steps (left and right)
    # timestep is duration of each waypoint transition
    # waypoints[0] is straight knees
    # waypoints[-1] is mirror of waypoints[1]

    # start fresh
    env.reset()

    # initial pose
    env.place_on_ground(waypoints[1], duration=SETTLE)
    init_pos, init_quat = env.get_base()

    # start walking
    for step in range(num_cycles*2):

        for waypoint in waypoints[2:]:
            if step % 2 == 1: waypoint = env.mirror_position(waypoint)
            env.track_trajectory([(timestep, env.angle_dict(waypoint))])
    
        # settle at last waypoint
        env.track_trajectory([(SETTLE, env.angle_dict(waypoint))])

    # done
    term_pos, term_quat = env.get_base()
    return init_pos, init_quat, term_pos, term_quat

if __name__ == "__main__":

    show = False
    num_cycles = 3
    timestep = 0.3
    relative_fall_tolerance = 0.05
    do_runs = False

    with open("knead_grid.pkl", "rb") as f:
        (param_ranges, success, waypoints) = pk.load(f)

    env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=show)

    if do_runs:

        base_info = {}
        start = perf_counter()
        for p, params in enumerate(success.keys()):
            if not success[params]: continue
    
            # run the gait
            base_info[params] = run_gait(env, waypoints[params], num_cycles, timestep)
            init_pos, init_quat, term_pos, term_quat = base_info[params]
    
            # check fall
            is_fall = (np.fabs(init_pos[2] - term_pos[2]) / init_pos[2]) > relative_fall_tolerance
    
            remaining = (perf_counter() - start) / (p+1) * (len(success) - (p+1)) / 60 / 60
            print(f"{p} of {len(success)}: {is_fall=} [~{remaining:.2f}h remaining]")
    
            if p % 10 == 1:
                with open(f"knead_runs_settle{SETTLE}.pkl", "wb") as f: pk.dump(base_info, f)
    
            # if p == 10: break    
            # input('.')
    
        with open(f"knead_runs_settle{SETTLE}.pkl", "wb") as f: pk.dump(base_info, f)

    env.close()

    with open(f"knead_runs_settle{SETTLE}.pkl", "rb") as f: base_info = pk.load(f)

    # organize results
    index_grid = np.array(list(it.product(*(range(len(r)) for r in param_ranges))))
    param_grid = np.array(list(it.product(*param_ranges)))
    falls = np.empty(len(param_grid), dtype=bool)
    rots = np.empty((len(param_grid), 3))
    rot_angs = np.empty(len(param_grid))
    term_pos = np.empty((len(param_grid), 3))

    for p, params in enumerate(param_grid):
        init_pos, init_quat, term_pos[p], term_quat = base_info[tuple(params)]

        falls[p] = (np.fabs(init_pos[2] - term_pos[p,2]) / init_pos[2]) > relative_fall_tolerance

        diff_quat = pb.getDifferenceQuaternion(init_quat, term_quat)
        axis, ang = pb.getAxisAngleFromQuaternion(diff_quat)
        rots[p] = np.array(axis) * ang
        rot_angs[p] = ang

    stands = ~falls

    print(f"{stands.sum()} stand, {falls.sum()} fall of {len(falls)} ({stands.mean()} success rate)")

    # fall robustness at each point
    robustness = np.zeros(len(stands))
    stand_idx = np.flatnonzero(stands)
    for p in stand_idx:
        # # shortest distance to a fall
        # dists = np.fabs(index_grid[p] - index_grid).max(axis=1) # all robusts 1
        # dists = np.fabs(index_grid[p] - index_grid).sum(axis=1) # a small fraction > 1
        dists = np.sum((index_grid[p] - index_grid)**2, axis=1)**.5 # a small fraction > 1
        robustness[p] = dists[falls].min()

        # average chance of fall at radius 1
        rad1 = np.fabs(index_grid[p] - index_grid).max(axis=1) == 1
        robustness[p] = falls[rad1].mean()

        # expected distance to a fall...

    print(f"{robustness[stands].min()}|{robustness[stands].mean()}|{robustness[stands].max()} min|mean|max stand robustness")

    # most robust params
    print("robust params: ", param_grid[robustness.argmax()])

    import matplotlib.pyplot as pt

    # # plot base poses in 3D
    # init_pos = np.stack([val[0] for val in base_info.values()])
    # ax = pt.gcf().add_subplot(projection='3d')
    # ax.plot(*term_pos.T, marker="+", color="k", linestyle="none")
    # ax.plot(*init_pos.T, marker=".", color="g", linestyle="none")
    # ax.quiver(*term_pos.T, *rots.T, length=0.1, normalize=True)
    # pt.show()

    # plot robustness vs base rotation angle
    pt.subplot(1,2,1)
    pt.plot(robustness[stands], rot_angs[stands], 'k.')
    pt.plot(robustness[robustness.argmax()], rot_angs[robustness.argmax()], 'go')
    pt.xlabel("Robust")
    pt.ylabel("Angle")
    pt.subplot(1,2,2)
    pt.plot(robustness[stands], term_pos[stands, 1], 'k.')
    pt.plot(robustness[robustness.argmax()], term_pos[robustness.argmax(), 1], 'go')
    pt.xlabel("Robust")
    pt.ylabel("Forward")
    pt.show()

    ## write the most robust trajectory to hardware format
    params = param_grid[robustness.argmax()]

    # populate trajectory
    trajectory = [(0., env.angle_dict(waypoints[tuple(params)][1]))] # initial pose
    for step in range(num_cycles*2):
        for waypoint in waypoints[tuple(params)][2:]:
            if step % 2 == 1: waypoint = env.mirror_position(waypoint)
            trajectory.append( (timestep, env.angle_dict(waypoint)) )

        # linger at last waypoint between steps
        trajectory.append( (1., env.angle_dict(waypoint)) )
    
    with open(os.environ["HOME"] + '/poppy-muffin/scripts/knead_trajectory.pkl', 'wb') as f:
        pk.dump(trajectory, f, protocol=2)


    # with open(os.environ["HOME"] + "/poindexter/knead.pkl", "rb") as f: waypoints = pk.load(f)
    # env = PoppyHumanoidEnv(pb.POSITION_CONTROL, show=True)
    # base_info = run_gait(env, waypoints, num_cycles, timestep)

    # env.place_on_ground(waypoints[1], duration=10)
    # state = pb.getLinkState(env.robot_id, env.joint_index["r_ankle_y"])
    # current_ankle_z = state[0][2]
    # print("current ankle z", current_ankle_z)
    # input('.')

    # for step in range(num_steps):

    #     for waypoint in waypoints[2:]:
    #         if step % 2 == 1: waypoint = env.mirror_position(waypoint)
    #         env.track_trajectory([(.3, env.angle_dict(waypoint))])
    #         # # briefly settle at current waypoint
    #         # env.track_trajectory([(.05, env.angle_dict(waypoint))])
    #         state = pb.getLinkState(env.robot_id, env.joint_index["l_ankle_y"])
    #         current_ankle_z = state[0][2]
    #         print("current ankle z", current_ankle_z)
    #         # input('.')
    
    #     # settle at last waypoint
    #     env.track_trajectory([(3., env.angle_dict(waypoint))])
    #     state = pb.getLinkState(env.robot_id, env.joint_index["l_ankle_y"])
    #     current_ankle_z = state[0][2]
    #     print("current ankle z", current_ankle_z)    
    #     # input('.')

    # input(']')
    
    # ## try straightening legs quickly to achieve slight liftoff ground
    # env.place_on_ground(waypoints[1])
    # state = pb.getLinkState(env.robot_id, env.joint_index["l_ankle_y"])
    # current_ankle_z = state[0][2]
    # print("current ankle z", current_ankle_z)

    # input('.')

    # env.track_trajectory([(.15, env.angle_dict(waypoints[0]))])
    # state = pb.getLinkState(env.robot_id, env.joint_index["l_ankle_y"])
    # current_ankle_z = state[0][2]
    # print("current ankle z", current_ankle_z)

    # input('.')

    # env.track_trajectory([(2, env.angle_dict(waypoints[0]))])
    # state = pb.getLinkState(env.robot_id, env.joint_index["l_ankle_y"])
    # current_ankle_z = state[0][2]
    # print("current ankle z", current_ankle_z)

    # input('.')

