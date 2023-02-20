import pickle as pk
import numpy as np
import poppy_wrapper as pw
try:
    from pypot.creatures import PoppyHumanoid as PH
    do_run = True
except:
    from mocks import PoppyHumanoid as PH
    do_run = False

# joints involved in walking
walk_names = ["%s_%s_y" % (lr, jnt) for lr in "lr" for jnt in ("hip", "ankle", "knee")]
walk_names += ["abs_y","bust_y"]
# walk_names += ["%s_hip_%s" % (lr, ax) for lr in "lr" for ax in "xz"]

# load planned trajectory
with open('pypot_traj1.pkl', "rb") as f: trajs = pk.load(f)
# get initial angles
init_angles = trajs[0][0][1]
# traj = trajs[0]
# durations, angles = zip(*traj)

# Run and don't show
if do_run:

    poppy = pw.PoppyWrapper(PH())
    
    # for python 2.7 on poppy
    if hasattr(__builtins__, 'raw_input'):
        input=raw_input
    
    input('[Enter] to turn off compliance')
    for m in poppy.motors: m.compliant = False

    # PID tuning
    K_p, K_i, K_d = 18.0, 4.0, 0.1
    # K_p, K_i, K_d = 20.0, 0.0, 0.0
    # for name in walk_names:
    #     getattr(poppy.robot, name).pid = (K_p, K_i, K_d)
    for m in poppy.motors:
        if hasattr(m, 'pid'): m.pid = (K_p, K_i, K_d)
    
    input('[Enter] for init angles (may want to hold up by strap)')
    init_buf = poppy.goto_position(init_angles, duration=1, bufsize=10, speed_ratio=-1) # don't abort for speed
    # init_buf2 = poppy.goto_position(init_angles, duration=1, bufsize=100, speed_ratio=-1) # to check more pid
    bufs = [init_buf] #, init_buf2

    # for t, traj in enumerate(trajs[1:5]):
    for t in range(1, 5):
        if t == 3: # swing
            # poppy.robot.l_ankle_y.pid = (30., 0., 0.)
            # poppy.robot.l_knee_y.pid = (30., 0., 0.)
            durfac = 1
        else:
            # poppy.robot.l_ankle_y.pid = (K_p, K_i, K_d)
            # poppy.robot.l_knee_y.pid = (K_p, K_i, K_d)
            durfac = 5
        traj = trajs[t]
        cmd = input('[Enter] for traj %d, [q] to quit: ' % t)
        if cmd == 'q': break
        for s, (duration, angles) in enumerate(traj):
            # input('[Enter] for step %d' % s)
            buf = poppy.goto_position(angles, durfac*duration, bufsize=10, speed_ratio=-1)
            bufs.append(buf)
            print('  success = %s' % buf[0])

    with open('scratchbuf.pkl', "wb") as f: pk.dump((poppy.motor_names, bufs), f)

    input('[Enter] to turn on compliance (may want to hold up by strap)')
    for m in poppy.motors: m.compliant = True
    
    poppy.close()
    print("closed")

# show and don't run
else:

    import matplotlib.pyplot as pt

    # load with latin1 encoding for numpy 2.x 3.x incompatibility
    with open('scratchbuf.real.pkl', 'rb') as f: (motor_names, bufs) = pk.load(f, encoding='latin1')
    # with open('scratchbuf.real.2.pkl', 'rb') as f: (motor_names, buffers, time_elapsed) = pk.load(f, encoding='latin1')
    # bufkeys = ("position", "speed", "load", "voltage", "temperature")

    successes, buffers, times_elapsed = zip(*bufs)
    step_times = np.cumsum([times_elapsed[s][len(buffers[s]['position'])-1] for s in range(len(bufs))])

    print("success: %s" % all(successes))

    registers = ['position'] #, 'load', 'voltage']
    # names = motor_names
    names = walk_names
    colors = 'bgrcmyk'

    # stand_time, maintain_time = stand[2], maintain[2]
    # maintain_time += stand_time[-1]
    # maintain = maintain[:2] + (maintain_time,)
    # target_angles = np.array([angles[0][name] for name in motor_names])

    for r,reg in enumerate(registers):
        pt.subplot(len(registers), 1, r+1)

        for b, (success, buffers, time_elapsed) in enumerate(bufs):
            if b > 0: time_elapsed += step_times[b-1]
            if not success: time_elapsed = time_elapsed[:len(buffers['position'])]
            for k,name in enumerate(names):
                j = motor_names.index(name)
                pt.plot(time_elapsed, buffers[reg][:,j], colors[k % len(colors)] + '+-')
                if reg == "position":
                    if b == 0:
                        pt.plot(time_elapsed[-1], buffers["target"][-1,j], colors[k % len(colors)] + 'o:', label=name)
                    else:
                        pt.plot(time_elapsed[-1], buffers["target"][-1,j], colors[k % len(colors)] + 'o:')

            if reg == 'position':
                # for k,name in enumerate(names):
                #     j = motor_names.index(name)
                    # pt.plot([stand_time[0], maintain_time[-1]], target_angles[[j,j]], colors[k % len(colors)] + 'o:', label=name)
                    # pt.plot(time_elapsed, buffers["target"][:,j], colors[k % len(colors)] + 'o:', label=name)
                pt.plot(step_times[[b, b]], [buffers[reg].min(), buffers[reg].max()], 'k:')

    pt.legend()
    pt.tight_layout()
    pt.show()

    # actual_angles = buffers['position']
    # print(actual_angles)
    # actual_loads = buffers['load']
    # actual_temperatures = buffers['temperature']
    # actual_time = time_elapsed
    
    # colors = 'bgrcmyk'
    # k = 0
    # for j in range(len(motor_names)):
    #     # if motor_names[j] not in names: continue
    #     pt.subplot(3,1,1)
    #     # pt.plot([durations[0], durations[0]], angles[[0,0],j], colors[k % len(colors)] + '+:')
    #     pt.plot(actual_time, actual_angles[:,j], colors[k % len(colors)] + '.-', label=motor_names[j])
    #     pt.subplot(3,1,2)
    #     pt.plot(actual_time, actual_loads[:,j], colors[k % len(colors)] + '.-', label=motor_names[j])
    #     pt.subplot(3,1,3)
    #     pt.plot(actual_time, actual_temperatures[:,j], colors[k % len(colors)] + '.-', label=motor_names[j])
    #     k += 1
    # for sp in range(1,4):
    #     pt.subplot(3,1,sp)
    #     pt.legend(loc='upper left')
    #     pt.xlabel('time')
    #     pt.ylabel(['pos', 'load', 'temp'][sp-1])
    
    # pt.tight_layout()
    # pt.show()

