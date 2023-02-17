import pickle as pk
import numpy as np
import poppy_wrapper as pw
try:
    from pypot.creatures import PoppyHumanoid as PH
    do_run = True
except:
    from mocks import PoppyHumanoid as PH
    do_run = False

# load planned trajectory
with open('pypot_traj1.pkl', "rb") as f: trajs = pk.load(f)
# get initial angles
traj = trajs[0]
durations, angles = zip(*traj)
print('traj[0] durations:')
print(durations)
print('traj[0] angles:')
print(angles)

# Run and don't show
if do_run:

    poppy = pw.PoppyWrapper(PH())
    
    # for python 2.7 on poppy
    if hasattr(__builtins__, 'raw_input'):
        input=raw_input
    
    input('[Enter] to turn off compliance')
    for m in poppy.motors: m.compliant = False
    
    input('[Enter] for motion (may want to hold off ground)')
    stand = poppy.goto_position(angles[0], durations[0]) # too long duration triggers restricted motion?
    maintain = poppy.goto_position(angles[0], duration=10, bufsize=100, speed_ratio=-1) # don't abort for speed

    with open('scratchbuf.pkl', "wb") as f: pk.dump((poppy.motor_names, stand, maintain), f)

    input('[Enter] to turn on compliance (may want to hold up by strap)')
    for m in poppy.motors: m.compliant = True
    
    poppy.close()
    print("closed")

# show and don't run
else:

    import matplotlib.pyplot as pt

    # load with latin1 encoding for numpy 2.x 3.x incompatibility
    with open('scratchbuf.real.pkl', 'rb') as f: (motor_names, stand, maintain) = pk.load(f, encoding='latin1')
    # with open('scratchbuf.real.2.pkl', 'rb') as f: (motor_names, buffers, time_elapsed) = pk.load(f, encoding='latin1')
    # bufkeys = ("position", "speed", "load", "voltage", "temperature")

    print("success: stand %s, maintain %s" % (stand[0], maintain[0]))

    registers = ['position'] #, 'load', 'voltage']
    names = ["%s_%s_y" % (lr, jnt) for lr in "lr" for jnt in ("hip", "ankle", "knee")]
    # names = motor_names
    colors = 'bgrcmyk'

    stand_time, maintain_time = stand[2], maintain[2]
    maintain_time += stand_time[-1]
    maintain = maintain[:2] + (maintain_time,)

    target_angles = np.array([angles[0][name] for name in motor_names])

    for r,reg in enumerate(registers):
        pt.subplot(len(registers), 1, r+1)

        for (_, buffers, time_elapsed) in (stand, maintain):
            for k,name in enumerate(names):
                j = motor_names.index(name)
                pt.plot(time_elapsed, buffers[reg][:,j], colors[k % len(colors)] + '+-')
                if reg == "position":
                    pt.plot(time_elapsed, buffers["target"][:,j], colors[k % len(colors)] + 'o:', label=name)


        if reg == 'position':
            # for k,name in enumerate(names):
            #     j = motor_names.index(name)
                # pt.plot([stand_time[0], maintain_time[-1]], target_angles[[j,j]], colors[k % len(colors)] + 'o:', label=name)
                # pt.plot(time_elapsed, buffers["target"][:,j], colors[k % len(colors)] + 'o:', label=name)
            pt.plot(stand_time[[-1, -1]], [stand[1][reg].min(), stand[1][reg].max()], 'k:')

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

