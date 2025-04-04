import ergo_jr_wrapper as ew

# for python 2.7 on poppy
if hasattr(__builtins__, 'raw_input'):
    input=raw_input

ergo = ew.ErgoJrWrapper(fps=10)

lift = {'m5': -6.0099999999999998, 'm4': 2.2000000000000002, 'm6': 6., 'm1': 0.44, 'm3': -8.9399999999999995, 'm2': -36.219999999999999}
grasp = {'m5': -4.8399999999999999, 'm4': 3.0800000000000001, 'm6': 6., 'm1': 0.44, 'm3': -10.119999999999999, 'm2': -51.469999999999999}


print("Created ergo with 10 fps camera.  Don't forget to ergo.close() before quit() when you are finished to clean up the motor state.")


