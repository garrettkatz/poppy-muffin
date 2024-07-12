cp ~/poindexter/cosine_trajectory.py ./

scp \
    poppy_wrapper.py \
    launch_poppy.py \
    cosine_trajectory.py \
    pypot_traj1_smoothed.pkl \
    pypot_traj1_bumped.pkl \
    run_walk_tweak.py \
    run_walk_experiment.py \
    poppy@poppy.local:scripts/

