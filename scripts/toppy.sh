cp ~/poindexter/cosine_trajectory.py ./

scp \
    poppy_wrapper.py \
    launch_poppy.py \
    cosine_trajectory.py \
    wobble.py \
    poppy@poppy.local:scripts/

