## This dataset contains joint and egocentric camera recordings from the Poppy humanoid during a walking motion.

There are 30 total repetitions of the experiment, all using the same sequence of joint trajectories.  In each repetition Poppy attempts a single step forward.  Due to sub-optimal joint trajectories and experimental noise, there are a comparable number of successful steps and failed steps where Poppy loses balance and falls, although the data is not perfectly balanced.

The data for repetition r is stored in the files results_-<r>.pkl and frames_-<r>.pkl.  The frames file contains the recorded camera images; the results file contains all other data like joint angle recordings, time elapsed, and success labels.

The walking gait trajectory is split into several transitions.  Labels are coarse and indicate the transition during which a fall occurred, but not precisely when it occurred during the transition.  Transitions after a fall should be ignored because the experimenter intervened to re-stabilize the robot and prevent hardware damage.

For an allocentric view of Poppy during a successful step, watch the video `IROS23_2793_VI_i.mp4`.  Repetitions were considered successful if the experimental harness remained slack for the entire duration of the trial.  If at any point Poppy lost balance and the harness became taut, the transition was marked as a failure.

Due to storage constraints on the robot's embedded system, all frames were downsampled by a factor of 5 in each dimension, resulting in image size 96x128.  The total number of frames across all repetitions is 9000, and the total size of the dataset is roughly 1GB.

For more details on the data format and examples of loading and parsing data, inspect and run the script `view_img_buf.py`.

