The data is grouped into sessions; each session is in a directory with the format YYYY_MM_DD_<stdev>, where stdev is the standard deviation of zero-centered Gaussian trajectory perturbations.  There is one "nominal" trajectory known to work well, but perturbed versions of this trajectory are used as the planned trajectories in different episodes.  Gaussian perturbations are applied to each angle in each time-step of the trajectory.

Each session directory contains trajectory and sensor data for several episodes.  These are in the files traj*pkl and bufs*pkl, respectively.

traj_<n>_<num_success>.pkl is the planned trajectory in the nth episode.  num_success is the number of successful steps before a fall.  You can load it with the python command

with open(traj_name, "rb") as f:
    trajectory = pickle.load(f, encoding='latin1')

trajectory[i] = (dur, ang) is the duration (in seconds) and angle dictionary for the ith waypoint in the trajectory. ang[m] is the angle (in degrees) for motor name m.  There are 6 foot steps, and 5 waypoints per step.  The 5th waypoint has a longer duration so that the robot takes a short pause after each footstep.

bufs_<n>_<num_success>.pkl are the buffers of actual data recorded during the robot motion.  Data is recorded at a higher frequency than the planned trajectory, so there are several buffer timepoints for each trajectory waypoint.  Several kinds of sensor data are saved including joint angles, camera images, and others.  You can load the data with the python command

with open(bufs_name, "rb") as f:
    (buffers, elapsed, waytime_points, all_motor_names) = pickle.load(f, encoding='latin1')

- buffers[kind][k] is the data at the kth timepoint.  kind is one of ("position", "speed", "load", "voltage", "temperature", "images", "target").
- elapsed[k] is the actual time elapsed at the kth timepoint
- waytime_points[n] is the actual time elapsed (in seconds) when the robot begins the motion to the nth trajectory waypoint.  Since buffer timepoints are recorded at higher frequency, waytime_points is a shorter array than elapsed.  waytime_points can be used to align buffer data with the planned trajectory.
- all_motor_names is a list of all motor names in the robot.

Some more information about buffers:  Each kind of data is recorded on a per-joint basis, except "images".  So for example buffers["position"][k][j] is the angle of the jth joint at the kth timepoint.

Images are numpy arrays of type uint8 with shape (rows, columns, 3), where the last array axis is the 3 color channels.  They are in reverse order (bgr instead of rgb).  The image data can be converted to a form suitable for matplotlib with the command

img = buffers["images"][k][:,:,[2,1,0]].astype(float)/255.

buffers["images"][k] is either an image with the format above, or None.  Many entries are None because images are recorded at a lower frame rate than the other buffers.  The framerate is approximately 16 frames per second.

The camera connection is somewhat unreliable, especially when the robot experiences significant jerk (for example, during a fall).  When this happens, the buffers contain copies of the last frame that was successfully received before connectivity was lost.  So the frames in the buffers after a fall should generally be ignored.

The script view_buffers.py has more examples of how to load and interact with the data.  The script should be run like:

python view_buffers.py <trajectory file> <buffer file>

