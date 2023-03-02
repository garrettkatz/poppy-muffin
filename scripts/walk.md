# Walking Gait Synthesis and Execution

Getting Poppy to autonomously take a step forward involves two stages:

- Synthesis of the joint trajectory

- Execution of the joint trajectory

Synthesis is done in an adjoining repository as described [here](https://github.com/garrettkatz/poppy-simulations/ergo/pybullet/tasks/walking/walk.md).  Execution is done using the code in this repository and the following steps:

1. Synthesized gaits will have been written to the following files.  Copy them into the `./walk_samples/` sub-directory here.

- `pypot_traj1.pkl` (the hand-tuned trajectory)
- `pypot_sample_trajectory_*.pkl` (the perturbed trajectories)

1. Copy these entire contents of this `scripts` directory to the embedded computer in the Poppy robot.

1. SSH onto the Poppy embedded computer and run the command

`$ python run_walk_experiment.py <sample>`

where `<sample>` is an integer.  If `<sample>` is negative, it uses the trajectory stored in `pypot_traj1.pkl`.  Otherwise, it uses the trajectory stored in the file `pypot_sample_trajectory_<sample>.pkl`.  In either case, it writes output data about the motor execution to a file `results<sample>.pkl`.

The python script will prompt you to take several actions before/after the robotic motion; make sure you appropriately and safely position the robot in between each one.

- First, it will prompt you to prepare the joints for non-compliance, which will fix them at their current angles.

- Next it will prompt you to suspend the robot while it moves to the initial pose of the step trajectory.  Make sure the robot's appendages have a clear, self-collision-free path from the non-compliant pose to the initial step pose.  It will not autonomously maintain balance during this motion, so you will need to suspend the robot above the ground and then manually plant its feet on the ground after the initial pose is reached.

- Next, it will wait for your signal before attempting an autonomous step.  Be prepared to catch the robot if it falls during the step attempt.

- Next, after the step is complete, it will prompt you to again suspend the robot so it can return to zero joint angles and then become compliant.  If you do not suspend it during this step, it will collapse under its own weight in a potentially damaging way as soon as motor compliance is restored.

- Finally, it will prompt you to enter the number of successful transitions during the step before Poppy lost balance and fell over.

1. Copy the result files back to your host computer.  To visualize the results, run the command

`$ python view_walk_experiment.py`

