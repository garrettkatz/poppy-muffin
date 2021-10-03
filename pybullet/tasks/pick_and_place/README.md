The code in this folder was used for experiments in the paper "Neurosymbolic Robotic Manipulation with a Neural Virtual Machine."

"Empirical Validation" Section experiments: Run `python nvm_rvm_compare.py` to reproduce these experiments.  For regenerating the result data, set the `run_exp` variable on line 94 to `True`.  To plot the result data in Figure 5, set the `plt_exp` variable on line 115 to `True`.  To plot the result data in Figure 6, run `sym_and_mp_fig.py`.

"Improving Performance" Section experiments: Run `python practice_all_cases_batched.py` to reproduce these experiments.  For regenerating result data, set the `run_exp` variable on line 147 to `True`.  For plotting the result data, set the `showresults` and `showtrained` variables on lines 148 and 152 to `True`.

