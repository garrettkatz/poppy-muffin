import os, glob
import numpy as np
from scipy.stats import mannwhitneyu, permutation_test

num_successes = {}
for suffix in ("ol","lqr"):
    # path = os.path.join(os.path.expanduser("~"), "Downloads", "poppy_walk_data", f"2026_03_02_{suffix}") # office
    path = os.path.join(os.path.expanduser("~"), "Downloads", "poppy_walk_data", f"2026_03_03_{suffix}") # casm

    num_successes[suffix] = []

    folders = glob.glob(path)
    for folder in folders:
        traj_files = glob.glob(os.path.join(path, folder, "traj*.pkl"))
        for traj_file in traj_files:
            # filename format is traj_<n>_<num_success>.pkl for the nth episode
            num_successes[suffix].append( int(traj_file[traj_file.rfind("_")+1:-4]) )
    
    num_successes[suffix] = np.array(num_successes[suffix])
    print(f"{suffix}: success rate = {(num_successes[suffix]==6).mean()} of {len(num_successes[suffix])}, avg steps = {num_successes[suffix].mean()}")

U1, p_val = mannwhitneyu(num_successes["ol"], num_successes["lqr"], method="asymptotic", alternative="less")
print(f"Mann-Whitney U test (asymptotic): {U1=}, {p_val=}")

U1, p_val = mannwhitneyu(num_successes["ol"], num_successes["lqr"], method="exact", alternative="less")
print(f"Mann-Whitney U test (exact, does not correct for ties): {U1=}, {p_val=}")


data = [num_successes["ol"], num_successes["lqr"]]
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)
res = permutation_test(data, statistic, permutation_type='independent', alternative='less')
print(f"permutation test pval={res.pvalue}")