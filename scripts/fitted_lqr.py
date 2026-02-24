import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import cvxpy as cp
from stitch_preanalysis import get_run_filepaths

if __name__ == "__main__":

    num_interp = 10 # number of interpolated timepoints

    do_chunk = False
    view_chunk = False
    do_dyn_fit = False
    view_dyn_fit = False
    do_cost_fit = True
    do_cost_var = False

    if do_chunk:
        # each observation is an interpolation of joint measurements within a waypoint window
        # cmd[r,n,j] is command for angle j, waypoint n, run r
        # obs[r,n,j,k] is kth interpolate of angle j, run r, *while* executing waypoint command n+1 (not before)
        # so obs[r,0] is observations at initial joint angles *before* issuing waypoint command n=0

        run_filepaths = get_run_filepaths()
        #run_filepaths = run_filepaths[:1]

        obs = np.empty((len(run_filepaths), 31, 25, num_interp))
        cmd = np.empty((len(run_filepaths), 30, 25))
        for r, (traj_file, buf_file, stdev, num_successes) in enumerate(run_filepaths):
            print(f"arrays for {r} of {len(run_filepaths)}")
           
            # load planned trajectory
            with open(traj_file, "rb") as f:
                trajectory = pk.load(f, encoding='latin1')
            
            # load the data
            with open(buf_file, "rb") as f:
                (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')
                elapsed = np.array(elapsed)
                waytime_points = np.array(waytime_points) # when the robot *begins* motion to the nth waypoint
          
            assert len(trajectory) == len(waytime_points) == 30 # 6 footsteps, 5 waypoints each

            # put final timepoint at the end for easier chunking
            waytime_points = np.append(waytime_points, elapsed[-1])
    
            # package trajectory into array
            durs, angs = zip(*trajectory)
            cmd[r] = np.array([[ang.get(name, 0.) for name in all_motor_names] for ang in angs])
            
            # setup initial observation by duplicating interpolants
            for k in range(num_interp):
                obs[r,0,:,k] = buffers["position"][0]

            # interpolate observations for each waypoint
            for n in range(1,31):
                mask = (waytime_points[n-1] <= elapsed) & (elapsed <= waytime_points[n])
                timepoints = np.linspace(waytime_points[n-1], waytime_points[n], num_interp)
                for j in range(25):
                    obs[r,n,j] = np.interp(timepoints, elapsed[mask], buffers["position"][mask, j])
        
        with open(f"lqr_chunks_ni{num_interp}.pkl","wb") as f: pk.dump((obs, cmd), f)

    if view_chunk:
        with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)
        pt.plot(np.arange(num_interp), obs[0,0].T, 'k:')
        pt.plot(np.arange(num_interp-1,2*num_interp-1), obs[0,1].T, 'b:')
        pt.plot([num_interp-1,2*num_interp-2], cmd[0,[0,0]], 'k-')
        pt.show()

    if do_dyn_fit:

        run_filepaths = get_run_filepaths()
        with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)
        
        # extract number of successful footsteps
        _, _, stdevs, nfsteps = zip(*run_filepaths)
        stdevs = np.array(stdevs)
        nfsteps = np.array(nfsteps)

        # flatten data for linear fits
        X = obs.reshape(len(run_filepaths), 31, -1)
        A = cmd.reshape(len(run_filepaths), 30, -1)

        # get target trajectory (average within noiseless success)
        X0 = X[(stdevs=="0.0") & (nfsteps==6)].mean(axis=0)
        A0 = A[(stdevs=="0.0") & (nfsteps==6)].mean(axis=0)

        # one linear model per waypoint timestep
        linmods, residuals, norms = [], [], []
        for n in range(1,30):
            # collect all chunks that do not fall while executing waypoint n
            # example: if nfstep == 1, then fall could have happened as early as when executing 5th waypoint
            # so fit transitions up to 3rd->4th (executing 4th waypoint), but not 4th->5th
            # 4//5 < 1, but 5//5 !< 1
            mask = (n//5 < nfsteps)
            N = mask.sum() # number of datapoints for linear fit

            # prepare regression data
            X_prev = X[mask, n-1] # observed while approaching previous waypoint
            A_curr = A[mask, n] # now new waypoint command is issued
            X_next = X[mask, n] # next observations while executing waypoint command
            
            # get deltas from nominal
            dX_prev = X_prev - X0[n-1]
            dA_curr = A_curr - A0[n]
            dX_next = X_next - X0[n]

            # do regression
            dXA = np.concatenate([dX_prev, dA_curr], axis=1)
            linmod = np.linalg.lstsq(dXA, dX_next, rcond=None)[0]

            # residual mad and norm
            residual = np.fabs(dXA @ linmod - dX_next).mean()
            norm = np.linalg.norm(linmod, ord=2)
            print(f"Model {n} ({N} datapoints, {dXA.shape[1]} dimensional) {residual=:.3e}, {norm=:.3f}")

            linmods.append(linmod)
            residuals.append(residual)
            norms.append(norm)
        
        with open(f"dynfit_ni{num_interp}.pkl","wb") as f: pk.dump((linmods, residuals, norms), f)

    if view_dyn_fit:

        num_interps = [2,3,5,10]
        fig = pt.figure(figsize=(8,4))
        for ni in num_interps:
            with open(f"dynfit_ni{ni}.pkl","rb") as f: (linmods, residuals, norms) = pk.load(f)
            
            pt.subplot(1,2,1)
            pt.plot(residuals, 'o-', label=f"K={ni}")
            
            pt.subplot(1,2,2)
            pt.plot(norms, 'o-', label=f"K={ni}")
        
        fig.supxlabel("waypoint timestep")
        
        pt.subplot(1,2,1)
        pt.title("MAD residual")
        pt.yscale("log")
        pt.legend()
        
        pt.subplot(1,2,2)
        pt.title("Matrix 2-norm")
        pt.yscale("log")
        pt.legend()
        
        pt.tight_layout()
        pt.savefig("lqr_dyn_fit.pdf")
        pt.show()
            
    if do_cost_fit:

        # try fitting quadratic cost on observations to fall labels
        run_filepaths = get_run_filepaths()
        with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)
        
        # extract number of successful footsteps
        _, _, stdevs, nfsteps = zip(*run_filepaths)
        stdevs = np.array(stdevs)
        nfsteps = np.array(nfsteps)

        # flatten data for quadratic fits
        X = obs.reshape(len(run_filepaths), 31, -1)
        A = cmd.reshape(len(run_filepaths), 30, -1)

        # get target trajectory (average within noiseless success)
        X0 = X[(stdevs=="0.0") & (nfsteps==6)].mean(axis=0)
        A0 = A[(stdevs=="0.0") & (nfsteps==6)].mean(axis=0)

        # one quadratic model per waypoint timestep
        # cost targets: footsteps left to 6 - remaining successful footsteps
        # exclude footsteps after the fall
        costmods, residuals = [], []
        for n in range(1, 30):
            print(f"Cost Model {n} ...")

            # mask footsteps after the fall
            current_footstep = n//5
            mask = (current_footstep <= nfsteps)

            # setup per-run cost targets
            costs = (6 - current_footstep) - (nfsteps[mask] - current_footstep)

            # prepare regression data
            X_prev = X[mask, n-1] # observed while approaching previous waypoint
            A_curr = A[mask, n] # now new waypoint command is issued
            
            # get deltas from nominal
            dX_prev = X_prev - X0[n-1]
            dA_curr = A_curr - A0[n]
            dXA = np.concatenate([dX_prev, dA_curr], axis=1)

            # quadratic fit with semi-definite programming
            Q = cp.Variable((dXA.shape[1], dXA.shape[1]), PSD=True)
            objective = cp.sum_squares(cp.hstack([
                cp.quad_form(row, Q) - cost
                for (row, cost) in zip(dXA, costs)
            ]))
            prob = cp.Problem(cp.Minimize(objective), constraints=[])
            prob.solve()

            costmod = Q.value
            residual = (objective.value / len(costs))**.5
            print(f"Problem status = {prob.status}, rmse={residual}")

            costmods.append(costmod)
            residuals.append(residual)

        with open(f"costfit_ni{num_interp}.pkl","wb") as f: pk.dump((costmods, residuals), f)

    if do_cost_var:
        # simpler approach that just inverts covariance within successful episodes? but will be low rank... add id...
        pass