import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import cvxpy as cp
from stitch_preanalysis import get_run_filepaths

if __name__ == "__main__":

    num_interp = 2 # number of interpolated timepoints

    do_cvx_psd = True # whether to enforce PSD in cvx cost
    do_cvx_sym = False # whether to enforce symmetric in cvx cost
    cvx_margin = -3. # margin for fall/success boundary (negative allows some slack, important for strictly pos-def)
    solver = cp.CLARABEL # solver to use
    do_lqr_lc = True # whether to use learned costs for lqr
    only_x = False # whether to include A in cvx cost
    avg_n = True # whether to average cost over time for cvx constraint

    # # these still achieves strict p.d. and stability
    # do_cvx_psd = True # whether to enforce PSD in cvx cost
    # do_cvx_sym = False # whether to enforce symmetric in cvx cost
    # cvx_margin = -.5 # margin for fall/success boundary (negative allows some slack, important for strictly pos-def)
    # solver = cp.CLARABEL # solver to use
    # do_lqr_lc = True # whether to use learned costs for lqr
    # only_x = True # whether to include A in cvx cost
    # avg_n = True # whether to average cost over time for cvx constraint

    do_metadata = False
    do_chunk = False
    view_chunk = False
    do_dyn_fit = False
    do_dyn_fit_pool = False
    view_dyn_fit = False
    do_cost_cvx = True
    do_cost_fit = False
    do_cost_var = False
    view_cost_var = False
    view_cost_cvx = True
    do_lqr = True
    view_lqr = True
    view_control_deviation = True
    do_repickle = False

    if do_metadata:
        run_filepaths = get_run_filepaths()
        _, _, stdevs, nfsteps = zip(*run_filepaths)
        stdevs = np.array(stdevs)
        nfsteps = np.array(nfsteps)
        with open("lqr_metadata.pkl","wb") as f: pk.dump((stdevs, nfsteps), f)

    if do_chunk:
        # each observation is an interpolation of joint measurements within a waypoint window
        # cmd[r,n,j] is command for angle j, waypoint n, run r
        # obs[r,n,j,k] is kth interpolate of angle j, run r, *before* executing waypoint command n (not while)
        # so obs[r,0] is observations at initial joint angles *before* issuing waypoint command n=0
        # obs[r,1] is observations *while* moving to waypoint n=0, *before* issuing waypoint command n=1
        # obs[r,30] is observations *while* moving to last waypoint n=29.

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
                # timepoints *while* moving to waypoint n-1, *before* executing waypoint n
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
        for n in range(30):

            # # this strategy may use transitions farther from the nominal trajectory:
            # # collect all chunks that definitely do not fall while executing waypoint n
            # # example: if nfstep == 1, then fall could have happened as early as while executing 5th waypoint
            # # so fit transitions up to 3rd->4th (executing 4th waypoint), but not 4th->5th
            # # 4//5 < 1, but 5//5 !< 1
            # mask = (n//5 < nfsteps)

            # this strategy may limit linearization data closer to the nominal trajectory:
            # just fit transitions within the noiseless success episodes
            mask = (stdevs=="0.0") & (nfsteps==6)

            N = mask.sum() # number of datapoints for linear fit

            # prepare regression data
            X_prev = X[mask, n] # observed *before* issuing waypoint n
            A_curr = A[mask, n] # now waypoint n command is issued
            X_next = X[mask, n+1] # next observations while moving to waypoint n
            
            # get deltas from nominal
            dX_prev = X_prev - X0[n]
            dA_curr = A_curr - A0[n]
            dX_next = X_next - X0[n+1]

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

    if do_dyn_fit_pool:
        # pool transition data across cycles

        run_filepaths = get_run_filepaths()
        with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)
        
        # extract number of successful footsteps
        _, _, stdevs, nfsteps = zip(*run_filepaths)
        stdevs = np.array(stdevs)
        nfsteps = np.array(nfsteps)
        print(f"{((stdevs=='0.0') & (nfsteps == 6)).sum()} noiseless success episodes")

        # flatten data for linear fits
        X = obs.reshape(len(run_filepaths), 31, -1)
        A = cmd.reshape(len(run_filepaths), 30, -1)

        # pool across footstep cycle
        X_pool = np.concatenate([
            X[:,0:11],
            X[:,10:21],
            X[:,20:31],
        ], axis=0)
        A_pool = np.concatenate([
            A[:,0:10],
            A[:,10:20],
            A[:,20:30],
        ], axis=0)
        stdevs_pool = np.concatenate([stdevs]*3)
        success_pool = np.concatenate([
            (nfsteps >= 2),
            (nfsteps >= 4),
            (nfsteps == 6),
        ])

        # get target trajectory for one cycle (average within noiseless success)
        X0 = X_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)
        A0 = A_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)

        # one linear model per waypoint timestep within one cycle
        linmods, residuals, norms = [], [], []
        for n in range(10):

            # # this strategy may use transitions farther from the nominal trajectory:
            # # collect all chunks that definitely do not fall while executing waypoint n
            # # example: if nfstep == 1, then fall could have happened as early as while executing 5th waypoint
            # # so fit transitions up to 3rd->4th (executing 4th waypoint), but not 4th->5th
            # # 4//5 < 1, but 5//5 !< 1
            # mask = (n//5 < nfsteps)

            # this strategy may limit linearization data closer to the nominal trajectory:
            # just fit transitions within the success episodes
            mask = success_pool

            # # this strategy doesn't work since dA=0 so B_n=0:
            # # just fit noiseless success
            # mask = (stdevs_pool=="0.0") & success_pool

            N = mask.sum() # number of datapoints for linear fit

            # prepare regression data
            X_prev = X_pool[mask, n] # observed *before* issuing waypoint n
            A_curr = A_pool[mask, n] # now waypoint n command is issued
            X_next = X_pool[mask, n+1] # next observations while moving to waypoint n
            
            # get deltas from nominal
            dX_prev = X_prev - X0[n]
            dA_curr = A_curr - A0[n]
            dX_next = X_next - X0[n+1]

            # do regression
            dXA = np.concatenate([dX_prev, dA_curr], axis=1)
            # if n == 9: # here dA_curr is zero since we never perturbed the initial joints of the cycle
            #     pt.imshow(dXA)
            #     pt.show()
            linmod = np.linalg.lstsq(dXA, dX_next, rcond=None)[0]

            # residual mad and norm
            residual = np.fabs(dXA @ linmod - dX_next).mean()
            norm = np.linalg.norm(linmod, ord=2)
            print(f"Model {n} ({N} datapoints, {dXA.shape[1]} dimensional) {residual=:.3e}, {norm=:.3f}")

            linmods.append(linmod)
            residuals.append(residual)
            norms.append(norm)
        
        with open(f"dynfit_pool_ni{num_interp}.pkl","wb") as f: pk.dump((linmods, residuals, norms), f)

    if view_dyn_fit:

        num_interps = [2,3,5,10]
        fig = pt.figure(figsize=(8,4))
        for ni in num_interps:
            #with open(f"dynfit_ni{ni}.pkl","rb") as f: (linmods, residuals, norms) = pk.load(f)
            with open(f"dynfit_pool_ni{ni}.pkl","rb") as f: (linmods, residuals, norms) = pk.load(f)
            
            dim_X = ni * 25
            print(f"{ni=}: max|B_n| = {max([np.fabs(M[dim_X:]).max() for M in linmods])}")

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
        #pt.savefig("lqr_dyn_fit.pdf")
        pt.savefig("lqr_dyn_fit_pool.pdf")
        pt.show()
        
        # visualize transition matrices for ni
        ni=5
        with open(f"lqr_chunks_ni{ni}.pkl","rb") as f: (obs, cmd) = pk.load(f)
        with open(f"dynfit_pool_ni{ni}.pkl","rb") as f: (linmods, residuals, norms) = pk.load(f)

        # flatten data for shape
        X = obs.reshape(len(obs), 31, -1)
        A = cmd.reshape(len(cmd), 30, -1)
        dim_X = X.shape[-1]

        n = 3
        print(dim_X)
        print(linmods[n].shape)
        A_n = linmods[n % len(linmods)][:dim_X].T
        B_n = linmods[n % len(linmods)][dim_X:].T
        print(f"max|B_n| = {np.fabs(B_n).max()}")
        pt.subplot(1,2,1)
        pt.imshow(A_n)
        pt.subplot(1,2,2)
        pt.imshow(B_n)
        pt.show()

    if do_cost_cvx:
        # objective: min dist to identity
        # constraints: higher cost on falls than success, PSD

        # load in data
        run_filepaths = get_run_filepaths()
        with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)

        _, _, stdevs, nfsteps = zip(*run_filepaths)
        stdevs = np.array(stdevs)
        nfsteps = np.array(nfsteps)

        # flatten data
        X = obs.reshape(len(run_filepaths), 31, -1)
        A = cmd.reshape(len(run_filepaths), 30, -1)
        dim_X = X.shape[-1]
        dim_A = A.shape[-1]
        dim_XA = dim_X + dim_A

        # pool across footstep cycle
        X_pool = np.concatenate([
            X[:,0:10],
            X[:,10:20],
            X[:,20:30],
        ], axis=0)
        A_pool = np.concatenate([
            A[:,0:10],
            A[:,10:20],
            A[:,20:30],
        ], axis=0)
        stdevs_pool = np.concatenate([stdevs]*3)
        success_pool = np.concatenate([
            (nfsteps >= 2),
            (nfsteps >= 4),
            (nfsteps == 6),
        ])

        # get nominal trajectory from pooled cycles (average within noiseless success)
        X0 = X_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)
        A0 = A_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)

        # duplicate back to full data length for residuals
        X0 = np.concatenate([X0]*3 + [X0[:1]], axis=0)
        A0 = np.concatenate([A0]*3, axis=0)
        dX = X - X0
        dA = A - A0

        if only_x:

            ## only X
    
            # setup optimization variables
            C = [cp.Variable((dim_X,)*2, PSD=do_cvx_psd, symmetric=do_cvx_sym) for _ in range(10)] # cost matrices
            d = cp.Variable() # success/fall threshold
    
            # setup objective
            objective = cp.sum_squares(cp.hstack([C[n] - np.eye(dim_X) for n in range(10)]))
    
            # setup constraints
            constraints = []
            for r, dx in enumerate(dX):
                n_F = 5 * nfsteps[r]
    
                cost = cp.quad_form(dx[n_F], C[n_F % 10], assume_PSD=do_cvx_psd)
                for n in range(n_F):
                    cost = cost + cp.quad_form(dx[n], C[n % 10], assume_PSD=do_cvx_psd)
    
                # average over timesteps?
                if avg_n: cost = cost / (n_F+1)
    
                if nfsteps[r] < 6:
                    constraints.append(cost >= d + cvx_margin)
                else:
                    constraints.append(cost <= d - cvx_margin)

        else:

            ## full XA
    
            # setup optimization variables
            C = [cp.Variable((dim_XA,)*2, PSD=do_cvx_psd, symmetric=do_cvx_sym) for _ in range(10)] # cost matrices
            d = cp.Variable() # success/fall threshold
    
            # setup objective
            objective = cp.sum_squares(cp.hstack([C[n] - np.eye(dim_XA) for n in range(10)]))
    
            # setup constraints
            constraints = []
            for r, (dx, da) in enumerate(zip(dX, dA)):
                n_F = 5 * nfsteps[r]
                dxa = np.concatenate([dx[:n_F], da[:n_F]], axis=-1)
    
                cost = cp.quad_form(dx[n_F], C[n_F % 10][:dim_X, :dim_X], assume_PSD=do_cvx_psd)
                for n in range(n_F):
                    cost = cost + cp.quad_form(dxa[n], C[n % 10], assume_PSD=do_cvx_psd)

                # average over timesteps?
                if avg_n: cost = cost / (n_F+1)
    
                if nfsteps[r] < 6:
                    constraints.append(cost >= d + cvx_margin)
                else:
                    constraints.append(cost <= d - cvx_margin)

        # solve problem
        prob = cp.Problem(cp.Minimize(objective), constraints)
        # prob = cp.Problem(cp.Minimize(objective), constraints=[]) # sanity check
        prob.solve(verbose=True, solver=solver)

        # save results
        C = [C[n].value for n in range(10)]
        with open(f"costcvx_ni{num_interp}.pkl","wb") as f: pk.dump(C, f)

    if view_cost_cvx:

        with open(f"costcvx_ni{num_interp}.pkl","rb") as f: C = pk.load(f)

        # check results
        max_sym_dev = -np.inf
        min_min_eig = np.inf
        for n in range(10):
            print(f"{n=}:")
            print(f"sym violation = {np.fabs(C[n] - C[n].T).max()}")

            eigs = np.linalg.eigvals(C[n])
            print(f"psd min eig = {eigs.min()} >= 0?")

            max_sym_dev = max(max_sym_dev, np.fabs(C[n] - C[n].T).max())
            min_min_eig = min(min_min_eig, eigs.min())

            # pt.subplot(1,2,1)
            # pt.imshow(C[n])
            # pt.subplot(1,2,2)
            # pt.plot(sorted(eigs))
            # pt.show()
            print("smallest eigs:")
            print(sorted(eigs)[:3])

        print(f"smallest eig over n = {min_min_eig}, max sym dev over n = {max_sym_dev}")

        # calculate costs on the runs

        # load in data
        run_filepaths = get_run_filepaths()
        with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)

        _, _, stdevs, nfsteps = zip(*run_filepaths)
        stdevs = np.array(stdevs)
        nfsteps = np.array(nfsteps)

        # flatten data
        X = obs.reshape(len(run_filepaths), 31, -1)
        A = cmd.reshape(len(run_filepaths), 30, -1)
        dim_X = X.shape[-1]
        dim_A = A.shape[-1]
        dim_XA = dim_X + dim_A

        # pool across footstep cycle
        X_pool = np.concatenate([
            X[:,0:10],
            X[:,10:20],
            X[:,20:30],
        ], axis=0)
        A_pool = np.concatenate([
            A[:,0:10],
            A[:,10:20],
            A[:,20:30],
        ], axis=0)
        stdevs_pool = np.concatenate([stdevs]*3)
        success_pool = np.concatenate([
            (nfsteps >= 2),
            (nfsteps >= 4),
            (nfsteps == 6),
        ])

        # get nominal trajectory from pooled cycles (average within noiseless success)
        X0 = X_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)
        A0 = A_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)

        # duplicate back to full data length for residuals
        X0 = np.concatenate([X0]*3 + [X0[:1]], axis=0)
        A0 = np.concatenate([A0]*3, axis=0)
        dX = X - X0
        dA = A - A0

        # get total costs on each run
        costs = np.empty(len(dX))
        if only_x:
            for r, dx in enumerate(dX):
                n_F = 5 * nfsteps[r]
    
                costs[r] = dx[n_F] @ C[n_F % 10] @ dx[n_F]
                for n in range(n_F):
                    costs[r] += dx[n].T @ C[n % 10] @ dx[n]

                if avg_n: costs[r] /= (n_F + 1)

        else:
            for r, (dx, da) in enumerate(zip(dX, dA)):
                n_F = 5 * nfsteps[r]

                dxa = np.concatenate([dx[:n_F], da[:n_F]], axis=-1)
    
                costs[r] = dx[n_F] @ C[n_F % 10][:dim_X, :dim_X] @ dx[n_F]
                for n in range(n_F):
                    costs[r] += dxa[n].T @ C[n % 10] @ dxa[n]

                if avg_n: costs[r] /= (n_F + 1)

        print(f"success cost <= {costs[nfsteps==6].max():.3f}, {costs[nfsteps<6].min():.3f} <= fall cost")

        pt.subplot(1,2,1)
        pt.hist(costs[nfsteps == 6], ec='k', fc='b')
        pt.title("Success")
        pt.subplot(1,2,2)
        pt.hist(costs[nfsteps < 6], ec='k', fc='r')
        pt.title("Fall")
        pt.show()

        # costmod = Q.value
        # residual = (objective.value / len(costs))**.5
        # print(f"Problem status = {prob.status}, rmse={residual}")


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
            X_prev = X[mask, n] # observed before issuing waypoint n
            A_curr = A[mask, n] # now new waypoint command is issued
            
            # get deltas from nominal
            dX_prev = X_prev - X0[n]
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
        # simpler approach that just inverts covariance within successful episodes, plus id for full rank

        # load data
        run_filepaths = get_run_filepaths()
        with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)
        
        # extract noise and success labels
        _, _, stdevs, nfsteps = zip(*run_filepaths)
        stdevs = np.array(stdevs)
        nfsteps = np.array(nfsteps)

        # flatten data at each run and timestep
        X = obs.reshape(len(run_filepaths), 31, -1)
        A = cmd.reshape(len(run_filepaths), 30, -1)

        # concatenate (x,a) pairs
        X_prev = X[:,:-1] # observed before issuing new waypoint command
        A_curr = A # now new waypoint command is issued
        XA = np.concatenate([X_prev, A_curr], axis=2) # (r, n, dim)

        # one quadratic model per waypoint timestep: covariance within the success episodes only
        mask = (nfsteps == 6) # success runs
        XA0 = XA[mask].mean(axis=0) # mean
        dXA = XA - XA0 # deltas from mean (r, n, dim)
        dim = dXA.shape[-1]
        costmods, consistencies = [], [] # consistency is how often failures have higher cost than success
        for n in range(30):
            print(f"Cost Model {n} ...")

            # get covariance matrix of datapoints within success runs
            dXA_n = dXA[mask, n,:] # (success runs, dim)
            cov = dXA_n.T @ dXA_n / mask.sum() # (dim, dim)
            cov = cov + .1*np.eye(dim) # make full rank? regularizes and avoids collapse to constant cost within success samples, but reduces consistency

            # invert (large cost for delta directions that tend to be small in success)
            Q = np.linalg.pinv(cov)

            # save cost model including offset
            costmods.append([Q, XA0[n]])

            # measure consistency
            all_costs = np.diagonal(dXA[:,n,:] @ Q @ dXA[:,n,:].T) # (r,dim)@(dim,dim)@(dim,r) = (r,dim)@(dim,r) = (r,r)
            consistency = (all_costs[mask,None] < all_costs[~mask]).mean()

            print(f"Cost model {n} {dim=} {consistency=:.3f}")
            consistencies.append(consistency)

        with open(f"costvar_ni{num_interp}.pkl","wb") as f: pk.dump((costmods, consistencies), f)

    if view_cost_var:

        fig = pt.figure(figsize=(8,4))

        # load data
        run_filepaths = get_run_filepaths()
        with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)
        with open(f"costvar_ni{num_interp}.pkl","rb") as f: (costmods, consistencies) = pk.load(f)

        # extract noise and success labels
        _, _, stdevs, nfsteps = zip(*run_filepaths)
        stdevs = np.array(stdevs)
        nfsteps = np.array(nfsteps)

        # flatten data at each run and timestep
        X = obs.reshape(len(run_filepaths), 31, -1)
        A = cmd.reshape(len(run_filepaths), 30, -1)

        # concatenate (x,a) pairs
        X_prev = X[:,:-1] # observed before issuing new waypoint command
        A_curr = A # now new waypoint command is issued
        XA = np.concatenate([X_prev, A_curr], axis=2) # (r, n, dim)

        # one quadratic model per waypoint timestep: covariance within the success episodes only
        mask = (nfsteps == 6) # success runs
        for n, (Q, XA0_n) in enumerate(costmods):

            # get costs
            dXA_n = (XA[:,n] - XA0_n) # (runs, dim)
            costs = np.diagonal(dXA_n @ Q @ dXA_n.T) # (r,dim)@(dim,dim)@(dim,r) = (r,dim)@(dim,r) = (r,r)
            # costs = ((dXA_n[:,:,None] * dXA_n[:,None,:]) * Q).sum(axis=(1,2))
            # pt.figure()
            # pt.imshow((dXA_n[:,:,None] * dXA_n[:,None,:])[0])
            # pt.show()
            # # print(costs)
            # input('.')

            # plot successes and failures in separate colors
            if n == 0:
                pt.plot(n + .5*np.random.rand(mask.sum()), costs[mask], 'b.', label='success')
                pt.plot(n + .5*np.random.rand((~mask).sum()), costs[~mask], 'r.', label='failure')
            else:
                pt.plot(n + .5*np.random.rand(mask.sum()), costs[mask], 'b.')
                pt.plot(n + .5*np.random.rand((~mask).sum()), costs[~mask], 'r.')

        pt.legend()
        pt.xlabel("Waypoint")
        pt.ylabel("Cost")
        pt.yscale("log")
        pt.tight_layout()
        pt.savefig("costvar.pdf")
        pt.show()


    if do_lqr:

        # load data
        run_filepaths = get_run_filepaths()
        with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)
        #with open(f"dynfit_ni{num_interp}.pkl","rb") as f: (linmods, residuals, norms) = pk.load(f)
        with open(f"dynfit_pool_ni{num_interp}.pkl","rb") as f: (linmods, residuals, norms) = pk.load(f)
        if do_lqr_lc:
            # with open(f"costvar_ni{num_interp}.pkl","rb") as f: (costmods, consistencies) = pk.load(f)
            with open(f"costcvx_ni{num_interp}.pkl","rb") as f: C = pk.load(f)

        # flatten data at each run and timestep
        X = obs.reshape(len(run_filepaths), 31, -1)
        A = cmd.reshape(len(run_filepaths), 30, -1)
        dim_X = X.shape[2]
        dim_A = A.shape[2]

        # final time-step cost
        if do_lqr_lc:
            if only_x:
                P = {30: C[0]}
            else:
                P = {30: C[0][:dim_X,:dim_X]}
        else:
            P = {30: np.eye(X.shape[2])}
        K = {}

        # backwards solve
        max_eigs = {}
        max_eigs_open = {}
        for n in reversed(range(30)):

            # dynamics coefficients
            A_n = linmods[n % len(linmods)][:dim_X].T
            B_n = linmods[n % len(linmods)][dim_X:].T

            # cost coefficients
            # quad, mu = costmods[n]
            # Q_n = quad[:dim_X, :dim_X]
            # R_n = quad[dim_X:, dim_X:]
            # S_n = quad[:dim_X, dim_X:]

            # sanity check, still can give eigs > 1
            if do_lqr_lc:
                if only_x:
                    Q_n = C[n % 10]
                    R_n = np.eye(dim_A)
                    S_n = np.zeros((dim_X, dim_A))
                else:
                    Q_n = C[n % 10][:dim_X, :dim_X]
                    R_n = C[n % 10][dim_X:, dim_X:]
                    S_n = C[n % 10][:dim_X, dim_X:]
            else:
                Q_n = np.eye(dim_X)
                R_n = np.eye(dim_A)
                S_n = np.zeros((dim_X, dim_A))

            # X = inv(A) * B <-> AX = B
            K[n] = - np.linalg.lstsq(R_n + B_n.T @ P[n+1] @ B_n, B_n.T @ P[n+1] @ A_n + S_n.T, rcond=None)[0]

            P[n] = Q_n + A_n.T @ P[n+1] @ (A_n + B_n @ K[n]) + S_n @ K[n]

            # check stability
            max_eigs[n] = np.abs(np.linalg.eigvals(A_n + B_n @ K[n])).max()
            max_eigs_open[n] = np.abs(np.linalg.eigvals(A_n)).max()
            print(f"Solved {n}: max eig = {max_eigs[n]} (vs {max_eigs_open[n]} open loop)")

        with open(f"lqr_ni{num_interp}.pkl","wb") as f: pk.dump((K, max_eigs, max_eigs_open), f)
    
    if view_lqr:

        num_interps = [2]#[2,3,5,10]
        fig = pt.figure(figsize=(8,4))
        for ni in num_interps:
            with open(f"lqr_ni{ni}.pkl","rb") as f: (K, max_eigs, max_eigs_open) = pk.load(f)
            max_eigs = np.array([max_eigs[n] for n in sorted(max_eigs.keys())])
            max_eigs_open = np.array([max_eigs_open[n] for n in sorted(max_eigs_open.keys())])
            for n in range(len(max_eigs)): print(f"{ni=}, {n=}: eig prod = {np.prod(max_eigs[:n+1])} (vs {np.prod(max_eigs_open[:n+1])} open)")
            pt.plot(max_eigs, 'o-', label=f"interp={ni}")
        
        pt.yscale("log")
        pt.xlabel("Waypoint timestep")
        pt.ylabel("Max eig magnitude")
        pt.legend()
        
        pt.tight_layout()
        pt.savefig("lqr_stability.pdf")
        pt.show()
    
    if view_control_deviation:
        # visualize how far the controller's generated commands are from the ones in the success data

        # load data
        run_filepaths = get_run_filepaths()
        with open(f"lqr_chunks_ni{num_interp}.pkl","rb") as f: (obs, cmd) = pk.load(f)
        #with open(f"dynfit_ni{num_interp}.pkl","rb") as f: (linmods, residuals, norms) = pk.load(f)
        with open(f"dynfit_pool_ni{num_interp}.pkl","rb") as f: (linmods, residuals, norms) = pk.load(f)
        # with open(f"costvar_ni{num_interp}.pkl","rb") as f: (costmods, consistencies) = pk.load(f)
        with open(f"lqr_ni{num_interp}.pkl","rb") as f: (K, max_eigs, _) = pk.load(f)

        # extract noise and success labels
        _, _, stdevs, nfsteps = zip(*run_filepaths)
        stdevs = np.array(stdevs)
        nfsteps = np.array(nfsteps)

        # flatten data at each run and timestep
        X = obs.reshape(len(run_filepaths), 31, -1)
        A = cmd.reshape(len(run_filepaths), 30, -1)

        # pool across footstep cycles for nominal trajectory
        X_pool = np.concatenate([
            X[:,0:11],
            X[:,10:21],
            X[:,20:31],
        ], axis=0)
        A_pool = np.concatenate([
            A[:,0:10],
            A[:,10:20],
            A[:,20:30],
        ], axis=0)
        stdevs_pool = np.concatenate([stdevs]*3)
        success_pool = np.concatenate([
            (nfsteps >= 2),
            (nfsteps >= 4),
            (nfsteps == 6),
        ])

        # get target trajectory for one cycle (average within noiseless success)
        X0 = X_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)
        A0 = A_pool[(stdevs_pool=="0.0") & success_pool].mean(axis=0)

        # print(K[9])
        # input('..')

        pt.figure(figsize=(8,4))

        # calculate controller outputs at each datapoint
        #mask = (stdevs_pool!="0.0") & success_pool
        mask = np.ones(len(stdevs_pool),dtype=bool)
        for n in range(10):
            dA_n = (X_pool[mask,n] - X0[n]) @ K[n].T

            # plot actual vs controller
            # actual = np.linalg.norm(A_pool[mask,n] - A0[n], axis=1) 
            # cntrol = np.linalg.norm(dA_n, axis=1)
            actual = np.fabs(A_pool[mask,n] - A0[n]).mean(axis=1) 
            cntrol = np.fabs(dA_n).mean(axis=1)
            if n == 0:
                pt.plot(n + .5*np.random.rand(len(actual)), actual, 'b.', label="dataset")
                pt.plot(n + .5*np.random.rand(len(cntrol)), cntrol, 'r.', label="controller")
            else:
                pt.plot(n + .5*np.random.rand(len(actual)), actual, 'b.')
                pt.plot(n + .5*np.random.rand(len(cntrol)), cntrol, 'r.')

        pt.xlabel("Waypoint timestep")
        pt.ylabel("Command delta MAD from noiseless (deg)")
        pt.legend()
        pt.tight_layout()
        pt.savefig("lqr_ctrl.pdf")
        pt.show()
    
    if do_repickle:
        # pickle with protocol 2 and lists for poppy hardware

        with open("lqr_ni%d.pkl" % num_interp,"rb") as f: (Kontrollers, max_eigs, _) = pk.load(f)
        Kontrollers = {n: K.tolist() for n,K in Kontrollers.items()}
        max_eigs = {n:M.tolist() for n,M in max_eigs.items()}
        with open("lqr_ni%d.pkl2" % num_interp,"wb") as f: pk.dump((Kontrollers, max_eigs), f, protocol=2)

        with open("lqr_metadata.pkl","rb") as f: (stdevs, nfsteps) = pk.load(f)
        stdevs = stdevs.tolist()
        nfsteps = nfsteps.tolist()
        print(nfsteps[0], type(nfsteps[0]))
        with open("lqr_metadata.pkl2","wb") as f: pk.dump((stdevs, nfsteps), f, protocol=2)

        with open("lqr_chunks_ni%d.pkl" % num_interp,"rb") as f: (obs, cmd) = pk.load(f)
        obs = obs.tolist()
        cmd = cmd.tolist()
        with open("lqr_chunks_ni%d.pkl2" % num_interp,"wb") as f: pk.dump((obs, cmd), f, protocol=2)
