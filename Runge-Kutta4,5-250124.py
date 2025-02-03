# %%
# This is a repeat of previous analyses to establish ODE fits with MCMC models, but this time using Runge-Kutta(4,5) solver in accordance with
# the reviewer instructions.
import os
import time
from multiprocessing import Pool
import scipy as sc
import scipy.integrate as integ
from scipy.integrate import quad
import scipy.stats as st
import scipy.optimize
import pandas as pd
import numpy as np

import sys

# Numerical differentiation package
# import numdifftools as ndt

# MCMC package
import emcee

# MCMC results visualization package
import corner

# Import pyplot for plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

# Seaborn, useful for graphics
# import seaborn as sns

# Bokeh stuff
# import bokeh.io

# Magic function to make matplotlib inline; other style specs must come AFTER
# %matplotlib inline

# This enables high res graphics inline (only use with static plots (non-Bokeh))
# SVG is preferred, but there is a bug in Jupyter with vertical lines
# %config InlineBackend.figure_formats = {'svg', 'retina'}

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2,
      'axes.labelsize': 18,
      'axes.titlesize': 18,
      'axes.facecolor': 'DFDFE5'}
# sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 2.5})
# sns.set_style('darkgrid', {'axes.facecolor': '(0.875, 0.875, 0.9)'})


# %%
# The equation to fit is for two gene modules, lambda1 and lambda2:


def f(t, G, a1, a2, a3, alpha1, beta1, alpha2):
    # t is timepoint
    # p is the parameter tuple
    # G is the lambda tuple
    G1, G2 = G
    if np.isnan(G1) or np.isnan(G2):
        return np.array([np.nan, np.nan])
    p = (a1, a2, a3, alpha1, beta1, alpha2)
    dG = np.zeros((2))
    dG[0] = formula1(G[0], G[1], p)
    dG[1] = formula2(G[0], G[1], p)
    return dG


def formula1(g1, g2, params):
    # ODE 1 for lambda1 (g1)
    a1, a2, a3, alpha1, beta1, alpha2 = params
    return alpha1 - beta1 * g1 - a1 * g2


def formula2(g1, g2, params):
    # ODE 2 for lambda2 (g2)
    a1, a2, a3, alpha1, beta1, alpha2 = params
    return alpha2 + a2 * g1 - a3 * g2


'''
# Plug the above into a Runge-Kutta(4-5) solver from scipy.integrate
# Scenarios are numbered 1 through 8
Greal = df[df['scenario'] == 1][['day', 'g1', 'g2']]

# %%
Greal

# %%
t_span = (min(Greal['day']), max(Greal['day']))
g_init = Greal.iloc[0, 1:]
# parameter values
p = (0.49, 0.09, 0.38, 8.39, -0.05, 6.33)
sol = integ.solve_ivp(f, t_span, g_init, method='RK45',
                      args=p, max_step=0.1, t_eval=Greal['day'])

# %%
sol

# %%
plt.plot(sol.t, sol.y[0])
plt.plot(sol.t, sol.y[1])

# %%
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
y = pd.DataFrame(x.transpose())
a = [0, 2, 3]
y
'''

# %%
# Theoretical model for populations
# Was previously Euler method. Now doing Runge-Kutta(4, 5) on the suggestion of the reviewer.


def popDynamics(p, d):
    data_times, (c01, c02) = d
    data_times = np.array(data_times)
    a1, a2, a3, alpha1, beta1, alpha2 = p
    max_step = 0.25  # This is now the max step size within the Runge-Kutta solver
    td = np.arange(data_times[0], data_times[-1] + max_step, max_step)
    # l_theor_full = np.zeros([len(td), len(d[1])])
    # l_theor_full[0] = np.array([[c01, c02]])
    l_theor = np.array([[c01, c02]])

    # New Runge-Kutta(4, 5)
    sol = integ.solve_ivp(f, (data_times[0], data_times[-1]), (c01, c02), method='RK45',
                          args=p, t_eval=data_times, max_step=max_step)
    l_theor_full = np.array([sol.t, sol.y[0], sol.y[1]]).transpose()
    l_theor = np.array([sol.y[0], sol.y[1]]).transpose()

    # Defunct Euler increments first order finite difference scheme
    # for i in range(1, len(td)):
    #    l_theor_full[i][0] = l_theor_full[i-1][0] + euler_step * (alpha1 - beta1 * l_theor_full[i-1][0] \
    #                                                             - a1 * l_theor_full[i-1][1])
    #    l_theor_full[i][1] = l_theor_full[i-1][1] + euler_step * (alpha2 + a2 * l_theor_full[i-1][0] \
    #                                                             - a3 * l_theor_full[i-1][1])
    #    if td[i] in data_times:
    #        l_theor = np.append(l_theor, [[l_theor_full[i][0], l_theor_full[i][1]]], axis=0)
    return l_theor_full, l_theor


def log_posterior(p, d, l):
    '''
    Assuming uniform priors for all of the parameters and Jeffreys prior for sigma. Assuming Gaussian 
    distributed measurements in the experiment. 
    '''
    data_times, (c01, c02) = d
    a1, a2, a3, alpha1, beta1, alpha2, s = p
    # Zero probability of having non-positive value of stdev
    if s <= 0:
        return -np.inf
    _, pops_theor = popDynamics(p[:-1], d)  # No s
    # Zero probability of having absurd population values
    for i in range(len(pops_theor)):
        for j in range(2):
            if np.abs(pops_theor[i][j]) > 150.0 or np.isnan(pops_theor[i][j]) or pops_theor[i][j] < -10.0:
                return -np.inf
    # for val in p[2:]:
    #    if val < 0:
    #        return -np.inf
    if beta1 > 3:
        # Large beta1 seems to induce oscillations
        return -np.inf
    if abs(alpha1) > 20 or abs(alpha2) > 20:
        # Large alpha values seem to encourage oscillations
        return -np.inf
    if a1 > 1 or a2 > 1 or a3 > 1:
        return -np.inf
    if a1 < -3 or a2 < -3 or a3 < -3:
        return -np.inf
    # Special measure for scenario 2: make sure lambda1 curves upward before going down
    # if pops_theor[0, 0] > pops_theor[1, 0]:
    #    return -np.inf

    # Nyquist frequency: 1 day^-1 so keep frequency below or equal to that
    if 1 / (2 * np.pi) * np.sqrt(np.sqrt(a1**2.0 * a3**2.0) + np.sqrt(beta1**2.0 * a2**2.0)) > 1.0:
        return -np.inf
    return -(2 * len(data_times) + 1) * np.log(s) - 0.5 / s**2.0 * (np.sum((l[:, 1]-pops_theor[:, 1])**2.0)
                                                                    + np.sum((l[:, 0]-pops_theor[:, 0])**2.0))


# main function
if __name__ == '__main__':
    # Import the df
    # Originally from 170924 MCMC Lambda gene expressions ipynb
    # Previous results of fittings from MCMC are summarized in 180422 new TF fittings summary.docx and in 180420 TF fittings summary.docx
    df = pd.read_excel('data/lambda_expressions_fitting/180420 TF fittings/average top gene for fitting.xlsx', comment='#',
                       header=0, sheet_name='tidy with dummy')

    n_dim = 7  # 7 parameters in the model
    n_walkers = 500  # Number of MCMC walkers
    n_burn = 500  # 500 burn-in steps
    n_steps = 1000  # Total number of steps after burn-in

    # Seeding random number generator
    np.random.seed(1)

    # Numpy automatic parallelization may cause problems with emcee, so we'll turn that off
    # os.environ["OMP_NUM_THREADS"] = "1"

    # Set up pool
    Pool()

    for scenario in [1, 3, 4, 5, 6, 7, 8]:  # range(1, 9):
        # p0[i, j] is the starting point for the ith walk for the jth variable
        p0 = np.empty((n_walkers, n_dim))

        # Set this per scenario with rough estimated ranges first
        if scenario == 1:
            p0[:, 0] = np.random.uniform(0.3, 0.7, n_walkers)
            p0[:, 1] = np.random.uniform(0.02, 0.15, n_walkers)
            p0[:, 2] = np.random.uniform(0.3, 0.5, n_walkers)
            p0[:, 3] = np.random.uniform(7, 10, n_walkers)
            p0[:, 4] = np.random.uniform(-0.1, -0.02, n_walkers)
            p0[:, 5] = np.random.uniform(5, 8, n_walkers)
        elif scenario == 2:
            p0[:, 0] = np.random.uniform(-1, 0, n_walkers)
            p0[:, 1] = np.random.uniform(-1, 0, n_walkers)
            p0[:, 2] = np.random.uniform(0, 1, n_walkers)
            p0[:, 3] = np.random.uniform(-10, 0, n_walkers)
            p0[:, 4] = np.random.uniform(-1, 0, n_walkers)
            p0[:, 5] = np.random.uniform(0, 10, n_walkers)
        elif scenario == 3:
            p0[:, 0] = np.random.uniform(-0.2, -0.1, n_walkers)
            p0[:, 1] = np.random.uniform(-0.15, -0.05, n_walkers)
            p0[:, 2] = np.random.uniform(0.1, 0.4, n_walkers)
            p0[:, 3] = np.random.uniform(-4, -3, n_walkers)
            p0[:, 4] = np.random.uniform(-0.1, 0, n_walkers)
            p0[:, 5] = np.random.uniform(5, 8, n_walkers)
        elif scenario == 4:
            p0[:, 0] = np.random.uniform(0.3, 0.4, n_walkers)
            p0[:, 1] = np.random.uniform(0.1, 0.2, n_walkers)
            p0[:, 2] = np.random.uniform(0.35, 0.45, n_walkers)
            p0[:, 3] = np.random.uniform(1, 4, n_walkers)
            p0[:, 4] = np.random.uniform(-0.15, 0, n_walkers)
            p0[:, 5] = np.random.uniform(0, 3, n_walkers)
        elif scenario == 5:
            p0[:, 0] = np.random.uniform(0.3, 0.6, n_walkers)
            p0[:, 1] = np.random.uniform(-0.35, -0.15, n_walkers)
            p0[:, 2] = np.random.uniform(0.25, 1.25, n_walkers)
            p0[:, 3] = np.random.uniform(5, 7, n_walkers)
            p0[:, 4] = np.random.uniform(0.1, 0.2, n_walkers)
            p0[:, 5] = np.random.uniform(8, 10, n_walkers)
        elif scenario == 6:
            p0[:, 0] = np.random.uniform(-0.25, -0.15, n_walkers)
            p0[:, 1] = np.random.uniform(-0.1, 0, n_walkers)
            p0[:, 2] = np.random.uniform(0.2, 0.3, n_walkers)
            p0[:, 3] = np.random.uniform(-5, -2, n_walkers)
            p0[:, 4] = np.random.uniform(0, 0.05, n_walkers)
            p0[:, 5] = np.random.uniform(5, 8, n_walkers)
        elif scenario == 7:
            p0[:, 0] = np.random.uniform(-3, 0, n_walkers)
            p0[:, 1] = np.random.uniform(0, 0.1, n_walkers)
            p0[:, 2] = np.random.uniform(0.1, 1.5, n_walkers)
            p0[:, 3] = np.random.uniform(-7, -5, n_walkers)
            p0[:, 4] = np.random.uniform(0.1, 0.3, n_walkers)
            p0[:, 5] = np.random.uniform(2, 5, n_walkers)
        elif scenario == 8:
            p0[:, 0] = np.random.uniform(0.3, 0.7, n_walkers)
            p0[:, 1] = np.random.uniform(0, 0.08, n_walkers)
            p0[:, 2] = np.random.uniform(0.05, 0.25, n_walkers)
            p0[:, 3] = np.random.uniform(8, 13, n_walkers)
            p0[:, 4] = np.random.uniform(0.01, 0.1, n_walkers)
            p0[:, 5] = np.random.uniform(2, 5, n_walkers)

        p0[:, n_dim-1] = np.random.exponential(0.1, n_walkers)  # sigma

        # Run the scenarios, 1 through 8
        Greal = df[df['scenario'] == scenario][['day', 'g1', 'g2']]

        t_span = (min(Greal['day']), max(Greal['day']))
        g_init = Greal.iloc[0, 1:]

        # Creating a MCMC sampler
        with Pool() as pool:
            emp_data = np.array([Greal['g1'], Greal['g2']])
            emp_data = np.transpose(emp_data)
            sampler_p = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior, args=((Greal['day'],
                                                                                      emp_data[0]), emp_data,), pool=pool, threads=16)

            # Do burn-in
            print('Starting burn-in.')
            start = time.time()
            pos, prob, state = sampler_p.run_mcmc(
                p0, n_burn, progress=True, store=False)
            end = time.time()
            multi_time = end - start
            print('Burn-in took {0:.1f} seconds'.format(multi_time))

            # Actual run
            print('Starting actual run.')
            start = time.time()
            print(os.environ.get('OMP_NUM_THREADS'))
            sampler_p.run_mcmc(pos, n_steps, log_prob0=prob,
                               progress=True, store=True)
            # for i, result in enumerate(sampler_p.sample(pos, log_prob0=prob, iterations=n_steps)):
            #    print("{0:5.1%}".format(float(i) / n_steps))
            end = time.time()
            multi_time = end - start
            print("Actual run took {0:.1f} seconds".format(multi_time))

        fig, ax = plt.subplots(7, 1, sharex=True)
        for i in range(7):
            ax[i].plot(sampler_p.chain[0, :, i], 'k-', lw=0.4)
            ax[i].plot([0, n_steps-1],
                       [sampler_p.chain[0, :, i].mean(), sampler_p.chain[0, :, i].mean()], 'r-', lw=0.4)

        ax[6].set_xlabel('sample number')
        plt.savefig('output/figures/walkers_scenario_' +
                    str(scenario) + '.pdf')
        plt.close()

        # Get the index of the most probable parameter set
        max_ind = np.argmax(sampler_p.flatlnprobability)

        # Pull out values.
        a1, a2, a3, alpha1, beta1, alpha2, s = sampler_p.flatchain[max_ind, :]
        # Calculate errors
        a1_e, a2_e, a3_e, alpha1_e, beta1_e, alpha2_e, s_e = sampler_p.flatchain.std(
            axis=0) / np.sqrt(n_walkers)

        # Print the results
        params_output = open('output/scenario_' + str(scenario) + '.txt', 'w')
        params_output.write('Most probable parameter values:' +
                            '\na1: ' + str(a1) + ' +- ' + str(a1_e) +
                            '\na2: ' + str(a2) + ' +- ' + str(a2_e) +
                            '\na3: ' + str(a3) + ' +- ' + str(a3_e) +
                            '\nalpha1: ' + str(alpha1) + ' +- ' + str(alpha1_e) +
                            '\nbeta1: ' + str(beta1) + ' +- ' + str(beta1_e) +
                            '\nalpha2: ' + str(alpha2) + ' +- ' + str(alpha2_e) +
                            '\ns: ' + str(s) + ' +- ' + str(s_e))
        params_output.close()

        # Summarize the fittings
        _ = plt.scatter(Greal['day'], Greal['g1'], color='green')
        _ = plt.scatter(Greal['day'], Greal['g2'], color='blue')
        print(str([a1, a2, a3, alpha1, beta1, alpha2]))
        fit_results = integ.solve_ivp(f, (Greal['day'][0], Greal['day'][0] + 80), g_init, method='RK45',
                                      args=(a1, a2, a3, alpha1, beta1, alpha2), max_step=0.1)

        # _ = plt.plot(td, popDynamics((-.496, -.016, .012, -19.73, 0.075, 1.98, s), (td, emp_data[0]))[0], lw=0.6, alpha=0.8)
        _ = plt.plot(fit_results.t, fit_results.y[0], 'g',
                     fit_results.t, fit_results.y[1], 'b', lw=0.6, alpha=0.8)
        _ = plt.legend([r'$\lambda_1$', r'$\lambda_2$'], loc='upper right')
        _ = plt.xlabel('Time (day)')
        _ = plt.ylabel('population')
        _ = plt.title('Scenario ' + str(scenario))
        _ = plt.savefig('output/figures/best_fit_scenario_' +
                        str(scenario) + '.pdf')
        plt.close()

        fit_results_df = pd.DataFrame(
            np.array([fit_results.t, fit_results.y[0], fit_results.y[1]]).transpose())
        fit_results_df.columns = ['day', 'g1', 'g2']
        fit_results_df.to_excel(
            'output/best_fit_scenario_' + str(scenario) + '.xlsx')
