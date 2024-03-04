#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Execution script for task 4 of the computational physics assignment 1
This script does basically the same as task 2 but bith an adaptive runge-kutta
method which was written in the numsolvers.runge_kutta module

**** README: ****
* MathTeX notation is used in comment formula explanations!
"""


############################## imports #######################################
import numpy as np
import matplotlib.pyplot as plt
from numsolvers import runge_kutta as rk
from tqdm import tqdm

######################### module level dunders ###############################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"
############################### script #######################################
# formula explanations:
# \vec{y} = ( \theta_1 , \theta_2 , p_1 , p_2 )
# \dot{\vec{y}} = ( \dot{\theta_1} , \dot{\theta_2} , \dot{p_1} , \dot{p_2} )
# index dict to make formulas more readable
i = {"theta_1": 0, "theta_2": 1, "p_1": 2, "p_2": 3}
inv_lat_i = {0: "\\theta_1", 1: "\\theta_2", 2: "p_1", 3: "p_2"}

# formulas for the double pendulum as provided on the assignment
# A and B are just to save some writing work
A = lambda t, y: (
    (y[i["p_1"]] * y[i["p_2"]] * np.sin(y[i["theta_1"]] - y[i["theta_2"]]))
    / (1 + np.sin(y[i["theta_1"]] - y[i["theta_2"]]) ** 2)
)
B = lambda t, y: (
    np.sin(y[i["theta_1"]] - y[i["theta_2"]])
    * np.cos(y[i["theta_1"]] - y[i["theta_2"]])
    * (
        y[i["p_1"]] ** 2
        + 2 * (y[i["p_2"]] ** 2)
        - 2
        * y[i["p_1"]]
        * y[i["p_2"]]
        * np.cos(y[i["theta_1"]] - y[i["theta_2"]])
    )
    / ((1 + np.sin(y[i["theta_1"]] - y[i["theta_2"]]) ** 2) ** 2)
)
# F is the functional F(t,\vec{y}(t)) = \dot{\vec{y}}
F = lambda t, y, w: np.array(
    [
        # \theta_1:
        (y[i["p_1"]] - y[i["p_2"]] * np.cos(y[i["theta_1"]] - y[i["theta_2"]]))
        / (1 + np.sin(y[i["theta_1"]] - y[i["theta_2"]]) ** 2),
        # \theta_2
        (
            2 * y[i["p_2"]]
            - y[i["p_1"]] * np.cos(y[i["theta_1"]] - y[i["theta_2"]])
        )
        / (1 + np.sin(y[i["theta_1"]] - y[i["theta_2"]]) ** 2),
        # p_1
        B(t, y) - A(t, y) - 2 * w**2 * np.sin(y[i["theta_1"]]),
        # p_2
        A(t, y) - B(t, y) - w**2 * np.sin(y[i["theta_2"]]),
    ]
)
# Function for the System Energy
T = lambda y, dy, m, l, g: (m * l**2 / 2) * (
    2 * dy[i["theta_1"]] ** 2
    + dy[i["theta_2"]] ** 2
    + 2
    * dy[i["theta_1"]]
    * dy[i["theta_2"]]
    * np.cos(y[i["theta_1"]] - y[i["theta_2"]])
)
V = lambda y, dy, m, l, g: (
    -m * g * l * (2 * np.cos(y[i["theta_1"]] + np.cos(y[i["theta_2"]])))
)
E = lambda y, dy, m, l, g: T(y, dy, m, l, g) + V(y, dy, m, l, g)

# Functions to map angles into a intervall [-pi,+pi]
calc_simp_theta = lambda th: np.where(
    th >= 0.0,
    np.where(th <= np.pi, th, th - 2 * np.pi),
    np.where(th >= -np.pi, th, th + 2 * np.pi),
)
map_theta = lambda theta: calc_simp_theta(np.fmod(theta, 2 * np.pi))

# pendulum constants
l1 = l2 = l = 1
m1 = m2 = m = 1
g = 9.81
w = np.sqrt(g / l)

# initial conditions each list has one entry for each starting condition:
n_cond = 4
display_time = 10  # time series data will be plottet up to this time
y_0_list = [  # list of starting condition
    np.array([0.0, 0.0, 4.0, 2.0]),
    np.array([0.0, 0.0, 0.0, 4.0]),
    np.array([0.0, 0.0, 8.0, 0.0]),
    np.array([0.4, 0.0, 0.0, 0.0]),
]
w_list = [w] * n_cond  # list of omegas for equation
t_min_list = [0] * n_cond  # list of starting times
t_max_list = [10] * n_cond  # list of ending times
dt_list = [0.1] * n_cond  # list of timesteps
n_tmax_timeplots = [  # calculate maximum index for
    int(display_time // dt_list[i_cond]) for i_cond in range(0, n_cond)
]

# result lists with one entry for each staring condition
t_list = [0 for n in range(0, n_cond)]
y_list = [0 for n in range(0, n_cond)]
dy_list = [0 for n in range(0, n_cond)]
E_list = [0 for n in range(0, n_cond)]
fE_list = [0 for n in range(0, n_cond)]
it_list = [0 for n in range(0, n_cond)]
err_list = [0 for n in range(0, n_cond)]
t_list_filtered = [0 for n in range(0, n_cond)]
y_list_filtered = [0 for n in range(0, n_cond)]

# plotting setup
fig = plt.figure(num=1, figsize=[22, 18])
fig.suptitle("task 4 adaptive rk", fontsize=20)
fig.subplots_adjust(
    left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.15, hspace=0.2
)

# this loop loops over the different starting conditions
# the function tqdm() gives a progress bar in the terminal
for i_cond in tqdm(range(0, n_cond), desc="condition-loop"):
    # plotting setup
    ax_txt = fig.add_subplot(7, n_cond, i_cond + 1)
    ax_theta = fig.add_subplot(7, n_cond, i_cond + 1 + n_cond)
    ax_p = fig.add_subplot(7, n_cond, i_cond + 1 + n_cond * 2)
    ax_E = fig.add_subplot(7, n_cond, i_cond + 1 + n_cond * 3)
    ax_it = fig.add_subplot(7, n_cond, i_cond + 1 + n_cond * 4)
    ax_err = fig.add_subplot(7, n_cond, i_cond + 1 + n_cond * 5)
    ax_poincare = fig.add_subplot(7, n_cond, i_cond + 1 + n_cond * 6)

    # plot text description
    ax_txt.set_xticks([])
    ax_txt.set_yticks([])
    ax_txt.text(
        0.5,
        0.5,
        "starting conditions\n$\\theta_{{1,0}}$={:2.2f}\n$\\theta_{{2,0}}$={:2.2f}\n$p_{{1,0}}$={:2.2f}\n$p_{{2,0}}$={:2.2f}".format(
            y_0_list[i_cond][i["theta_1"]],
            y_0_list[i_cond][i["theta_2"]],
            y_0_list[i_cond][i["p_1"]],
            y_0_list[i_cond][i["p_2"]],
        ),
        verticalalignment="center",
        horizontalalignment="center",
    )

    # execute the adaptive runge-kutta 4th order
    (
        t_list[i_cond],
        y_list[i_cond],
        it_list[i_cond],
        err_list[i_cond],
    ) = rk.adpt_rk4_method(
        lambda t, y: F(t, y, w_list[i_cond]),
        y_0_list[i_cond],
        t0=t_min_list[i_cond],
        t_max=t_max_list[i_cond],
        e0=dt_list[i_cond],
    )

    # calculate derivatives for system energy
    dy_list[i_cond] = F(t_list[i_cond], y_list[i_cond], w_list[i_cond])
    # calculate system energy
    E_list[i_cond] = E(y_list[i_cond], dy_list[i_cond], m, l, g)

    # map the angles to [-pi,pi] since for high start impulses,
    # the pendulum makes loops endlessly and angles increase continiuously
    y_list[i_cond][i["theta_1"]] = map_theta(y_list[i_cond][i["theta_1"], :])
    y_list[i_cond][i["theta_2"]] = map_theta(y_list[i_cond][i["theta_2"], :])

    t_indizes = np.where(t_list[i_cond] < display_time)

    # plot the angles
    ax_theta.plot(
        t_list[i_cond][t_indizes],
        y_list[i_cond][i["theta_1"]][t_indizes],
        label=r"$\theta_1$",
        color="red",
    )
    ax_theta.plot(
        t_list[i_cond][t_indizes],
        y_list[i_cond][i["theta_2"]][t_indizes],
        label=r"$\theta_2$",
        color="blue",
    )
    ax_theta.legend()

    # plot the impulses
    ax_p.plot(
        t_list[i_cond][t_indizes],
        y_list[i_cond][i["p_1"]][t_indizes],
        label=r"$p_1$",
        color="red",
    )
    ax_p.plot(
        t_list[i_cond][t_indizes],
        y_list[i_cond][i["p_2"]][t_indizes],
        label=r"$p_2$",
        color="blue",
    )
    ax_p.legend()

    # plot the system energy
    ax_E.plot(
        t_list[i_cond][t_indizes],
        E_list[i_cond][t_indizes],
        label=r"$E(t)$",
        color="green",
    )
    ax_E.legend()

    # plot the adaptive rk iterations per datapoint
    ax_it.plot(
        t_list[i_cond][t_indizes],
        it_list[i_cond][t_indizes],
        label="it/t_step",
        color="green",
    )
    ax_it.legend()

    # plot the aproximated error per phasespace for each coordinate
    for i_err in range(4):
        ax_err.plot(
            t_list[i_cond][t_indizes],
            err_list[i_cond][i_err][t_indizes],
            label="$err_{{{}}}$".format(inv_lat_i[i_err]),
        )
    ax_err.set_yscale("log")
    ax_err.legend()

    # calculate and plot the poincare maps
    # this is done in a try catch, since the datapointselection
    # for theta2 = 0 don't always give enough values, since we have to
    # actually compare 0.0-epsilon > theta2 > 0.0+epsilon
    try:
        # select indizes with valid datapoints for poincare map
        indizes = np.where(
            np.logical_and(
                np.isclose(
                    y_list[i_cond].T[:, i["theta_2"]],
                    0.0,
                    rtol=dt_list[i_cond],
                    atol=dt_list[i_cond],
                ),
                y_list[i_cond].T[:, i["p_2"]] > 0.0,
            )
        )

        # get t and y for all timepoints which are valid
        t_list_filtered[i_cond] = t_list[i_cond][indizes]
        y_list_filtered[i_cond] = np.squeeze(
            np.transpose(y_list[i_cond][:, indizes], [0, 2, 1])
        )  # here we had to do some array formatting so we get the correct shape

        # plot poincare map
        ax_poincare.plot(
            y_list_filtered[i_cond][i["theta_1"], :],
            y_list_filtered[i_cond][i["p_1"], :],
            ".",
            markersize=1,
            label=r"($\theta_1$,$p_1$)",
            color="blue",
        )
        ax_poincare.legend()
    except:
        print(
            "plotting of conditions ",
            i_cond,
            " not possible due to missing poincare data",
        )
fig.savefig("./Ex1/plots/plot_4.pdf", dpi="figure")
