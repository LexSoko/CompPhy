#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Execution script for task 3 of the computational physics assignment 1

**** README: ****
* MathTeX notation is used in comment formula explanations!
"""


############################## imports #######################################
import numsolvers.runge_kutta as rk
import numsolvers.phase_space_difference as ps
import numsolvers.ODE_Functions as ode
import numpy as np
import matplotlib.pyplot as plt

######################### module level dunders ###############################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

############################# functions ######################################
def funcarray(*funcs):
    """
    A wrapper for code lengh reduction
    lets you input any amount of functions so that a y matrix can be given as input for the functions

    Returns:
        returns a lambda expression as a numpy array
    """
    return lambda t, y: np.array([f(t, y) for f in funcs])


# defining constants
g = 9.81
l = 1


# slightly deviating initial conditions

# difference between initial conditions
epsilon = 5e-2

# first sets of chaotic systems
y0_1a = np.array([0.5, 0, 0, 0])
y0_1b = y0_1a + epsilon
y0_2a = np.array([0, 0.5, 0.1, 0.2])
y0_2b = y0_2a - epsilon


y0_3a = np.array([0, 0, 0, 2])
y0_3b = y0_3a + epsilon


# list of initial conditions for easier plotting
y0_matrix = np.array([[y0_1a, y0_1b], [y0_2a, y0_2b], [y0_3a, y0_3b]])
y0_matrix = np.array(
    [
        [[0.5, 0.0, -8.0, 2.0], [0.55, 0.0, -7.95, 2.05]],
        [[0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 4.05]],
        [[0.0, 0.0, 8.0, 0.0], [0.0, 0.0, 8.05, 0]],
        [[0.0, 0.0, 4.0, 2.0], [0.0, 0.0, 4.05, 2.05]],
        [[0.4, 0.0, 0.0, 0.0], [0.45, 0.0, 0.0, 0]],
        [[0.5, 0.0, 0.0, 0.0], [0.55, 0.0, 0.0, 0.0]],
    ]
)

# computation and plotting of difference with the sets of initial values

# plotting configuration
figure, axis = plt.subplots(len(y0_matrix), 1, figsize=[22, 18])
figure.subplots_adjust(
    left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.15, hspace=0.4
)

# main title
figure.suptitle("Exercise 3.")
# configuration variables
e = 1e-3  # timestep-size
t0 = 0  # initial timepoint
t_max = 60  # final timepoint
maxdelta = 10  # maximum distance between two trajectories at which a system is considered chaotic
# this value is only set to 10 for easier plotting, a system is considered chaotic if the delta follows an upward trend

# main computation of 12 total ODES
for i in range(len(y0_matrix)):
    d = []
    t_a, y_a = rk.expl_rk_method(
        F=ode.DP_ODES(l, g),
        y0=y0_matrix[i][0],
        a_ij=rk.rk4_a_ij,
        bj=rk.rk4_b_j,
        cj=rk.rk4_c_j,
        t0=t0,
        t_max=t_max,
        e=e,
    )
    t_b, y_b = rk.expl_rk_method(
        F=ode.DP_ODES(l, g),
        y0=y0_matrix[i][1],
        a_ij=rk.rk4_a_ij,
        bj=rk.rk4_b_j,
        cj=rk.rk4_c_j,
        t0=t0,
        t_max=t_max,
        e=e,
    )
    delta = ps.delta(y_a, y_b)
    # titles of the subplots
    if max(delta) > maxdelta:

        axis[i].set_title("Chaotic System")
    else:
        axis[i].set_title("Stable System")

    # plotting the difference of the trajectory vectors in phase space
    axis[i].plot(
        t_a,
        delta,
        label="$\\delta(t)_{"
        + str(i)
        + "}$ \n "
        + "$\\theta_{1,0}$ = "
        + str(y0_matrix[i][0][0])
        + " $\\theta_{2,0}$ = "
        + str(y0_matrix[i][0][1])
        + " $p_{1,0}$ = "
        + str(y0_matrix[i][0][2])
        + " $p_{2,0}$ = "
        + str(y0_matrix[i][0][3])
        + " \n "
        + "$\\theta_{1,0}'$ = "
        + str(y0_matrix[i][1][0])
        + " $\\theta_{2,0}'$ = "
        + str(y0_matrix[i][1][1])
        + " $p_{1,0}'$ = "
        + str(y0_matrix[i][1][2])
        + " $p_{2,0}'$ = "
        + str(y0_matrix[i][1][3]),
    )
    axis[i].legend(loc="best")


plt.savefig(
    "./Ex1/plots/plot_3.pdf",
    dpi="figure",
    orientation="landscape",
)
# plt.show()
