#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Execution script for task 1 of the computational physics assignment 1

**** README: ****
* MathTeX notation is used in comment formula explanations!
"""


import numsolvers.euler_method as eum
import numsolvers.runge_kutta as rk
import numsolvers.lightness as li

import numpy as np
import matplotlib.pyplot as plt
import colorsys
import matplotlib
import copy

__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

# %%
# Task 1.a
""" The solution for task 1.a is found in the runge_kutta.py file in the numsolver file. 
    The function is called expl_rk_method()"""


# %%
# Task 1.b

# Define constants
g = 9.81
l = 1
m = 1
w = g / l

# Define starting values and equationsystem
y0 = np.array([0, 2 * w])
t0 = 0
t_max = 10
t_delta = 1
# a_ij = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
# b_j = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
# c_j = np.array([0, 0.5, 0.5, 1])
f = lambda t, y: np.array([y[1], -((w) ** 2) * np.sin(y[0])])


# call RK-function
t, y = rk.expl_rk_method(
    F=f,
    y0=y0,
    t0=0,
    t_max=t_max,
    e=t_delta,
    a_ij=rk.rk4_a_ij,
    bj=rk.rk4_b_j,
    cj=rk.rk4_c_j,
)

# y is the solution of the differential equation, where y[0] = theta and y[1] = $\dot\theta$

# %%
# Task 1.c

# Define constants
g = 9.81
l = 1
m = 1
w = g / l

# define starting conditions
y0 = np.array([0, 2 * w])
t0 = 0
t_max = 25

# define increments
t_delta_list = [0.01, 0.001]

# define  variation variables of the starting conditions
theta_delta = 10
v_delta = w / 2
starts = 1  # the starting conditions get variated by 2*starts times

# define equations
f = lambda t, y: np.array([y[1], -((w) ** 2) * np.sin(y[0])])
E = (
    lambda v, deta: 1 / 2 * m * v**2 + l * (1 - np.cos(deta)) * m * g
)  # totall energie equation


# generate placeholder lists to fill later
inner_list = [0] * (1 + 2 * starts)
outer_list = [0] * len(t_delta_list)

# deepcopies are used, so that two lists aren't the same one by reference
for i in range(0, len(outer_list)):
    outer_list[i] = copy.deepcopy(inner_list)

# first index gives chooses the increment, second index chooses the starting conditions
t_list = copy.deepcopy(outer_list)
y_eu = copy.deepcopy(outer_list)
y_rk = copy.deepcopy(outer_list)
y0_arr = copy.deepcopy(outer_list)


# call RK- & Euler-method for all increments and starting conditions
for j in range(0, len(t_delta_list)):  # iterates over different timesteps
    for k in [  # iterates over starting condition deviations
        *range(-starts, starts + 1)
    ]:  # * to unpack range type into list of values
        y0_temp = np.array([y0[0] + k * theta_delta, y0[1] + k * v_delta])

        t_list[j][k], y_eu[j][k] = eum.euler_method(
            f, y0_temp, t0=t0, t_max=t_max, e=t_delta_list[j]
        )
        y0_arr[j][k] = copy.deepcopy(y0_temp)
        t, y_rk[j][k] = rk.rk4_method(f, y0_temp, t_vect=t_list[j][k])
        # t, y_rk[j][k] = rk.rk4_method(
        #    f, y0_temp, t0=t0, t_max=t_max, e=t_delta_list[j]
        # )


# Plotting

# variables for subplots; will later genrate rows*cols subplots
rows = 4
cols = len(t_delta_list)

# size and shape of the figure
fig = plt.figure(figsize=[22, 11], num=1)
fig.subplots_adjust(
    left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.15, hspace=0.55
)
fig.suptitle("Exercise 1.")

# this code block is used for brightnes variation in different plot graphs
# define the ploting color of the two methods and convert them to rgb
color_rk = matplotlib.colors.ColorConverter.to_rgb("red")
color_eu = matplotlib.colors.ColorConverter.to_rgb("blue")
# generate values for later lightness variaton of the colors
n = np.linspace(0.5, 1.5, num=2 * starts + 1)

# plot the method solutions
for j in range(1, len(t_delta_list) + 1):
    l = j - 1  # index for list adressing, since axes index starts with 1

    # Euler theta
    ax = fig.add_subplot(rows, cols, j)
    ax.set_title(
        f"$\\theta$ für Eulermethode & $\\epsilon = {t_delta_list[l]}$ "
    )
    for k in range(0, starts * 2 + 1):
        c = li.scale_lightness(
            color_eu, n[k]
        )  # set the lightness of the color relative to k
        ax.plot(
            t_list[l][k],
            y_eu[l][k][0],
            color=c,
            linewidth=2,
            label=f"$\\theta_0 = {round(y0_arr[l][k][0], 2)}$, $\\dot \\theta_0 = {round(y0_arr[l][k][1], 2)}$",
        )
    ax.legend()
    ax.grid()

    # RK4 theta
    ax = fig.add_subplot(rows, cols, cols + j)
    ax.set_title(
        f"$\\theta$ für RK4-Methode & $\\epsilon = {t_delta_list[l]}$ "
    )
    for k in range(0, starts * 2 + 1):
        c = li.scale_lightness(
            color_rk, n[k]
        )  # set the lightness of the color relative to k
        ax.plot(
            t_list[l][k],
            y_rk[l][k][0],
            color=c,
            linewidth=2,
            label=f"$\\theta_0 = {round(y0_arr[l][k][0], 2)}$, $\\dot \\theta_0 = {round(y0_arr[l][k][1], 2)}$",
        )
    ax.legend()
    ax.grid()

    # Euler Energy
    ax = fig.add_subplot(rows, cols, cols * 2 + j)
    ax.set_title(f"$E$ für Eulermethode & $\\epsilon = {t_delta_list[l]}$ ")
    for k in range(0, starts * 2 + 1):
        c = li.scale_lightness(
            color_eu, n[k]
        )  # set the lightness of the color relative to k
        ax.plot(
            t_list[l][k],
            E(y_eu[l][k][1], y_eu[l][k][0]),
            linewidth=2,
            color=c,
            label=f"$\\theta_0 = {round(y0_arr[l][k][0], 2)}$, $\\dot \\theta_0 = {round(y0_arr[l][k][1], 2)}$",
        )
    ax.legend()
    ax.grid()

    # RK4 Energy
    ax = fig.add_subplot(rows, cols, cols * 3 + j)
    ax.set_title(f"$E$ für RK4-Methode & $\\epsilon = {t_delta_list[l]}$ ")
    for k in range(0, starts * 2 + 1):
        c = li.scale_lightness(
            color_rk, n[k]
        )  # set the lightness of the color relative to k
        ax.plot(
            t_list[l][k],
            E(y_rk[l][k][1], y_rk[l][k][0]),
            linewidth=2,
            color=c,
            label=f"$\\theta_0 = {round(y0_arr[l][k][0], 2)}$, $\\dot \\theta_0 = {round(y0_arr[l][k][1], 2)}$",
        )
    ax.legend()
    ax.grid()

fig.savefig("./Ex1/plots/plot_1c.pdf", dpi="figure")

""" 
The average total energy is conserved, while the total energy oszilates periodicly.
Due to our research, the Runge-Kutta-Methods don´t conserve the total energy.
We couldn´t find out what the exact technical reason for that is. 
Our assumption is, that due to the periodic oszillation of the used differential 
equations, the error of the RK-method oszillates between negative and positive direction.
This leads to a periodic increase and decrease of the total energy of the system. 
"""


#%%
# Task 1.d

# Define constants
g = 9.81
l = 1
m = 1
w = g / l

# define equations
f = lambda t, y: np.array([y[1], -((w) ** 2) * np.sin(y[0])])
T_analytisch = (
    2 * np.pi * np.sqrt(l / g)
)  # analytic solution for the period with small angle approximation

# define starting conditions
y0 = np.array([np.pi / 3, 0])
t0 = 0
t_max = 1000
t_delta = 0.001

# call RK-4-Method
t, y = rk.rk4_method(f, y0, t0=t0, t_max=t_max, e=t_delta)

# FFT to find the period of the theta oszilation
ft = np.fft.rfft(y[0])
freqs = np.fft.rfftfreq(
    len(y[0]), t[1] - t[0]
)  # get the frequency axis from the time axis
mags = abs(
    ft
)  # since no complex phase information needed, we only take real numbers

# get the freq with the highest amplitude
i = mags.argmax(axis=0)
f_num = freqs[i]

T_delta = abs(T_analytisch - (1 / f_num))


# plot
fig = plt.figure(num=2)
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Exercise 1.d")
ax.plot([], [], color="white", label=f"$\\Delta T =${round(T_delta, 3)}")
ax.plot(freqs, mags, color="blue", label=f"FFT von $\\theta$ für RK4-Methode")
ax.axvline(
    x=(1 / T_analytisch),
    color="red",
    label=f"Frequenz $f$ von $\\theta$ für Kleinwinkelnäherung",
)
ax.set_xlim(0, 10)
ax.legend()
ax.grid()
fig.savefig("./Ex1/plots/plot_1d.pdf", dpi="figure")


# %%
