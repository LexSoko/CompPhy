#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Assignment 3 Task 3 Split Operator Method
"""


############################## imports #########################################
import timeit
import numpy as np
import matplotlib.pyplot as plt
import numsolvers.split_operator as spop
import numsolvers.cranck_nicolson as crnic
from tqdm import tqdm

######################### module level dunders #################################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

########################## execution script ####################################
#%%
############################### Task 3.a #######################################
"""
Solve the free particle (V(x) = 0) time-dependent Schrödinger Equ with the
split operator method. As initial wave function choose Gaussian wave packets
with sigma = 20, q = 2, 20 and x_0 = 0.
Use library for fft function.

the split operator method was accomplished in following module:
numsolvers.split_operator.py
"""

# set up Wafefunction Psi at t=0
h = spop.h
Nx = 9000
offset_x = 400
dx = 0.1
Nt = 4000
dt = 0.1
V_real = lambda x: 0.0 * x
sig = 20.0
x_0 = 0.0
q = 2.0
N_psi_t0 = 1 / (np.sqrt(sig * np.sqrt(np.pi)))
psit0 = (
    lambda x: N_psi_t0
    * np.exp(-((x - x_0) ** 2) / (2 * sig**2))
    * np.exp((1j / spop.h) * q * x)
)
x_space, t_space, psi = spop.exec_split_operator_method(
    Nx, dx, Nt, dt, psit0, V_real, shift_x=offset_x
)

#%%
############################### Task 3.b #######################################
"""
Make a plot of |psi|**2 at t = 400 and compare it with |psi|**2 at t = 400 from
the Crank-Nicolson method. Which of these two deviates less from the analytical
evolution of the gaussian wave packet?
"""
if True:
    # analytical solution
    psi_an = (
        lambda x, t, x0, sig, q0: np.sqrt(
            1
            / (
                np.sqrt(2 * np.pi * sig**2)
                * (1 + ((1j * t) / (2 * sig**2)))
            )
        )
        * np.exp(
            -(
                (x - x0 - (q0 * t)) ** 2
                / (4 * sig**2 * (1 + ((1j * t) / (2 * sig**2))))
            )
        )
        * np.exp(1j * q0 * x)
        * np.exp(-1j * q0**2 * t / 2)
    )

    # crank nicelson solution
    psi_crnic, x_crnic, t_crnic = crnic.crack_nicolson_func(
        Nx, dx, Nt + 1, dt, psit0, V_real, shift_x=offset_x
    )

    # plotting all the propabilities
    fig1 = plt.figure(num=1, figsize=[22, 18])
    fig1.suptitle("task 3b split operator comparison", fontsize=20)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(
        x_space, np.abs(psi[0]) ** 2, label=r"$|\psi(t=0)|^2$", color="yellow"
    )
    ax1.plot(
        x_space,
        np.abs(psi[int(400 / dt)]) ** 2,
        label=r"$|\psi(t=400)|^2$",
        color="red",
    )
    ax1.plot(
        x_space,
        np.abs(psi_an(x_space, 400, 0.0, (sig / np.sqrt(2)), 2.0)) ** 2,
        label=r"$|\psi_{an}(t=400)|^2$",
        linestyle="dashed",
        color="black",
    )
    ax1.plot(
        x_crnic,
        np.abs(psi_crnic[int(400 / dt)]) ** 2,
        label=r"$|\psi_{crnic}(t=400)|^2$",
        linestyle="dotted",
        color="green",
    )
    ax1.legend()
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$|\psi|$")
    ax1.set_xlim(offset_x - (Nx * dx / 2), offset_x + (Nx * dx / 2))
    fig1.savefig("./Ex3/plots/plot_3_b.pdf", dpi="figure")


#%%
############################### Task 3.c #######################################
"""
Solve the Schrödinger equation for the Potentials V1 and V2 and the correspond-
ing parameters from the previous tasl and compare your results graphically.
Which algorithm is in your implementation more efficient?
"""
if False:
    # set up wafefunction Psi at t=0
    h = spop.h
    Nx = 5000
    dx = 0.1
    offset_x = 200
    Nt = 4000
    dt = 0.034
    sig = 20.0
    x_0 = 0.0
    q = 2.0
    N_psi_t0 = 1 / (np.sqrt(sig * np.sqrt(np.pi)))
    psi_t0 = (
        N_psi_t0
        * np.exp(-((x_space - x_0) ** 2) / (2 * sig**2))
        * np.exp((1j / h) * q * x_space)
    )

    # set up potential operators
    V_0_list = np.array([1.5, 2.0, 2.5])
    # V_0_list = np.array([1.5])
    a = 100
    b = 200
    d = 10
    V_x_1 = lambda x, V_0: V_0 * (
        np.heaviside(x - a, 0.5) - np.heaviside(x - (a + d), 0.5)
    )
    V_x_2 = lambda x, V_0: V_x_1(x, V_0) + V_0 * (
        np.heaviside(x - b, 0.5) - np.heaviside(x - (b + d), 0.5)
    )

    psi_split_V1 = []
    texec_split_V1 = []
    psi_split_V2 = []
    texec_split_V2 = []
    psi_crnic_V1 = []
    texec_crnic_V1 = []
    psi_crnic_V2 = []
    texec_crnic_V2 = []
    # calculate the wave functions with split operator and crank nicolson
    # and measure the calculation execution time
    for V_0_i in tqdm(V_0_list):
        starttime = timeit.default_timer()
        x_space, t_space, psi_1 = spop.exec_split_operator_method(
            Nx, dx, Nt, dt, psit0, lambda x: V_x_1(x, V_0_i), shift_x=offset_x
        )
        psi_split_V1.append(psi_1)
        endtime = timeit.default_timer()
        texec_split_V1.append(endtime - starttime)

        starttime = timeit.default_timer()
        x_space, t_space, psi_2 = spop.exec_split_operator_method(
            Nx, dx, Nt, dt, psit0, lambda x: V_x_2(x, V_0_i), shift_x=offset_x
        )
        psi_split_V2.append(psi_2)
        endtime = timeit.default_timer()
        texec_split_V2.append(endtime - starttime)

        starttime = timeit.default_timer()
        psi_crnic_1, x_crnic, t_crnic = crnic.crack_nicolson_func(
            Nx,
            dx,
            Nt + 1,
            dt,
            psit0,
            lambda x: V_x_1(x, V_0_i),
            shift_x=offset_x,
        )
        psi_crnic_V1.append(psi_crnic_1)
        endtime = timeit.default_timer()
        texec_crnic_V1.append(endtime - starttime)

        starttime = timeit.default_timer()
        psi_crnic_2, x_crnic, t_crnic = crnic.crack_nicolson_func(
            Nx,
            dx,
            Nt + 1,
            dt,
            psit0,
            lambda x: V_x_2(x, V_0_i),
            shift_x=offset_x,
        )
        psi_crnic_V2.append(psi_crnic_2)
        endtime = timeit.default_timer()
        texec_crnic_V2.append(endtime - starttime)

    # plotting all the propabilities for potential V1
    fig2 = plt.figure(num=2, figsize=[22, 18])
    sub_title_V1 = (
        "t_exec_spop = {:5.2f}s\t {:5.2f}s\t {:5.2f}s\n".format(
            *texec_split_V1
        )
        + "t_exec_crnic = {:5.2f}s\t {:5.2f}s\t {:5.2f}s".format(
            *texec_crnic_V1
        )
    ).replace("\t", "                                            ")
    fig2.suptitle(
        "task 3c split-operator(spop) / crank-nicolson(crnic) comparison with $V_1$ \n"
        + sub_title_V1,
        fontsize=20,
    )

    i_pot = 0
    for i, t in enumerate([25, 50, 60, 75, 100]):
        for i_pot, V_0_i in enumerate(V_0_list):
            ax1 = fig2.add_subplot(5, 3, i * 3 + 1 + i_pot)
            ax1.plot(
                x_space,
                np.abs(psi_split_V1[i_pot][int(t / dt)]) ** 2,
                label=r"$|\psi_{spop}(t=" + str(t) + r")|^2$",
                color="red",
            )
            ax1.plot(
                x_crnic,
                np.abs(psi_crnic_V1[i_pot][int(t / dt)]) ** 2,
                label=r"$|\psi_{crnic}(t=" + str(t) + r")|^2$",
                color="blue",
            )
            ax1.plot(
                x_space,
                V_x_1(x_space, V_0_i) / 100,
                label=r"$V_1(V_0=" + str(V_0_i) + r")/100$",
                color="green",
            )
            ax1.legend()
            ax1.set_xlabel("x")
            ax1.set_ylabel(r"$|\psi|$")
            ax1.set_xlim(offset_x - (Nx * dx / 2), offset_x + (Nx * dx / 2))
    fig2.savefig("./Ex3/plots/plot_3_c_V1.pdf", dpi="figure")

    # plotting all the propabilities for potential V2
    fig3 = plt.figure(num=3, figsize=[22, 18])
    sub_title_V2 = (
        "t_exec_spop = {:5.2f}s\t {:5.2f}s\t {:5.2f}s\n".format(
            *texec_split_V2
        )
        + "t_exec_crnic = {:5.2f}s\t {:5.2f}s\t {:5.2f}s".format(
            *texec_crnic_V2
        )
    ).replace("\t", "                                            ")
    fig3.suptitle(
        "task 3c split-operator(spop) / crank-nicolson(crnic) comparison with $V_2$ \n"
        + sub_title_V1,
        fontsize=20,
    )

    i_pot = 0
    for i, t in enumerate([25, 50, 60, 75, 100, 136]):
        for i_pot, V_0_i in enumerate(V_0_list):
            ax2 = fig3.add_subplot(6, 3, i * 3 + 1 + i_pot)
            ax2.plot(
                x_space,
                np.abs(psi_split_V2[i_pot][int(t / dt)]) ** 2,
                label=r"$|\psi_{spop}(t=" + str(t) + r")|^2$",
                color="red",
            )
            ax2.plot(
                x_crnic,
                np.abs(psi_crnic_V2[i_pot][int(t / dt)]) ** 2,
                label=r"$|\psi_{crnic}(t=" + str(t) + r")|^2$",
                color="blue",
            )
            ax2.plot(
                x_space,
                V_x_2(x_space, V_0_i) / 100,
                label=r"$V_2(V_0=" + str(V_0_i) + r")/100$",
                color="green",
            )
            ax2.legend()
            ax2.set_xlabel("x")
            ax2.set_ylabel(r"$|\psi|$")
            ax2.set_xlim(offset_x - (Nx * dx / 2), offset_x + (Nx * dx / 2))
    fig3.savefig("./Ex3/plots/plot_3_c_V2.pdf", dpi="figure")
