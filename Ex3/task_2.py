#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Execution script for task 2 of the computational physics assignment 3

**** README: ****
* MathTeX notation is used in comment formula explanations!
"""
from typing import Callable, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter, FFMpegWriter
import time
from tqdm import tqdm
from scipy.optimize import curve_fit


######################### module level dunders ###############################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

############################# functions ######################################


def gaussian_wave_packet_fit(x, sigma, mean):
    C = 1 / np.sqrt(sigma * np.sqrt(np.pi))
    func = (
        np.abs(
            C
            * np.exp(((-1) * ((x - mean) ** 2)) / (2 * sigma**2))
            * np.exp((1j / hbar) * 2 * x)
        )
        ** 2
    )
    return func


def gaussian_wave_packet(x, sigma, mean, q):
    C = 1 / np.sqrt(sigma * np.sqrt(np.pi))
    func = (
        C
        * np.exp(((-1) * ((x - mean) ** 2)) / (2 * sigma**2))
        * np.exp((1j / hbar) * q * x)
    )
    return func


def wave_packet_width(t, sigma, mass):
    delta = t / (mass * sigma**2)
    return sigma * np.sqrt(1 + delta**2)


def potential(x, V0, a_b, d):
    pot = V0 * (np.heaviside(x - a_b, 0) - np.heaviside(x - (a_b + d), 0))
    return pot


hbar = 1
m = 1
v = 2
############################# 2.a) ######################################


def crack_nicolson(
    Nx: int, dx: float, Nt: int, dt: float, psi0: list, V: list
):
    """
    Cranck-Nicolson Method for solving timedepending differential Equations

    This function takes psi0 and the Potential V as lists

    Arguments:
        Nx -- Number of Space-Gridpoints
        dx -- spacing between Space-Gridpoints
        Nt -- Number of Time-Gridpoints
        dt -- spacing between Time-Gridpoints
        psi0 -- the wave function at t = 0 as thhe initial condition
        V -- Potential

    Returns:
        psi(x,t) list(list())
    """
    a_k_func = lambda Vk, dx_a, dt_a: 2 * (
        1
        + ((m * dx_a**2) / (hbar**2)) * Vk
        - (1j * 2 * m * dx_a**2) / (hbar * dt_a)
    )

    Omega_func = lambda Vk, dx_a, dt_a: 2 * (
        1
        + ((m * dx_a**2) / (hbar**2)) * Vk
        + (1j * 2 * m * dx_a**2) / (hbar * dt_a)
    )

    a_k = np.squeeze(np.zeros((1, Nx), dtype=np.complex_))

    a_k[1] = a_k_func(V[1], dx, dt)

    for k in tqdm(range(2, Nx - 1), desc="ak-loop"):
        a_k[k] = a_k_func(V[k], dx, dt) - 1 / a_k[k - 1]

    psi = np.zeros((Nt, Nx), dtype=np.complex_)
    psi[0] = psi0
    psi[0][0] = 0 + 0j
    psi[0][Nx - 1] = 0 + 0j
    for n in tqdm(range(0, Nt - 1), desc="time-loop"):
        Omega = np.squeeze(np.zeros((1, Nx), dtype=np.complex_))
        b_n = np.squeeze(np.zeros((1, Nx), dtype=np.complex_))
        for k in range(1, Nx - 1):
            Omega[k] = Omega_func(V[k], dx, dt) * psi[n][k] - (
                psi[n][k - 1] + psi[n][k + 1]
            )
        b_n[1] = Omega[1]
        for k in range(2, Nx - 1):
            b_n[k] = b_n[k - 1] / a_k[k - 1] + Omega[k]

        k_index = np.arange(1, Nx - 1, 1)
        k_index = k_index[::-1]

        psi[n + 1][0] = 0
        psi[n + 1][Nx - 1] = 0
        for k in k_index:
            psi[n + 1][k] = (1 / a_k[k]) * (psi[n + 1][k + 1] - b_n[k])

    return psi


############################# 2.b) ######################################
# initiating x array of psi
x_b = np.arange(-50, 50, 1e-2)
# initiating psi_0

psi_0 = gaussian_wave_packet(x_b, 10, 0, 2)


# velocity of the packet
def velocity(psi: list, x_1: list, dt: float):
    """
    Velocity calculator

    Arguments:
        psi -- psi(t,x)
        x_1 -- x list associated to psi(t,x)
        dt -- timestep

    Returns:
        list(velocities),list(time)
    """

    psi_t1 = np.abs(psi) ** 2
    delta_pos = []
    x_b_c = x_1
    for i in range(0, len(psi) - 1):
        # searches for maximal points psi data set
        # and calculates differential coefficient
        delta_pos.append(
            np.abs(
                x_b_c[np.argmax(psi_t1[i + 1])] - x_b_c[np.argmax(psi_t1[i])]
            )
            / dt
        )
    t = np.arange(0, dt * len(psi) - dt, dt)
    return delta_pos, t


# lists of different gridsizes and timesteps to analyse v(dt)
dt_list = [1, 0.5, 0.2, 1e-2]
Nt_list = [10, 20, 50, 1000]
v = 2


def plot_velocities(plot=False):
    figure_2_b_v, axis_2_b_v = plt.subplots(1, 1, figsize=[15, 15])
    axis_2_b_v.set_ylim(0.5, 2.5)
    axis_2_b_v.set_xlabel("t / seconds")
    axis_2_b_v.set_ylabel("v(t) / Velocity")
    figure_2_b_v.suptitle(
        "Velocity of the Wavepacket with different timesteps"
    )

    if plot == True:
        for dt_1, Nt_1 in zip(dt_list, Nt_list):
            psi_t = crack_nicolson(
                len(psi_0), 1e-2, Nt_1, dt_1, psi_0, [0] * len(psi_0)
            )
            delta_pos, t_array = velocity(psi_t, x_b, dt_1)
            print(delta_pos)
            axis_2_b_v.plot(
                t_array,
                delta_pos,
                label="Nt = " + str(Nt_1) + " dt = " + str(dt_1),
            )

    axis_2_b_v.legend(loc="best")
    figure_2_b_v.savefig(
        "./Ex3/plots/plot_2_b_velocites_dt.pdf",
        dpi="figure",
        orientation="landscape",
    )


plot_velocities(plot=False)

# psi for calculation of the width
# psi_b = crack_nicolson(len(psi_0), 1e-2, 450, 0.0333, psi_0, [0] * len(psi_0))
# width_time_array = np.arange(0, len(psi_b) * 0.0333, 0.0333)


# width of the packet
def width(psi: list, sigma: float, calc=False, plot=False, plot_save=False):
    """
    A function that fits sigma with the psi_0 function
    and compares it to the analytical model
    Also plots it

    Arguments:
        psi --  psi(t,x)
        sigma -- standart deviation of the gauss packet at psi_0

    Keyword Arguments:
        calc -- if True calculates stuff (default: {False})
        plot -- if True plots (default: {False})
        plot_save -- if True saves (default: {False})
    """
    fittet_widths = []
    # each psi at every grid point is fitted using the initial psi0 function
    # the estimated sigmas are stored and displayed
    if calc == True:
        for i in range(0, len(width_time_array)):
            psi_t1 = np.abs(psi[i]) ** 2
            gaussian_params, _ = curve_fit(
                gaussian_wave_packet_fit,
                x_b,
                psi_t1,
            )
            fittet_widths.append(gaussian_params[0])
        # the analytical solution is also calculated and displayed
        predicted_widths = wave_packet_width(width_time_array, sigma, m)

        for i, timepoints in enumerate(width_time_array):
            print(
                "Predicted Widths: "
                + str(predicted_widths[i])
                + " | Fittet Widths: "
                + str(fittet_widths[i])
                + "| timepoint: "
                + str(timepoints)
            )

        if plot == True:
            # plots thhe predicted and observed widths
            figure_2_b, axis_2_b = plt.subplots(1, 1, figsize=[15, 15])
            axis_2_b.plot(
                width_time_array, fittet_widths, label="Observed Widths"
            )
            axis_2_b.plot(
                width_time_array, predicted_widths, label="Predicted Widths"
            )
            axis_2_b.legend(loc="best")
            axis_2_b.set_ylabel("$\\sigma$ / m ")
            axis_2_b.set_xlabel("$t$ / seconds")
            figure_2_b.suptitle("Wavepacket Width Time Evolution ")
            plt.show()
            if plot_save == True:
                figure_2_b.savefig(
                    "./Ex3/plots/plot_2_b_width.pdf",
                    dpi="figure",
                    orientation="landscape",
                )
            plt.cla()


# width(psi_b, 10, calc=False, plot=True, plot_save=True)


############################# 2.c) ######################################
# progressbar for gif saving
pbar = tqdm(desc="gif-saving", total=0)


class Framesaver:
    # helper class to use last_frame in update function
    last_frame = 0


fs = Framesaver()
# animation of the free guassian wave packet
def animation_(
    psi: list, x: list, dt: float, plot_name: str, potential: list, fps_: int
):
    """
    Animates a wavepacket and potential

    Arguments:
        psi -- psi(t,x)
        x -- x-array associated to psi(t,x)
        dt -- timestep size
        plot_name -- plot name
        potential -- potential that should be plotted
        fps_ -- Frames per Second of the gif

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(xlim=(min(x), max(x)), ylim=(0, 0.15))
    # ax.set_aspect("equal")
    (line,) = ax.plot([], [], lw=3)
    ax.plot(x, potential, label="Potential")
    ax.legend(loc="best")
    time_template = "time = %.1fs"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        psi_prob = np.abs(psi[i]) ** 2
        # psi_prob = psi_t[i]
        line.set_data(x, psi_prob)
        time_text.set_text(time_template % (i * dt))
        return (line,)

    # animation
    ani = animation.FuncAnimation(
        fig, animate, len(psi), interval=len(psi), blit=True, init_func=init
    )

    ax.plot(x, potential, label="Potential")

    ### mp4 saving
    def update_progress(cur_fr: int, tot_fr: int):
        if cur_fr == 0:
            pbar.reset(tot_fr)
        fr_diff = cur_fr - fs.last_frame
        fs.last_frame = cur_fr
        pbar.update(fr_diff)

    writergif = animation.PillowWriter(fps=fps_)

    ani.save(
        "./Ex3/plots/wp" + plot_name + ".gif",
        writer=writergif,
        progress_callback=update_progress,
    )
    plt.cla()


# initializing psi_0 for animation and probability conserving plot
x_c = np.arange(-50, 300, 1e-2)
psi_0_c = gaussian_wave_packet(x_c, 10, 0, 2)
psi_c = crack_nicolson(
    len(psi_0_c), 1e-2, 450, 0.0333, psi_0_c, [0] * len(psi_0_c)
)
# animated thhe free moving packet
animation_(
    psi_c,
    x_c,
    0.0333,
    "gaussianwavepacket",
    psi_0_c,
    [0] * len(psi_0_c),
    33,
)
# probability checking
def probability_conserv(psi: list, dx: float, plot=False, plot_save=False):
    """
    Plots the total Probability at all timepoints

    Arguments:
        psi -- |psi(t,x)|^2
        dx -- spacing between Space-Gridpoints

    Keyword Arguments:
        plot -- if True plots (default: {False})
        plot_save -- if True saves (default: {False})
    """
    # in order to get the total probability a sum over all y-values is done
    # because we use equidistant gridpoints this sum is multiplied be x-step -> area = xi * delta x

    if plot == True:
        total_prob = []
        time_array_prob = [i for i in range(0, len(psi))]
        for current_psi in psi:
            total_prob.append(np.sum(current_psi))
        total_prob = np.array(total_prob) * dx
        figure_2_c, axis_2_c = plt.subplots(1, 1, figsize=[15, 15])
        axis_2_c.set_ylim(0.95, 1.05)
        figure_2_c.suptitle("Conservation of total Probability")
        axis_2_c.set_xlabel("n / timestep ")
        axis_2_c.set_ylabel("a.U /1")
        axis_2_c.plot(
            time_array_prob,
            total_prob,
            "b+",
            label="Total Probability for each timestep (dt=0.0333)",
        )
        axis_2_c.legend(loc="best")

        if plot_save == True:
            figure_2_c.savefig(
                "./Ex3/plots/plot_2_c_probconserv.pdf",
                dpi="figure",
                orientation="landscape",
            )
        plt.show()


# probability_conserv(np.abs(psi_t) ** 2, 1e-2, plot=False, plot_save=True)


############################# 2.d) & 2.e) ######################################
# parameters for potentials that get used for scattering the gaussian wave packet
Potentials = []
V0 = [1.5, 2.0, 2.5]
a = 100
b = 200
d = 10
# the potentials are located at bigger distances so the observed space is set to (-100,300)
x_d = np.arange(-50, 250, 0.1)
psi_0_2_d = gaussian_wave_packet(x_d, 20, 0, 2)
psi_p1 = np.abs(psi_0) ** 2

Potentials_names = []
# preparing the used potentials
for i in V0:
    Potentials.append(potential(x_d, i, a, d))
    Potentials_names.append(str(i) + "_" + str(a) + "_" + str(d))
for i in V0:
    Potentials.append(potential(x_d, i, a, d) + potential(x_d, i, b, d))
    Potentials_names.append(str(i) + "_" + str(a) + "_" + str(d) + "_2")


def wave_packet_scattering(
    potentials: list, calc=False, plot=False, animate=False
):

    if calc == True:
        wave_functions_t = (
            []
        )  # initializing of array were the time evolved wave functions are stored
        for pot_n, pot_N in tqdm(
            zip(potentials, Potentials_names),
            desc="calculating time evolution",
        ):
            # calculating the time evolution with every Potential
            psi_n = crack_nicolson(
                len(psi_0_2_d), 0.1, 1900, 0.0666, psi_0_2_d, pot_n
            )
            wave_functions_t.append(psi_n)
            if animate == True:
                # the size of the timesteps is chosen corresponding to the fps 1/15
                # animates the scattering
                animation_(psi_n, x_d, 0.0666, pot_N, pot_n, 15)

        if plot == True:
            # plots the wavefunction scattering of potentials at different timesteps
            # not very useful
            figure_2_e, axis_2_e = plt.subplots(3, 2, figsize=[15, 15])
            for w, (wave_func, Potential) in tqdm(
                enumerate(zip(wave_functions_t, Potentials)),
                desc="creating plots",
            ):
                if w < 3:
                    axis_2_e[w, 0].plot(
                        x_d,
                        Potential,
                        "r--",
                        label="Potential = $V_{1}(x)$ = "
                        + str(np.max(Potential))
                        + "[$\\theta(x-100) - \\theta(x-110)$]",
                    )

                    for timepoint in range(60, 300, 40):
                        axis_2_e[w, 0].plot(
                            x_d,
                            np.abs(wave_func[timepoint]) ** 2,
                            label=str(timepoint) + " seconds",
                        )
                    axis_2_e[w, 0].legend(loc="best")
                if w > 2:
                    axis_2_e[w - 3, 1].plot(
                        x_d,
                        Potential,
                        "r--",
                        label="Potential = $V_{1}(x)$ = "
                        + str(np.max(Potential))
                        + "[$\\theta(x-100) - \\theta(x-110)$]",
                    )

                    for timepoint in range(60, 300, 40):
                        axis_2_e[w - 3, 1].plot(
                            x_d,
                            np.abs(wave_func[timepoint]) ** 2,
                            label=str(timepoint) + " seconds",
                        )
                    axis_2_e[w - 3, 1].legend(loc="best")
            figure_2_e.suptitle(
                "$V_{1}(x) = V_{0}[\\theta(x-a) - \\theta(x-(a+d))]$                   $V_{2}(x) = V_{1}(x) + V_{0}[\\theta(x-b) - \\theta(x-(b+d))]$"
            )
            plt.show()

        return wave_functions_t


w_f = wave_packet_scattering(Potentials, calc=False, plot=False, animate=True)
