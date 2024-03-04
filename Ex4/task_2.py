#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Execution script for task 2 of the computational physics assignment 3

**** README: ****
* MathTeX notation is used in comment formula explanations!
"""
import numpy as np
from numsolvers import runge_kutta as rk
from numsolvers import bodys_ODES as bd
from typing import Callable, Sequence
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.signal import find_peaks

######################### module level dunders ###############################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

############################# functions ######################################
def center_of_mass(masses: list, positions: list):
    masses = np.array(masses)
    positions = np.array(positions)
    Rs = (
        masses[0] * positions[0]
        + masses[1] * positions[1]
        + masses[2] * positions[2]
    ) / np.sum(masses, 0)

    return np.array(Rs)


def y0_generator(*initcond):
    y0 = []
    for i0 in initcond:
        for j0 in i0:
            y0.append(j0)
    return y0


def Hamilton_Energy(
    masses: list, velocity: list, positions: list, time_array: list
):
    G = 6.67408 * 10 ** (-11)
    masses = np.array(masses)
    velocity = np.array(velocity)
    positions = np.array(positions)
    E_pot = (
        lambda m1, m2, ri, rj: G * m1 * m2 / (np.sqrt(np.sum((ri - rj) ** 2)))
    )
    Ekin = []
    Epot = []
    Momentum = []
    for i in range(len(time_array)):
        Momentum.append(
            np.sqrt(
                sum(
                    (
                        masses[0] * np.cross(positions[0][i], velocity[0][i])
                        + masses[1] * np.cross(positions[1][i], velocity[1][i])
                        + masses[0] * np.cross(positions[2][i], velocity[2][i])
                    )
                    ** 2
                )
            )
        )
        Ekin.append(
            (masses[0] / 2) * sum(velocity[0][i] ** 2)
            + (masses[1] / 2) * sum(velocity[1][i] ** 2)
            + (masses[2] / 2) * sum(velocity[2][i] ** 2)
        )
        Epot.append(
            E_pot(masses[0], masses[1], positions[0][i], positions[1][i])
            + E_pot(masses[0], masses[2], positions[0][i], positions[2][i])
            + E_pot(masses[1], masses[2], positions[1][i], positions[2][i])
        )
    return np.array(Ekin) + np.array(Epot), Momentum


############################# 2.a) ######################################

m_neptune = 1.024 * 10**26
m_sun = 1.98847 * 10 ** (30)
m_merk = 3.285 * 10**23
m_mars = 6.417 * 10**23
m_jupiter = 1.898 * 10**27

r_earth = 149.6 * 10 ** (9)

masses_merk_mars = [m_sun, m_merk, m_mars]
masses_jupiter_neptune = [m_sun, m_jupiter, m_neptune]
masses_merk_neptune = [m_sun, m_merk, m_neptune]
r_sonne = [0, 0, 0]
v_sonne = [0, 0, 0]
# 5-Jan-2023 parameter
r_merkur = (
    np.array(
        [
            -4.524814158658526 * 1e07,
            2.705886664410684 * 1e07,
            6.269490511141302 * 1e06,
        ]
    )
    * 10**3
)
v_merkur = (
    np.array(
        [
            -3.554212768207265 * 1e01,
            -3.944538486926456 * 1e01,
            3.814923736153730 * 1e-02,
        ]
    )
    * 10**3
)

r_mars = (
    np.array(
        [
            -2.041896047545896 * 1e07,
            2.356513199273445 * 1e08,
            5.438144012945265 * 1e06,
        ]
    )
    * 10**3
)
v_mars = (
    np.array(
        [
            -2.323084098109877 * 1e01,
            8.845694655923159 * 1e-02,
            5.721465908501231 * 1e-01,
        ]
    )
    * 10**3
)
r_neptune = (
    np.array(
        [
            4.451405939664204 * 1e09,
            4.336422412316926 * 1e08,
            -9.36571678694674 * 1e07,
        ]
    )
    * 10**3
)
v_neptune = (
    np.array(
        [
            4.905510201390117 * 1e-01,
            5.441266848510163,
            -1.235795034030671 * 1e-01,
        ]
    )
    * 10**3
)
r_jupiter = (
    np.array(
        [
            7.187374198670086 * 1e08,
            1.725286952958207 * 1e08,
            -1.679586429145517 * 1e07,
        ]
    )
    * 10**3
)
v_jupiter = (
    np.array(
        [
            -3.197773328387303 * 1e00,
            1.331864503608595 * 1e01,
            1.631874570169511 * 1e-02,
        ]
    )
    * 10**3
)
y0_mars_merk = y0_generator(
    r_sonne, r_merkur, r_mars, v_sonne, v_merkur, v_mars
)
y0_merk_neptune = y0_generator(
    r_sonne, r_merkur, r_neptune, v_sonne, v_merkur, v_neptune
)
y0_jupiter_neptune = y0_generator(
    r_sonne, r_jupiter, r_neptune, v_sonne, v_jupiter, v_neptune
)


def mars_merk(calc=False):
    if calc == True:
        t_t_m_m, y_t_m_m = rk.rk4_method(
            F=bd.Newton_ODES(m_sun, m_merk, m_mars),
            y0=y0_mars_merk,
            t0=0,
            t_max=59356800 * 3,
            e=86400,
        )
        return t_t_m_m, y_t_m_m


def jupiter_neptune(calc=False):
    if calc == True:
        t_t_j_n, y_t_j_n = rk.rk4_method(
            F=bd.Newton_ODES(m_sun, m_jupiter, m_neptune),
            y0=y0_jupiter_neptune,
            t0=0,
            t_max=6307200000,
            e=86400,
        )
    return t_t_j_n, y_t_j_n


def merk_neptune(calc=False):
    if calc == True:
        t_t_m_n, y_t_m_n = rk.rk4_method(
            F=bd.Newton_ODES(m_sun, m_merk, m_neptune),
            y0=y0_merk_neptune,
            t0=0,
            t_max=6307200000,
            e=86400,
        )
    return t_t_m_n, y_t_m_n


# mars_merk(calc=False)


def rk_output_converter(y_t):

    velocities = []
    positions = []
    for i in range(2, 9, 3):
        positions.append(np.array(y_t[i - 2 : i + 1]).T)
        velocities.append(np.array(y_t[i + 7 : i + 10]).T)
    return positions, velocities


t_t_m_m, y_t_m_m = mars_merk(calc=True)
# t_t_j_n, y_t_j_n = jupiter_neptune(calc=True)
pos_m_m, vel_m_m = rk_output_converter(y_t_m_m)
# pos_j_n, vel_j_n = rk_output_converter(y_t_j_n)
#
#
positions_merk_mars = pos_m_m
velocities_merk_mars = vel_m_m
names_merk_mars = ["Sun", "Mercury", "Mars"]
#
# positions_jupiter_neptune = pos_j_n
# velocities_jupiter_neptune = vel_j_n
names_jupiter_neptune = ["Sun", "Jupiter", "Neptune"]


lim_neptune = np.array([-1.3871 * 30 * r_earth, 1.3871 * 30 * r_earth])
############################# 2.b) ######################################
def show_orbits(
    planets_pos: list,
    planets_names: list,
    save_ani=False,
    lim=np.array([-1.3871 * 1.2 * r_earth, 1.3871 * 1.2 * r_earth]),
):

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(projection="3d")
    ax.set_xlim3d(lim)
    ax.set_xlabel("X / m")

    ax.set_ylim3d(lim)
    ax.set_ylabel("Y / m")

    ax.set_zlim3d(lim / 4)
    ax.set_zlabel("Z / m")

    def update_lines(num, walks, lines):
        for line, walk in zip(lines, walks):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(walk[: num * 100, :2].T)
            line.set_3d_properties(walk[: num * 100, 2])
        return lines

    lines = [
        ax.plot(
            [], [], [], marker="o", linestyle="none", markersize=3, label=n
        )[0]
        for _, n in zip(planets_pos, planets_names)
    ]

    ax.legend(loc="best")

    ani = animation.FuncAnimation(
        fig,
        update_lines,
        int(len(planets_pos[0]) / 100),
        fargs=(planets_pos, lines),
        interval=300,
        blit=False,
    )
    plt.show()
    if save_ani == True:
        pbar = tqdm(desc="mp4-saving", total=0)

        class Framesaver:
            # helper class to use last_frame in update function
            last_frame = 0

        fs = Framesaver()
        # update function for the progress bar
        def update_progress(cur_fr: int, tot_fr: int):
            if cur_fr == 0:
                pbar.reset(tot_fr)
            fr_diff = cur_fr - fs.last_frame
            fs.last_frame = cur_fr
            pbar.update(fr_diff)

        # mp4 saving
        writermp4 = animation.FFMpegWriter(fps=10)
        ani.save(
            "./Ex4/plots/"
            + planets_names[0]
            + "_"
            + planets_names[1]
            + "_"
            + planets_names[2]
            + "3D.mp4",
            writer=writermp4,
            progress_callback=update_progress,
        )


# show_orbits(positions_merk_mars, names_merk_mars, save_ani=True)
# show_orbits(
#    positions_jupiter_neptune,
#    names_jupiter_neptune,
#    save_ani=True,
#    lim=lim_neptune,
# )

#t_t_m_n, y_t_m_n = merk_neptune(calc=True)
#pos_m_n, vel_m_n = rk_output_converter(y_t_m_n)
#positions_merk_neptune = pos_m_n
#velocities_merk_neptune = vel_m_n
names_merk_neptune = ["Sun", "Mercury", "Neptune"]
# show_orbits(
#    positions_merk_neptune, names_merk_neptune, save_ani=True, lim=lim_neptune
# )


def distance_mass_center(
    planets_pos: list,
    planets_names: list,
    planets_masses: list,
    time_array: list,
    e=86400,
    xlabel="days",
    show_period=False,
):
    figure, axis = plt.subplots(len(planets_pos), 1, figsize=[15, 12])
    figure.subplots_adjust(
        left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.15, hspace=0.4
    )
    figure.suptitle("Distance r(t) to the center of mass ")
    Rs = center_of_mass(planets_masses, planets_pos)
    colors = ["blue", "green", "red"]
    for pos, nam, col, i in zip(
        planets_pos, planets_names, colors, range(len(planets_pos))
    ):
        axis[i].set_xlim(0, 165)
        axis[i].set_xlabel("t/ " + xlabel)
        if show_period == True:
            if nam == "Mercury":
                axis[i].set_xlim(0, 88)
                e = 86400
                axis[i].set_xlabel("t/ days")
            if nam == "Mars":
                axis[i].set_xlim(0, 687)
                e = 86400
                axis[i].set_xlabel("t/ days")
            if nam == "Jupiter":
                axis[i].set_xlim(0, 11.87)
            if nam == "Neptune":
                axis[i].set_xlim(0, 165)
                e = 86400 * 365
                axis[i].set_xlabel("t/ years")
        r_rel = np.array(pos) - Rs
        r_t = [np.sqrt(sum(rel**2)) for rel in r_rel]
        axis[i].plot(np.array(time_array) / e, r_t, label=nam, color=col)
        axis[i].legend(loc="best")
        axis[i].set_ylabel("r(t) / m")

    plt.savefig(
        "./Ex4/plots/"
        + planets_names[0]
        + "_"
        + planets_names[1]
        + "_"
        + planets_names[2]
        + "_masscenter_nach_abgabe_orbitperiod.pdf",
        dpi="figure",
        orientation="landscape",
    )


#distance_mass_center(positions_merk_mars, names_merk_mars,masses_merk_mars, t_t_m_m , show_period=True)
# distance_mass_center(
#    positions_jupiter_neptune,
#    names_jupiter_neptune,
#    masses_jupiter_neptune,
#    t_t_j_n,
#    e=365 * 86400,
#    xlabel="years",
# )
# distance_mass_center(
#    positions_merk_neptune,
#    names_merk_neptune,
#    masses_merk_neptune,
#    t_t_m_n,
#    e=365 * 86400,
#    xlabel="years",
#    show_period=True
# )
############################# 2.c) ######################################
def show_energy_momentum_conserv(
    planets_pos: list,
    planets_vel: list,
    planets_names: list,
    planets_masses: list,
    time_array: list,
    e=86400,
    xlabel="days",
):
    Rs = center_of_mass(planets_masses, planets_pos)
    print(Rs)
    figure2, axis2 = plt.subplots(2, 1, figsize=[15, 12])
    figure2.subplots_adjust(
        left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.15, hspace=0.4
    )
    figure2.suptitle("Total Energy and Momentum (barycentric system) ")
    planets_pos = np.array(planets_pos) - np.array([Rs, Rs, Rs])
    Eges, Momentum_ges = Hamilton_Energy(
        planets_masses, planets_vel, planets_pos, time_array
    )

    axis2[0].plot(
        np.array(time_array) / e,
        Eges,
        label="$\\mathcal{H}(t)$ = Total Energy of "
        + planets_names[0]
        + "-"
        + planets_names[1]
        + "-"
        + planets_names[2]
        + "-System",
        color="blue",
    )
    axis2[0].legend()
    axis2[0].set_ylabel("Energy / J")
    axis2[0].set_xlabel("t / " + xlabel)

    axis2[1].plot(
        np.array(time_array) / e,
        Momentum_ges,
        label="$|L(t)|$ = Total Momentum of "
        + planets_names[0]
        + "-"
        + planets_names[1]
        + "-"
        + planets_names[2]
        + "-System",
        color="red",
    )
    axis2[1].legend()
    axis2[1].set_ylabel("Momentum / Js")
    axis2[1].set_xlabel("t / " + xlabel)

    plt.savefig(
        "./Ex4/plots/"
        + planets_names[0]
        + "_"
        + planets_names[1]
        + "_"
        + planets_names[2]
        + "_Energy_Momentum_nach_abgabe.pdf",
        dpi="figure",
        orientation="landscape",
    )


# show_energy_momentum_conserv(
#    positions_merk_mars,
#    velocities_merk_mars,
#    names_merk_mars,
#    masses_merk_mars,
#    t_t_m_m,
# )
# show_energy_momentum_conserv(
#    positions_jupiter_neptune,
#    velocities_jupiter_neptune,
#    names_jupiter_neptune,
#    masses_jupiter_neptune,
#    t_t_j_n,
#    e=86400 * 365,
#    xlabel="years",
# )
# show_energy_momentum_conserv(
#    positions_merk_neptune,
#    velocities_merk_neptune,
#    names_merk_neptune,
#    masses_merk_neptune,
#    t_t_m_n,
#    e=86400 * 365,
#    xlabel="years",
# )
