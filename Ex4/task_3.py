#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Assignment 3 Task 3 Split Operator Method
"""


############################## imports #########################################
import numpy as np
from numpy.fft import fft, ifft as inv_fft
import matplotlib.pyplot as plt
import numsolvers.Frauenhofer_diffraction as Diff


######################### module level dunders #################################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"


nm = 1e-9
mum = 1e-6
mm = 1e-3
spaltbreite = 50 * mum
spaltabstand = 0.25 * mm
Lxy = 20 * spaltabstand
Nxy = 1000
lambd = 633 * nm

single_slit = lambda x, y: (
    (
        np.heaviside(x + spaltbreite / 2, 0.5)
        - np.heaviside(x - spaltbreite / 2, 0.5)
    )
    * (
        np.heaviside(y + spaltabstand, 0.5)
        - np.heaviside(y - spaltabstand, 0.5)
    )
)

double_slit = lambda x, y: (
    (
        (
            np.heaviside(x + (spaltabstand / 2 + spaltbreite / 2), 0.5)
            - np.heaviside(x + (spaltabstand / 2 - spaltbreite / 2), 0.5)
        )
        + (
            np.heaviside(x - (spaltabstand / 2 - spaltbreite / 2), 0.5)
            - np.heaviside(x - (spaltabstand / 2 + spaltbreite / 2), 0.5)
        )
    )
    * (
        np.heaviside(y + spaltabstand, 0.5)
        - np.heaviside(y - spaltabstand, 0.5)
    )
)
grating = lambda x, y: (
    (
        (
            np.heaviside(x + spaltbreite * 11 / 2, 0.5)
            - np.heaviside(x + spaltbreite * 9 / 2, 0.5)
        )
        + (
            np.heaviside(x + spaltbreite * 7 / 2, 0.5)
            - np.heaviside(x + spaltbreite * 5 / 2, 0.5)
        )
        + (
            np.heaviside(x + spaltbreite * 3 / 2, 0.5)
            - np.heaviside(x + spaltbreite / 2, 0.5)
        )
        + (
            np.heaviside(x - spaltbreite / 2, 0.5)
            - np.heaviside(x - spaltbreite * 3 / 2, 0.5)
        )
        + (
            np.heaviside(x - spaltbreite * 5 / 2, 0.5)
            - np.heaviside(x - spaltbreite * 7 / 2, 0.5)
        )
        + (
            np.heaviside(x - spaltbreite * 9 / 2, 0.5)
            - np.heaviside(x - spaltbreite * 11 / 2, 0.5)
        )
    )
    * (
        np.heaviside(y + spaltabstand, 0.5)
        - np.heaviside(y - spaltabstand, 0.5)
    )
)
circle = lambda x, y: np.heaviside((spaltbreite) ** 2 - x**2 - y**2, 0.5)
circles = (
    lambda x, y: np.heaviside((spaltbreite) ** 2 - x**2 - y**2, 0.5)
    + np.heaviside(
        (spaltbreite) ** 2 - (x - spaltabstand) ** 2 - (y - spaltabstand) ** 2,
        0.5,
    )
    + np.heaviside(
        (spaltbreite) ** 2 - (x + spaltabstand) ** 2 - (y + spaltabstand) ** 2,
        0.5,
    )
    + np.heaviside(
        (spaltbreite) ** 2 - (x + spaltabstand) ** 2 - (y - spaltabstand) ** 2,
        0.5,
    )
    + np.heaviside(
        (spaltbreite) ** 2 - (x - spaltabstand) ** 2 - (y + spaltabstand) ** 2,
        0.5,
    )
)


func_list = [single_slit, double_slit, grating, circle, circles]
fig = plt.figure(figsize=(40, 40))

for i in range(5):
    # calculating diffraction patterns from aperture
    diff = Diff.Diffraction(Lxy, Lxy, Nxy, Nxy, lambd)
    diff.optical_mask(func_list[i], calc=True)
    # finding the interesting areas automatically
    app_xmin, app_xmax, app_ymin, app_ymax = diff.get_apperture_bounds(0.01)
    diff_xmin, diff_xmax, diff_ymin, diff_ymax = diff.get_diffraction_bounds(
        0.001
    )
    app_xymax = np.max([app_xmax, app_ymax])
    app_xymax = app_xymax + app_xymax / 4
    app_xymin = np.min([app_xmin, app_ymin])
    app_xymin = app_xymin + app_xymin / 4

    # calculating diffraction aperture patterns from diffraction
    app = Diff.Diffraction(Lxy * 10, Lxy * 10, Nxy, Nxy, lambd)
    app.intensity_distribution(lambda x, y: func_list[i](x, y), calc=True)

    # finding the interesting areas automatically
    app_xmin1, app_xmax1, app_ymin1, app_ymax1 = app.get_apperture_bounds(
        0.001
    )
    (
        diff_xmin1,
        diff_xmax1,
        diff_ymin1,
        diff_ymax1,
    ) = app.get_diffraction_bounds(0.1)

    diff_xymax1 = np.max([diff_xmax1, diff_ymax1])
    diff_xymax1 = diff_xymax1 + diff_xymax1 / 4
    diff_xymin1 = np.min([diff_xmin1, diff_ymin1])
    diff_xymin1 = diff_xymin1 + diff_xymin1 / 4

    ax1 = fig.add_subplot(5, 4, 1 + i * 4)
    ax2 = fig.add_subplot(5, 4, 2 + i * 4)
    ax3 = fig.add_subplot(5, 4, 3 + i * 4)
    ax4 = fig.add_subplot(5, 4, 4 + i * 4)
    ax1.imshow(
        diff.apperture,
        cmap="inferno",
        extent=[
            diff.space_grid_app[0].min() / mm,
            diff.space_grid_app[0].max() / mm,
            diff.space_grid_app[1].min() / mm,
            diff.space_grid_app[1].max() / mm,
        ],
    )
    ax1.set_xlabel("x / mm")
    ax1.set_ylabel("y / mm")
    ax1.set_xlim(app_xymin / mm, app_xymax / mm)
    ax1.set_ylim(app_xymin / mm, app_xymax / mm)

    ax2.imshow(
        np.abs(diff.diffraction_wave),
        cmap="inferno",
        extent=[
            diff.space_grid_diff[0].min() * 3600,
            diff.space_grid_diff[0].max() * 3600,
            diff.space_grid_diff[1].min() * 3600,
            diff.space_grid_diff[1].max() * 3600,
        ],
    )
    ax2.set_xlabel(r"$\phi$ / ''")
    ax2.set_ylabel(r"$\theta$ / ''")
    ax2.set_xlim(diff_xmin * 3600, diff_xmax * 3600)
    ax2.set_ylim(diff_ymin * 3600, diff_ymax * 3600)

    ax3.imshow(
        np.abs(app.diffraction_wave),
        cmap="inferno",
        extent=[
            app.space_grid_diff[0].min() * 3600,
            app.space_grid_diff[0].max() * 3600,
            app.space_grid_diff[1].min() * 3600,
            app.space_grid_diff[1].max() * 3600,
        ],
    )
    ax3.set_xlabel(r"$\phi$ / ''")
    ax3.set_ylabel(r"$\theta$ / ''")
    ax3.set_xlim(diff_xymin1 * 3600, diff_xymax1 * 3600)
    ax3.set_ylim(diff_xymin1 * 3600, diff_xymax1 * 3600)

    ax4.imshow(
        app.apperture,
        cmap="inferno",
        extent=[
            app.space_grid_app[0].min() / mm,
            app.space_grid_app[0].max() / mm,
            app.space_grid_app[1].min() / mm,
            app.space_grid_app[1].max() / mm,
        ],
    )
    ax4.set_xlabel("x / mm")
    ax4.set_ylabel("y / mm")
    ax4.set_xlim(app_xmin1 / mm, app_xmax1 / mm)
    ax4.set_ylim(app_ymin1 / mm, app_ymax1 / mm)


# plt.show()
plt.savefig("./Ex4/plots/task_3.pdf", dpi="figure")
