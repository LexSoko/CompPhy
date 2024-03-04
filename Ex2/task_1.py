#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Assignment 2
Task 1

**** README: ****
* MathTeX notation is used in comment formula explanations!
"""


############################## imports #######################################
import numpy as np
import numsolvers.gauss_seidel as gsm
import numsolvers.jacobi as jam
import toolbox.symetric_array as symarr
import matplotlib.pyplot as plt

######################### module level dunders ###############################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

############################# Task 1 ######################################

size = [100, 1000]  # list of different sizes for the symetric 2D arrays
limit_low = 0  # lower limit for array values
limit_high = 10  # upper limit for array values


x_j = [0] * len(size)
x_gs = [0] * len(size)
a_j = [0] * len(size)
a_gs = [0] * len(size)
b_diff_j = [0] * len(size)
b_diff_gs = [0] * len(size)


for count, value in enumerate(size):
    A = symarr.arr_2Dsym_dig0(
        value, limit_low, limit_high
    )  # get random symetric 2D array without zero on the diagonal
    b = np.random.randint(low=limit_low, high=limit_high + 1, size=value)
    x_gs[count], a_gs[count], b_diff_gs[count] = gsm.gauss_seidel_method(
        A, b, caldiff=True
    )
    x_j[count], a_j[count], b_diff_j[count] = jam.jacobi_method(
        A, b, caldiff=True
    )


############################### plotting #######################################
# plotting setup
fig = plt.figure(num=1, figsize=[22, 18])
fig.suptitle("Task 1 ", fontsize=20)
fig.subplots_adjust(
    left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.15, hspace=0.25
)
rows = 2
cols = int(-(-len(size) // rows))  # divide and round result up


for count, value in enumerate(size):
    globals()[f"ax_{count}"] = fig.add_subplot(rows, cols, count + 1)
    globals()[f"ax_{count}"].plot(
        range(b_diff_gs[count].size),
        b_diff_gs[count],
        linewidth=2,
        color="blue",
        linestyle="--",
        marker=".",
        label=f"Gauss-Seidel method: $L2-norm(b^p - b)$",
    )

    globals()[f"ax_{count}"].plot(
        range(b_diff_j[count].size),
        b_diff_j[count],
        linewidth=2,
        color="red",
        linestyle="--",
        marker=".",
        label=f"Jacobi method: $L2-norm(b^p - b$)",
    )
    globals()[f"ax_{count}"].plot(
        [],
        [],
        color="white",
        label=f"{size[count]} x {size[count]} Matrix",
    )
    globals()[f"ax_{count}"].set_yscale("log")
    globals()[f"ax_{count}"].set_xlabel("iteration / $i$")
    globals()[f"ax_{count}"].grid()
    globals()[f"ax_{count}"].legend()


plt.show()
fig.savefig("./Ex2/plots/plot_1.pdf", dpi="figure")
