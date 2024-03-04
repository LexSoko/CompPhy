#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Execution script for the Animation
**** README: ****
* MathTeX notation is used in comment formula explanations!
"""


############################## imports #######################################
from pickle import NONE
import numsolvers.runge_kutta as rk
import numsolvers.ODE_Functions as ode
import numpy as np
from numpy import sin, cos
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter, FFMpegWriter
from collections import deque
from tqdm import tqdm

######################### module level dunders ###############################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

############################### script #######################################

# constants
g = 9.83  # acceleration due to gravity, in m/s^2
l = 1.0  # length of pendulum 1 and 2

# configuration variables
y0 = np.array([0.5, 0.5, 0, 0])  # initial conditions
t0 = 0  # initial timepoint
t_max = 60  # maximum timepoint
fps = 60  # frames per second of the gif or mp4
e = t_max / (t_max * fps)  # timestep size


# solving the ODE system
# dt_a, y_a = rk.expl_rk_method(
#    F=ode.DP_ODES(l, g),
#    y0=y0,
#    a_ij=rk.rk4_a_ij,
#    bj=rk.rk4_b_j,
#    cj=rk.rk4_c_j,
#    t0=t0,
#    t_max=t_max,
#    e=e,
# )
dt_b, y_b = rk.rk4_method(
    F=ode.DP_ODES(l, g),
    y0=y0,
    t0=t0,
    t_max=t_max,
    e=e,
)
dt_a, y_a, itper, errper = rk.adpt_rk4_method(
    F=ode.DP_ODES(l, g),
    y0=y0,
    t0=t0,
    t_max=t_max,
    e0=e,
)
plt.plot(dt_a, y_a[1], label="adapt")
plt.plot(dt_b, y_b[1], label="normal")
plt.legend()
plt.show()
# convert into cartesian coordinates
x1 = l * sin(y_b[0])
y1 = -l * cos(y_b[0])

x2 = l * sin(y_b[1]) + x1
y2 = -l * cos(y_b[1]) + y1

# plotting configuration
history_len = 100  # how many trajectory points to display
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(
    autoscale_on=False, xlim=(-2 * l, 2 * l), ylim=(-3 * l, 3 * l)
)

# ax.set_facecolor("black")  # sets the background color to black
ax.set_aspect("equal")  # Set the aspect ratio of the axes scaling
ax.grid()

(line,) = ax.plot([], [], "o-", lw=2)
(trace,) = ax.plot([], [], "r.-", lw=1, ms=2)
time_template = "time = %.1fs"
time_text = ax.text(
    0.05, 0.9, "", transform=ax.transAxes
)  # plots current time
history_x, history_y = deque(maxlen=history_len), deque(
    maxlen=history_len
)  # deque allows faster append and pop operations

# animation function that displays the i-th element
def animate(i):
    """
    animation function

    This function defines how the animations should take place

    Arguments:
        i -- frames

    Returns:
        tuple[Line2D,Line2D,text]
    """
    thisx = [0, x1[i], x2[i]]  # current x at i-th frame
    thisy = [0, y1[i], y2[i]]  # current y at i-th frame

    # first frame erases all previous positions for trace
    if i == 0:
        history_x.clear()
        history_y.clear()

    # previous positions of the lower mass are saved
    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy)  # plotting lines between the three points
    trace.set_data(history_x, history_y)  # plotting previous points as a trace
    time_text.set_text(time_template % (i * e))  # updates displayed time

    return line, trace, time_text


# generating the animation
ani = animation.FuncAnimation(
    fig, animate, len(y_a[0]), interval=t_max, blit=True
)
# showing the animation
plt.show()
# writergif = PillowWriter(fps=30)
# ani.save("double_pendelum.gif", writer=writergif)
# FFwriter = animation.FFMpegWriter(fps=10)
# ani.save("animation.mp4", writer=FFwriter)

# progress bar for gif/mp4 generation
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
writermp4 = FFMpegWriter(fps=fps)
#ani.save(
#    "./Ex1/plots/double_pendelum2.mp4",
#    writer=writermp4,
#    progress_callback=update_progress,
#)
