#!/usr/bin/python
# -*- coding: utf-8 -*-
############################## imports #######################################
import numpy as np


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


def DP_ODES(l: float, g: float):
    """
    ODES for double pendelum

    For given lengh and gravitational acceleration this function generates an array of functions
    which takes in a vector of y positions

    Arguments:
        l -- length of the double pendelum l1=l2
        g -- gravitational acceleration

    Returns:
        list[Callables]
    """
    # defining the angle velocity
    w = np.sqrt(g / l)

    # system of differential equations used for the double pendelum
    # y[0] = \theta_1
    # y[1] = \theta_2
    # y[2] = momentum of the first mass \p_1
    # y[3] = momentum of the second mass \p_2

    theta_1 = lambda t, y: (
        (y[2] - y[3] * np.cos(y[0] - y[1])) / (1 + (np.sin(y[0] - y[1]) ** 2))
    )
    theta_2 = lambda t, y: (
        (2 * y[3] - y[2] * np.cos(y[0] - y[1]))
        / (1 + (np.sin(y[0] - y[1])) ** 2)
    )
    momentum_1 = lambda t, y: (
        -(y[2] * y[3] * np.sin(y[0] - y[1])) / (1 + (np.sin(y[0] - y[1])) ** 2)
        + (
            (y[2] ** 2 + 2 * y[3] ** 2 - 2 * y[2] * y[3] * np.cos(y[0] - y[1]))
            * np.sin(y[0] - y[1])
            * np.cos(y[0] - y[1])
        )
        / (1 + (np.sin(y[0] - y[1])) ** 2)
        - (2) * w**2 * np.sin(y[0])
    )
    momentum_2 = lambda t, y: (
        (y[2] * y[3] * np.sin(y[0] - y[1])) / (1 + (np.sin(y[0] - y[1])) ** 2)
        - (
            (y[2] ** 2 + 2 * y[3] ** 2 - 2 * y[2] * y[3] * np.cos(y[0] - y[1]))
            * np.sin(y[0] - y[1])
            * np.cos(y[0] - y[1])
        )
        / ((1 + (np.sin(y[0] - y[1])) ** 2) ** 2)
        - w**2 * np.sin(y[1])
    )
    return funcarray(theta_1, theta_2, momentum_1, momentum_2)


print(np.sum(np.abs(np.array([1, 2, 3]) - np.array([4, 1, 0])) ** 3))
