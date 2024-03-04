#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Assignment 2
Toolbox for discretized n dimensional coordinate magic

def nb(k,**kwargs):
    calculates neighboring k indizes for given k as requested in assignment 2

def get_index_permutations(*xi):
    get all index permutations for given coordinate index lists xi
    
def get_roled_up_index(*ni , dbg=False, **Ni):
    calculates rolled up index from given indices ni with given coordinate intervall counts Ni
    with dbg=True the function prints out debug info to console

def get_unrolled_indizes(k: int, dbg=False, **Ni: int) -> list(int):
    inverse function of get_rolled_up_index()

**** README: ****
* MathTeX notation is used in comment formula explanations!
"""


############################## imports #########################################
import copy
from typing import Any, Tuple
import numpy as np
import re

######################### module level dunders #################################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

########################## helper functions ####################################


# TODO: update docstring
#
def nb(k: int, bound="dirichlet", **kwargs: int) -> Any:
    """
    function for calculating rolled up indices of neighbouring points in gridspace.
    
    This function has two signatures(k is a rolled up index):
        nb(1, mu=1,**Ni) -> int -- will return one neigbouring rolled up index where the spacial direction is given by mu this is limited to d=2\\
        nb(1, d=3,bound="periodic",**Ni) -> list(int) -- will give a list of all neighbouring rolled up indices in 3 dimensions with periodic boundary contitions

    Arguments:
        k -- a rolled up index as integer

    Keyword Arguments(only one of both allowed):
        mu -- integer defining the spatial direction in d dimensions where mu can have the values 1,2,...,d for positive spatial direction and d+1,d+2,...,2d for negative spatial direction.
        d -- integer defining the number of dimensions
        bound(str) -- either \"dirichlet\" for neighbours in dirichlet borders or \"periodic\" in periodic borders
        Ni -- where i has to be a number in the keyword, total number of intervalls on each axis as integers
    
    Returns:
        k_nb -- either one or a list of neighbouring rolled up indices depending on the function call (-1 if there are to many kwargs, -2 if neither mu nor d is present in kwargs)
    """

    # filter out dimensionalities Ni with regex "^N\d" for all keys starting with "N<digit>"
    Ni = {
        key: value
        for (key, value) in kwargs.items()
        if not (re.match(r"^N\d", key) is None)
    }

    # prepare mu list depending on given key word arguments
    if "mu" in kwargs:
        mu_list = [kwargs.get("mu")]
        d = 2
    elif "d" in kwargs:
        mu_list = [i + 1 for i in range(2 * kwargs.get("d"))]
        d = kwargs.get("d")
    else:
        return -2

    # check if dimentionality fits Ni size
    if len(Ni) != d:
        return -3

    if bound == "periodic":
        # x,y,z,... indices of point k
        indices = list(get_unrolled_indizes(k, **Ni))
        k_nb_list = []
        # iterates through each direction in which a neighbour can be
        for i, mu_i in enumerate(mu_list):
            # sign gives direction on axis
            sign = 1 if mu_i <= d else -1
            # dim gives axis in which to calculate neighbour
            dim = (mu_i - 1) % d

            # add axis direction to indices of k
            indices_nb = copy.deepcopy(indices)
            new_ind = indices_nb[dim] + sign
            # this deals with ring like indexing in periodic corder conditions
            indices_nb[dim] = (new_ind + Ni["N" + str(dim + 1)]) % Ni[
                "N" + str(dim + 1)
            ]
            k_nb = get_roled_up_index(*indices_nb, **Ni)

            k_nb_list.append(k_nb)

        return k_nb_list

    elif bound == "dirichlet":
        # x,y,z,... indices of point k
        ni = list(get_unrolled_indizes(k, **Ni))
        # iterates through each index of k and checks if it is at the border of
        # the grid
        for j, n_j in enumerate(ni):
            # 0 border mark this mu with -1
            if n_j == 0:
                mu_list[d + j] = -1
            # max border mark this mu with -1
            elif n_j == Ni["N" + str(j + 1)] - 1:
                mu_list[j] = -1
        # throw out all mu's with -1 so we only have neighbours in the grid
        mu_list = [mu_i for mu_i in mu_list if mu_i != -1]

        k_nb_list = []

        # iterates through each direction in which a neighbour can be
        for i, mu_i in enumerate(mu_list):
            # sign gives direction on axis
            sign = 1 if mu_i <= d else -1
            # dim gives axis in which to calculate neighbour
            dim = ((mu_i - 1) % d) + 1

            # calculate dimensionality product
            p = 1
            # use the formula for k from the script to calculate the neighbouring k
            for j, N_j in enumerate(Ni.values()):
                if j == dim - 1:
                    break
                p = p * N_j

            # calculate neighboring k
            k_nb_list.append(k + sign * p)

        return k_nb_list


def get_index_combinations(*xi):
    """
    returns all index combinations for given room xi

    Arguments:
        *xi -- the indices lists for the i-th coordinate axis

    Returns:
        all different permutations of the different coordinate index or in other
        terms a list of all possible index combination for the given spatial
        grid
    """
    d = len(xi)
    return np.array(np.meshgrid(*xi, indexing="ij")).T.reshape(-1, d)


# TODO: rolled
def get_roled_up_index(*ni: int, dbg=False, **Ni: int) -> int:
    """
    Calculates the roled up index adress of a discrete :math:`d`-dimensional\\
    coordinate system.
    
    For given indicess :math:`n_1,n_2, ... ,n_d` with coordinate bounds\\
    :math:`N_1,N_2, ... ,N_d` a distinct \"roled up\" index adress
    :math:`k` will be calculated in the following way:\\
    :math:`k = n_1 + n_2 * N_1 + n_3 * N_1 * N_2 + ... + n_d * N_1 * ... * N_{d-1}`
    
    sum(1<=i<=d){n_i* prod()}

    Arguments:
        *ni -- Tuple of coordinate indices for one point as integers
    
    Keyword Arguments:
        dbg -- if this flag is set to true, this function makes debug prints to console (default: {False})\\
        **Ni -- kwargs for the total number of intervalls on each axis as integers
        
    Returns:
        k -- rolled up index as integer (
            -1 if there wasn't the same amount of ni and Ni)
    """
    if len(ni) != len(Ni):
        return -1

    d = len(ni)
    k = 0
    if dbg:
        for i, n_i in enumerate(ni):
            p = 1
            string = "+ n" + str(i + 1)
            for j, N_j in enumerate(Ni.values()):
                if j == i:
                    break
                string = string + "*N" + str(j + 1)
                p = p * N_j
            if dbg:
                print(string)
            k = k + n_i * p
    else:
        # this code implements the formula for k from the script
        for i, n_i in enumerate(ni):
            p = 1
            for j, N_j in enumerate(Ni.values()):
                if j == i:
                    break
                p = p * N_j
            k = k + n_i * p
    return k


def get_unrolled_indizes(k: int, dbg=False, **Ni: int) -> Tuple[int, ...]:
    """
    Unrolls the rolled up index :math:`k` into the indices :math`n_i`.
    
    The unrolling is achieved by inverting the calculation:
    :math:`k = n_1 + n_2 * N_1 + n_3 * N_1 * N_2 + ... + n_d * N_1 * ... * N_{d-1}`
    which gives back the tuple:
    :math:`n_i = (n_1, n_2, ... , n_d)`
    where the tuple is calculated with:
    :math:`(k \\mod N_1, \\frac{k - n_1}{N_1} \\mod N_2, ... , \\frac{\\frac{\\frac{k - n_1}{N_1} - ... }{...} - n_{d-1} }{N_{d-1}})`

    Arguments:
        k -- rolled up index as integer 

    Keyword Arguments:
        dbg -- if this flag is set to true, this function makes debug prints to console (default: {False})\\
        **Ni -- kwargs for the total number of intervalls on each axis as integers

    Returns:
        *ni -- Tuple of coordinate indices for one point as integers
    
    Notes:
        This function per definition is not able to check if the correct number of **Ni are passed!
    """
    d = len(Ni)
    ni = []
    kj = k

    for j, Nj in enumerate(Ni.values()):
        nj = kj % Nj
        kj = int((kj - nj) / Nj)
        ni.append(nj)

    return tuple(ni)
