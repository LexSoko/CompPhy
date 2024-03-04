#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
TODO: Title einfügen

**** README: ****
* MathTeX notation is used in comment formula explanations!
"""


############################## imports #######################################
import numpy as np
import copy
from tqdm import tqdm

######################### module level dunders ###############################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

############################# functions ######################################
def jacobi_method(
    A, b, conv_crit=10e-10, abort_crit=10e5, caldiff=False, div_crit=10e100, disable_ldb=False
):
    """
    Jacobi method

    Jacobi method for solving strictly diagonally dominant linear equation systems.
    Convergates if A strictly diagonally dominant.

    Arguments:
        A(array) -- Symetrical (n x n) 2D-Array, with no zeros on the diagonal
        b(array) -- 1D-Array of size n

    Keyword Arguments:
        conv_crit(float) -- absolut error between x**(p) and x**(p-1) below which convergenz is asumed (default: {10e-10})
        abort_crit(float) -- max. number of iterations until function is stopped (default: {10e5})
        caldiff(bool) -- If True, function will also calculate the L2-norm of the differenc between b and b**(p) (default: {False})
        div_crit(float) -- maximal value for |x^p-x^(p-1)|, afterwards the calculation is considerd divergent
        disable_ldb(bool) -- If True, no loadingbars whil be displayed while calculating

    Returns:
        x(array) -- solution vector
        a(bool) -- status parameter; when False, function did not convergate, or function stopped due to itteration max.
        b_diff_norm(array) -- L2-norm of the differenc between b and b**(p) for each iteration; only given if calldiff == True
    """

    rows, cols = np.shape(A)

    # check if array is symetrical
    if rows != cols:
        print("Error: Given matrix A is not symetrical")
        return False, False, False

    n = rows

    # get the diagonal of A as an numpy array
    A_dia = []
    for i in range(0, n):
        A_dia.append(A[i][i])
    A_dia = np.array(A_dia)

    # check if any of the diagonal values is 0
    if np.any(A_dia == 0):
        print(
            "Error jacobi_method: Given Matrix A has 0 as its value on the diagonal"
        )
        return

    x = [1.0] * n  # created starting vektor
    x = np.array(x, dtype=np.complex_)

    err = (
        conv_crit + 1
    )  # create starting error, which is bigger than the convergenc criteria

    sum1 = 0
    sum2 = 0
    b_diff_norm = []
    p = 0  # p is the iteration count
    while (
        np.any(err > conv_crit) and np.any(err < div_crit) and p < abort_crit
    ):
        x_p_1 = copy.deepcopy(x)
        for i in tqdm(range(0, n), desc="jacobi row-loop", disable=disable_ldb):
            for j in range(0, i):
                sum1 += A[i][j] * x_p_1[j]
            for j in range(i + 1, n):
                sum2 += A[i][j] * x_p_1[j]

            x[i] = 1 / A[i][i] * (b[i] - sum1 - sum2)

            sum1 = 0
            sum2 = 0

        # absolut error
        err = np.abs(x - x_p_1)

        # calculate differenz between current b and real b
        if caldiff:
            b_diff_norm.append(np.linalg.norm(np.dot(A, x) - b))
        if (
            any(ele == np.inf for ele in b_diff_norm)
            or any(ele == np.nan for ele in b_diff_norm)
        ):
            p = abort_crit
        p += 1

    if p < abort_crit:
        a = True  # the algorithm calculated the result within the given iteration max
    else:
        a = False  # the algorithm did not calculated the result within the given iteration max

    if caldiff:
        b_diff_norm = np.array(b_diff_norm)
        b_diff_norm = np.nan_to_num(b_diff_norm, posinf=1.7e155)
        return x, a, b_diff_norm

    return x, a
