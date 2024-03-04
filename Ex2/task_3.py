#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Assignment 2 Task 3

**** README: ****
* MathTeX notation is used in comment formula explanations!
"""


############################## imports #########################################
import timeit
import coordinate_tools.rolled_up_index as roupin
import numpy as np
import matplotlib.pyplot as plt
import numsolvers.gauss_seidel as gs
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

######################### module level dunders #################################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

########################## execution script ####################################
#%%
############################### Task 3.a #######################################
"""
Task 3.a was achieved in the package coordinate_tools.rolled_up_index

Ps.: the function nb(k,d) was choosen to be implemented, because it has a more
general use case and as additional arguments the axis intervall counts Ni had to
be added to the function, otherwise it would not have been possible to detect
space boundaries and it also would not have been possible to calculate
neighboring k indices`.
"""
#%%
############################### Task 3.b #######################################
"""
This section creates a N1 x N2 Matrix which describes the Laplace Operator
for a discretized spatial grid with periodic boundary conditions
"""

# the Poisson matrix calculation was programmed in a function, because it is
# used for the benchmarks in 3.d and 3.e
def get_pooisson_matrix(**Ni):
    """
    Creates a Poisson matrix for periodic boundary conditions in the 2D
    gridspace defined by **Ni

    Keyword Arguments:
        **Ni -- kwargs for the total number of intervalls on each axis as integers

    Returns:
        A_ij , nb_list -- A_ij is the discrete poisson matrix, nb_list is the
        list of lists of neighbouring k indices for each k index
    """
    # our rolled up indexing functions work with a dictionary for the different Ni's
    dimensions = len(Ni)

    # gives a list of the different coordinate intevalls, for the 2D case this will
    # be [nparray([0,1,2,...,N1-1]),nparray([0,1,2,...,N2-1])]
    ni_lists = [np.arange(0, Ni) for Ni in Ni.values()]

    # now generate every possible index combination aka the indexes for every point
    # in the 2D Grid
    ix_iy_indexes = roupin.get_index_combinations(*ni_lists)
    # Map indices to rolled up k indices
    k_indizes = [
        roupin.get_roled_up_index(xy[0], xy[1], **Ni) for xy in ix_iy_indexes
    ]
    # poisson array size in one dimension
    K = len(k_indizes)
    # prepare empty poisson array
    A_ij = np.zeros(tuple([K] * 2))
    # get all neighbour k's for each k
    nb_list = [
        roupin.nb(j, bound="periodic", d=dimensions, **Ni) for j in k_indizes
    ]

    # set values of the poisson array, -4 for i=j, 1 for j in nb(i)
    for k in k_indizes:
        A_ij[k, k] = -4.0
        A_ij[k, roupin.nb(k, bound="periodic", d=dimensions, **Ni)] = 1.0

    return A_ij, nb_list, k_indizes


# calculate poisson matrix for 50x50 grid
Ni_dict = {"N1": 50, "N2": 50}
A_ij, _, k_indizes = get_pooisson_matrix(**Ni_dict)

# Plotting of the Poisson matrix
fig_3b = plt.figure()
fig_3b.suptitle(
    "3.b) Poisson Matrix (periodic boundaries, "
    + str(Ni_dict["N1"])
    + "x"
    + str(Ni_dict["N2"])
    + " grid)"
)
ax_3b = fig_3b.add_subplot(111)
matshow_ax_3b = ax_3b.matshow(A_ij)
cbar_3b = fig_3b.colorbar(matshow_ax_3b)
cbar_3b.set_label("Matrix values", rotation=90)


#%%
############################### Task 3.c #######################################
"""
This section solves the poisson problem for a dipole arangement using the
gauss-seidl method
"""

# the dipole calculation was programmed in a function, because it is
# used for the benchmarks in 3.d and 3.e
def calc_dipole_prob(
    lapl_matrix,
    use_nonzero_gs=False,
    nonzero_indizes_list=None,
    disable_gs_bar=True,
    **Ni,
):
    """
    Calculates a solution for dipole charges with given laplace matrix
    lapl_matrix and gridspace **Ni

    Arguments:
        lapl_matrix -- Matrix that describes the discrete Laplace operator

    Keyword Arguments:
        use_nonzero_gs -- if this flag is True, the gauss-seidl method, which is
        optimized for a big number of zero values in the Matrix is used
        nonzero_indizes_list -- list of all k indicess which are not zero
        **Ni -- kwargs for the total number of intervalls on each axis as integers

    Returns:
        Voltage U, charge rho -- both as matrices, directly representing the
        values of Voltage and charge in the gridspace
    """
    # coordinate setup for dipole
    ix_rho1 = int(Ni_dict["N1"] * 8 / 17)
    ix_rho2 = Ni_dict["N1"] - ix_rho1
    iy_rho = int(Ni_dict["N2"] / 2)

    # prepare rho vector
    rho = np.zeros((np.product(np.array(list(Ni_dict.values())))))
    rho[roupin.get_roled_up_index(ix_rho1, iy_rho, **Ni_dict)] = 1.0
    rho[roupin.get_roled_up_index(ix_rho2, iy_rho, **Ni_dict)] = -1.0

    # calculate Potential U with gauss-seidl
    if use_nonzero_gs:
        U, success = gs.gauss_seidel_method_nonzero(
            lapl_matrix,
            rho,
            nonzero_indizes_list,
            conv_crit=10e-4,
            disable_ldb=True,
            disable_gs_errorbar=disable_gs_bar,
        )
    else:
        U, success = gs.gauss_seidel_method(
            lapl_matrix,
            rho,
            conv_crit=10e-4,
            disable_ldb=True,
            disable_gs_errorbar=disable_gs_bar,
        )

    if not success:
        print("gauss seidl not successfull")

    # reshaping to easyly show voltage and charge with plt.matshow()
    U = U.reshape((Ni_dict["N2"], Ni_dict["N1"]))
    # gauss-seidl works with complex numbers, so we only get the real part
    U = np.real(U)
    rho = rho.reshape((Ni_dict["N2"], Ni_dict["N1"]))
    return U, rho


U, rho = calc_dipole_prob(A_ij, disable_gs_bar=False, **Ni_dict)

# plotting the dipole potential
x, y = np.meshgrid(range(Ni_dict["N1"]), range(Ni_dict["N2"]))
fig_3c = plt.figure(figsize=(15, 5))
fig_3c.suptitle(
    "3.c) Dipole Problem (periodic boundaries, "
    + str(Ni_dict["N1"])
    + "x"
    + str(Ni_dict["N2"])
    + " grid)"
)
# plot charge density
ax_rho_3c = fig_3c.add_subplot(121, projection="3d")
surf_rho_3c = ax_rho_3c.plot_surface(
    x,
    y,
    rho,
    cmap=cm.coolwarm,
    linewidth=0.1,
    edgecolors="black",
)
ax_rho_3c.set(
    title="Charge Distribution",
    xlabel=r"$i_x$",
    ylabel=r"$i_y$",
    zlabel=r"$\rho_(x,y)$",
)
ax_rho_3c.view_init(-160, 80)
cbar_rho_3c = fig_3c.colorbar(surf_rho_3c, shrink=0.5, aspect=5)
cbar_rho_3c.set_label("Charge Values", rotation=90)

# plot potential
ax_pot_3c = fig_3c.add_subplot(122, projection="3d")
surf_pot_3c = ax_pot_3c.plot_surface(
    x,
    y,
    U,
    cmap=cm.coolwarm,
    linewidth=0.1,
    edgecolors="black",
)
ax_pot_3c.set(
    title="Potential",
    xlabel=r"$i_x$",
    ylabel=r"$i_y$",
    zlabel=r"$\Phi(x,y)$",
)
ax_pot_3c.view_init(-160, 80)
cbar_pot_3c = fig_3c.colorbar(surf_pot_3c, shrink=0.5, aspect=5)
cbar_pot_3c.set_label("Potential Values", rotation=90)
fig_3c.savefig("./Ex2/plots/plot_3c.pdf", dpi="figure")

#%%
############################### Task 3.d #######################################
"""
This section is a benchmark for computation time in relation to spatial grid size
"""
timescalingplot = True
if timescalingplot:
    # prepare N list for the NxN grids and the time list which we both plot later
    N_list = np.arange(5, 100)
    t_list = np.zeros_like(N_list)
    for i, N in enumerate(tqdm(N_list)):
        # for each NxN grid measure time before and after calculaton of the dipole
        # problem and then add the timediff to the time list
        starttime = timeit.default_timer()
        Ni_dict = {"N1": N, "N2": N}
        A, _, _ = get_pooisson_matrix(**Ni_dict)
        U, rho = calc_dipole_prob(A, **Ni_dict)
        endtime = timeit.default_timer()
        t_list[i] = endtime - starttime

#%%
############################### Task 3.e #######################################
"""
This section is a benchmark for computation time in relation to spatial grid size
"""
if timescalingplot:
    N_list_nz = np.arange(5, 100)
    t_list_nz = np.zeros_like(N_list_nz)

    for i, N in enumerate(tqdm(N_list_nz)):
        # for each NxN grid measure time before and after calculaton of the dipole
        # problem and then add the timediff to the time list
        # but now we do calculations with a faster gauss seidl
        starttime = timeit.default_timer()
        Ni_dict = {"N1": N, "N2": N}
        A, nb_list, _ = get_pooisson_matrix(**Ni_dict)
        U, rho = calc_dipole_prob(
            A, use_nonzero_gs=True, nonzero_indizes_list=nb_list, **Ni_dict
        )
        endtime = timeit.default_timer()
        t_list_nz[i] = endtime - starttime

    # plotting of timesacaling benchmarks of 3.d and 3.e
    fig_3de = plt.figure()
    fig_3de.suptitle(
        "3d),3e): Time scaling comparison for normal Gauss Seidl\n and Nonzero matrix element optimised Gauss Seidl"
    )
    ax_3de = fig_3de.add_subplot(111)
    ax_3de.plot(N_list, t_list, color="red", label=r"$t(N x N)$")
    ax_3de.plot(
        N_list_nz, t_list_nz, color="green", label=r"$t_{Non 0}(N x N)$"
    )
    ax_3de.legend()
    ax_3de.set(xlabel=r"$N$", ylabel=r"$t$")
    fig_3de.savefig("./Ex2/plots/plot_3de.pdf", dpi="figure")

#%%
############################### Task 3.f #######################################
"""
This section creates a simulation for a faraday cage(25x25)
in an electrical potential (50x50)
Phi(x,0) = U, Phi(x,L_y) = -U, U=1
and periodic on the other axis
Phi(0,y)=Phi(L_x,y)

The Main code parts are the solution constraints for the cage and  the boundary
conditions Phi(x,0) and Phi(x,L_y).
For this a mask array is created which holds True values where the value of
Phi(x,y) is predetermined. A second array mask holds the predetermined values of
Phi(x,y)
In the gaus seidl algorithm, the array mask with predetermined Phi(x,y) values
is then used as start solution vector and the boolean array mask is used to skip
calculations for predetermined valunes, so they always stay the same.
"""
Ni_dict = {"N1": 50, "N2": 50}
A, nb_list, _ = get_pooisson_matrix(**Ni_dict)

# create complete grid
ni_lists = [np.arange(0, Ni) for Ni in Ni_dict.values()]
# now generate every possible index combination aka the indexes for every point
# in the 2D Grid
ix_iy_indexes = roupin.get_index_combinations(*ni_lists)
k_indizes = [
    roupin.get_roled_up_index(xy[0], xy[1], **Ni_dict) for xy in ix_iy_indexes
]
mask_False = np.full((Ni_dict["N1"], Ni_dict["N2"]), False)
mask_True = np.full((Ni_dict["N1"], Ni_dict["N2"]), True)

# cage indices for the outer bounds of the faraday cage
# walls paralell to y axis
ix_yb_cage = np.arange(12, 12 + 25)
iy_yb_cage = np.array([12, 12 + 25 - 1])
ixy_yb_cage = roupin.get_index_combinations(ix_yb_cage, iy_yb_cage)

# walls paralell to x axis
ix_xb_cage = np.array([12, 12 + 25 - 1])
iy_xb_cage = np.arange(12, 12 + 25)
ixy_xb_cage = roupin.get_index_combinations(ix_xb_cage, iy_xb_cage)

# add wall indices lists together
ixy_b_cage = np.concatenate((ixy_yb_cage, ixy_xb_cage))
k_b_cage = [
    roupin.get_roled_up_index(xy[0], xy[1], **Ni_dict) for xy in ixy_b_cage
]

# create boolean mask for the whole grid with True elements where the cage walls are
mask_b_cage = np.isin(k_indizes, k_b_cage)


# Phi boundary masks are calculated the same way as cage bounds
# Phi+ indices
ix_y0_Phi = np.arange(0, Ni_dict["N1"])
iy_y0_Phi = np.array([0])
ixy_y0_Phi = roupin.get_index_combinations(ix_y0_Phi, iy_y0_Phi)
k_y0_Phi = [
    roupin.get_roled_up_index(xy[0], xy[1], **Ni_dict) for xy in ixy_y0_Phi
]
# Phi+ mask
mask_y0_Phi = np.isin(k_indizes, k_y0_Phi)

# Phi- indices
ix_yL_Phi = np.arange(0, Ni_dict["N1"])
iy_yL_Phi = np.array([Ni_dict["N2"] - 1])
ixy_yL_Phi = roupin.get_index_combinations(ix_yL_Phi, iy_yL_Phi)
k_yL_Phi = [
    roupin.get_roled_up_index(xy[0], xy[1], **Ni_dict) for xy in ixy_yL_Phi
]
# Phi- mask
mask_yL_Phi = np.isin(k_indizes, k_yL_Phi)

# build binding Phi mask with the Voltage values of the bound grid points
binding_Phi = np.zeros(Ni_dict["N1"] * Ni_dict["N2"])
binding_Phi[mask_y0_Phi] = 1
binding_Phi[mask_yL_Phi] = -1
binding_Phi[mask_b_cage] = 0
binding_mask = np.logical_or(
    np.logical_or(mask_y0_Phi, mask_yL_Phi), mask_b_cage
)

# build charge density
rho = np.zeros(Ni_dict["N1"] * Ni_dict["N2"], dtype=np.float64)

# calculate potential
U, success = gs.gauss_seidel_method_nonzero(
    A,
    rho,
    nb_list,
    conv_crit=10e-4,
    disable_ldb=True,
    binding_x=binding_Phi,
    binding_mask=binding_mask,
)
print("success=", success)

# get real valued array for plotting
U = np.real(U.reshape((Ni_dict["N2"], Ni_dict["N1"])))
# get xy mesh for plotting
x, y = np.meshgrid(range(Ni_dict["N1"]), range(Ni_dict["N2"]))

# plotting of the Potential U
fig_3f = plt.figure()
fig_3f.suptitle(
    "3.c) Faraday cage Problem (periodic x boundaries,\n dirichlet y boundaries "
    + str(Ni_dict["N1"])
    + "x"
    + str(Ni_dict["N2"])
    + " grid)"
)
ax_3f = fig_3f.add_subplot(111, projection="3d")
surf_3f = ax_3f.plot_surface(
    x, y, U, cmap=cm.coolwarm, linewidth=0.1, edgecolors="black"
)
ax_3f.set(
    title="Potential",
    xlabel=r"$i_x$",
    ylabel=r"$i_y$",
    zlabel=r"$\Phi(x,y)$",
)
ax_3f.view_init(-160, 80)
cbar_pot_3f = fig_3f.colorbar(surf_3f, shrink=0.5, aspect=5)
cbar_pot_3f.set_label("Potential Values", rotation=90)
fig_3f.savefig("./Ex2/plots/plot_3f.pdf", dpi="figure")

plt.show()
