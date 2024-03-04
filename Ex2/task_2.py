#!/usr/bin/python
# -*- coding: utf-8 -*-


############################## imports #######################################
import numpy as np
from scipy.linalg import eigvals
from scipy.linalg import lu
from numsolvers import gauss_seidel as gs
from numsolvers import jacobi as j
from itertools import permutations as pm
import matplotlib.pyplot as plt
from tqdm import tqdm

######################### module level dunders ###############################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"
############################# Input Values ######################################
N_ = 6  # Number of Sites
t_ = -2.6  # Probability of transitioning between Site alpha -> beta
E_ = [-6, 6]  # Energy Interval
delta_E_ = 1e-2  # spacing between Energys
delta_ = 0.5  # Energy Broadening
alpha_1 = 1  # alpha site of first configuration
alpha_2 = 1  # alpha site of second configuration
beta_1 = 3  # beta site of first configuration
beta_2 = 4  # beta site of second configuration

############################# functions ######################################

# Identity matrix for solving the linear system
I = np.identity(6, dtype=np.complex_)

############################# Task 2.b ######################################
def ring(N: int, t: float, E: float, delta: float, alpha: int, beta: int):
    """
    This Function creates the Hamiltionian of the Ring System with N-Sites under the Condition
    that the Transmission of the Electron occurs between the alpha and beta Site.


    Arguments:
        N -- Number of Sites
        t -- Probability of transitioning between Site alpha -> beta
        E -- Energy of the Electron
        delta -- Energy Broadening
        alpha -- Site Alpha
        beta -- Site Beta

    Returns:
        ndarray N x N
    """

    Hr = np.zeros((N, N))
    Delta_t = np.zeros((N, N), dtype=np.complex_)
    I = np.identity(N)

    Sites = [i + 1 for i in range(N)]  # array of all possible Sites
    Sys_Perm = list(pm(Sites, 2))  # permutation of all posible tuples
    Rel_Perm = []  # relevant Permutations
    for (
        i
    ) in (
        Sys_Perm
    ):  # filtering of all possible jumps between the electrodes under the restriction that only 1 step can be made
        if np.abs(i[0] - i[1]) == 1 or np.abs(i[0] - i[1]) == N - 1:
            Rel_Perm.append(i)

    for r in Rel_Perm:  # generating the Ring Hamaltionian
        Hr[r[0] - 1, r[1] - 1] = t

    Delta_t[alpha - 1, alpha - 1] = 1j * delta  # energy broadening
    Delta_t[beta - 1, beta - 1] = 1j * delta

    return Hr - E * I + Delta_t


############################# Task 2.c ######################################
# generating matrizes to store the Hamiltonians at every Energy Input
H_Sys_14 = []
H_Sys_13 = []


Energy_Range = np.arange(E_[0], E_[1], delta_E_)


for i in Energy_Range:
    H_Sys_14.append(ring(N_, t_, i, delta_, alpha_2, beta_2))
    H_Sys_13.append(ring(N_, t_, i, delta_, alpha_1, beta_1))
# h = np.append(H_Sys_13[0], I[0], axis=0)
p, l, u = lu(H_Sys_13[1])
print("h_sys", H_Sys_13[1])
print("p", p)
print("l", l)
print("u", u)
erg = np.matmul(p, np.matmul(l, u))
print("ergebniss", np.matmul(p, np.matmul(l, u)))
t = 0
for i in l:
    print(t)
    print(i)
    t += 1
# for i, n in zip(H_Sys_13[1], erg):
#    print(t)
#    print("hsys", i)
#    print("erg", n)

# array with solved G matrizes for every Energy input
G_Sys_13 = []
G_Sys_14 = []

for current_H_13, current_H_14 in tqdm(
    zip(H_Sys_13, H_Sys_14), "Calculating G"
):
    current_G_13 = []
    current_G_14 = []

    for ei in I:
        # generating a augmented matrix to track the b vector change
        current_H_13_1 = np.append(current_H_13, [ei], axis=0)
        current_H_14_1 = np.append(current_H_14, [ei], axis=0)

        # transpose to get the right shape
        current_H_13_1 = current_H_13_1.T
        current_H_14_1 = current_H_14_1.T

        # using LU dicomposition to generate a triangular matrix
        _, _, tria_13 = lu(current_H_13_1)
        _, _, tria_14 = lu(current_H_14_1)

        # create right data structure
        tria13 = np.array(tria_13, dtype=np.complex_).T
        tria14 = np.array(tria_14, dtype=np.complex_).T

        # extracting of the triangular matrix
        current_H_13_2 = tria13[:6].T
        current_H_14_2 = tria14[:6].T
        # extracting of the new b vector which is rearanged according  to the LU decomposition
        e_new_13 = tria13[6]
        e_new_14 = tria14[6]

        # using the now diagonal dominant matrix to solve with the gauß algo
        g13, a13 = gs.gauss_seidel_method(
            current_H_13_2,
            e_new_13,
            disable_ldb=True,
            disable_gs_errorbar=True,
        )
        g14, a14 = gs.gauss_seidel_method(
            current_H_14_2,
            e_new_14,
            disable_ldb=True,
            disable_gs_errorbar=True,
        )
        current_G_13.append(g13)
        current_G_14.append(g14)

    G_Sys_13.append(current_G_13)
    G_Sys_14.append(current_G_14)


############################# Task 2.d ######################################
# calculating the eigenvalues of H_R
H_R = ring(N_, t_, 0, 0, 1, 1)
H_R_Eig = eigvals(H_R)


# generating matrizes for absolute square value of G_{\alpha,\beta}
G_ab_13 = []
G_ab_14 = []
# filtering for the Greens Functions corresponding to jumps between alpha and beta sites
for i, l in zip(G_Sys_13, G_Sys_14):
    G_ab_13.append(i[alpha_1 - 1][beta_1 - 1])
    G_ab_14.append(l[alpha_2 - 1][beta_2 - 1])

# calculating the transitions probability between sites alpha and beta
G_ab_13 = np.array(G_ab_13)
G_ab_14 = np.array(G_ab_14)

G_ab_p_13 = np.abs(G_ab_13) ** 2
G_ab_p_14 = np.abs(G_ab_14) ** 2

# plotting configuration
box_style = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
plt_1 = plt.figure(figsize=(10, 6))
# plt.style.use(["seaborn-darkgrid"])
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
    }
)

plt.plot(Energy_Range, G_ab_p_13, label="$\\alpha$ = 1 , $\\beta$ = 3 ")
plt.plot(Energy_Range, G_ab_p_14, label="$\\alpha$ = 1 , $\\beta$ = 4")
plt.vlines(
    H_R_Eig,
    ymin=0,
    ymax=1,
    colors="green",
    linestyles="dashed",
    label="$H_{R}$ Eigenvalues: "
    + str(round(H_R_Eig[0].real, 1))
    + ", "
    + str(round(H_R_Eig[1].real, 1))
    + ", "
    + str(round(H_R_Eig[2].real, 1))
    + ", "
    + str(round(H_R_Eig[3].real, 1))
    + ", "
    + str(round(H_R_Eig[4].real, 1))
    + ", "
    + str(round(H_R_Eig[5].real, 1)),
)
plt.title("Transition Probability between Sites $\\alpha$ and $\\beta$")
plt.ylim(0, 1.3)
plt.xlabel("E / arbitrary unit")
plt.ylabel("$|G_{\\alpha,\\beta}|^{2}$ / 1")
plt.legend(loc="upper right")
plt.show()
plt.savefig(
    "./Ex2/plots/plot_2_d_new.pdf",
    dpi="figure",
    orientation="landscape",
)
