import numpy as np
import matplotlib.pyplot as plt
import numsolvers.runge_kutta as rk
from tqdm import tqdm


# define constants
eta = 0.001
psi_0 = [0, 1]  # [psi, psi_derivation], psi_dev(s = -1/2) = 1
v01 = 10
v02 = 100
v1 = lambda s: 0
v2 = lambda s: v01 * np.exp(s**2 / 0.08)
v3 = lambda s: v02 * np.heaviside(s - 0.3, 0.5) * np.heaviside(0.35 - s, 0.5)
v4 = lambda s: -v02 * np.heaviside(0.25 - s, 0.5) * np.heaviside(s + 0.25, 0.5)


def erwartungswert(s, psi):
    x = s * psi[0]
    return np.sum(x)


def varianz(s, psi):
    x2 = s**2 * psi[0]
    x_2 = s * psi[0]
    return np.sum(x2) - np.sum(x_2) ** 2


# berechnung der Wellenfunktion
def get_psi(epsilon, v, psi_0, t0=-1 / 2, t_max=1 / 2, t_delta=0.01):
    f = lambda s, psi, epsilon, v: np.array(
        [psi[1], 2 * (v(s) - epsilon) * psi[0]]
    )
    s, psi = rk.rk4_method_asg3(
        F=f, y0=psi_0, t0=t0, t_max=t_max, e=t_delta, epsilon=epsilon, v=v
    )
    return s, psi


# pseudo code implementation
def find_epsilon(v, psi_0, eta, eigenvalue_num=5, increment=0.1):

    epsilon = []
    epsilon_aver = 0

    # step 1
    epsilon_trial = [0, 0]

    for j in tqdm(range(eigenvalue_num), desc="pseudocode"):

        # step 1 and 2
        i = 0
        psi1 = get_psi(epsilon_trial[0], v, psi_0)[1][0][-1]
        psi2 = get_psi(epsilon_trial[1], v, psi_0)[1][0][-1]

        while (psi1 * psi2) > 0:
            epsilon_trial[1] = epsilon_trial[0] + eta * np.exp(i)
            i += 0.1
            psi1 = get_psi(epsilon_trial[0], v, psi_0)[1][0][-1]
            psi2 = get_psi(epsilon_trial[1], v, psi_0)[1][0][-1]

        # step 3
        while (np.abs(epsilon_trial[0] - epsilon_trial[1])) > eta:
            epsilon_aver = (epsilon_trial[0] + epsilon_trial[1]) / 2
            if (
                (get_psi(epsilon_trial[0], v, psi_0)[1][0][-1])
                * (get_psi(epsilon_aver, v, psi_0)[1][0][-1])
            ) < 0:
                epsilon_trial[1] = epsilon_aver
            else:
                epsilon_trial[0] = epsilon_aver
        epsilon.append(epsilon_aver)
        epsilon_trial = [epsilon_aver + eta, epsilon_aver + 2 * eta]

    return np.array(epsilon)


# analytische eigenwerte für das unendliche kastenpotenzial
def eigenvalue_analytic(n=1):
    return np.pi**2 * (n) ** 2 / 2


########### c.) ###################

# plotting
fig, axs = plt.subplots(5)
fig.subplots_adjust(
    left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.15, hspace=0.55
)
fig.suptitle("Eigenfunktionen für das unendliche Kastenpotential")

eigenvalue = find_epsilon(v1, psi_0, eta, eigenvalue_num=5)

for ind, val in enumerate(eigenvalue):
    axs[ind].set_title(f"n={ind+1}")
    val_analytic = eigenvalue_analytic(ind + 1)
    s, psi = get_psi(val, v1, psi_0)
    psi_norm = [psi[0] + np.abs(np.amin(psi[0])), psi[1]]
    s_analytic, psi_analytic = get_psi(val_analytic, v1, psi_0)
    psi_analytic_norm = [
        psi_analytic[0] + np.abs(np.amin(psi_analytic[0])),
        psi_analytic[1],
    ]
    ew = round(erwartungswert(s, psi_norm), 3)
    var = round(varianz(s, psi_norm), 3)
    ew_a = round(erwartungswert(s_analytic, psi_analytic_norm), 3)
    var_a = round(varianz(s_analytic, psi_analytic_norm), 3)
    axs[ind].scatter(
        s,
        psi_norm[0],
        color="red",
        s=20,
        alpha=0.9,
        label="Numerische Lösung" + f"<s> = {ew}, var(s) = {var}",
    )
    axs[ind].plot(
        s_analytic,
        psi_analytic_norm[0],
        label="Analytische Lösung" + f"<s> = {ew_a}, var(s) = {var_a}",
    )
    axs[ind].legend()
# fig.savefig("./plots/plot_1_b.pdf", dpi="figure")


plt.show()

# ############ d.) / e.) ###################


# plotting
fig, axs = plt.subplots(5)
fig.subplots_adjust(
    left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.15, hspace=0.55
)
fig.suptitle("Eigenfunktionen für verschiedene Potenziale")

eigenvalue2 = find_epsilon(v2, psi_0, eta, eigenvalue_num=5)
eigenvalue3 = find_epsilon(v3, psi_0, eta, eigenvalue_num=5)
eigenvalue4 = find_epsilon(v4, psi_0, eta, eigenvalue_num=5)

for i in range(5):
    axs[i].set_title(f"n={i+1}")
    axs[i].set_xlabel("s / L")
    s2, psi2 = get_psi(eigenvalue2[i], v2, psi_0)
    s3, psi3 = get_psi(eigenvalue3[i], v3, psi_0)
    s4, psi4 = get_psi(eigenvalue4[i], v4, psi_0)
    psi2_norm = [psi2[0] + np.abs(np.amin(psi2[0])), psi2[1]]
    psi3_norm = [psi3[0] + np.abs(np.amin(psi3[0])), psi3[1]]
    psi4_norm = [psi4[0] + np.abs(np.amin(psi4[0])), psi4[1]]
    ew2 = round(erwartungswert(s2, psi2_norm), 3)
    ew3 = round(erwartungswert(s3, psi3_norm), 3)
    ew4 = round(erwartungswert(s4, psi4_norm), 3)
    var2 = round(varianz(s2, psi2_norm), 3)
    var3 = round(varianz(s3, psi3_norm), 3)
    var4 = round(varianz(s4, psi4_norm), 3)

    axs[i].scatter(
        s2,
        psi2_norm[0],
        color="red",
        s=20,
        alpha=0.9,
        label=f"v = {v01}"
        + "$\cdot e^{s^2 / 0.08}$ \n"
        + f"<s> = {ew2}, var(s) = {var2}",
    )
    axs[i].scatter(
        s3,
        psi3_norm[0],
        color="blue",
        s=20,
        alpha=0.9,
        label=f"v = {v02}"
        + "$\cdot \\theta(s - 0.3) \cdot \\theta(0.35 - s)$"
        + f"<s> = {ew3}, var(s) = {var3}",
    )
    axs[i].scatter(
        s4,
        psi4_norm[0],
        color="green",
        s=20,
        alpha=0.9,
        label=f"v = -{v02}"
        + "$\cdot \\theta(0.25 - s) \cdot \\theta(s + 0.25)$"
        + f"<s> = {ew4}, var(s) = {var4}",
    )

    axs[i].legend()
# fig.savefig("./plots/plot_1_e.pdf", dpi="figure")
plt.show()
