#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Assignment 4 Task 1 Singular value decomposition of a drawing
"""


############################## imports #########################################
from PIL import Image
import numpy as np
import numsolvers.eigenvector_methods as eigm
from tqdm import tqdm
import matplotlib.pyplot as plt


######################### module level dunders #################################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"

########################## execution script ####################################

#####           TASK 1.A            #####
#%%
# siehe numsolvers/eigenvector_method

#####           TASK 1.B            #####
#%%


img = Image.open(".\EX4\\Lady_with_Ermine.bmp")  # load image
A = np.array(img, dtype="float64")  # convert to numpy array
A_r = A[:, :, 0]  # extract matrix for red

B = A_r.transpose().conjugate().dot(A_r)  # calculate square matrix B
eigenvalues, eigenvectors, residuum = eigm.eig_n(
    B, 10
)  # calculate eigenvectors and eigenvalues of B


####           plotting            #####
fig, ax = plt.subplots(1, 3, figsize=(20, 15))
fig.suptitle("Powermethod eigenvector computation", fontsize="20")
for i in range(3):
    xdata = np.arange(0, len(residuum[i][:]))
    ax[i].set_title(f"Eigenvector {i+1}")
    ax[i].set_xlabel("iteration")
    ax[i].set_ylabel("residuum")
    ax[i].semilogy(xdata, residuum[i][:], "-", color="red", alpha=1, label=r"")
    ax[i].grid(True, which="both")
fig.savefig(".\\EX4\\plots\\task_1\\task_1b.pdf")


# #####           TASK 1.C            #####
#%%

n = [1, 5, 10, 20]  # define number of eigenvectors to be calculated
input = ["Vitruvian"]

for j in tqdm(range(len(input)), desc=f"compressing picture"):
    img = Image.open(".\EX4\\" + input[j] + ".bmp")  # import image
    A = np.array(img, dtype="float64")  # convert to numpy array

    if (
        A.shape[1] < A.shape[0]
    ):  # check orientation of the array, so that matrix B maximized
        A = A.swapaxes(0, 1)
        swap = True
    else:
        swap = False

    # svd-compress array
    for i in tqdm(
        range(0, len(n)), desc="computing different compressionrates"
    ):
        img_comp = eigm.svd_compression(A, n[i])  # svd-compress the array

        if (
            swap == True
        ):  # if array was transformed before the svd compression, transform back
            img_comp = img_comp.swapaxes(0, 1)

        img_com = Image.fromarray(
            img_comp.astype(np.uint8), mode="RGB"
        )  # convert array to image
        img_com.save(
            f".\\EX4\\plots\\task_1\\Leonardo_{n[i]}.jpg"
        )  # save image


#####           TASK 1.D            #####
#%%

comp_rate = [
    *np.linspace(0.2, 1, num=50)
]  # compressionrate is here defined as the ratio between the calculated eigenvektors and the maximum number of eigenvektors of the matrix B
input = ["Codice_Atlantico", "Vitruvian"]


for j in tqdm(range(len(input)), desc=f"compressing picture"):
    img = Image.open(".\EX4\\" + input[j] + ".bmp")  # import image
    A = np.array(img, dtype="float64")  # convert to array

    if (
        A.shape[1] < A.shape[0]
    ):  # check orientation of the array, so that matrix B maximized
        A = A.swapaxes(0, 1)
        swap = True
    else:
        swap = False

    nmax = A.shape[1]

    for i in tqdm(
        range(0, len(comp_rate)),
        desc="computing different compressionrates",
    ):
        n = int(
            comp_rate[i] * nmax
        )  # calculate the number of eigenvektors for the given compression rate

        img_comp = eigm.svd_compression(
            A, n
        )  # compute svd-compression of array

        if (
            swap == True
        ):  # if array was transformed before the svd compression, transform back
            img_comp = img_comp.swapaxes(0, 1)

        img_com = Image.fromarray(
            img_comp.astype(np.uint8), mode="RGB"
        )  # convert array to image
        img_com.save(
            f".\\EX4\\plots\\task_1\\c\\{input[j]}_{np.round(comp_rate[i], 3)}_n-{n}-SVD.jpg"
        )  # save image

""" Depending on the starting image and its color, a compression rate (definition given above) of 70% is needed for a fairly sharp compressed image."""
