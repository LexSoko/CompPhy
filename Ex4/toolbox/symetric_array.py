import numpy as np
from tqdm import tqdm

# create symetric matrix without 0 in its diagonal
def arr_2Dsym_dig0(arr_size, low=0, high=100):
    A_dia = np.array([0], dtype="float")

    while np.any(A_dia == 0):
        A = np.random.uniform(
            low=float(low),
            high=float(high),
            size=(arr_size, arr_size),
        )
        A_dia = []
        for i in tqdm(range(0, arr_size), desc='arr_2D_diagonal-loop'):
            A_dia.append(A[i][i])
        A_dia = np.array(A_dia)

    return A
