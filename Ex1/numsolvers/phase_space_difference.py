############################## imports #######################################
import numpy as np
from typing import Sequence, Callable

######################### module level dunders ###############################
__author__ = "Max Jost, Aleksey Sokolov, Martin Ferdinand Steiner"


############################# functions ######################################
def delta(
    first_trajectory: Sequence[float], second_trajectory: Sequence[float]
):
    """
    This Function calculates the distance between two trajectory matrices for every timestep y(t)



    Arguments:
        first_trajectory -- first trajectory matrix
        second_trajectory -- second trajectory matrix

    Returns:
        The Distance between two vectors for every timestep -> list[float]
    """
    # calculating every delta for each timestep
    diff = first_trajectory - second_trajectory
    d = []
    # transposing the difference matrix to get every to get delta y(t) for every timestep
    for i in diff.T:
        d.append(np.linalg.norm(i))

    return d
