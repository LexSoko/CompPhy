from typing import Callable, Sequence
from numpy.typing import ArrayLike
import numpy as np


def euler_method(
    f: Callable[[ArrayLike, float], ArrayLike],
    y0: ArrayLike,
    t_vec: Sequence[float] = None,
    t0: float = None,
    t_max: float = None,
    e: float = None,
) -> Sequence[float]:
    """
    Euler-method

    calculates the euler method for fixed and variable timesteps.
    Either accepts a prepared time point list t_vec or otherwise
    a starttime t0 endtime t_max and a time intervall

    Arguments:
        f -- Functional F(t,y(t)) for Equation y`(t) = F(t,y(t))
        y0 -- start condition vector

    Keyword Arguments:
        t_vec -- time points (default: {None}) should be in form of array([t0, t1, t2,...,tn])
        t0 -- start time (default: {None})
        t_max -- end time (default: {None})
        e -- time intervall (default: {None})

    Returns:
        tuple (t, y(t) as numpy array)
    """
    # check in what way t is given
    if t_vec != None:
        t = t_vec
    elif t_max != None and e != None and t0 != None:
        t = np.arange(start=t0, stop=t_max + 1 + e, step=e)
    else:
        print("Time not given in an accaptable way")
        return

    y = [y0]
    y_n = y0

    for n in range(0, len(t) - 1):
        e = t[n + 1] - t[n]
        y_n1 = y_n + f(t[n], y_n) * e
        y.append(y_n1)
        y_n = y_n1

    y = np.transpose(y)

    return t, y
