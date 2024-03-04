import numsolvers.jacobi as jb
import numpy as np


def test_jacobi_method():
    A = np.array([[16, 3], [7, -11]], dtype="float")
    b = np.array([11, 13], dtype="float")

    x, a = jb.jacobi_method(A, b)

    print(
        f"""Solution of the method: {x}
          Checked solution: [0.8122, -0.6650]"""
    )
