import numsolvers.gauss_seidel as gsm
import numpy as np
import timeit


def test_gauss_seidel_method():
    A = np.array([[16, 3], [7, -11]], dtype="float")
    b = np.array([11, 13], dtype="float")

    x, a = gsm.gauss_seidel_method(A, b)

    print(
        f"""Solution of the method: {x}
          Checked solution: [0.8122, -0.6650]"""
    )


def test_summs2():
    dim = 10
    A = np.random.rand(dim, dim)
    b = np.random.rand(dim)
    for i in range(0, dim):
        sum1 = 0
        sum2 = 0
        gsm.summs(A, dim, i, b, sum1, sum2)

        sum1_n = 0
        sum2_n = 0
        gsm.summs2(A, dim, i, b, sum1_n, sum2_n)
        assert sum1 == sum1_n
        assert sum2 == sum2_n
