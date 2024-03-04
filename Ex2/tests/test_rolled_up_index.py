import coordinate_tools.rolled_up_index as rui
import numpy as np
from tqdm import tqdm


def test_pythonic_rolled_up_index():
    ni_tup = (1, 2, 3)
    ni_list = [1, 2, 3]

    Ni_dic = {"N1": 10, "N2": 10, "N3": 10}

    k = 321
    
    k1 = rui.get_roled_up_index(1, 2, 3, dbg=True, N1=10, N2=10, N3=10)
    k2 = rui.get_roled_up_index(1, 2, 3, dbg=True, **Ni_dic)
    k3 = rui.get_roled_up_index(*ni_tup, dbg=True, N1=10, N2=10, N3=10)
    k4 = rui.get_roled_up_index(*ni_tup, dbg=True, **Ni_dic)
    k5 = rui.get_roled_up_index(*ni_list, dbg=True, N1=10, N2=10, N3=10)
    k6 = rui.get_roled_up_index(*ni_list, dbg=True, **Ni_dic)

    assert k == k1
    assert k == k2
    assert k == k3
    assert k == k4
    assert k == k5
    assert k == k6


def test_numpythic_rolled_up_index():
    ni_arr = np.array([1, 2, 3])

    Ni_dic = {"N1": 10, "N2": 10, "N3": 10}

    k = 321

    k1 = rui.get_roled_up_index(*ni_arr, dbg=True, **Ni_dic)
    assert k == k1


def test_rolled_up_index():
    debugging = False
    Ni_dic = {"N1": 7, "N2": 3, "N3": 13, "N4": 4}
    Ni_arr = np.array(list(Ni_dic.values()))
    Ni_prod = np.prod(Ni_arr)
    ni_list = np.arange(0, Ni_prod)

    nj_list = [np.arange(0, Nj) for Nj in Ni_arr]
    nj_array = rui.get_index_combinations(*nj_list)

    for i in range(Ni_prod):
        perm = nj_array[i, :]
        k = rui.get_roled_up_index(*perm, dbg=debugging, **Ni_dic)
        assert k == ni_list[i]

def test_get_unrolled_indizes():
    Ni_dic = {"N1": 10, "N2": 10, "N3": 10}
    k_l = [17,573,126]
    ni_l = [(7,1,0),(3,7,5),(6,2,1)]
    
    for l,kl in enumerate(k_l):
        nil = ni_l[l]
        assert nil == rui.get_unrolled_indizes(kl,dbg=True,**Ni_dic)

def test_nb():
    Ni_dic = {"N1": 10, "N2": 10, "N3": 10}
    dimensions = 3
    
    k_list = [
        9,      #point in korner
        490,    #point on edge
        830,    #point on surface
        456     #point in room
    ]    
    nb_list = [ #neighbor lists for each point
        [109,19,8],
        [390,590,480,491],
        [730,930,820,840,831],
        [356,556,446,466,455,457]
    ]
    
    for j,kj in enumerate(k_list):
        nb_j = rui.nb(kj,d=dimensions,**Ni_dic)
        if not (type(nb_j) is list):
            assert False, "errorcode:"+str(nb_j)
            
        assert set(nb_list[j]) == set(nb_j)
    