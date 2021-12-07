import numpy as np
import copy
from Vector import *


def get_functions_dict():
    return {
        'init_method_data__mx33mult_np': init_method_data__mx33mult_np,

        'calc_result__mx33mult_np_dot': calc_result__mx33mult_np_dot,
        'calc_result__mx33mult_np_matmul': calc_result__mx33mult_np_matmul,

    }




#############################################################

#
#
#
def init_method_data__mx33mult_np(input_data, method):
    return input_data



#
#
#
def calc_result__mx33mult_np_matmul(test_data):

    loop_count = test_data['loop_count']
    #matrices_count = test_data['matrices_count']

    A = test_data['A']
    B = test_data['B']
    C = np.zeros_like(test_data['C'])

    for loop_i in range(loop_count):
        C = np.matmul(A, B)

    test_data['C'] = C

    return test_data


#
#
#
def calc_result__mx33mult_np_dot(test_data):

    loop_count = test_data['loop_count']
    matrices_count = test_data['matrices_count']

    A = test_data['A']
    B = test_data['B']
    C = np.zeros_like(test_data['C'])

    for loop_i in range(loop_count):
        for i in range(matrices_count):
            C[i] = np.dot(A[i], B[i])

    test_data['C'] = C

    return test_data



# def verify_result__mx33mult_np(data_to_verify, correct_data, method=None):
#     return np.allclose(data_to_verify['C'], correct_data['C'], rtol=1e-05, atol=0.01)



###############################################################################


#
#
#
def calc_result__mx33multInc_etalon(loop_count, input_data):

    A = copy.deepcopy(input_data['A'])      # A меняется в данной функции в процесее расчета , а нам его еще использовать
    B = input_data['B']

    C = np.zeros_like(input_data['A'])      # input_data['C'] менять не будем, пусть там останется 0
    test_size = input_data['test_size']

    for loop_i in range(loop_count):
        for i in range(test_size):
            C[i] = np.dot(A[i], B[i])

        A[..., int(loop_i / 3) % 3, loop_i % 3] += 1

    return input_data, {'C': C}

#############################################################


#
#
#
def calc_result__mx33multInc_np_matmul(loop_count, input_data):

    A = input_data['A']
    B = input_data['B']
    C = None

    for loop_i in range(loop_count):
        C = np.matmul(A, B)

        A[..., int(loop_i / 3) % 3, loop_i % 3] += 1

    input_data['C'] = C

    return input_data



