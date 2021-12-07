#import numpy as np


def get_functions_dict():
    return {
        'init_method_data__mx33mult_py': init_method_data__mx33mult_py,

        'calc_result__mx33mult_py_f1': calc_result__mx33mult_py_f1,
        'calc_result__mx33mult_py_f2': calc_result__mx33mult_py_f2,
        'calc_result__mx33mult_py_f3': calc_result__mx33mult_py_f3,
    }

#
#
#
def init_method_data__mx33mult_py(test_data, method):

    test_data['A'] = test_data['A'].tolist()
    test_data['B'] = test_data['B'].tolist()
    test_data['C'] = test_data['C'].tolist()

    # test_data['matrices_count'] = test_data['matrices_count']
    # test_data['loop_count'] = test_data['loop_count']

    return test_data


#
#
#
def calc_result__mx33mult_py_f1(test_data):

    loop_count = test_data['loop_count']
    matrices_count = test_data['matrices_count']


    A = test_data['A']
    B = test_data['B']
    C = test_data['C']


    for _ in range(loop_count):
        for matrix_i in range(matrices_count):

            ####################################################

            # Вариант1. перемножение матриц.
            # к каждому элементу каждый раз адресуемся адресуемся через 3 индекса

            for row in range(3):
                for col in range(3):
                    elem = 0
                    for n in range(3):
                        elem += A[matrix_i][row][n] * B[matrix_i][n][col]

                    C[matrix_i][row][col] = elem


    return test_data


#
# Вариант2. перемножение матриц.
# К каждому элементу  адресуемся через 2 индекса
#
def calc_result__mx33mult_py_f2(test_data):

    loop_count = test_data['loop_count']
    matrices_count = test_data['matrices_count']

    A = test_data['A']
    B = test_data['B']
    C = test_data['C']

    for _ in range(loop_count):
        for matrix_i in range(matrices_count):
            A_cur_mx33 = A[matrix_i]
            B_cur_mx33 = B[matrix_i]
            C_cur_mx33 = C[matrix_i]

            for row in range(3):
                for col in range(3):
                    elem = 0
                    for n in range(3):
                        elem += A_cur_mx33[row][n] * B_cur_mx33[n][col]

                    C_cur_mx33[row][col] = elem

    return test_data


#
# Вариант3. перемножение матриц.
#
def calc_result__mx33mult_py_f3(test_data):

    loop_count = test_data['loop_count']
    matrices_count = test_data['matrices_count']


    A = test_data['A']
    B = test_data['B']
    C = test_data['C']


    for _ in range(loop_count):
        for matrix_i in range(matrices_count):
            A_cur_mx33 = A[matrix_i]
            B_cur_mx33 = B[matrix_i]
            C_cur_mx33 = C[matrix_i]

            C_cur_mx33[0][0] = A_cur_mx33[0][0] * B_cur_mx33[0][0] + A_cur_mx33[0][1] * B_cur_mx33[1][0] + A_cur_mx33[0][2] * B_cur_mx33[2][0]
            C_cur_mx33[0][1] = A_cur_mx33[0][0] * B_cur_mx33[0][1] + A_cur_mx33[0][1] * B_cur_mx33[1][1] + A_cur_mx33[0][2] * B_cur_mx33[2][1]
            C_cur_mx33[0][2] = A_cur_mx33[0][0] * B_cur_mx33[0][2] + A_cur_mx33[0][1] * B_cur_mx33[1][2] + A_cur_mx33[0][2] * B_cur_mx33[2][2]

            C_cur_mx33[1][0] = A_cur_mx33[1][0] * B_cur_mx33[0][0] + A_cur_mx33[1][1] * B_cur_mx33[1][0] + A_cur_mx33[1][2] * B_cur_mx33[2][0]
            C_cur_mx33[1][1] = A_cur_mx33[1][0] * B_cur_mx33[0][1] + A_cur_mx33[1][1] * B_cur_mx33[1][1] + A_cur_mx33[1][2] * B_cur_mx33[2][1]
            C_cur_mx33[1][2] = A_cur_mx33[1][0] * B_cur_mx33[0][2] + A_cur_mx33[1][1] * B_cur_mx33[1][2] + A_cur_mx33[1][2] * B_cur_mx33[2][2]

            C_cur_mx33[2][0] = A_cur_mx33[2][0] * B_cur_mx33[0][0] + A_cur_mx33[2][1] * B_cur_mx33[1][0] + A_cur_mx33[2][2] * B_cur_mx33[2][0]
            C_cur_mx33[2][1] = A_cur_mx33[2][0] * B_cur_mx33[0][1] + A_cur_mx33[2][1] * B_cur_mx33[1][1] + A_cur_mx33[2][2] * B_cur_mx33[2][1]
            C_cur_mx33[2][2] = A_cur_mx33[2][0] * B_cur_mx33[0][2] + A_cur_mx33[2][1] * B_cur_mx33[1][2] + A_cur_mx33[2][2] * B_cur_mx33[2][2]

    return test_data


# #
# # Вариант3. перемножение матриц.
# #
# def calc_result__mx33multInc_py_f3(loop_count, input_data):
#
#     A = input_data['A']
#     B = input_data['B']
#     C = input_data['C']
#     test_size = input_data['test_size']
#
#     for loop_i in range(loop_count):
#         for matrix_i in range(test_size):
#             A_cur_mx33 = A[matrix_i]
#             B_cur_mx33 = B[matrix_i]
#             C_cur_mx33 = C[matrix_i]
#
#             C_cur_mx33[0][0] = A_cur_mx33[0][0] * B_cur_mx33[0][0] + A_cur_mx33[0][1] * B_cur_mx33[1][0] + A_cur_mx33[0][2] * B_cur_mx33[2][0]
#             C_cur_mx33[0][1] = A_cur_mx33[0][0] * B_cur_mx33[0][1] + A_cur_mx33[0][1] * B_cur_mx33[1][1] + A_cur_mx33[0][2] * B_cur_mx33[2][1]
#             C_cur_mx33[0][2] = A_cur_mx33[0][0] * B_cur_mx33[0][2] + A_cur_mx33[0][1] * B_cur_mx33[1][2] + A_cur_mx33[0][2] * B_cur_mx33[2][2]
#
#             C_cur_mx33[1][0] = A_cur_mx33[1][0] * B_cur_mx33[0][0] + A_cur_mx33[1][1] * B_cur_mx33[1][0] + A_cur_mx33[1][2] * B_cur_mx33[2][0]
#             C_cur_mx33[1][1] = A_cur_mx33[1][0] * B_cur_mx33[0][1] + A_cur_mx33[1][1] * B_cur_mx33[1][1] + A_cur_mx33[1][2] * B_cur_mx33[2][1]
#             C_cur_mx33[1][2] = A_cur_mx33[1][0] * B_cur_mx33[0][2] + A_cur_mx33[1][1] * B_cur_mx33[1][2] + A_cur_mx33[1][2] * B_cur_mx33[2][2]
#
#             C_cur_mx33[2][0] = A_cur_mx33[2][0] * B_cur_mx33[0][0] + A_cur_mx33[2][1] * B_cur_mx33[1][0] + A_cur_mx33[2][2] * B_cur_mx33[2][0]
#             C_cur_mx33[2][1] = A_cur_mx33[2][0] * B_cur_mx33[0][1] + A_cur_mx33[2][1] * B_cur_mx33[1][1] + A_cur_mx33[2][2] * B_cur_mx33[2][1]
#             C_cur_mx33[2][2] = A_cur_mx33[2][0] * B_cur_mx33[0][2] + A_cur_mx33[2][1] * B_cur_mx33[1][2] + A_cur_mx33[2][2] * B_cur_mx33[2][2]
#
#
#             A_cur_mx33[int(loop_i / 3) % 3][loop_i % 3] += 1
#
#
#     return input_data




# #
# #
# #
# def verify_result__mx33mult_py(data_to_verify, correct_data, method=None):
#     return np.allclose(np.array(data_to_verify['C'], dtype=np.double), correct_data['C'], rtol=1e-05, atol=0.01)


