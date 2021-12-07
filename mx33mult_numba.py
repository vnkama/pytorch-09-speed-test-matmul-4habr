import numpy as np
from numba import cuda          # Библиотека Nvidia для работы с GPU
import numba

def get_functions_dict():
    return {

        'init_method_data__mx33mult_numba': init_method_data__mx33mult_numba,

        'upload__mx33mult_numba_f1': upload__mx33mult_numba_f1,
        'upload__mx33mult_numba_f2': upload__mx33mult_numba_f2,
        'calc_result__mx33mult_numba_f1': calc_result__mx33mult_numba_f1,
        'calc_result__mx33mult_numba_f2': calc_result__mx33mult_numba_f2,
        'download__mx33mult_numba_f1': download__mx33mult_numba_f1,
        'download__mx33mult_numba_f2': download__mx33mult_numba_f2,
        'convert_to_np__mx33mult_numba_f2': convert_to_np__mx33mult_numba_f2,
    }


def init_method_data__mx33mult_numba(test_data, method):
    # объекты cuda создадим сразу в функции upload
    return test_data


def upload__mx33mult_numba_f1(test_data):
    # test_data['loop_count'] = test_data['loop_count']
    # test_data['matrices_count'] = test_data['matrices_count']

    test_data['A'] = cuda.to_device(test_data['A'])
    test_data['B'] = cuda.to_device(test_data['B'])
    test_data['C'] = cuda.to_device(test_data['C'])


    cuda.synchronize()
    return test_data


def upload__mx33mult_numba_f2(test_data):

    # test_data['loop_count'] = test_data['loop_count']
    # test_data['matrices_count'] = test_data['matrices_count']


    # число элементов массива
    size = test_data['A'].size
    A = np.resize(test_data['A'], size)
    B = np.resize(test_data['B'], size)
    C = np.resize(test_data['C'], size)

    test_data['A'] = cuda.to_device(A)
    test_data['B'] = cuda.to_device(B)
    test_data['C'] = cuda.to_device(C)


    cuda.synchronize()
    return test_data



@cuda.jit
def mx33mult_f1_cuda(A, B, C):

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    bx = cuda.blockIdx.x    # номер блока = номер перемножаемой матрицы

    A_mx33 = A[bx]
    B_mx33 = B[bx]
    C[bx, tx, ty] = A_mx33[tx, 0] * B_mx33[0, ty] + A_mx33[tx, 1] * B_mx33[1, ty] + A_mx33[tx, 2] * B_mx33[2, ty]


def calc_result__mx33mult_numba_f1(test_data):
    A = test_data['A']
    B = test_data['B']
    C = test_data['C']

    loop_count = test_data['loop_count']
    matrices_count = test_data['matrices_count']

    blocks_per_grid = matrices_count       # blocks_per_grid
    threads_per_block = (3, 3)             # блок имет размер 3x3 те он соовтетсвует марице

    for _ in range(loop_count):
        mx33mult_f1_cuda[blocks_per_grid, threads_per_block](A, B, C)
        cuda.synchronize()

    return test_data


@cuda.jit
def mx33mult_f2_cuda(A, B, C, loop_count):

    # 1 матрица 3x3 занимает на GPU 1блок = 9 потоков

    # thread index in block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    #tpb - thread per block
    tpb_x = cuda.blockDim.x         # 3
    tpb_y = cuda.blockDim.y
    tpb_count = tpb_x * tpb_y       # 9 threads в каждом блоке (константа)

    bx = cuda.blockIdx.x            # номер текущего блока = номер перемножаемой матрицы 3x3

    #C_shared = numba.cuda.shared.array(shape=tpb, dtype=numba.float32)

    # тоотальный массив - линейный массив(1D) в котором развернуты все матрицы 3x3
    # позиция данномго потока в линейном массиве, когда все представлены
    pos = (bx * tpb_count) + (ty * tpb_x + tx)

    # позиция начала матрицы3x3 к которой относится данный поток
    i00 = bx * tpb_count

    A_0_ty = A[i00 + ty*3 + 0]
    A_1_ty = A[i00 + ty*3 + 1]
    A_2_ty = A[i00 + ty*3 + 2]

    B_tx_0 = B[i00 + 0 + tx]
    B_tx_1 = B[i00 + 3 + tx]
    B_tx_2 = B[i00 + 6 + tx]

    for _ in range(loop_count):
        C[pos] = A_0_ty * B_tx_0 + A_1_ty * B_tx_1 + A_2_ty * B_tx_2
        cuda.syncthreads()


def calc_result__mx33mult_numba_f2(test_data):
    A = test_data['A']
    B = test_data['B']
    C = test_data['C']

    loop_count = test_data['loop_count']
    matrices_count = test_data['matrices_count']

    bpg = matrices_count       # blocks_per_grid
    tpb = (3, 3)

    mx33mult_f2_cuda[bpg, tpb](A, B, C, loop_count)

    cuda.synchronize()
    return test_data



def download__mx33mult_numba_f1(test_data):
    # input_data['A'] = input_data['A'].copy_to_host()
    # input_data['B'] = input_data['B'].copy_to_host()
    test_data['C'] = test_data['C'].copy_to_host()
    cuda.synchronize()

    return test_data

def download__mx33mult_numba_f2(test_data):
    # input_data['A'] = input_data['A'].copy_to_host()
    # input_data['B'] = input_data['B'].copy_to_host()
    test_data['C'] = test_data['C'].copy_to_host()
    cuda.synchronize()

    return test_data


def convert_to_np__mx33mult_numba_f2(test_data):
    size = test_data['A'].size
    test_data['C'] = np.resize(test_data['C'], (size//9, 3, 3))

    return test_data


