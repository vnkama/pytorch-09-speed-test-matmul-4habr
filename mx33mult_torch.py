import numpy as np
import torch as t

import time

cuda_device = None

def get_functions_dict():
    return {

        'init_method_data__mx33mult_torch': init_method_data__mx33mult_torch,

        #'upload__mx33mult_torch_gpu': upload__mx33mult_torch_gpu,
        'calc_result__mx33mult_torch_cpu': calc_result__engine2d_torch_cpu_f1,
        'calc_result__mx33mult_torch_gpu': calc_result__engine2d_torch_gpu_f1,
        'download__mx33mult_torch_gpu': download__mx33mult_torch_gpu,
        'convert_to_np__mx33mult_torch': convert_to_np__mx33mult_torch,
    }


def init_method_data__mx33mult_torch(test_data, method):
    global cuda_device

    if method == 'cpu':
        cuda_device = t.device('cpu')
    elif method == 'gpu':
        cuda_device = t.device('cuda:0')


    test_data['A'] = t.tensor(test_data['A'], dtype=t.float32, device=cuda_device, requires_grad=False)
    test_data['B'] = t.tensor(test_data['B'], dtype=t.float32, device=cuda_device, requires_grad=False)
    test_data['C'] = t.tensor(test_data['C'], dtype=t.float32, device=cuda_device, requires_grad=False)

    return test_data




# def upload__mx33mult_torch_gpu(test_data):
#
#     cuda_device = t.device('cuda:0')
#
#     test_data = {
#         'A': test_data['A'].to(cuda_device),
#         'B': test_data['B'].to(cuda_device),
#         'C': test_data['C'].to(cuda_device),
#         'test_size': test_data['test_size'],
#     }
#
#     t.cuda.synchronize()
#
#     return test_data

def calc_result__engine2d_torch_cpu_f1(test_data):
    return calc_result__mx33mult_torch_f1(test_data, 'cpu')

def calc_result__engine2d_torch_gpu_f1(test_data):
    return calc_result__mx33mult_torch_f1(test_data, 'gpu')

def calc_result__mx33mult_torch_f1(test_data, method):

    loop_count = test_data['loop_count']
    #matrices_count = test_data['matrices_count']

    A = test_data['A']
    B = test_data['B']
    C = test_data['C']

    for _ in range(loop_count):
        C = t.bmm(A, B)

    t.cuda.synchronize()
    test_data['C'] = C

    return test_data



def download__mx33mult_torch_gpu(input_data):
    input_data['C'] = input_data['C'].cpu()
    return input_data


def convert_to_np__mx33mult_torch(test_data):
    test_data['A'] = test_data['A'].numpy()
    test_data['B'] = test_data['B'].numpy()
    test_data['C'] = test_data['C'].numpy()

    return test_data


# def verify_result__mx33mult_torch(input_data, correct_data, method):
#     C_correct = correct_data['C']
#     C_to_test = input_data['C'].numpy()
#
#     return np.allclose(C_to_test, C_correct, rtol=1e-05, atol=0.01)


# def calc_result__mx33multInc_torch_cpu(loop_count, input_data):
#     A = input_data['A']
#     B = input_data['B']
#     #C = input_data['C']
#
#     test_size = input_data['test_size']
#
#     for loop_i in range(loop_count):
#         C = t.bmm(A, B)
#
#         x = int(loop_i / 3) % 3
#         y = loop_i % 3
#
#         A[..., x, y] += 1
#
#     input_data['C'] = C
#
#     return input_data


