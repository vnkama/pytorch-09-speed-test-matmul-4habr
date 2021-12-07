import numpy as np


def get_functions_dict():
    return {
        'init_shape_data__mx33mult': init_shape_data__mx33mult,
        'verify_result__mx33mult': verify_result__mx33mult,
        'print_result__mx33mult': print_result__mx33mult,
    }

#
# подготовка данных
#
def init_shape_data__mx33mult(shape_test_cfg):
    rng = None

    data_gene_mode = shape_test_cfg['main_props']['data_gene_mode']
    rng_seed = shape_test_cfg['main_props']['rng_seed']

    shape_props = shape_test_cfg['shape_props']

    rng = np.random.default_rng(rng_seed)

    loop_count = shape_props[0]   # число агентов, т.е число движущихся фигур
    matrices_count = shape_props[1]     # длинна теста, т.е число шагов для каждого агента



    test_data = {
        'data_gene_mode': data_gene_mode,
        'loop_count': loop_count,
        'matrices_count': matrices_count,
    }

    if data_gene_mode == 'const_full':
        test_data = _init_shape_data__const_full(test_data)

    elif data_gene_mode == 'random_full':
        test_data = _init_shape_data__random_full(test_data, rng)


    return test_data


def _init_shape_data__const_full(test_data):
    A = np.array(
            [
                [
                    [1, 2, -3],
                    [4, -5, 6],
                    [-7, 8, 9],
                ],
                [
                    [-11, 21, 31],
                    [41, -51, 61],
                    [71, 81, -91],
                ]
            ],
            dtype=np.float32
    )

    B = np.array(
            [
                [
                    [10, -20, 0.30],
                    [-40, 0.50, -60],
                    [0.70, -80, 90],
                ],
                [
                    [-0.10, -20, -30],
                    [0.40, 50, 60],
                    [-0.70, -80, -90],
                ]
            ],
            dtype=np.float32
    )

    # результат перемножния первой пары матриц
    #   [[-72.1, 221, -389.7]. [244.2, -562.5, 841.2], [-383.7,	-576, 327.9]]

    # результат перемножния второй пары матриц
    #  [[-12.2	-1210	-1200],
    # [-67.2	-8250	-9780],
    # [89	9910	10920],]

    test_data['A'] = A
    test_data['B'] = B
    test_data['C'] = np.zeros_like(A)
    test_data['matrices_count'] = A.shape[0]
    test_data['loop_count'] = 1

    return test_data


def _init_shape_data__random_full(test_data, rng):
    size_of_array = (test_data['matrices_count'], 3, 3)

    test_data['A'] = np.random.uniform(low=-100, high=100, size=size_of_array)
    test_data['B'] = np.random.uniform(low=-100, high=100, size=size_of_array)
    test_data['C'] = np.random.uniform(low=-100, high=100, size=size_of_array)

    return test_data



#
#
#
def verify_result__mx33mult(test_data, correct_data, data_gene_mode):

    # при data_gene_mode=True, для проверки берутся correct_data как эталонные
    # при data_gene_mode=False, дл япроверки используются константные данные, correct_data - игнорируется

    if data_gene_mode == 'random_full':
        correct_data_4_test = correct_data['C']

    elif data_gene_mode == 'const_full':

        correct_data_4_test = [
            [
                [-72.1, 221, -389.7], [244.2, -562.5, 841.2], [-383.7, -576, 327.9]
            ],
            [
                [-12.2, -1210, -1200], [-67.2, -8250, -9780], [89, 9910, 10920]    # 10920
            ]
        ]
    else:
        raise Exception

    return np.allclose(
        test_data['C'],
        correct_data_4_test,
        rtol=1e-05,
        atol=0.01
    )



#
#
#
def getAlgorithmTestCfg():

    libraries = [
        {
            # это эталонный замер
            'library_name': 'np',
            'methods': ['dot', 'matmul'],
        },
        # {
        #     'library_name': 'py',
        #     'methods': ['f1', 'f2', 'f3'],
        # },
        {
            'library_name': 'torch',
            'methods': ['cpu', 'gpu'],
        },
        {
            'library_name': 'numba',
            'methods': ['f1', 'f2'],
        },
    ]

    # pilot_n : pilot_necessary_params
    # Набор данных для пилота - пробоного теста
    # данные
    const_full = {
        'is_print': True,
        'data_gene_mode': 'const_full',           # const
        'rng_seed': 1002,
    }

    # debug_full = {
    #     'is_print': True,
    #     'data_gene_mode': 'debug_full',         # частично генерируетьс по правилам, остальное random_full
    #     'rng_seed': 1002,
    # }
    #
    random_full = {
        'is_print': True,
        'data_gene_mode': 'random_full',     # генерируются все перемещения, сохраняются и проверяются все апозиции
        'rng_seed': 1003,
    }
    #
    # random_part = {
    #     'is_print': True,
    #     'data_gene_mode': 'random_part',
    #     'rng_seed': 1002,
    # }


    # Формат массива
    # matrces_count  loop_count
    shapes_cfgs = [
        {'main_props': const_full,  'shape_props': [0, 0]},
        {'main_props': random_full, 'shape_props': [1, 1]},
        {'main_props': random_full, 'shape_props': [10, 10]},
        {'main_props': random_full, 'shape_props': [10000, 1000]},


        #######################################
        # {'main_props': random_full, 'shape_props': [10, 10]},
        # {'main_props': random_full, 'shape_props': [10, 100]},
        # {'main_props': random_full, 'shape_props': [10, 1000]},
        # {'main_props': random_full, 'shape_props': [10, 10000]},
        # {'main_props': random_full, 'shape_props': [10, 100000]},
        #
        # {'main_props': random_full, 'shape_props': [100, 10]},
        # {'main_props': random_full, 'shape_props': [100, 100]},
        # {'main_props': random_full, 'shape_props': [100, 1000]},
        # {'main_props': random_full, 'shape_props': [100, 10000]},
        #
        # {'main_props': random_full, 'shape_props': [1000, 10]},
        # {'main_props': random_full, 'shape_props': [1000, 100]},
        # {'main_props': random_full, 'shape_props': [1000, 1000]},
        #######################################

        # {'main_props': random_part, 'shape_props': [10, 10]},
        # {'main_props': random_part, 'shape_props': [10, 100]},
        # {'main_props': random_part, 'shape_props': [10, 1000]},
        # {'main_props': random_part, 'shape_props': [10, 10000]},
        # {'main_props': random_part, 'shape_props': [10, 100000]},

        # {'main_props': random_part, 'shape_props': [100, 10]},
        # {'main_props': random_part, 'shape_props': [100, 100]},
        # {'main_props': random_part, 'shape_props': [100, 1000]},
        # {'main_props': random_part, 'shape_props': [100, 10000]},
        # {'main_props': random_part, 'shape_props': [100, 100000]},

        # {'main_props': random_part, 'shape_props': [1000, 10]},
        # {'main_props': random_part, 'shape_props': [1000, 100]},
        # {'main_props': random_part, 'shape_props': [1000, 1000]},
        # {'main_props': random_part, 'shape_props': [1000, 10000]},
        # {'main_props': random_part, 'shape_props': [1000, 100000]},

        # {'main_props': random_part, 'shape_props': [10000, 10]},
        # {'main_props': random_part, 'shape_props': [10000, 100]},
        # {'main_props': random_part, 'shape_props': [10000, 1000]},
        # {'main_props': random_part, 'shape_props': [10000, 10000]},
        # {'main_props': random_part, 'shape_props': [10000, 100000]},

        # {'main_props': random_part, 'shape_props': [100000, 10]},
        # {'main_props': random_part, 'shape_props': [100000, 100]},
        #{'main_props': random_part, 'shape_props': [100000, 1000]},
        #{'main_props': random_part, 'shape_props': [100000, 10000]},

        #{'main_props': random_part, 'shape_props': [1000000, 10]},


    ]

    return {
        'libraries': libraries,
        'shapes_cfgs': shapes_cfgs,
    }


def print_result__mx33mult(postfix, algo, test_data, is_test_ok,
                 start_time, init_data_end_time, upload_data_end_time, calc_data_end_time, download_data_end_time):

    print('#' * 80)
    print(f'{algo}')
    print(f'{postfix}')

    test_message = ", test: \033[32mOK\033[0m" if is_test_ok == True else (", \033[31mTEST ERROR !!!!!\033[0m" if is_test_ok == False else "")
    print(f'loop_count: {test_data["loop_count"]}, matrices_count: {test_data["matrices_count"]}{test_message}')

    #init_dur = init_data_end_time - start_time
    upload_dur = upload_data_end_time - init_data_end_time
    calc_dur = calc_data_end_time - upload_data_end_time
    download_dur = download_data_end_time - calc_data_end_time

    # calc_max_dur = download_data_end_time - init_data_end_time
    total_dur = download_data_end_time - start_time

    print(f'Duration (mks):')
    print('tot: {:6.1f}'.format(total_dur*1e6))
    print('upl: {:6.1f}, clc: {:6.1f}, dwl: {:6.1f}'.format(upload_dur*1e6, calc_dur*1e6, download_dur*1e6))


    #print('t/1: {:6.3f}, c/1: {:6.3f}'.format(total_dur / loop_count / test_size * 1e6, calc_dur / loop_count / test_size * 1e6))
    print('\r\n')

