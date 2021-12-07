

# библиотеки/технологии данных которые тестируем
#   py1     python          обычный питон
#   python-fast     питон, оптимизируем код
#   np      numpy
#   numba-cuda
#   pytorch-cpu-1   матрицы перемножаются по очереди
#   pytorch-cpu-2   матрицы перемножаются паралельно
#   pytorch-gpu
#

########################################################

import copy
import time
import matplotlib.pyplot as plt

#ALGORITHM = 'mx33mult'
import engine2d


ALGORITHM = 'engine2d'      #   mx33mult    engine2d

functions = {}

if ALGORITHM == 'mx33mult':
    import mx33mult as algorithm_module
    import mx33mult_python
    import mx33mult_numpy
    import mx33mult_torch
    import mx33mult_numba

    functions = {
        **algorithm_module.get_functions_dict(),

        **mx33mult_python.get_functions_dict(),
        **mx33mult_numpy.get_functions_dict(),
        **mx33mult_torch.get_functions_dict(),
        **mx33mult_numba.get_functions_dict(),
    }


elif ALGORITHM == 'engine2d':
    import engine2d as algorithm_module
    import engine2d_numpy
    import engine2d_torch
    import engine2d_numba

    functions = {
        **algorithm_module.get_functions_dict(),

        **engine2d_numpy.get_functions_dict(),
        **engine2d_torch.get_functions_dict(),
        **engine2d_numba.get_functions_dict(),
    }




#
# провести все тесты для алгоритма (например алгоритм : engine2d)
# тесты разными библиотеками, функциями и разные размеры
#
# algorithm_test_cfg  - наборы данных (датасеты), на которые проводится группа тестов.
#                   на каждый датасет
def runAlgorithmTest(algorithm_test_cfg):


    shapes_tests_cfgs = algorithm_test_cfg['shapes_cfgs']
    libraries = algorithm_test_cfg['libraries']

    is_algo_error_found = False

    # первый запуск, он нужен тк многие библиотеки запускаются на первом тесте очень долго,
    # чтол приводит к неверному таймингу
    # результаты замера не выводятся

    graphs = {}



    for _, shape_test_cfg in enumerate(shapes_tests_cfgs):

        is_print = shape_test_cfg['main_props']['is_print']
        data_gene_mode = shape_test_cfg['main_props']['data_gene_mode']

        print(shape_test_cfg)

        # тест на shape, включая для данного shape все тесты разных лаqбрари и их методов


        shape_data_original = functions[f'init_shape_data__{ALGORITHM}'](shape_test_cfg)
        correct_test_sindex = None
        correct_test = None

        for _, library in enumerate(libraries):
            library_name = library['library_name']

            for __, method in enumerate(library['methods']):
                if is_print:
                    print('#' * 80)
                    print(library_name, method)

                # вспомогательный постфиксы
                LM = f'{library_name}_{method}'
                ALM = f'{ALGORITHM}_{library_name}_{method}'

                if not (LM in graphs):
                    graphs[LM] = {}
                    graphs[LM]['X'] = []
                    graphs[LM]['Y'] = []


                start_time = time.perf_counter()


                test_data = copy.deepcopy(shape_data_original)
                test_data = functions[f'init_method_data__{ALGORITHM}_{library_name}'](test_data, method)
                init_end_time = time.perf_counter()


                # функция 'upload__' есть не у всех методов
                upload_end_time = init_end_time
                func_name = f'upload__{ALM}'
                if func_name in functions:
                    test_data = functions[func_name](test_data)
                    upload_end_time = time.perf_counter()

                test_data = functions[f'calc_result__{ALM}'](test_data)
                calc_end_time = time.perf_counter()

                # выгрузка с GPU в CPU
                download_end_time = calc_end_time
                func_name = f'download__{ALM}'
                if func_name in functions:
                    test_data = functions[func_name](test_data)
                    download_end_time = time.perf_counter()

                func_name = f'convert_to_np__{ALM}'
                if func_name in functions:
                    test_data = functions[func_name](test_data)


                if data_gene_mode == 'const_full':
                    # тест проводился по заранее подгтовленым захардкоженым константным данным,
                    # а не по случайным/сгенерированным а
                    # соответсвенно проверять будем по константным ответам
                    is_method_verified = functions[f'verify_result__{ALGORITHM}'](test_data, None, data_gene_mode)

                elif data_gene_mode == 'debug_full' or data_gene_mode == 'random_full' or data_gene_mode == 'random_part':
                    if correct_test_sindex is None:
                        # первый расчет в рамках одного шейпа для режимов debug_full и random_full считается всегда правильным.
                        # первый - имеется в виду считается независимо от библиотеки метода итп
                        # вызвали первый libary+method : numpy_f5 - они будет первый
                        # остальные будем проверять по нему

                        # индекс в формате
                        correct_test_sindex = LM        # префикс для отсыл к правильному
                        correct_test = test_data
                        is_method_verified = True

                    else:
                        # проведем сравним данные
                        is_method_verified = functions[f'verify_result__{ALGORITHM}'](test_data, correct_test, data_gene_mode)

                else:
                    raise Exception

                is_algo_error_found = is_algo_error_found or not is_method_verified

                if is_print:
                    functions[f'print_result__{ALGORITHM}'](
                            f'{library_name}_{method}',
                            ALGORITHM,
                            test_data,
                            is_method_verified,
                            start_time, init_end_time, upload_end_time, calc_end_time, download_end_time,
                            graphs[LM]
                    )


    plt_fig = plt.figure(figsize=[12, 5])

    plt_ax_plain = plt_fig.add_subplot(111)
    #plt_ax_log = plt_fig.add_subplot(122)


    plt_ax_plain.set_title(engine2d.GRAPH_TITLE)
    #plt_ax_log.set_title(engine2d.GRAPH_TITLE)

    plt_ax_plain.set_xlabel(engine2d.GRAPH_X_TITLE)
    #plt_ax_log.set_xlabel(engine2d.GRAPH_X_TITLE)

    plt_ax_plain.grid()

    plt_ax_plain.set_xscale("log")
    plt_ax_plain.set_yscale("log")

    #plt_ax_log.grid()
    #plt_ax_log.set_xscale("log")

    for _, key in enumerate(graphs):
        color = None
        label = key
        if key == 'np_f4':
            color = 'b-D'
            label = 'numpy'

        elif key == 'np_f1':
            color = 'c-D'
            label = 'numpy_f1'

        elif key == 'torch_cpu_f2':
            color = 'r-D'
            label = 'Torch CPU'

        elif key == 'torch_gpu_f2':
            color = 'm-D'
            label = 'Torch GPU'

        elif key == 'numba_f1':
            color = 'g-D'
            label = 'Numba'

        graph = graphs[key]
        plt_ax_plain.plot(graph['X'], graph['Y'], color, label=label)
        #plt_ax_log.plot(graph['X'], graph['Y'], color, label=label)

    plt_ax_plain.legend()
    #plt_ax_log.legend()


    plt.show()


    if is_algo_error_found:
        print('\033[31mERROR ERROR ERROR ERROR ERROR\033[0m')
    else:
        print('\033[32mAll OK\033[0m')




########################################################

if __name__ == '__main__':

    # взять настройки теста из текущего модуля
    algorithm_test_cfg = algorithm_module.getAlgorithmTestCfg()

    # провести тесты
    runAlgorithmTest(algorithm_test_cfg)


