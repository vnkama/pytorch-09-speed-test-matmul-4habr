#import numpy as np
import torch as t
from Vector import *
from vector_torch import *
import time
import timeit

cuda_device = None

def get_functions_dict():
    return {
        'init_method_data__engine2d_torch': init_method_data__engine2d_torch,
        'calc_result__engine2d_torch_cpu_f2': calc_result__engine2d_torch_cpu_f2,
        'calc_result__engine2d_torch_gpu_f2': calc_result__engine2d_torch_gpu_f2,
        #'convert_to_np__engine2d_torch_cpu': convert_to_np__engine2d_torch_cpu,
    }


def init_method_data__engine2d_torch(test_data, method):
    global cuda_device

    if method == 'cpu' or method == 'cpu_f2':
        cuda_device = t.device('cpu')
    elif method == 'gpu' or method == 'gpu_f2':
        cuda_device = t.device('cuda:0')

    if test_data['data_gene_mode'] == 'const_full' or test_data['data_gene_mode'] == 'debug_full' or test_data['data_gene_mode'] == 'random_full':
        data_gene_mode_int = 1
    elif test_data['data_gene_mode'] == 'random_part':
        data_gene_mode_int = 4
        routes_len_4_random_part = test_data['routes_len_4_random_part']

    test_data = {
        'data_gene_mode_int': data_gene_mode_int,
        'data_gene_mode_int_t': t.tensor(data_gene_mode_int, dtype=t.int32, device=cuda_device, requires_grad=False),

        'agents': test_data['agents'],
        'agents_count': test_data['agents_count'],
        'agents_count_t': t.tensor(test_data['agents_count'], dtype=t.int32, device=cuda_device, requires_grad=False),
        'figures': test_data['figures'],

        'figures_count': t.tensor(test_data['figures_count'], dtype=t.int32, device=cuda_device, requires_grad=False),
        'figures_max_points_count': t.tensor(test_data['figures_max_points_count'], dtype=t.int32, device=cuda_device, requires_grad=False),
        'figures_points_count': test_data['figures_points_count'],
        'movements': t.tensor(test_data['movements'], dtype=t.float32, device=cuda_device, requires_grad=False),
        'routes_len': test_data['routes_len'],
        'routes_len_t': t.tensor(test_data['routes_len'], dtype=t.int32, device=cuda_device, requires_grad=False),
    }

    if data_gene_mode_int == 4:
        test_data['routes_len_4_random_part'] = routes_len_4_random_part
        test_data['routes_len_4_random_part_t'] = \
            t.tensor(test_data['routes_len_4_random_part'], dtype=t.int32, device=cuda_device, requires_grad=False)


    return test_data


def calc_result__engine2d_torch_cpu_f2(test_data):
    return calc_result__engine2d_torch_f2(test_data, 'cpu')

def calc_result__engine2d_torch_gpu_f2(test_data):
    return calc_result__engine2d_torch_f2(test_data, 'gpu')






def calc_result__engine2d_torch_f2(test_data, method):
    global cuda_device

    time_func_start = time.perf_counter()
    time_sum = 0
    time_1 = 0
    cou = 0



    ##############
    # 1 mks
    data_gene_mode_int = test_data['data_gene_mode_int']
    data_gene_mode_int_t = test_data['data_gene_mode_int_t']
    agents = test_data['agents']
    agents_count = test_data['agents_count']
    agents_count_t = test_data['agents_count_t']
    figures = test_data['figures']
    figures_max_points_count_t = test_data['figures_max_points_count']
    figures_points_count = test_data['figures_points_count']
    routes_len = test_data['routes_len']
    routes_len_t = test_data['routes_len_t']
    movements_t = test_data['movements']


    ############
    # 0 mks
    routes_len_4_random_part = None
    if data_gene_mode_int == 4:
        routes_len_4_random_part = test_data['routes_len_4_random_part']
        routes_len_4_random_part_t = test_data['routes_len_4_random_part_t']
    ############


    ############
    # 176 mks
    figures_t = t.from_numpy(figures)
    agents_t = t.from_numpy(agents)
    figures_points_count_t = t.from_numpy(figures_points_count)

    if method == 'gpu':
        figures_t = figures_t.to(cuda_device)
        agents_t = agents_t.to(cuda_device)
        figures_points_count_t = figures_points_count_t.to(cuda_device)
    #############



    # 64 mks        !!!!!!!!!!!!!!!
    agents_positions_t = t.zeros(
            size=(agents_count_t, 2),
            dtype=t.float32,
            device=cuda_device
    )



    # 47 mks        !!!!!!!!!!!!!!!!
    agents_angles_t = t.zeros(
            size=(agents_count_t,),
            dtype=t.float32,
            device=cuda_device
    )



    # все фигуры всех перемещений всех агентов.
    # это и будет результат расчета
    # на каждую фигрур точек закладываем по максимуму, т.е. для маленьких фигур часть точек не используется


    #############
    # 75 mks
    if data_gene_mode_int == 1:
        routes_figures_t = t.zeros(
                size=(routes_len_t, agents_count_t, figures_max_points_count_t, 2),
                dtype=t.float32,
                device=cuda_device,
        )
    elif data_gene_mode_int == 4:
        routes_figures_t = t.zeros(
                size=(routes_len_4_random_part, agents_count_t, figures_max_points_count_t, 2),
                dtype=t.float32,
                device=cuda_device,
        )
    #############


    # индексы фигур, для каждого агента.
    # разные агенты могут иметь одну и туже фигуру, т.е. индексы совпадут

    figures_inds_by_agents_t = agents_t[:, 3].long()        #30 mks

    # это ссылки на фигуры
    figures_by_agents_t = figures_t[figures_inds_by_agents_t]       # 28 mks !!!!!!!!!!!

    # кол-во точек в фигуре для каждого агента
    #points_count_by_agents_t = figures_points_count_t[figures_inds_by_agents_t]     #17 mks


    # zeros  cpu 36mks, gpu 99mks
    raw_points_2D_mxNN31 = t.ones(
        size=(agents_count_t, figures_max_points_count_t, 3, 1),
        device=cuda_device,
    )


    # print(agents_count_t.device)
    # print(raw_points_2D_mxNN31.device)
    # print(points_count_by_agents_t.device)
    # print(figures_by_agents_t.device)


    raw_points_2D_mxNN31[:, ..., 0:2, 0] = figures_by_agents_t[:]   #cpu 170mks, gpu 26 mks

    # # Продожительносьт всего цикла по агентам cpu: 271ms, gpu 663 ms        может нам сразу cpu использовать ??
    # # 10000 агентов
    # # цикл agent_i рабоатет в дав раза ьыстрее чем на agent_i_t
    # agent_i_t = None
    # for agent_i in range(agents_count):
    #
    #     # loop 137 mks      agent_i_t
    #     # loop 68 mks       agent_i
    #
    #     # 14 mks
    #     # if agent_i_t is not None:
    #     #     agent_i_t = t.add(agent_i_t, 1)
    #     # else:
    #     #     agent_i_t = t.zeros(size=(1,), dtype=t.int64, device=cuda_device)       # t.int64, used by indexes
    #
    #     # 283 mks, 182 mks, 120 mks
    #
    #     # 37mks
    #     #A = agent_i_t.item()
    #
    #     # 38 mks
    #     F = figures_by_agents_t[agent_i]          # 5mks
    #
    #     # 49 mks
    #     V = vctt_pointsToMatrix_mxN31(F)
    #
    #     raw_points_2D_mxNN31[agent_i] = V     # 11 mks

    to_print = True

    movement_i_t = None
    for movement_i in range(routes_len_t):

        ##############
        # 18 mks
        if movement_i_t is not None:
            movement_i_t = t.add(movement_i_t, 1)
        else:
            movement_i_t = t.zeros(size=(1,), dtype=t.int32, device=cuda_device)
        ##############

        # if to_print:
        #     to_print = False
        #     # print(agents_positions_t.device)
        #     print(movement_i_t.device)
        #     print(movements_t.device)
        #     # print(agents_angles_t.device)

        cur_movements_t = None


        ##############
        # movement_i            9mks / 9mks,
        # movement_i_t.item()   11 mks / 47mks
        if data_gene_mode_int == 1:
            cur_movements_t = movements_t[:, movement_i]
        elif data_gene_mode_int == 4:
            cur_movements_t = movements_t[:, (movement_i % 10)]        #
        ##############

        ##############
        # cpu/gpu       218mks/46 mks, (on 1 movement)
        agents_positions_t[:] += cur_movements_t[:, 0:2]
        agents_angles_t[:] += cur_movements_t[:, 2]
        ##############

        # cpu/gpu       180 mks / 295mks        использовал при формированиии вектора t.eye
        # cpu/gpu       112 mks / 196mks        без eye
        # cpu/gpu       177 mks / 63mks        c eye, но без вызова функции


        ########################################
        # MACRO формируем матриwы трансляции

        # аналогично вызову функции
        #translate_mxN33_by_agents_t = vctt_offsToTranslateMx_mxN33(agents_positions_t)

        translate_mxN33_by_agents_t = t.empty(
                size=(agents_count, 3, 3),
                dtype=t.float32,
                device=cuda_device,
        )
        translate_mxN33_by_agents_t[:] = t.eye(3, dtype=t.float32, device=cuda_device)
        translate_mxN33_by_agents_t[:, 0:2, 2] = agents_positions_t[:, 0:2]
        ########################################





        ########################################
        # MACRO формируем матриwу поворота
        # cpu/gpu   202 mks / 389 mks     - использовал функцию
        # rotate_mxN33_by_agents_t = vctt_anglesToRotateMx_mxN33(agents_angles_t)

        # cpu/gpu   197 mks / 141 mks     - убрал функцию

        cos_alfas = t.cos(agents_angles_t)
        sin_alfas = t.sin(agents_angles_t)

        # еденичные матрицы
        rotate_mxN33_by_agents_t = t.empty(
                size=(agents_count, 3, 3),
                dtype=t.float32,
                device=cuda_device,
        )
        rotate_mxN33_by_agents_t[:] = t.eye(3, dtype=t.float32, device=cuda_device)

        rotate_mxN33_by_agents_t[:, 0, 0] = cos_alfas[:]
        rotate_mxN33_by_agents_t[:, 1, 1] = cos_alfas[:]
        rotate_mxN33_by_agents_t[:, 0, 1] = -sin_alfas[:]
        rotate_mxN33_by_agents_t[:, 1, 0] = sin_alfas[:]
        ########################################




        # cpu/gpu   115mks / 38 mks
        transform_mxN33_by_agents_t = t.matmul(
                translate_mxN33_by_agents_t,
                rotate_mxN33_by_agents_t,
        )

        transform_mxN33_by_agents_t = transform_mxN33_by_agents_t[:, np.newaxis]    #6 mks / 7mks



        # 722 mks / 45mks   (!!! cpu тормозит)
        points_new_2D_mxN31 = t.matmul(transform_mxN33_by_agents_t, raw_points_2D_mxNN31)

        if data_gene_mode_int == 1:
            # все варианты кроме random_part
            routes_figures_t[movement_i] = points_new_2D_mxN31[..., 0:2, 0]

        elif data_gene_mode_int == 4:
            # зписываем каждый 1000й (0, 1000, 2000, etc), а также поседний цикл
            if movement_i % 1000 == 0:
                routes_figures_t[int(movement_i / 1000)] = points_new_2D_mxN31[..., 0:2, 0]

            elif movement_i + 1 == routes_len:
                # для последнего записываемого перемещения
                routes_figures_t[int(movement_i / 1000) + 1] = points_new_2D_mxN31[..., 0:2, 0]


        time_1 = time.perf_counter()
        # t.cuda.synchronize()
        time_sum += time.perf_counter() - time_1
        cou = cou + 1


    if method == 'gpu':
        test_data['routes_figures'] = routes_figures_t.to('cpu')
        #test_data['routes_len'] = routes_len_t.to('cpu')
        #test_data['agents_count'] = agents_count_t.to('cpu')

    else:
        test_data['routes_figures'] = routes_figures_t

    time_func_duration = time.perf_counter() - time_func_start



    # print('func time ms: {:.1f}'.format(time_func_duration*1000))
    # print('count       : {}'.format(cou))
    # print('sect time ms: {:.1f}'.format(time_sum*1000))
    # print('sect(med) ms: {:.3f}'.format(time_sum*1000 / cou))


    return test_data


# def convert_to_np__engine2d_torch_cpu(test_data):
#
#
#     test_data['routes_figures'] = test_data['routes_figures'].numpy()
#
#     return test_data

