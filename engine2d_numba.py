import numpy as np
from numba import cuda          # Библиотека Nvidia для работы с GPU
import numba
from Vector import *

def get_functions_dict():
    return {
        'init_method_data__engine2d_numba': init_method_data__engine2d_numba,
        'upload__engine2d_numba_f1': upload__engine2d_numba_f1,
        'calc_result__engine2d_numba_f1': calc_result__engine2d_numba_f1,
        'download__engine2d_numba_f1': download__engine2d_numba_f1
    }



def init_method_data__engine2d_numba(test_data, method):
    return test_data


def upload__engine2d_numba_f1(test_data):

    data_gene_mode_int = None
    if test_data['data_gene_mode'] == 'const_full' or test_data['data_gene_mode'] == 'debug_full' or test_data['data_gene_mode'] == 'random_full':
        data_gene_mode_int = 1
    elif test_data['data_gene_mode'] == 'random_part':
        data_gene_mode_int = 4
        routes_len_4_random_part = test_data['routes_len_4_random_part']


    #print('ZZZZZZZZZZZZZZZZZZZZZZZZZZZZzz ', data_gene_mode_int)

    test_data = {
        'data_gene_mode_int': data_gene_mode_int,
        #'data_gene_mode_int_g': data_gene_mode_int,
        #'data_gene_mode_int_g': cuda.to_device(data_gene_mode_int),

        'agents': test_data['agents'],
        #'agents_g': cuda.to_device(test_data['agents']),

        'agents_count': test_data['agents_count'],
        #'agents_count_g': cuda.to_device(test_data['agents_count']),


        'figures': test_data['figures'],
        #'figures_g': cuda.to_device(test_data['figures']),

        'figures_count': test_data['figures_count'],
        #'figures_count_g': cuda.to_device(test_data['figures_count']),

        'figures_max_points_count': test_data['figures_max_points_count'],
        #'figures_max_points_count_g': cuda.to_device(test_data['figures_max_points_count']),

        'figures_points_count': test_data['figures_points_count'],
        'figures_points_count_g': cuda.to_device(test_data['figures_points_count']),

        'movements': test_data['movements'],
        'movements_g': cuda.to_device(test_data['movements']),

        'routes_len': test_data['routes_len'],
        #'routes_len_g': cuda.to_device(test_data['routes_len']),
    }

    if data_gene_mode_int == 4:
        test_data['routes_len_4_random_part'] = routes_len_4_random_part
        test_data['routes_len_4_random_part_g'] = cuda.to_device(routes_len_4_random_part)

    cuda.synchronize()
    return test_data


@cuda.jit
def f1_cuda(
        routes_len,
        result_g,
        movements_g,
        agents_positions_g,
        agents_angles_g,
        routes_figures_g,
        points_count_by_agents_g,
        raw_points_2D_mxN31_g,
        data_gene_mode_int,
):

    # 1 thread = 1 block = 1 agent
    # если у нас 3 агента то будет 3 Потока и 3 блока

    # thread index in block
    tx = cuda.threadIdx.x                       # всегода 0, т.к. у на все блоки размером в один поток (
    ty = cuda.threadIdx.y                       # 0

    # block index in grid
    agent_i = bx = cuda.blockIdx.x              # agent_i   0-based     0 1 2
    by = cuda.blockIdx.x                        # 0

    block_size_x = cuda.blockDim.x              # threads in block      1
    block_size_y = cuda.blockDim.y              # 1
    threads_in_block = block_size_x * block_size_y      # 1


    # матрица перемещения для агента
    translate_mx33_g = cuda.local.array(shape=(3, 3), dtype='f4')
    rotate_mx33_g = cuda.local.array(shape=(3, 3), dtype='f4')
    transform_mx33_g = cuda.local.array(shape=(3, 3), dtype='f4')

    point_A_mx31_g = cuda.local.array(shape=3, dtype='f4')

    raw_points_2D_mx31_g = raw_points_2D_mxN31_g[agent_i]

    for movement_i in range(routes_len):

        if data_gene_mode_int == 1:
            cur_movements_g = movements_g[agent_i, movement_i]
        elif data_gene_mode_int == 4:
            cur_movements_g = movements_g[agent_i, (movement_i % 10)]


        agents_positions_g[agent_i, 0] += cur_movements_g[0]
        agents_positions_g[agent_i, 1] += cur_movements_g[1]

        agents_angles_g[agent_i] += cur_movements_g[2]

        translate_mx33_g[0, 0] = 1
        translate_mx33_g[0, 1] = 0
        translate_mx33_g[0, 2] = agents_positions_g[agent_i, 0]
        translate_mx33_g[1, 0] = 0
        translate_mx33_g[1, 1] = 1
        translate_mx33_g[1, 2] = agents_positions_g[agent_i, 1]
        translate_mx33_g[2, 0] = 0
        translate_mx33_g[2, 1] = 0
        translate_mx33_g[2, 2] = 1

        cos_alfa = math.cos(agents_angles_g[agent_i])
        sin_alfa = math.sin(agents_angles_g[agent_i])

        rotate_mx33_g[0, 0] = cos_alfa
        rotate_mx33_g[0, 1] = -sin_alfa
        rotate_mx33_g[0, 2] = 0
        rotate_mx33_g[1, 0] = sin_alfa
        rotate_mx33_g[1, 1] = cos_alfa
        rotate_mx33_g[1, 2] = 0
        rotate_mx33_g[2, 0] = 0
        rotate_mx33_g[2, 1] = 0
        rotate_mx33_g[2, 2] = 1

        # rotate_mx33_g = [
        #     [cos_alfa, -sin_alfa, 0],
        #     [sin_alfa, cos_alfa, 0],
        #     [0, 0, 1],
        # ]

        transform_mx33_g[0][0] = translate_mx33_g[0][0] * rotate_mx33_g[0][0] + translate_mx33_g[0][1] * rotate_mx33_g[1][0] + translate_mx33_g[0][2] * rotate_mx33_g[2][0]
        transform_mx33_g[0][1] = translate_mx33_g[0][0] * rotate_mx33_g[0][1] + translate_mx33_g[0][1] * rotate_mx33_g[1][1] + translate_mx33_g[0][2] * rotate_mx33_g[2][1]
        transform_mx33_g[0][2] = translate_mx33_g[0][0] * rotate_mx33_g[0][2] + translate_mx33_g[0][1] * rotate_mx33_g[1][2] + translate_mx33_g[0][2] * rotate_mx33_g[2][2]
        transform_mx33_g[1][0] = translate_mx33_g[1][0] * rotate_mx33_g[0][0] + translate_mx33_g[1][1] * rotate_mx33_g[1][0] + translate_mx33_g[1][2] * rotate_mx33_g[2][0]
        transform_mx33_g[1][1] = translate_mx33_g[1][0] * rotate_mx33_g[0][1] + translate_mx33_g[1][1] * rotate_mx33_g[1][1] + translate_mx33_g[1][2] * rotate_mx33_g[2][1]
        transform_mx33_g[1][2] = translate_mx33_g[1][0] * rotate_mx33_g[0][2] + translate_mx33_g[1][1] * rotate_mx33_g[1][2] + translate_mx33_g[1][2] * rotate_mx33_g[2][2]
        transform_mx33_g[2][0] = translate_mx33_g[2][0] * rotate_mx33_g[0][0] + translate_mx33_g[2][1] * rotate_mx33_g[1][0] + translate_mx33_g[2][2] * rotate_mx33_g[2][0]
        transform_mx33_g[2][1] = translate_mx33_g[2][0] * rotate_mx33_g[0][1] + translate_mx33_g[2][1] * rotate_mx33_g[1][1] + translate_mx33_g[2][2] * rotate_mx33_g[2][1]
        transform_mx33_g[2][2] = translate_mx33_g[2][0] * rotate_mx33_g[0][2] + translate_mx33_g[2][1] * rotate_mx33_g[1][2] + translate_mx33_g[2][2] * rotate_mx33_g[2][2]


        # перемещаем все точки данного агента/фигуры

        # число точек в фигуре данного агента. (1 thread = 1 agent)
        points_count = points_count_by_agents_g[agent_i]


        for point_i in range(points_count):
            point_mx31 = raw_points_2D_mx31_g[point_i]

            point_A_mx31_g[0] = transform_mx33_g[0][0] * point_mx31[0, 0] + transform_mx33_g[0][1] * point_mx31[1, 0] + transform_mx33_g[0][2] * point_mx31[2, 0]
            point_A_mx31_g[1] = transform_mx33_g[1][0] * point_mx31[0, 0] + transform_mx33_g[1][1] * point_mx31[1, 0] + transform_mx33_g[1][2] * point_mx31[2, 0]
            point_A_mx31_g[2] = transform_mx33_g[2][0] * point_mx31[0, 0] + transform_mx33_g[2][1] * point_mx31[1, 0] + transform_mx33_g[2][2] * point_mx31[2, 0]

            if data_gene_mode_int == 1:
                # все варианты кроме random_part
                routes_figures_g[movement_i, agent_i, point_i, 0] = point_A_mx31_g[0]
                routes_figures_g[movement_i, agent_i, point_i, 1] = point_A_mx31_g[1]


            elif data_gene_mode_int == 4:
                # зписываем каждый 1000й, а также поседний movement

                if movement_i % 1000 == 0:
                    movement_shorted_i = int(movement_i / 1000)
                    routes_figures_g[movement_shorted_i, agent_i, point_i, 0] = point_A_mx31_g[0]
                    routes_figures_g[movement_shorted_i, agent_i, point_i, 1] = point_A_mx31_g[1]

                elif movement_i + 1 == routes_len:
                    # для последнего записываемого перемещения
                    movement_shorted_i = int(movement_i / 1000) + 1


                    routes_figures_g[movement_shorted_i, agent_i, point_i, 0] = point_A_mx31_g[0]
                    routes_figures_g[movement_shorted_i, agent_i, point_i, 1] = point_A_mx31_g[1]


#
#
#
def calc_result__engine2d_numba_f1(test_data):

    data_gene_mode_int = test_data['data_gene_mode_int']
    #data_gene_mode_int_g = test_data['data_gene_mode_int_g']
    agents = test_data['agents']
    agents_count = test_data['agents_count']
    figures = test_data['figures']
    figures_max_points_count = test_data['figures_max_points_count']
    figures_points_count = test_data['figures_points_count']
    figures_points_count_g = test_data['figures_points_count_g']
    routes_len = test_data['routes_len']
    movements_g = test_data['movements_g']

    routes_len_4_random_part = None

    if data_gene_mode_int == 4:
        routes_len_4_random_part = test_data['routes_len_4_random_part']

    # agents_positions = np.zeros(shape=(agents_count, 2), dtype="f4")
    agents_positions_g = cuda.to_device(np.zeros(shape=(agents_count, 2), dtype="f4"))

    # agents_angles = np.zeros(shape=agents_count, dtype="f4")
    agents_angles_g = cuda.to_device(np.zeros(shape=agents_count, dtype="f4"))

    routes_figures_g = None
    if data_gene_mode_int == 1:
        #routes_figures = np.zeros(shape=(routes_len, agents_count, figures_max_points_count, 2), dtype="f4")
        routes_figures_g = cuda.to_device(np.zeros(shape=(routes_len, agents_count, figures_max_points_count, 2), dtype="f4"))
    elif data_gene_mode_int == 4:
        #routes_figures = np.zeros(shape=(routes_len_4_random_part, agents_count, figures_max_points_count, 2), dtype="f4")
        routes_figures_g = cuda.to_device(np.zeros(shape=(routes_len_4_random_part, agents_count, figures_max_points_count, 2), dtype="f4"))


    # индексы фигур, для каждого агента.
    figures_inds_by_agents = agents[:, 3].astype("i4", copy=False)

    # ссылки на фигуры(наборы точек), для каждого агента
    figures_by_agents = figures[figures_inds_by_agents]

    # кол-во точек в фигуре для каждого агента
    points_count_by_agents = figures_points_count[figures_inds_by_agents]
    points_count_by_agents_g = cuda.to_device(points_count_by_agents)

    # для каждого агента, фигура в нулевой точке пути, включает все точки относящиеся у фигуре
    # каждая точка матрица [3,1] (именно [3,1] а не [3])
    raw_points_2D_mxN31 = np.zeros((agents_count, figures_max_points_count, 3, 1), 'f4')

    for agent_i in range(agents_count):
        raw_points_2D_mxN31[agent_i][0:points_count_by_agents[agent_i]] = vctm_pointToMx_mxN31(
                    figures_by_agents[agent_i][0:points_count_by_agents[agent_i]]
            )


    raw_points_2D_mxN31_g = cuda.to_device(raw_points_2D_mxN31)


    result_g = cuda.to_device(np.zeros(shape=200, dtype='f4'))

    bpg = agents_count
    tpb = 1
    f1_cuda[bpg, tpb](
            routes_len,
            result_g,
            movements_g,
            agents_positions_g,
            agents_angles_g,
            routes_figures_g,
            points_count_by_agents_g,
            raw_points_2D_mxN31_g,
            data_gene_mode_int,
    )

    test_data['routes_figures_g'] = routes_figures_g

    return test_data



def download__engine2d_numba_f1(test_data):
    test_data['routes_figures'] = test_data['routes_figures_g'].copy_to_host()
    test_data['routes_figures_g'] = None

    cuda.synchronize()

    # print('NUMBA:')
    # print(test_data['routes_figures'])

    return test_data