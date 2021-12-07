import numpy as np
import time

from Vector import *



def get_functions_dict():
    return {
        'init_method_data__engine2d_np': init_method_data__engine2d_np,

        'calc_result__engine2d_np_f1': calc_result__engine2d_np_f1,
        'calc_result__engine2d_np_f2': calc_result__engine2d_np_f2,
        'calc_result__engine2d_np_f3': calc_result__engine2d_np_f3,
        'calc_result__engine2d_np_f4': calc_result__engine2d_np_f4,
    }




##########################################################################

def init_method_data__engine2d_np(test_data, method):
    return test_data


def calc_result__engine2d_np_f1(test_data):

    def calc_figure(figure, figure_points_count, agent_position, figure_in_route, movement):

        agent_position[0:2] = agent_position[0:2] + movement[0:2]
        agent_position[2] = agent_position[2] + movement[2]


        translate_mx33 = vct_offsToTranslateMx_mx33(agent_position[0:2])
        rotate_mx33 = vct_angleToRotateMx_mx33(agent_position[2])

        for point_i in range(figure_points_count):
            point_mx31 = vct_pointToMx_mx31(figure[point_i])
            point_mx31 = np.dot(rotate_mx33, point_mx31)
            point_mx31 = np.dot(translate_mx33, point_mx31)

            figure_in_route[point_i] = point_mx31[0:2]

        return figure_in_route, agent_position


    agents = test_data['agents']
    agents_count = test_data['agents_count']
    figures = test_data['figures']
    figures_max_points_count = test_data['figures_max_points_count']
    figures_points_count = test_data['figures_points_count']
    routes_len = test_data['routes_len']
    movements = test_data['movements']




    # все центры, всех фигур, всех путей, всех агентов
    # agents_position[0:2]  - координаты фигнуры
    # agents_position[2]    - угол
    agents_positions = np.zeros((agents_count, 3), "f4")

    # все фигуры всех путей всех агентов
    routes_figures = np.zeros(shape=(routes_len, agents_count, figures_max_points_count, 2), dtype="f4")



    for movement_i in range(routes_len):

        for agent_i, agent in enumerate(agents):

            #
            #     # action[0], action[1] - перемещение dx, dy
            #     # action[2] - поворот на угол
            #
            # шаблон фигруры

            routes_figures[movement_i][agent_i], agents_positions[agent_i] = calc_figure(
                    figures[round(agent[3])],               # шаблон фигуры для данного агента
                    figures_points_count[round(agent[3])],  # кол-во точек в шаблоне
                    agents_positions[agent_i],              # стартовая позиция (x, y, угол)
                    routes_figures[movement_i][agent_i],
                    movements[agent_i][movement_i],
            )

    test_data['routes_figures'] = routes_figures

    return test_data


#
#
#
def calc_result__engine2d_np_f2(loop_count, test_data):
    agents = test_data['agents']
    agents_count = test_data['agents_count']
    figures = test_data['figures']
    figures_max_points_count = test_data['figures_max_points_count']
    figures_points_count = test_data['figures_points_count']
    routes_len = test_data['routes_len']
    movements = test_data['movements']


    # все центры, всех фигур, всех путей, всех агентов
    # agents_position[0:2]  - координаты фигнуры
    # agents_position[2]    - угол
    agents_positions = np.zeros(shape=(agents_count, 3), dtype="f4")

    # все фигуры всех перемещений всех агентов
    routes_figures = np.zeros(shape=(routes_len, agents_count, figures_max_points_count, 2), dtype="f4")

    # индексы фигур, для агентов.
    figures_indices_by_agents = agents[:, 3]
    figures_indices_by_agents = figures_indices_by_agents.astype("i4", copy=False)


    figures_4_agents = np.empty(shape=agents_count, dtype=object)
    points_count_by_agents = np.empty(shape=(agents_count), dtype="i4")

    for agent_i in range(agents_count):
        figures_4_agents[agent_i] = figures[figures_indices_by_agents[agent_i]]
        points_count_by_agents[agent_i] = figures_points_count[figures_indices_by_agents[agent_i]]


    for movement_i in range(routes_len):

        #print(f'movement_i:{movement_i}')
        # действеи
        routes_figures_cur_movement = routes_figures[movement_i]

        # данные на перемещения для даннго  movement_i, для всех агентов
        cur_movements = movements[:, movement_i]

        agents_positions[:, 0:3] = agents_positions[:, 0:3] + cur_movements[:, 0:3]

        # матрица преобразований для всех агентов
        # shape=(agents_count, 3,3)
        translate_mxN33_by_agents = vctm_offsToTranslateMx_mx33(agents_positions)
        rotate_mxN33_by_agents = vctm_angleToRotateMx_mx33(agents_positions[:, 2])
        transform_mxN33_by_agents = np.matmul(
                translate_mxN33_by_agents,
                rotate_mxN33_by_agents,
        )

        for agent_i in range(agents_count):

            points_mxN31 = vctm_pointToMx_mxN3(figures_4_agents[agent_i][:points_count_by_agents[agent_i]])

            # установим размерность массива как (4,3,1) а не (4,3), иначе matmul рабоате некорректно
            points_mxN31 = np.reshape(points_mxN31, newshape=(points_count_by_agents[agent_i], 3, 1))

            trans = np.empty(shape=(points_mxN31.shape[0],3,3), dtype="f4")
            trans[:] = transform_mxN33_by_agents[agent_i]

            points_new_mxN31 = np.matmul(
                    trans,
                    points_mxN31
            )

            routes_figures_cur_movement[agent_i][:points_count_by_agents[agent_i]] = \
                points_new_mxN31[:points_count_by_agents[agent_i], 0:2, 0]


    test_data['routes_figures'] = routes_figures

    return test_data



#
# для одного перемещения, все точки всех агентов выстриваем в один 1D-массив
# учитываем что фигуры имеют разное кол-во точек
#
def calc_result__engine2d_np_f3(loop_count, test_data):

    agents = test_data['agents']
    agents_count = test_data['agents_count']
    figures = test_data['figures']
    figures_max_points_count = test_data['figures_max_points_count']
    figures_points_count = test_data['figures_points_count']
    routes_len = test_data['routes_len']
    movements = test_data['movements']

    agents_positions = np.zeros(shape=(agents_count, 2), dtype="f4")
    agents_angles = np.zeros(shape=agents_count, dtype="f4")

    # все фигуры всех перемещений всех агентов.
    # это и будет результат расчета
    routes_figures = np.zeros(shape=(routes_len, agents_count, figures_max_points_count, 2), dtype="f4")

    # индексы фигур, для каждого агента агентов.
    figures_inds_by_agents = agents[:, 3].astype("i4", copy=False)

    # ссылки на фигуры
    figures_by_agents = figures[figures_inds_by_agents]

    # кол-во точек в фигуре для каждого агента
    points_count_by_agents = figures_points_count[figures_inds_by_agents]

    # количесвто точек для всех агентов
    points_count = np.sum(points_count_by_agents)


    # стартовые индексы точек для каждого агента в points_mxN31_for_agents
    # массив на 1 больше чем число агентов, чтобы для последнего агента модно было взять значение для agent_i+1
    points_inds_by_agents = np.hstack(
            ([0], np.cumsum(points_count_by_agents))
    )

    points_intervals_by_agents = np.empty(agents_count, object)

    # номер агента для каждой точки
    agents_by_points = np.empty(points_count, 'i4')

    # для каждого агента, фигура в нулевой точке
    raw_points_mxN31 = np.empty((points_count, 3, 1), 'f4')

    for agent_i in range(agents_count):
        points_intervals_by_agents[agent_i] = np.arange(start=points_inds_by_agents[agent_i], stop=points_inds_by_agents[agent_i + 1])

        agents_by_points[points_intervals_by_agents[agent_i]] = agent_i

        raw_points_mxN31[points_intervals_by_agents[agent_i]] =\
            vctm_pointToMx_mxN31(
                    figures_by_agents[agent_i][0:points_count_by_agents[agent_i]]
            )


    for movement_i in range(routes_len):
        # данные на перемещения для даннго  movement_i, для всех агентов
        cur_movements = movements[:, movement_i]

        agents_positions[:] += cur_movements[:, 0:2]
        agents_angles[:] += cur_movements[:, 2]

        # матрица преобразований для всех агентов
        # shape=(agents_count, 3,3)
        translate_mxN33_by_agents = vctm_offsToTranslateMx_mx33(agents_positions)
        rotate_mxN33_by_agents = vctm_angleToRotateMx_mx33(agents_angles)
        transform_mxN33_by_agents = np.matmul(
                translate_mxN33_by_agents,
                rotate_mxN33_by_agents,
        )

        transform_mxN33_by_points = transform_mxN33_by_agents[agents_by_points]

        points_new_mxN31 = np.matmul(transform_mxN33_by_points, raw_points_mxN31)

        for agent_i in range(agents_count):
            interval = points_intervals_by_agents[agent_i]
            routes_figures[movement_i][agent_i][0:interval.shape[0]] = points_new_mxN31[interval][:, 0:2, 0]

    test_data['routes_figures'] = routes_figures

    return test_data

#
# для одного перемещения, все точки всех агентов выстриваем в один 1D-массив
# учитываем что фигуры имеют разное кол-во точек
#
def calc_result__engine2d_np_f4(test_data):

    data_gene_mode = test_data['data_gene_mode']
    agents = test_data['agents']
    agents_count = test_data['agents_count']
    figures = test_data['figures']
    figures_max_points_count = test_data['figures_max_points_count']
    figures_points_count = test_data['figures_points_count']
    movements = test_data['movements']
    routes_len = test_data['routes_len']

    if data_gene_mode == 'random_part':
        routes_len_4_random_part = test_data['routes_len_4_random_part']


    agents_positions = np.zeros(shape=(agents_count, 2), dtype="f4")
    agents_angles = np.zeros(shape=agents_count, dtype="f4")

    # все фигуры всех перемещений всех агентов.
    # это и будет результат расчета
    if data_gene_mode == 'const_full' or data_gene_mode == 'debug_full' or data_gene_mode == 'random_full':
        routes_figures = np.zeros(shape=(routes_len, agents_count, figures_max_points_count, 2), dtype="f4")
    elif data_gene_mode == 'random_part':
        routes_figures = np.zeros(shape=(routes_len_4_random_part, agents_count, figures_max_points_count, 2), dtype="f4")

    # индексы фигур, для каждого агента.
    figures_inds_by_agents = agents[:, 3].astype("i4", copy=False)

    # ссылки на фигуры(наборы точек), для каждого агента
    figures_by_agents = figures[figures_inds_by_agents]

    # кол-во точек в фигуре для каждого агента
    points_count_by_agents = figures_points_count[figures_inds_by_agents]

    # для каждого агента, фигура в нулевой точке (координаты точек)
    raw_points_2D_mxN31 = np.zeros((agents_count, figures_max_points_count, 3, 1), 'f4')


    for agent_i in range(agents_count):
        raw_points_2D_mxN31[agent_i][0:points_count_by_agents[agent_i]] = vctm_pointToMx_mxN31(
                    figures_by_agents[agent_i][0:points_count_by_agents[agent_i]]
            )


    for movement_i in range(routes_len):

        # данные на перемещения для даннго  movement_i, для всех агентов
        if data_gene_mode == 'const_full' or data_gene_mode == 'debug_full' or data_gene_mode == 'random_full':
            cur_movements = movements[:, movement_i]

        elif data_gene_mode == 'random_part':
            cur_movements = movements[:, (movement_i % 10)]

        agents_positions[:] += cur_movements[:, 0:2]
        agents_angles[:] += cur_movements[:, 2]

        # матрица преобразований для всех агентов
        # shape=(agents_count, 3,3)
        translate_mxN33_by_agents = vctm_offsToTranslateMx_mx33(agents_positions)
        rotate_mxN33_by_agents = vctm_angleToRotateMx_mx33(agents_angles)

        transform_mxN33_by_agents = np.matmul(
                translate_mxN33_by_agents,
                rotate_mxN33_by_agents,
        )

        transform_mxN33_by_agents = transform_mxN33_by_agents[:, np.newaxis]


        points_new_2D_mxN31 = np.matmul(transform_mxN33_by_agents, raw_points_2D_mxN31)

        if data_gene_mode == 'const_full' or data_gene_mode == 'debug_full' or data_gene_mode == 'random_full':
            # все варианты кроме random_part
            routes_figures[movement_i] = points_new_2D_mxN31[..., 0:2, 0]
            #print(movement_i, points_new_2D_mxN31[0,0,0])

        elif data_gene_mode == 'random_part':


            # зписываем каждый 1000й, а также поседний цикл
            if movement_i % 1000 == 0:
                routes_figures[int(movement_i / 1000)] = points_new_2D_mxN31[..., 0:2, 0]

            elif movement_i + 1 == routes_len:
                # для последнего записываемого перемещения
                routes_figures[int(movement_i / 1000) + 1] = points_new_2D_mxN31[..., 0:2, 0]
                #print(movement_i, points_new_2D_mxN31[0,0,0])


    test_data['routes_figures'] = routes_figures


    return test_data

