import numpy as np

#import time


###################
# 1 из 2х
# это занчение меняется (по оси X)
GRAPH_X_NAME = None
GRAPH_X_NAME = 'AGENTS_COUNT'
#GRAPH_X_NAME = 'ROUTES_LEN'


GRAPH_AGENTS_COUNT_CONST = 25000
GRAPH_ROUTES_LEN_CONST = 1000

GRAPH_TITLE = 'unknown title'
GRAPH_X_TITLE = ''

# сбросим лишние константы
if GRAPH_X_NAME == 'AGENTS_COUNT':
    GRAPH_AGENTS_COUNT_CONST = None
    GRAPH_TITLE = 'Циклов: ' + str(GRAPH_ROUTES_LEN_CONST)
    GRAPH_X_TITLE = 'Кол-во агентов'

elif GRAPH_X_NAME == 'ROUTES_LEN':
    GRAPH_ROUTES_LEN_CONST = None
    GRAPH_TITLE = 'Агентов: ' + str(GRAPH_AGENTS_COUNT_CONST)
    GRAPH_X_TITLE = 'Кол-во циклов'





def get_functions_dict():
    return {
        'init_shape_data__engine2d': init_shape_data__engine2d,
        'verify_result__engine2d': verify_result__engine2d,
        'print_result__engine2d': print_result__engine2d,
    }


def _init_shape_data__const_full(test_data):
    # агентов 3

    # формат agents
    # [
    #   номер агента,
    #   {0-x координата агента, 1-y координата агента, 2-угол поворота агента, 3-тип фигуры связаной с агентом} ]
    agents = np.array(
            [
                [0., 0., 0., 0],
                [0., 0., 0., 1],
                [0., 0., 0., 2],
            ],
            dtype='f4'
    )
    agents_count = agents.shape[0]

    # формат массива movements : [agents_count, routes_len ]
    # [3, 5] - три агента на 5 ходов
    movements = np.array(
            [
                [
                    # agent 0 (5 movements)
                    [10, 0, 0.1],  # Перемещение по x на 10
                    [11, 0, 0.2],  # перемещение по y на 10
                    [12, 0, -0.1],
                    [13, 1, 0.1],  # поворот на 0.1
                    [14, 1, 0.2],
                ],
                [
                    # agent 1
                    [0, 20, 0.0],
                    [-5, 21.1, 0.1],
                    [-3, 22, 0.1],
                    [-2, 23, 0.1],
                    [1, 24, 0.15],
                ],
                [
                    # agent 2
                    [5, 3.3, 0.0],
                    [6, 2.3, 0.1],
                    [7, 0.1, 0.3],
                    [-1, -0.10, -0.1],
                    [-1.5, 0, -0.1],
                ],
            ],
            dtype='f4'
    )

    routes_len = movements.shape[1]

    if agents_count != movements.shape[0]:
        raise Exception

    test_data['agents'] = agents
    test_data['agents_count'] = agents_count
    test_data['movements'] = movements
    test_data['routes_len'] = routes_len

    return test_data



def _init_shape_data__debug_full(test_data, rng):

    routes_len = 10

    # [номер агента, {0-x агента 1-y агента, 2-угол поворота агента, 3-тип фигуры связаной с агентом} ]
    agents = np.array(
            [
                [0., 0., 0., 0],
                [0., 0., 0., 1],
                [0., 0., 0., 2],
                [0., 0., 0., 3],
                [0., 0., 0., 4],
            ],
            dtype='f4'
    )
    agents_count = agents.shape[0]

    movements = np.empty(shape=(agents_count, routes_len, 3), dtype="f4")

    # формируем movements для agent 0
    # движение только по оси X
    cur_movements = np.zeros(shape=(routes_len, 3), dtype="f4")
    cur_movements[:, 0] = rng.uniform(0, 10, size=routes_len)
    movements[0] = cur_movements

    # движение то лько по оси Y
    cur_movements = np.zeros(shape=(routes_len, 3), dtype="f4")
    cur_movements[:, 1] = rng.uniform(0, 10, size=routes_len)
    movements[1] = cur_movements

    # движение по осям X, Y, без вращения
    cur_movements = np.zeros(shape=(routes_len, 3), dtype="f4")
    cur_movements[:, 0] = rng.uniform(0, 10, size=routes_len)
    cur_movements[:, 1] = rng.uniform(0, 10, size=routes_len)
    movements[2] = cur_movements

    # движение по оси X и вращение
    cur_movements = np.zeros(shape=(routes_len, 3), dtype="f4")
    cur_movements[:, 0] = rng.uniform(0, 10, size=routes_len)
    cur_movements[:, 2] = rng.uniform(-1., 1., size=routes_len)
    movements[3] = cur_movements

    # вращение на месте
    cur_movements = np.zeros(shape=(routes_len, 3), dtype="f4")
    cur_movements[:, 2] = rng.uniform(-1., 1., size=routes_len)
    movements[4] = cur_movements

    test_data['agents'] = agents
    test_data['agents_count'] = agents_count
    test_data['movements'] = movements
    test_data['routes_len'] = routes_len

    return test_data



def _init_shape_data__random_full(test_data, rng):
    agents_count = test_data['agents_count']
    routes_len = test_data['routes_len']

    agents = np.zeros(shape=(agents_count, 4), dtype='f4')
    agents[:, 3] = rng.integers(
            low=0,
            high=test_data['figures_count'],
            size=agents_count,
            dtype='i4',
    )

    movements = np.empty(shape=(agents_count, routes_len, 3), dtype="f4")
    movements[..., 0:2] = (10 * rng.random(size=(agents_count, routes_len, 2), dtype='f4')) - 3
    movements[..., 2] = rng.random(size=(agents_count, routes_len), dtype='f4') - 0.2


    test_data['agents'] = agents
    test_data['movements'] = movements

    return test_data



def _init_shape_data__random_part(test_data, rng):
    agents_count = test_data['agents_count']
    routes_len = test_data['routes_len']

    agents = np.zeros(shape=(agents_count, 4), dtype='f4')
    agents[:, 3] = rng.integers(
            low=0,
            high=test_data['figures_count'],
            size=agents_count,
            dtype='i4',
    )
    ##################################################
    # генерим случайные числа по 3 на каждого агента
    #randoms = rng.random(size=(agents_count, 3), dtype='f4')

    if routes_len >= 10:
        movements_len = 10
    else:
        movements_len = routes_len

    # movements асчитываем на 10 шагов максимум
    movements = np.empty(shape=(agents_count, movements_len, 3), dtype="f4")
    movements[..., 0:2] = (10 * rng.random(size=(agents_count, movements_len, 2), dtype='f4')) - 3
    movements[..., 2] = rng.random(size=(agents_count, movements_len), dtype='f4') - 0.2



    ##################################################
    test_data['movements'] = movements
    test_data['agents'] = agents

    test_data['routes_len_4_random_part'] = int(1 + (routes_len-1) / 1000 + (1 if (routes_len % 1000 != 1) else 0))
                # записываетс каждое 1000е перемещение, а также самое последнеепоследнее
                # 1, 1001, 2001, 3001 итд       (для 1-based)
                # если routes_len не кратно 1000 то надо добавить 1
                # возможно что одно и тоже перемещение будет записано 2 раза
                # routes_len=1      то  routes_len_4_random_part = 1
                # routes_len=100    то  routes_len_4_random_part = 2
                # routes_len=1000   то  routes_len_4_random_part = 2        1й, 1000й и 1000й как послений
                # routes_len=1002   то  routes_len_4_random_part = 3        1, 1001, 1002


    return test_data

#
# подготовка данных
#
def init_shape_data__engine2d(shape_test_cfg):
    #rng = None

    data_gene_mode = shape_test_cfg['main_props']['data_gene_mode']
    rng_seed = shape_test_cfg['main_props']['rng_seed']

    shape_props = shape_test_cfg['shape_props']

    rng = np.random.default_rng(rng_seed)

    agents_count = shape_props[0]   # число агентов, т.е число движущихся фигур
    routes_len = shape_props[1]     # длинна теста, т.е число шагов для каждого агента


    figures = np.array(
            [
                # x, y,
                [[5, 0], [0, 5], [0, -5], [1e6, 0],     [1e6, 0], [1e6, 0]],
                [[5, 2], [-5, 2], [-5, -2], [5, -2],    [1e6, 0], [1e6, 0]],
                [[5, 1], [2, 5], [-3, 2], [-4, -2],     [0, -2.2], [1e6, 0]],
                [[7, 1], [4, 5], [1, 3], [-2, 1],       [0, -1], [7, -5]],
                [[8, 0], [7, 5], [3, 3], [-2, 1],       [0, -1], [7, -5]],
            ],
            dtype="f4"
    )
    figures_count = figures.shape[0]
    figures_max_points_count = figures[0].shape[0]

    # записываем размеры фигур
    figures_points_count = np.empty(shape=figures_count, dtype="i4")

    for figure_i in range(figures_count):

        # для начала считаем что в фигуре точек максимальное число
        figures_points_count[figure_i] = figures_max_points_count

        for point_i in range(figures_max_points_count):
            if figures[figure_i][point_i][0] > 1000:
                # фигура иммет точек меньше чем максимально возможно
                figures_points_count[figure_i] = point_i
                break

    test_data = {
        'data_gene_mode': data_gene_mode,
        'figures': figures,
        'figures_points_count': figures_points_count,
        'figures_max_points_count': figures_max_points_count,
        'figures_count': figures_count,
        'agents_count': agents_count if agents_count else None,
        'routes_len': routes_len if routes_len else None,
    }




    if data_gene_mode == 'const_full':
        test_data = _init_shape_data__const_full(test_data)


    elif data_gene_mode == 'debug_full':
        # генерируем данные частично рандомно , частично по определенным пправмилам
        # применяется при отладке
        test_data = _init_shape_data__debug_full(test_data, rng)

    elif data_gene_mode == 'random_full':
        test_data = _init_shape_data__random_full(test_data, rng)

    elif data_gene_mode == 'random_part':
        test_data = _init_shape_data__random_part(test_data, rng)

    return test_data


def verify_result__engine2d(test_data, correct_data, data_gene_mode):
    # data_gene_mode:        const random_full random_100


    if data_gene_mode == 'random_full' or data_gene_mode == 'debug_full' or data_gene_mode == 'random_part':
        correct_data_4_test = correct_data['routes_figures']

    elif data_gene_mode == 'const_full':
        correct_data_4_test = np.array(
            [
                [
                    # [[14.975021, 0.4991671], [9.500833, 4.975021], [10.499167, -4.975021], [0., 0.], [0., 0.], [0., 0.]],
                    [[14.875021, 0.4991671], [9.500833, 4.975021], [10.499167, -4.975021], [0., 0.], [0., 0.], [0., 0.]],
                    [[5., 22.], [-5., 22.], [-5., 18.], [5., 18.], [0., 0.], [0., 0.]],
                    [[10., 4.3], [7., 8.3], [2., 5.3], [1., 1.3], [5., 1.0999999], [0., 0.]],
                ],
                [

                    [[25.776682, 1.477601], [19.5224, 4.7766824], [22.4776, -4.7766824], [0., 0.], [0., 0.], [0., 0.]],
                    [[-0.22464609, 43.589172], [-10.174688, 42.59084], [-9.775354, 38.610825], [0.17468786, 39.609158], [0., 0.], [0., 0.]],
                    [[15.875187, 7.094171], [12.490841, 10.774688], [7.815321, 7.2905083], [7.2196503, 3.2106578], [11.219633, 3.4109907], [0., 0.]],
                ],
                [
                    [[37.900333, 0.9933467], [32.006653, 4.900333], [33.993347, -4.900333], [0., 0.], [0., 0.], [0., 0.]],
                    [[-3.497006, 66.05348], [-13.297672, 64.06679], [-12.502995, 60.14652], [-2.7023282, 62.133213], [0., 0.], [0., 0.]],
                    [[22.215887, 8.568152], [17.895031, 11.084141], [14.45798, 6.373867], [15.094593, 2.3002045], [18.85672, 3.6736655], [0., 0.]],
                ],
                [
                    [[50.776684, 2.477601], [44.5224, 5.7766824], [47.4776, -3.7766824], [0., 0.], [0., 0.], [0., 0.]],
                    [[ -5.814358, 89.48827], [-15.367723, 86.53307], [-14.185642, 82.71172], [-4.632277, 85.66692], [0., 0.], [0., 0.]],
                    [[21.481163, 8.032938], [17.433071, 10.967723], [13.54295, 6.624112], [13.769694, 2.507246], [17.650145, 3.4982595], [0., 0.]],
                ],
                [
                    [[64.38791, 4.3971276], [57.60287, 6.3879128], [62.39713, -2.3879128], [0., 0.], [0., 0.], [0., 0.]],
                    [[-5.367696, 114.07572], [-14.372167, 109.72607], [-12.632304, 106.124275], [-3.6278334, 110.47393], [0., 0.], [0., 0.]],
                    [[20.201664, 7.573413 ], [16.466787, 10.897672 ], [12.162461, 6.964125 ], [11.977072, 2.8451893], [15.937073, 3.4438534], [0., 0.]],
                ]
            ],
            dtype='f4'
        )

    else:
        raise Exception


    RTOL = 0.01
    ATOL = 0.01

    res_total = True

    #DEBUG


    # routes_len это полная длинна пути
    # saved_routes_len  - кол-во сохраненых перемещений, т.к. random_part в сохранены не все
    if data_gene_mode == 'const_full' or data_gene_mode == 'debug_full' or data_gene_mode == 'random_full':
        saved_routes_len = test_data['routes_len']
    elif data_gene_mode == 'random_part':
        saved_routes_len = test_data['routes_len_4_random_part']

    for save_movement_i in range(saved_routes_len):
        for agent_i in range(test_data['agents_count']):
            figure_index = int(test_data['agents'][agent_i][3])
            point_count = test_data['figures_points_count'][figure_index]

            res = np.allclose(
                    test_data['routes_figures'][save_movement_i][agent_i][0:point_count],
                    correct_data_4_test[save_movement_i][agent_i][0:point_count],
                    rtol=RTOL,
                    atol=ATOL
            )
            res_total = res_total and res

    return res_total


#
#
#
def getAlgorithmTestCfg():

    libraries = [
        {
            'library_name': 'np',
            'methods': ['f4', 'f1'],
        },
        {
            'library_name': 'torch',
            'methods': ['cpu_f2', 'gpu_f2'],
        },
        {
            'library_name': 'numba',
            'methods': ['f1'],
        },
    ]

    # pilot_n : pilot_necessary_params
    # Набор данных для пилота - пробоного теста
    # данные
    const_full = {
        'is_print': False,
        'data_gene_mode': 'const_full',           # const
        'rng_seed': 1002,
    }

    debug_full = {
        'is_print': False,
        'data_gene_mode': 'debug_full',         # частично генерируетьс по правилам, остальное random_full
        'rng_seed': 1002,
    }

    random_full = {
        'is_print': True,
        'data_gene_mode': 'random_full',     # генерируются все перемещения, сохраняются и проверяются все апозиции
        'rng_seed': 1003,
    }

    random_part = {
        'is_print': True,
        'data_gene_mode': 'random_part',
        'rng_seed': 1003,
    }


    # Формат массива
    # agents_count  routes_len
    shapes_cfgs = [

        {'main_props': const_full,  'shape_props': [0, 0]},         # agent_counts, routes_len - задаются при генераци исходных данных
        # {'main_props': debug_full,  'shape_props': [0, 0]},         # agent_counts, routes_len - задется при гененарции  исходных данных
        # {'main_props': random_full, 'shape_props': [100, 100]},
        # {'main_props': random_part, 'shape_props': [100, 100]},

    ]



    if GRAPH_X_NAME == 'AGENTS_COUNT':
        shapes_cfgs.extend([
                                                        # agents_count   routes_len
            #{'main_props': random_part, 'shape_props': [10000, GRAPH_ROUTES_LEN_CONST]},

            # {'main_props': random_part, 'shape_props': [1, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [5, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [10, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [20, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [30, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [40, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [50, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [60, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [70, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [80, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [90, GRAPH_ROUTES_LEN_CONST]},
            {'main_props': random_part, 'shape_props': [100, GRAPH_ROUTES_LEN_CONST]},
            {'main_props': random_part, 'shape_props': [150, GRAPH_ROUTES_LEN_CONST]},
            {'main_props': random_part, 'shape_props': [200, GRAPH_ROUTES_LEN_CONST]},
            {'main_props': random_part, 'shape_props': [250, GRAPH_ROUTES_LEN_CONST]},
            {'main_props': random_part, 'shape_props': [300, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [400, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [600, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [700, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [800, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [900, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [1000, GRAPH_ROUTES_LEN_CONST]},

            # {'main_props': random_part, 'shape_props': [1000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [1500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [2000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [2500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [3000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [3500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [4000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [4500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [5000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [5500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [6000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [6500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [7000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [7500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [8000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [8500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [9000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [9500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [10000, GRAPH_ROUTES_LEN_CONST]},

            # {'main_props': random_part, 'shape_props': [1000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [1200, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [1500, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [5000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [10000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [15000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [20000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [25000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [30000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [40000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [60000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [80000, GRAPH_ROUTES_LEN_CONST]},
            #{'main_props': random_part, 'shape_props': [200000, GRAPH_ROUTES_LEN_CONST]},

            # {'main_props': random_part, 'shape_props': [150000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [200000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [250000, GRAPH_ROUTES_LEN_CONST]},
            # {'main_props': random_part, 'shape_props': [300000, GRAPH_ROUTES_LEN_CONST]},


            # WORK
            # GRAPH_ROUTES_LEN_CONST = 10000

            # 1000 агентов, numpy опускается на 3 место     # 1 -numba, 2-torch-cpu, 3 numpy , 4 torch GPU


            # GRAPH_ROUTES_LEN_CONST = 10000
            # 500-1000  агентов, numpy опускается на 3 местов       # 1 -numba, 2-torch-cpu, 3 numpy, 4 torch GPU
            # 5000 - 10000 агентов, numpy опускается на 4 место     # 1 -numba, 2-torch-cpu, 3 torch GPU, 4 numpy

        ])

    elif GRAPH_X_NAME == 'ROUTES_LEN':
        shapes_cfgs.extend([
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 5]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 10]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 15]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 20]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 25]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 30]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 40]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 60]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 80]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 100]},

            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 120]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 140]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 160]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 180]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 200]},

            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 220]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 240]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 260]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 280]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 300]},

            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 400]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 500]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 600]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 700]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 800]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 900]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 1000]},

            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 2000]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 3000]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 4000]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 10000]},

            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 20000]},

            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 30000]},

            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 100000]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 150000]},

            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 200000]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 300000]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 500000]},
            {'main_props': random_part, 'shape_props': [GRAPH_AGENTS_COUNT_CONST, 1000000]},
        ])

    return {
        'libraries': libraries,
        'shapes_cfgs': shapes_cfgs,
    }


#
#
#
def print_result__engine2d(postfix, algo, test_data, is_test_ok,
                 start_time, init_data_end_time, upload_data_end_time, calc_data_end_time, download_data_end_time, graph):

    print('#' * 80)
    print(f'{algo}')
    print(f'{postfix}')


    test_message = ", test: \033[32mOK\033[0m" if is_test_ok == True else (", \033[31mTEST ERROR !!!!!\033[0m" if is_test_ok == False else "")
    print(f'agents_count: {test_data["agents_count"]}, routes_len: {test_data["routes_len"]}{test_message}')

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

    if GRAPH_X_NAME == 'AGENTS_COUNT':
        graph['X'].append(test_data["agents_count"])
    else:
        graph['X'].append(test_data["routes_len"])

    graph['Y'].append(total_dur)
