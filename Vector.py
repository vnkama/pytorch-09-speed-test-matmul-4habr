import math
import numpy as np
import pygame as pg
#from fw.functions import *


PI=3.1415926535
PI_2 = PI*2     #PI_2
PI_d2 = PI/2

# Префикс vectm_ означает что операция ведется парралельно сразу над несколькими входными данными ссведенными

# вектор от p1 к p2
def vct_getVectorFromTwoPoints(p1, p2):
    return (p2[0]-p1[0], p2[1]-p1[1])



#угол между векторами
def vct_calcAngle(a, b):
    return (a[0] * b[0] + a[1] * b[1]) / (math.sqrt(a[0] ** 2 + a[1] ** 2) * math.sqrt(b[0] ** 2 + b[1] ** 2))

#
# вектор от точки p1 на точку p2
#
def vct_calcVectorPointToPoint_mx31(p1, p2):
    return np.array([p2[0] - p1[0], p2[1] - p1[1], 0], float)


# нормализованый вектор от точки p1 на точку p2
def vct_calcNormalizeVectorPointToPoint_f(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    len = math.sqrt(x ** 2 + y ** 2)
    return (x / len, y / len)

###########################################


# def nd2_getPoint(nd2):
#     return [nd2[0],nd2[1]]

# def nd2_getPointInt(nd2):
#     return [int(nd2[0]),int(nd2[1])]

#
# Преобразуем вектор вида touple(x,y) в матрицу
#
# третье значение в матрице 0, т.к.  матрица дает направление
#
def vct_getVectorMatrix_mx31(pos):
    return np.array([pos[0], pos[1], 0], float)

#
# ковертируем угол (в радианах) в
#
def vct_rad2NormalizeVector_mx31(alfa):
    return np.array([math.cos(alfa), math.sin(alfa), 0], float)



#=============================================================
#
# Преобразуем координату вида (x,y) в матрицу
#
# третье значение в матрице 1 если матрица дает координату
#
def vct_pointToMx_mx31(p):
    return np.array([p[0], p[1], 1], dtype="f4")


def vctm_pointToMx_mxN3(p):
    # p.shape = (N,2)
    # где N число точек преобразуемых в вектор
    A = np.empty(shape=(p.shape[0], 3), dtype="f4")
    A[:, 0:2] = p
    A[:, 2] = 1

    return A

def vctm_pointToMx_mxN31(p):
    # p.shape = (N,2)
    # где N число точек преобразуемых в вектор
    A = np.empty(shape=(p.shape[0], 3, 1), dtype="f4")
    A[:, 0:2, 0] = p[:, 0:2]
    A[:, 2, 0] = 1

    return A

#=============================================================

def vct_mx31ToIntD2(mx):
    return (round(mx[0]), round(mx[1]))

#=============================================================
#
# матрица смещения      vct_offsToMx33
#
def vct_offsToTranslateMx_mx33(offs):
    return np.array(
            [
                [1, 0, offs[0]],
                [0, 1, offs[1]],
                [0, 0, 1],
            ],
            dtype="f4"
    )

def vctm_offsToTranslateMx_mx33(offs):
    # offs - массив numpy
    # offs.shape = (N,2)
    # где N - число матриц перемещения, которые нам надо создать

    A = np.zeros(shape=(offs.shape[0], 3, 3), dtype="f4")
    A[:] = np.eye(3)
    A[:, 0:2, 2] = offs[:, 0:2][:]

    return A


#=============================================================


#
# матрица масштабирвания 2D
#
def vct_getScaleMatrix_mx33(k):
    return np.array([
            [k, 0, 0],
            [0, k, 0],
            [0, 0, 1],
        ],
        float
    )



#
# масштабируем вектор. те изменим его длинну не меняя направления
# k - коефф масшиабирования
# v - вектор
#
#
#
#          [ 1.1    0       0 ]     [35]        [38.5]
#          [ 0      1.1     0 ]  X  [25] =      [27.5]
#          [ 0      0       1 ]     [0]         [0]
#
def vct_scaleVector_mx31(v, k):
    return np.array(
        [
            [k, 0, 0],
            [0, k, 0],
            [0, 0, 1],
        ],
        float
    ) @ v



#
# преобразуем нормализованый вектор в матрицу поворота на этот вектор
#
def vct_NormalizeVector2RotateMatrix_mx33(vector):
    return np.array(
        [
            [vector[0],     -vector[1],     0],
            [vector[1],     vector[0],      0],
            [0,             0,              1],
        ],
        float,
    )

###########################################################
#
# формирует матрицу поворота на любой угол
# угол передается в радианах - alfa_radian
#
def vct_angleToRotateMx_mx33(alfa_radian):

    cos_alfa = math.cos(alfa_radian)
    sin_alfa = math.sin(alfa_radian)

    return np.array(
        [
            [cos_alfa,     -sin_alfa,       0],
            [sin_alfa,     cos_alfa,        0],
            [0,             0,              1],
        ],
        "f4",
    )


def vctm_angleToRotateMx_mx33(alfas_radian):
    # alfa_radian - массив numpy
    # alfa_radian.shape = (N)
    # где N - число матриц поворота, которые нам надо создать

    cos_alfas = np.cos(alfas_radian)
    sin_alfas = np.sin(alfas_radian)

    # еденичные матрицы
    A = np.zeros(shape=(alfas_radian.shape[0], 3, 3), dtype="f4")
    A[:] = np.eye(3)

    A[:, 0, 0] = cos_alfas[:]
    A[:, 1, 1] = cos_alfas[:]
    A[:, 0, 1] = -sin_alfas[:]
    A[:, 1, 0] = sin_alfas[:]

    return A

###########################################################

#
# матрица поворот 180
#
def vct_getRotateMatrix180_mx33():
    return np.array(
        [
            [-1,     0,     0],
            [0,     -1,     0],
            [0,     0,      1],
        ],
        float
    )


#
# поворотная матрица поворот на 90 по часовой стрелки (ось Y на экране вниз)
#
def vct_getRotateMatrixRight90_mx33():
    return np.array(
        [
            [0,     -1,     0],
            [1,     0,      0],
            [0,     0,      1],
        ],
        float
    )


#
# поворачиваем вектор на 90 по часовой стрелке
#
def vct_rotateVectorRight90_mx31(v):
    return np.array(
        [
            [0,     -1,     0],
            [1,     0,      0],
            [0,     0,      1],
        ],
        float
    ) @ v




#
# поворотная матрица поворот на 90 против часовой стрелки (Y направлена вниз)
#
def vct_getRotateMatrixLeft90_mx33():
    return np.array(
        [
            [0,     1,      0],
            [-1,    0,      0],
            [0,     0,      1],
        ],
        float
    )

#
# поворачиваем вектор на 90 против часовой стрелке (Y направлена вниз)
#
def vct_rotateVectorLeft90_mx31(v):
    return np.array(
        [
            [0,     1,     0],
            [-1,     0,      0],
            [0,     0,      1],
        ],
        float
    ) @ v


def nd2_vector_len(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2)

#
# Нормализовать вектор
#
def vct_normalizeVector(a):
    len = math.sqrt(a[0] ** 2 + a[1] ** 2)
    return (a[0] / len, a[1] / len)


def vct_normalizeVector_mx31(a):
    len = math.sqrt(a[0] ** 2 + a[1] ** 2)
    a[0] = a[0] / len
    a[1] = a[1] / len
    return a



#
# угол поворта вектора относительно оси X
# дает только положительные значения
#
# in: a [x,y] - угол поворота вектора
#
def getAngleFromPoint(a):
    if abs(a[0]) < 0.0000001:
        #x=0. находимся на оси Y
        return (PI_d2 if a[1] > 0 else -PI_d2)
    else:
        if a[0] >= 0:
            if a[1] >= 0:
                return math.atan(a[1] / a[0])
            else:
                #(a[1] < 0):
                return PI_2 + math.atan(a[1] / a[0])
        else:
            if a[1] >= 0:
                return PI  + math.atan(a[1] / a[0])
            else:
                #(a[1] < 0):
                return PI  + math.atan(a[1] / a[0])


#
# вектор минус вектор
#
def nd2_minus(v1,v2):
    return [
        v1[0]-v2[0],
        v1[1]-v2[1],
        v1[2]-v2[2]]


def vct_getLinesIntersectionPoint_mx31(p1, p2):
    #ищет точку пересечения двух прямых (НЕ отрезков)
    # a1, b1, c1 - общее уравение прямой

    # ВНИМАНИЕ. Если прямые парралельны то мы получим деление на ноль !!!!!
    # проверяем заранее

    # несокращеный код
    # matrix_D = np.array([
    #     [a1, b1],
    #     [a2, b2],
    # ])
    #
    # matrix_Dx = np.array([
    #     [-c1, b1],
    #     [-c2, b2],
    # ])
    #
    # matrix_Dy = np.array([
    #     [a1, -c1],
    #     [a2, -c2],
    # ])
    #
    # D = getDeterminant_mx22(matrix_D)
    # Dx = getDeterminant_mx22(matrix_Dx)
    # Dy = getDeterminant_mx22(matrix_Dy)
    #
    # return np.array((Dx / D, Dy / D ,1))

    # с окращеный код
    D = getDeterminant_mx22(np.array([
        [p1[0], p1[1]],
        [p2[0], p2[1]],
    ]))

    Dx = getDeterminant_mx22(np.array([
        [-p1[2], p1[1]],
        [-p2[2], p2[1]],
    ]))

    Dy = getDeterminant_mx22(np.array([
        [p1[0], -p1[2]],
        [p2[0], -p2[2]],
    ]))

    if (abs(D) < 1e-6):
        return False, None
    else:
        return True, np.array((Dx / D, Dy / D, 1))



#
# ищем точку пересчения отрезков
# return    False - точки пересечения отрезков нет
#           mx31 - координаты точки пересечения
#
def vct_caclLinesegsIntersectionPoint_mx31(ls1_p1_mx31, ls1_p2_mx31, ABC1, ls2_p1_mx31, ls2_p2_mx31, ABC2):
    # код пересечения прямых

    # проверка на коллинераность
    # TODO совпадение прямых не проверям,
    if vct_isCollinearVector(ABC1, ABC2):
        return None


    D = getDeterminant_mx22(np.array([
        [ABC1[0], ABC1[1]],
        [ABC2[0], ABC2[1]],
    ]))

    Dx = getDeterminant_mx22(np.array([
        [-ABC1[2], ABC1[1]],
        [-ABC2[2], ABC2[1]],
    ]))

    Dy = getDeterminant_mx22(np.array([
        [ABC1[0], -ABC1[2]],
        [ABC2[0], -ABC2[2]],
    ]))

    # точка пересечения прямых. НО НЕ ОТРЕЗКОВ !!
    intersect_p_mx31 = np.array((Dx / D, Dy / D, 1))


    s1 = signFloat(intersect_p_mx31[0] - ls1_p1_mx31[0])
    s2 = signFloat(intersect_p_mx31[0] - ls1_p2_mx31[0])
    ls1_x_flag = abs(s1 + s2) < 1.9

    s1 = signFloat(intersect_p_mx31[1] - ls1_p1_mx31[1])
    s2 = signFloat(intersect_p_mx31[1] - ls1_p2_mx31[1])
    ls1_y_flag = abs(s1 + s2) < 1.9

    s1 = signFloat(intersect_p_mx31[0] - ls2_p1_mx31[0])
    s2 = signFloat(intersect_p_mx31[0] - ls2_p2_mx31[0])
    ls2_x_flag = abs(s1 + s2) < 1.9

    s1 = signFloat(intersect_p_mx31[1] - ls2_p1_mx31[1])
    s2 = signFloat(intersect_p_mx31[1] - ls2_p2_mx31[1])
    ls2_y_flag = abs(s1 + s2) < 1.9

    # если пара s1 s2 имееет вид (-1 -1) или (1 1) занчит отрезки не пересекаются
    # пары s1 s2 должны быть или с разыным занком (-1 1) или (0 0) или (1 0) или (-1 0)
    # если s1 s2 Имеет вид (1 1) или (-1 -1), значит отрезки пересекаются.
    if not (ls1_x_flag and ls1_y_flag and ls2_x_flag and ls2_y_flag):
        # отрезки не пересекаются
        return None

    # отрезки пересекаются, вернем точку пересечения отрезков

    return intersect_p_mx31



#
# проверяет что два вектора коллинеарны
# считаем модуль произведения векторов
#
def nd2_getVectorLen4VectorsMult(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

def vct_isCollinearVector(v1, v2):
    return abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-6


#
# конвертируем две точки прямой в коефыфиценты прямой ABC (общее уранвение прямой)
#
def vct_2PointsToLineEquationABC(p1, p2):
    # 1я точка прямой: p1 [x1,y1]
    # 2я точка прямой: p2 [x2,y2]

    # A = p2[1] - p1[1]       # y2-y1
    # B = p1[0] - p2[0]       # x1-x2
    # C = p1[0]*(p1[1]-p2[1]) + p1[1]*(p2[0]-p1[0])   #x1(y1-y2) + y1(x2-x1)
    # return (A, B, C)

    return (
        p2[1] - p1[1],
        p1[0] - p2[0],
        p1[0] * (p1[1] - p2[1]) + p1[1] * (p2[0] - p1[0])
    )



def isLinesIntersect(start1, end1, start2, end2):
    vector1 = (end2[0] - start2[0]) * (start1[1] - start2[1]) - (end2[1] - start2[1]) * (start1[0] - start2[0])
    vector2 = (end2[0] - start2[0]) * (end1[1] - start2[1]) - (end2[1] - start2[1]) * (end1[0] - start2[0])
    vector3 = (end1[0] - start1[0]) * (start2[1] - start1[1]) - (end1[1] - start1[1]) * (start2[0] - start1[0])
    vector4 = (end1[0] - start1[0]) * (end2[1] - start1[1]) - (end1[1] - start1[1]) * (end2[0] - start1[0])
    return (vector1 * vector2 <= 0) and (vector3 * vector4 <= 0)

####################################################


# детерминант матрицы   getDeterminant_mx2
def getDeterminant_mx22(a):
    return a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]

def signFloat(f):
    return (1 if (f > 1e-6) else (-1 if (f < -1e-6) else 0))


# расстояние между двумя точками
def vct_calcDistanceBetween2Points(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

def vct_grad2rad(grad):
    return grad * 0.0174532922

def vct_rad2grad(rad):
    return rad * 57.29578049


# возращает заготовку для pgrect
def vct_getWrapperRect(p1_d2, p2_d2):
    x_min = min(p1_d2[0], p2_d2[0])
    x_max = max(p1_d2[0], p2_d2[0])
    y_min = min(p1_d2[1], p2_d2[1])
    y_max = max(p1_d2[1], p2_d2[1])

    return pg.Rect(
        x_min - 2,
        y_min - 2,
        x_max - x_min + 3,
        y_max - y_min + 3,
    )
