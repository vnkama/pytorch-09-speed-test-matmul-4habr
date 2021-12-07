import torch as t


def vctt_pointsToMatrix_mxN31(points_mxN2):
    # p.shape = (N,2)
    # где N число точек преобразуемых в вектор
    A = t.empty(
            size=(points_mxN2.shape[0], 3, 1),
            dtype=t.float32,
            device=points_mxN2.device
    )
    A[:, 0:2, 0] = points_mxN2[:, 0:2]
    A[:, 2, 0] = 1

    return A

def vctt_offsToTranslateMx_mxN33(offs_tensN2):
    # offs - массив torch
    # offs.shape = (N,2)
    # где N - число матриц перемещения, которые нам надо создать

    A = t.zeros(
            size=(offs_tensN2.shape[0], 3, 3),
            dtype=t.float32,
            device=offs_tensN2.device,
    )
    #A[:] = t.eye(3, dtype=t.float32)
    A[:, 0:2, 2] = offs_tensN2[:, 0:2][:]
    A[:, 0, 0] = 1
    A[:, 1, 1] = 1
    A[:, 2, 2] = 1

    return A

def vctt_anglesToRotateMx_mxN33(angles):
    # alfa_radian - массив numpy
    # alfa_radian.shape = (N)
    # где N - число матриц поворота, которые нам надо создать

    cos_alfas = t.cos(angles)
    sin_alfas = t.sin(angles)

    # еденичные матрицы
    A = t.empty(
            size=(angles.shape[0], 3, 3),
            dtype=t.float32,
            device=angles.device,
    )
    A[:] = t.eye(3)

    A[:, 0, 0] = cos_alfas[:]
    A[:, 1, 1] = cos_alfas[:]
    A[:, 0, 1] = -sin_alfas[:]
    A[:, 1, 0] = sin_alfas[:]

    return A
