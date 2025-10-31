import numpy as np
from numba import njit


###################### NUMBA 加速 ######################
@njit
def build_adjacent_faces_numba(face_adjacency):
    """
    使用 Numba 加速构建邻接面片数组。
    :param face_adjacency: (N, 2) numpy 数组，包含邻接面片对。
    :return:
        - adj_list: 一维数组，存储所有邻接面片。
        - offsets: 一维数组，记录每个面片的邻接起始位置。
    """
    n_faces = np.max(face_adjacency) + 1  # 总面片数
    n_edges = face_adjacency.shape[0]  # 总邻接边数

    # 第一步：统计每个面片的邻接数量（度数）
    degrees = np.zeros(n_faces, dtype=np.int32)
    for i in range(n_edges):
        f1, f2 = face_adjacency[i]
        degrees[f1] += 1
        degrees[f2] += 1
    max_degree = np.max(degrees)  # 最大度数

    adjacent_faces = np.ones((n_faces, max_degree), dtype=np.int32) * -1  # 邻接面片数组
    adjacent_faces_count = np.zeros(n_faces, dtype=np.int32)  # 邻接面片计数器
    for i in range(n_edges):
        f1, f2 = face_adjacency[i]
        adjacent_faces[f1, adjacent_faces_count[f1]] = f2
        adjacent_faces_count[f1] += 1
        adjacent_faces[f2, adjacent_faces_count[f2]] = f1
        adjacent_faces_count[f2] += 1
    return adjacent_faces


###################### NUMBA 加速 ######################
