import torch
import random
import trimesh
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from ppp_sam.Module.timer import Timer


def normalize_pc(pc):
    """
    pc: (N, 3)
    """
    max_, min_ = np.max(pc, axis=0), np.min(pc, axis=0)
    center = (max_ + min_) / 2
    scale = (max_ - min_) / 2
    scale = np.max(np.abs(scale))
    pc = (pc - center) / (scale + 1e-10)
    return pc


@torch.no_grad()
def get_feat(model, points, normals):
    data_dict = {
        "coord": points,
        "normal": normals,
        "color": np.ones_like(points),
        "batch": np.zeros(points.shape[0], dtype=np.int64),
    }
    data_dict = model.transform(data_dict)
    for k in data_dict:
        if isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k].cuda()
    point = model.sonata(data_dict)
    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    feat = point.feat  # [M, 1232]
    feat = model.mlp(feat)  # [M, 512]
    feat = feat[point.inverse]  # [N, 512]
    feats = feat
    return feats


@torch.no_grad()
def get_mask(model, feats, points, point_prompt, iter=1):
    """
    feats: [N, 512]
    points: [N, 3]
    point_prompt: [K, 3]
    """
    point_num = points.shape[0]
    prompt_num = point_prompt.shape[0]
    feats = feats.unsqueeze(1)  # [N, 1, 512]
    feats = feats.repeat(1, prompt_num, 1).cuda()  # [N, K, 512]
    points = torch.from_numpy(points).float().cuda().unsqueeze(1)  # [N, 1, 3]
    points = points.repeat(1, prompt_num, 1)  # [N, K, 3]
    prompt_coord = (
        torch.from_numpy(point_prompt).float().cuda().unsqueeze(0)
    )  # [1, K, 3]
    prompt_coord = prompt_coord.repeat(point_num, 1, 1)  # [N, K, 3]

    feats = feats.transpose(0, 1)  # [K, N, 512]
    points = points.transpose(0, 1)  # [K, N, 3]
    prompt_coord = prompt_coord.transpose(0, 1)  # [K, N, 3]

    mask_1, mask_2, mask_3, pred_iou = model(feats, points, prompt_coord, iter)

    mask_1 = mask_1.transpose(0, 1)  # [N, K]
    mask_2 = mask_2.transpose(0, 1)  # [N, K]
    mask_3 = mask_3.transpose(0, 1)  # [N, K]

    mask_1 = mask_1.detach().cpu().numpy() > 0.5
    mask_2 = mask_2.detach().cpu().numpy() > 0.5
    mask_3 = mask_3.detach().cpu().numpy() > 0.5

    org_iou = pred_iou.detach().cpu().numpy()  # [K, 3]

    return mask_1, mask_2, mask_3, org_iou


def cal_iou(m1, m2):
    return np.sum(np.logical_and(m1, m2)) / np.sum(np.logical_or(m1, m2))


def cal_single_iou(m1, m2):
    return np.sum(np.logical_and(m1, m2)) / np.sum(m1)


def iou_3d(box1, box2, signle=None):
    """
    计算两个三维边界框的交并比 (IoU)

    参数:
        box1 (list): 第一个边界框的坐标 [x1_min, y1_min, z1_min, x1_max, y1_max, z1_max]
        box2 (list): 第二个边界框的坐标 [x2_min, y2_min, z2_min, x2_max, y2_max, z2_max]

    返回:
        float: 交并比 (IoU) 值
    """
    # 计算交集的坐标
    intersection_xmin = max(box1[0], box2[0])
    intersection_ymin = max(box1[1], box2[1])
    intersection_zmin = max(box1[2], box2[2])
    intersection_xmax = min(box1[3], box2[3])
    intersection_ymax = min(box1[4], box2[4])
    intersection_zmax = min(box1[5], box2[5])

    # 判断是否有交集
    if (
        intersection_xmin >= intersection_xmax
        or intersection_ymin >= intersection_ymax
        or intersection_zmin >= intersection_zmax
    ):
        return 0.0  # 无交集

    # 计算交集的体积
    intersection_volume = (
        (intersection_xmax - intersection_xmin)
        * (intersection_ymax - intersection_ymin)
        * (intersection_zmax - intersection_zmin)
    )

    # 计算两个盒子的体积
    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])

    if signle is None:
        # 计算并集的体积
        union_volume = box1_volume + box2_volume - intersection_volume
    elif signle == "1":
        union_volume = box1_volume
    elif signle == "2":
        union_volume = box2_volume
    else:
        raise ValueError("signle must be None or 1 or 2")

    # 计算 IoU
    iou = intersection_volume / union_volume if union_volume > 0 else 0.0
    return iou


def cal_point_bbox_iou(p1, p2, signle=None):
    min_p1 = np.min(p1, axis=0)
    max_p1 = np.max(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p2 = np.max(p2, axis=0)
    box1 = [min_p1[0], min_p1[1], min_p1[2], max_p1[0], max_p1[1], max_p1[2]]
    box2 = [min_p2[0], min_p2[1], min_p2[2], max_p2[0], max_p2[1], max_p2[2]]
    return iou_3d(box1, box2, signle)


def cal_bbox_iou(points, m1, m2):
    p1 = points[m1]
    p2 = points[m2]
    return cal_point_bbox_iou(p1, p2)


def clean_mesh(mesh):
    """
    mesh: trimesh.Trimesh
    """
    # 1. 合并接近的顶点
    mesh.merge_vertices()

    # 2. 删除重复的顶点
    # 3. 删除重复的面片
    mesh.process(True)
    return mesh


def remove_outliers_iqr(data, factor=1.5):
    """
    基于 IQR 去除离群值
    :param data: 输入的列表或 NumPy 数组
    :param factor: IQR 的倍数（默认 1.5）
    :return: 去除离群值后的列表
    """
    data = np.array(data, dtype=np.float32)
    q1 = np.percentile(data, 25)  # 第一四分位数
    q3 = np.percentile(data, 75)  # 第三四分位数
    iqr = q3 - q1  # 四分位距
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)].tolist()


def better_aabb(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x = remove_outliers_iqr(x)
    y = remove_outliers_iqr(y)
    z = remove_outliers_iqr(z)
    min_xyz = np.array([np.min(x), np.min(y), np.min(z)])
    max_xyz = np.array([np.max(x), np.max(y), np.max(z)])
    return [min_xyz, max_xyz]


def fix_label(face_ids, adjacent_faces, use_aabb=False, mesh=None, show_info=False):
    if use_aabb:

        def _cal_aabb(face_ids, i, _points_org):
            _part_mask = face_ids == i
            _faces = mesh.faces[_part_mask]
            _faces = np.reshape(_faces, (-1))
            _points = mesh.vertices[_faces]
            min_xyz, max_xyz = better_aabb(_points)
            _part_mask = (
                (_points_org[:, 0] >= min_xyz[0])
                & (_points_org[:, 0] <= max_xyz[0])
                & (_points_org[:, 1] >= min_xyz[1])
                & (_points_org[:, 1] <= max_xyz[1])
                & (_points_org[:, 2] >= min_xyz[2])
                & (_points_org[:, 2] <= max_xyz[2])
            )
            _part_mask = np.reshape(_part_mask, (-1, 3))
            _part_mask = np.all(_part_mask, axis=1)
            return i, [min_xyz, max_xyz], _part_mask

        with Timer("计算aabb"):
            aabb = {}
            unique_ids = np.unique(face_ids)
            # print(max(unique_ids))
            aabb_face_mask = {}
            _faces = mesh.faces
            _vertices = mesh.vertices
            _faces = np.reshape(_faces, (-1))
            _points = _vertices[_faces]
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = []
                for i in unique_ids:
                    if i < 0:
                        continue
                    futures.append(executor.submit(_cal_aabb, face_ids, i, _points))
                for future in futures:
                    res = future.result()
                    aabb[res[0]] = res[1]
                    aabb_face_mask[res[0]] = res[2]

            # _faces = mesh.faces
            # _vertices = mesh.vertices
            # _faces = np.reshape(_faces, (-1))
            # _points = _vertices[_faces]
            # aabb_face_mask = cal_aabb_mask(_points, face_ids)

    with Timer("合并mesh"):
        loop_cnt = 1
        changed = True
        progress = tqdm(disable=not show_info)
        no_mask_ids = np.where(face_ids < 0)[0].tolist()
        faces_max = adjacent_faces.shape[0]
        while changed and loop_cnt <= 50:
            changed = False
            # 获取无色面片
            new_no_mask_ids = []
            for i in no_mask_ids:
                # if face_ids[i] < 0:
                # 找邻居
                if not (0 <= i < faces_max):
                    continue
                _adj_faces = adjacent_faces[i]
                _adj_ids = []
                for j in _adj_faces:
                    if j == -1:
                        break
                    if face_ids[j] >= 0:
                        _tar_id = face_ids[j]
                        if use_aabb:
                            _mask = aabb_face_mask[_tar_id]
                            if _mask[i]:
                                _adj_ids.append(_tar_id)
                        else:
                            _adj_ids.append(_tar_id)
                if len(_adj_ids) == 0:
                    new_no_mask_ids.append(i)
                    continue
                _max_id = np.argmax(np.bincount(_adj_ids))
                face_ids[i] = _max_id
                changed = True
            no_mask_ids = new_no_mask_ids
            # print(loop_cnt)
            progress.update(1)
            # progress.set_description(f"合并mesh循环：{loop_cnt} {np.sum(face_ids < 0)}")
            loop_cnt += 1
    return face_ids


def save_mesh(save_path, mesh, face_ids, color_map):
    face_colors = np.zeros((len(mesh.faces), 3), dtype=np.uint8)
    for i in tqdm(range(len(mesh.faces)), disable=True):
        _max_id = face_ids[i]
        if _max_id == -2:
            continue
        face_colors[i, :3] = color_map[_max_id]

    mesh_save = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    mesh_save.visual.face_colors = face_colors
    mesh_save.export(save_path)
    mesh_save.export(save_path.replace(".glb", ".ply"))
    # print('保存mesh完成')

    scene_mesh = trimesh.Scene()
    scene_mesh.add_geometry(mesh_save)
    unique_ids = np.unique(face_ids)
    aabb = []
    for i in unique_ids:
        if i == -1 or i == -2:
            continue
        _part_mask = face_ids == i
        _faces = mesh.faces[_part_mask]
        _faces = np.reshape(_faces, (-1))
        _points = mesh.vertices[_faces]
        min_xyz, max_xyz = better_aabb(_points)
        center = (min_xyz + max_xyz) / 2
        size = max_xyz - min_xyz
        box = trimesh.path.creation.box_outline()
        box.vertices *= size
        box.vertices += center
        box_color = np.array([[color_map[i][0], color_map[i][1], color_map[i][2], 255]])
        box_color = np.repeat(box_color, len(box.entities), axis=0).astype(np.uint8)
        box.colors = box_color
        scene_mesh.add_geometry(box)
        min_xyz = np.min(_points, axis=0)
        max_xyz = np.max(_points, axis=0)
        aabb.append([min_xyz, max_xyz])
    scene_mesh.export(save_path.replace(".glb", "_aabb.glb"))
    aabb = np.array(aabb)
    np.save(save_path.replace(".glb", "_aabb.npy"), aabb)
    np.save(save_path.replace(".glb", "_face_ids.npy"), face_ids)


def get_aabb_from_face_ids(mesh, face_ids):
    unique_ids = np.unique(face_ids)
    aabb = []
    for i in unique_ids:
        if i == -1 or i == -2:
            continue
        _part_mask = face_ids == i
        _faces = mesh.faces[_part_mask]
        _faces = np.reshape(_faces, (-1))
        _points = mesh.vertices[_faces]
        min_xyz = np.min(_points, axis=0)
        max_xyz = np.max(_points, axis=0)
        aabb.append([min_xyz, max_xyz])
    return np.array(aabb)


def calculate_face_areas(mesh):
    """
    计算每个三角形面片的面积
    :param mesh: trimesh.Trimesh 对象
    :return: 面片面积数组 (n_faces,)
    """
    return mesh.area_faces
    # # 提取顶点和面片索引
    # vertices = mesh.vertices
    # faces = mesh.faces

    # # 获取所有三个顶点的坐标
    # v0 = vertices[faces[:, 0]]
    # v1 = vertices[faces[:, 1]]
    # v2 = vertices[faces[:, 2]]

    # # 计算两个边向量
    # edge1 = v1 - v0
    # edge2 = v2 - v0

    # # 计算叉积的模长（向量面积的两倍）
    # cross_product = np.cross(edge1, edge2)
    # areas = 0.5 * np.linalg.norm(cross_product, axis=1)

    # return areas


def get_connected_region(face_ids, adjacent_faces, return_face_part_ids=False):
    vis = [False] * len(face_ids)
    parts = []
    face_part_ids = np.ones_like(face_ids) * -1
    for i in range(len(face_ids)):
        if vis[i]:
            continue
        _part = []
        _queue = [i]
        while len(_queue) > 0:
            _cur_face = _queue.pop(0)
            if vis[_cur_face]:
                continue
            vis[_cur_face] = True
            _part.append(_cur_face)
            face_part_ids[_cur_face] = len(parts)
            if not (0 <= _cur_face < adjacent_faces.shape[0]):
                continue
            _cur_face_id = face_ids[_cur_face]
            _adj_faces = adjacent_faces[_cur_face]
            for j in _adj_faces:
                if j == -1:
                    break
                if not vis[j] and face_ids[j] == _cur_face_id:
                    _queue.append(j)
        parts.append(_part)
    if return_face_part_ids:
        return parts, face_part_ids
    else:
        return parts


def aabb_distance(box1, box2):
    """
    计算两个轴对齐包围盒（AABB）之间的最近距离。
    :param box1: 元组 (min_x, min_y, min_z, max_x, max_y, max_z)
    :param box2: 元组 (min_x, min_y, min_z, max_x, max_y, max_z)
    :return: 最近距离（浮点数）
    """
    # 解包坐标
    min1, max1 = box1
    min2, max2 = box2

    # 计算各轴上的分离距离
    dx = max(0, max2[0] - min1[0], max1[0] - min2[0])  # x轴分离距离
    dy = max(0, max2[1] - min1[1], max1[1] - min2[1])  # y轴分离距离
    dz = max(0, max2[2] - min1[2], max1[2] - min2[2])  # z轴分离距离

    # 如果所有轴都重叠，则距离为0
    if dx == 0 and dy == 0 and dz == 0:
        return 0.0

    # 计算欧几里得距离
    return np.sqrt(dx**2 + dy**2 + dz**2)


def aabb_volume(aabb):
    """
    计算轴对齐包围盒（AABB）的体积。
    :param aabb: 元组 (min_x, min_y, min_z, max_x, max_y, max_z)
    :return: 体积（浮点数）
    """
    # 解包坐标
    min_xyz, max_xyz = aabb

    # 计算体积
    dx = max_xyz[0] - min_xyz[0]
    dy = max_xyz[1] - min_xyz[1]
    dz = max_xyz[2] - min_xyz[2]
    return dx * dy * dz


def find_neighbor_part(parts, adjacent_faces, parts_aabb=None, parts_ids=None):
    face2part = {}
    for i, part in enumerate(parts):
        for face in part:
            face2part[face] = i
    neighbor_parts = []
    for i, part in enumerate(parts):
        neighbor_part = set()
        for face in part:
            if not (0 <= face < adjacent_faces.shape[0]):
                continue
            for adj_face in adjacent_faces[face]:
                if adj_face == -1:
                    break
                if adj_face not in face2part:
                    continue
                if face2part[adj_face] == i:
                    continue
                if parts_ids is not None and parts_ids[face2part[adj_face]] in [-1, -2]:
                    continue
                neighbor_part.add(face2part[adj_face])
        neighbor_part = list(neighbor_part)
        if (
            parts_aabb is not None
            and parts_ids is not None
            and (parts_ids[i] == -1 or parts_ids[i] == -2)
            and len(neighbor_part) == 0
        ):
            min_dis = np.inf
            min_idx = -1
            for j, _part in tqdm(enumerate(parts)):
                if j == i:
                    continue
                if parts_ids[j] == -1 or parts_ids[j] == -2:
                    continue
                aabb_1 = parts_aabb[i]
                aabb_2 = parts_aabb[j]
                dis = aabb_distance(aabb_1, aabb_2)
                if dis < min_dis:
                    min_dis = dis
                    min_idx = j
                elif dis == min_dis:
                    if aabb_volume(parts_aabb[j]) < aabb_volume(parts_aabb[min_idx]):
                        min_idx = j
            neighbor_part = [min_idx]
        neighbor_parts.append(neighbor_part)
    return neighbor_parts


def do_post_process(
    face_areas, parts, adjacent_faces, face_ids, threshold=0.95, show_info=False
):
    # # 获取邻接面片
    # mesh_save = mesh.copy()
    # face_adjacency = mesh.face_adjacency
    # adjacent_faces = {}
    # for face1, face2 in face_adjacency:
    #     if face1 not in adjacent_faces:
    #         adjacent_faces[face1] = []
    #     if face2 not in adjacent_faces:
    #         adjacent_faces[face2] = []
    #     adjacent_faces[face1].append(face2)
    #     adjacent_faces[face2].append(face1)

    # parts = get_connected_region(face_ids, adjacent_faces)

    unique_ids = np.unique(face_ids)
    if show_info:
        print(f"连通区域数量：{len(parts)}")
        print(f"ID数量：{len(unique_ids)}")

    # face_areas = calculate_face_areas(mesh)
    total_area = np.sum(face_areas)
    if show_info:
        print(f"总面积：{total_area}")
    part_areas = []
    for i, part in enumerate(parts):
        part_area = np.sum(face_areas[part])
        part_areas.append(float(part_area / total_area))

    sorted_parts = sorted(zip(part_areas, parts), key=lambda x: x[0], reverse=True)
    parts = [x[1] for x in sorted_parts]
    part_areas = [x[0] for x in sorted_parts]
    integral_part_areas = np.cumsum(part_areas)

    neighbor_parts = find_neighbor_part(parts, adjacent_faces)

    new_face_ids = face_ids.copy()

    for i, part in enumerate(parts):
        if integral_part_areas[i] > threshold and part_areas[i] < 0.01:
            if len(neighbor_parts[i]) > 0:
                max_area = 0
                max_part = -1
                for j in neighbor_parts[i]:
                    if integral_part_areas[j] > threshold:
                        continue
                    if part_areas[j] > max_area:
                        max_area = part_areas[j]
                        max_part = j
                if max_part != -1:
                    if show_info:
                        print(f"合并mesh：{i} {max_part}")
                    parts[max_part].extend(part)
                    parts[i] = []
                    target_face_id = face_ids[parts[max_part][0]]
                    for face in part:
                        new_face_ids[face] = target_face_id

    return new_face_ids


def do_no_mask_process(parts, face_ids):
    # # 获取邻接面片
    # mesh_save = mesh.copy()
    # face_adjacency = mesh.face_adjacency
    # adjacent_faces = {}
    # for face1, face2 in face_adjacency:
    #     if face1 not in adjacent_faces:
    #         adjacent_faces[face1] = []
    #     if face2 not in adjacent_faces:
    #         adjacent_faces[face2] = []
    #     adjacent_faces[face1].append(face2)
    #     adjacent_faces[face2].append(face1)
    # parts = get_connected_region(face_ids, adjacent_faces)

    unique_ids = np.unique(face_ids)
    max_id = np.max(unique_ids)
    if -1 or -2 in unique_ids:
        new_face_ids = face_ids.copy()
        for i, part in enumerate(parts):
            if face_ids[part[0]] == -1 or face_ids[part[0]] == -2:
                for face in part:
                    new_face_ids[face] = max_id + 1
                max_id += 1
        return new_face_ids
    else:
        return face_ids


def union_aabb(aabb1, aabb2):
    min_xyz1 = aabb1[0]
    max_xyz1 = aabb1[1]
    min_xyz2 = aabb2[0]
    max_xyz2 = aabb2[1]
    min_xyz = np.minimum(min_xyz1, min_xyz2)
    max_xyz = np.maximum(max_xyz1, max_xyz2)
    return [min_xyz, max_xyz]


def aabb_increase(aabb1, aabb2):
    min_xyz_before = aabb1[0]
    max_xyz_before = aabb1[1]
    min_xyz_after, max_xyz_after = union_aabb(aabb1, aabb2)
    min_xyz_increase = np.abs(min_xyz_after - min_xyz_before) / np.abs(min_xyz_before)
    max_xyz_increase = np.abs(max_xyz_after - max_xyz_before) / np.abs(max_xyz_before)
    return min_xyz_increase, max_xyz_increase


def sort_multi_list(multi_list, key=lambda x: x[0], reverse=False):
    """
    multi_list: [list1, list2, list3, list4, ...], len(list1)=N, len(list2)=N, len(list3)=N, ...
    key: 排序函数，默认按第一个元素排序
    reverse: 排序顺序，默认降序
    return:
        [list1, list2, list3, list4, ...]: 按同一个顺序排序后的多个list
    """
    sorted_list = sorted(zip(*multi_list), key=key, reverse=reverse)
    return zip(*sorted_list)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
