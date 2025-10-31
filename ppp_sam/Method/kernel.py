import os
import copy
import shutil
import trimesh
import fpsample
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from ppp_sam.Method.utils import *
from ppp_sam.Method.numba import build_adjacent_faces_numba
from ppp_sam.Module.timer import Timer


def mesh_sam(
    model,
    mesh,
    save_path,
    point_num=100000,
    prompt_num=400,
    save_mid_res=False,
    show_info=False,
    post_process=False,
    threshold=0.95,
    clean_mesh_flag=True,
    seed=42,
    prompt_bs=32,
):
    with Timer("加载mesh"):
        model, model_parallel = model
        if clean_mesh_flag:
            mesh = clean_mesh(mesh)
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    if show_info:
        print(f"点数：{mesh.vertices.shape[0]} 面片数：{mesh.faces.shape[0]}")

    point_num = 100000
    prompt_num = 400
    with Timer("获取邻接面片"):
        face_adjacency = mesh.face_adjacency
    with Timer("处理邻接面片"):
        adjacent_faces = build_adjacent_faces_numba(face_adjacency)

    with Timer("采样点云"):
        _points, face_idx = trimesh.sample.sample_surface(mesh, point_num, seed=seed)
        _points_org = _points.copy()
        _points = normalize_pc(_points)
        normals = mesh.face_normals[face_idx]
    if show_info:
        print(f"点数：{point_num} 面片数：{mesh.faces.shape[0]}")

    with Timer("获取特征"):
        _feats = get_feat(model, _points, normals)
    if show_info:
        print("预处理特征")

    if save_mid_res:
        feat_save = _feats.float().detach().cpu().numpy()
        data_scaled = feat_save / np.linalg.norm(feat_save, axis=-1, keepdims=True)
        pca = PCA(n_components=3)
        data_reduced = pca.fit_transform(data_scaled)
        data_reduced = (data_reduced - data_reduced.min()) / (
            data_reduced.max() - data_reduced.min()
        )
        _colors_pca = (data_reduced * 255).astype(np.uint8)
        pc_save = trimesh.points.PointCloud(_points, colors=_colors_pca)
        pc_save.export(os.path.join(save_path, "point_pca.glb"))
        pc_save.export(os.path.join(save_path, "point_pca.ply"))
        if show_info:
            print("PCA获取特征颜色")

    with Timer("FPS采样提示点"):
        fps_idx = fpsample.fps_sampling(_points, prompt_num)
        _point_prompts = _points[fps_idx]
    if save_mid_res:
        trimesh.points.PointCloud(_point_prompts, colors=_colors_pca[fps_idx]).export(
            os.path.join(save_path, "point_prompts_pca.glb")
        )
        trimesh.points.PointCloud(_point_prompts, colors=_colors_pca[fps_idx]).export(
            os.path.join(save_path, "point_prompts_pca.ply")
        )
    if show_info:
        print("采样完成")

    with Timer("推理"):
        bs = prompt_bs
        step_num = prompt_num // bs + 1
        mask_res = []
        iou_res = []
        for i in tqdm(range(step_num), disable=not show_info):
            cur_propmt = _point_prompts[bs * i : bs * (i + 1)]
            pred_mask_1, pred_mask_2, pred_mask_3, pred_iou = get_mask(
                model_parallel, _feats, _points, cur_propmt
            )
            pred_mask = np.stack(
                [pred_mask_1, pred_mask_2, pred_mask_3], axis=-1
            )  # [N, K, 3]
            max_idx = np.argmax(pred_iou, axis=-1)  # [K]
            for j in range(max_idx.shape[0]):
                mask_res.append(pred_mask[:, j, max_idx[j]])
                iou_res.append(pred_iou[j, max_idx[j]])
    mask_res = np.stack(mask_res, axis=-1)  # [N, K]
    if show_info:
        print("prmopt 推理完成")

    with Timer("根据IOU排序"):
        iou_res = np.array(iou_res).tolist()
        mask_iou = [[mask_res[:, i], iou_res[i]] for i in range(prompt_num)]
        mask_iou_sorted = sorted(mask_iou, key=lambda x: x[1], reverse=True)
        mask_sorted = [mask_iou_sorted[i][0] for i in range(prompt_num)]
        iou_sorted = [mask_iou_sorted[i][1] for i in range(prompt_num)]

    with Timer("NMS"):
        clusters = defaultdict(list)
        with ThreadPoolExecutor(max_workers=20) as executor:
            for i in tqdm(range(prompt_num), desc="NMS", disable=not show_info):
                _mask = mask_sorted[i]
                futures = []
                for j in clusters.keys():
                    futures.append(executor.submit(cal_iou, _mask, mask_sorted[j]))

                for j, future in zip(clusters.keys(), futures):
                    if future.result() > 0.9:
                        clusters[j].append(i)
                        break
                else:
                    clusters[i].append(i)

    if show_info:
        print(f"NMS完成，mask数量：{len(clusters)}")

    if save_mid_res:
        part_mask_save_path = os.path.join(save_path, "part_mask")
        if os.path.exists(part_mask_save_path):
            shutil.rmtree(part_mask_save_path)
        os.makedirs(part_mask_save_path, exist_ok=True)
        for i in tqdm(clusters.keys(), desc="保存mask", disable=not show_info):
            cluster_num = len(clusters[i])
            cluster_iou = iou_sorted[i]
            cluster_area = np.sum(mask_sorted[i])
            if cluster_num <= 2:
                continue
            mask_save = mask_sorted[i]
            mask_save = np.expand_dims(mask_save, axis=-1)
            mask_save = np.repeat(mask_save, 3, axis=-1)
            mask_save = (mask_save * 255).astype(np.uint8)
            point_save = trimesh.points.PointCloud(_points, colors=mask_save)
            point_save.export(
                os.path.join(
                    part_mask_save_path,
                    f"mask_{i}_iou_{cluster_iou:.5f}_area_{cluster_area:.5f}_num_{cluster_num}.glb",
                )
            )

    # 过滤只有一个mask的cluster
    with Timer("过滤只有一个mask的cluster"):
        filtered_clusters = []
        other_clusters = []
        for i in clusters.keys():
            if len(clusters[i]) > 2:
                filtered_clusters.append(i)
            else:
                other_clusters.append(i)
    if show_info:
        print(
            f"过滤前：{len(clusters)} 个cluster，"
            f"过滤后：{len(filtered_clusters)} 个cluster"
        )

    # 再次合并
    with Timer("再次合并"):
        filtered_clusters_num = len(filtered_clusters)
        cluster2 = {}
        is_union = [False] * filtered_clusters_num
        for i in range(filtered_clusters_num):
            if is_union[i]:
                continue
            cur_cluster = filtered_clusters[i]
            cluster2[cur_cluster] = [cur_cluster]
            for j in range(i + 1, filtered_clusters_num):
                if is_union[j]:
                    continue
                tar_cluster = filtered_clusters[j]
                if (
                    cal_bbox_iou(
                        _points, mask_sorted[tar_cluster], mask_sorted[cur_cluster]
                    )
                    > 0.5
                ):
                    cluster2[cur_cluster].append(tar_cluster)
                    is_union[j] = True
    if show_info:
        print(f"再次合并，合并数量：{len(cluster2.keys())}")

    with Timer("计算没有mask的点"):
        no_mask = np.ones(point_num)
        for i in cluster2:
            part_mask = mask_sorted[i]
            no_mask[part_mask] = 0
    if show_info:
        print(
            f"{np.sum(no_mask == 1)} 个点没有mask,"
            f" 占比：{np.sum(no_mask == 1) / point_num:.4f}"
        )

    with Timer("修补遗漏mask"):
        # 查询漏掉的mask
        for i in tqdm(range(len(mask_sorted)), desc="漏掉mask", disable=not show_info):
            if i in cluster2:
                continue
            part_mask = mask_sorted[i]
            _iou = cal_single_iou(part_mask, no_mask)
            if _iou > 0.7:
                cluster2[i] = [i]
                no_mask[part_mask] = 0
                if save_mid_res:
                    mask_save = mask_sorted[i]
                    mask_save = np.expand_dims(mask_save, axis=-1)
                    mask_save = np.repeat(mask_save, 3, axis=-1)
                    mask_save = (mask_save * 255).astype(np.uint8)
                    point_save = trimesh.points.PointCloud(_points, colors=mask_save)
                    cluster_iou = iou_sorted[i]
                    cluster_area = int(np.sum(mask_sorted[i]))
                    cluster_num = 1
                    point_save.export(
                        os.path.join(
                            part_mask_save_path,
                            f"mask_{i}_iou_{cluster_iou:.5f}_area_{cluster_area:.5f}_num_{cluster_num}.glb",
                        )
                    )
    if show_info:
        print(f"修补遗漏mask：{len(cluster2.keys())}")

    with Timer("计算点云最终mask"):
        final_mask = list(cluster2.keys())
        final_mask_area = [int(np.sum(mask_sorted[i])) for i in final_mask]
        final_mask_area = [
            [final_mask[i], final_mask_area[i]] for i in range(len(final_mask))
        ]
        final_mask_area_sorted = sorted(
            final_mask_area, key=lambda x: x[1], reverse=True
        )
        final_mask_sorted = [
            final_mask_area_sorted[i][0] for i in range(len(final_mask_area))
        ]
        final_mask_area_sorted = [
            final_mask_area_sorted[i][1] for i in range(len(final_mask_area))
        ]
    if show_info:
        print(f"最终mask数量：{len(final_mask_sorted)}")

    with Timer("点云上色"):
        # 生成color map
        color_map = {}
        for i in final_mask_sorted:
            part_color = np.random.rand(3) * 255
            color_map[i] = part_color
        # print(color_map)

        result_mask = -np.ones(point_num, dtype=np.int64)
        for i in final_mask_sorted:
            part_mask = mask_sorted[i]
            result_mask[part_mask] = i
    if save_mid_res:
        # 保存点云结果
        result_colors = np.zeros_like(_colors_pca)
        for i in final_mask_sorted:
            part_color = color_map[i]
            part_mask = mask_sorted[i]
            result_colors[part_mask, :3] = part_color
        trimesh.points.PointCloud(_points, colors=result_colors).export(
            os.path.join(save_path, "auto_mask_cluster.glb")
        )
        trimesh.points.PointCloud(_points, colors=result_colors).export(
            os.path.join(save_path, "auto_mask_cluster.ply")
        )
        if show_info:
            print("保存点云完成")

    with Timer("投影Mesh并统计label"):
        # 保存mesh结果
        face_seg_res = {}
        for i in final_mask_sorted:
            _part_mask = result_mask == i
            _face_idx = face_idx[_part_mask]
            for k in _face_idx:
                if k not in face_seg_res:
                    face_seg_res[k] = []
                face_seg_res[k].append(i)
        _part_mask = result_mask == -1
        _face_idx = face_idx[_part_mask]
        for k in _face_idx:
            if k not in face_seg_res:
                face_seg_res[k] = []
            face_seg_res[k].append(-1)

        face_ids = -np.ones(len(mesh.faces), dtype=np.int64) * 2
        for i in tqdm(face_seg_res, leave=False, disable=True):
            _seg_ids = np.array(face_seg_res[i])
            # 获取最多的seg_id
            _max_id = np.argmax(np.bincount(_seg_ids + 2)) - 2
            face_ids[i] = _max_id
        face_ids_org = face_ids.copy()
    if show_info:
        print("生成face_ids完成")

    with Timer("第一次修复face_ids"):
        face_ids += 1
        face_ids = fix_label(face_ids, adjacent_faces, mesh=mesh, show_info=show_info)
        face_ids -= 1
    if show_info:
        print("修复face_ids完成")

    color_map[-1] = np.array([255, 0, 0], dtype=np.uint8)

    if save_mid_res:
        save_mesh(
            os.path.join(save_path, "auto_mask_mesh.glb"), mesh, face_ids, color_map
        )
        save_mesh(
            os.path.join(save_path, "auto_mask_mesh_org.glb"),
            mesh,
            face_ids_org,
            color_map,
        )
        if show_info:
            print("保存mesh结果完成")

    with Timer("计算连通区域"):
        face_areas = calculate_face_areas(mesh)
        mesh_total_area = np.sum(face_areas)
        parts = get_connected_region(face_ids, adjacent_faces)
        connected_parts, _face_connected_parts_ids = get_connected_region(
            np.ones_like(face_ids), adjacent_faces, return_face_part_ids=True
        )
    if show_info:
        print(f"共{len(parts)}个mesh")
    with Timer("排序连通区域"):
        parts_cp_idx = []
        for x in parts:
            _face_idx = x[0]
            parts_cp_idx.append(_face_connected_parts_ids[_face_idx])
        parts_cp_idx = np.array(parts_cp_idx)
        parts_areas = [float(np.sum(face_areas[x])) for x in parts]
        connected_parts_areas = [float(np.sum(face_areas[x])) for x in connected_parts]
        parts_cp_areas = [connected_parts_areas[x] for x in parts_cp_idx]
        parts_sorted, parts_areas_sorted, parts_cp_areas_sorted = sort_multi_list(
            [parts, parts_areas, parts_cp_areas], key=lambda x: x[1], reverse=True
        )

    with Timer("去除面积过小的区域"):
        filtered_parts = []
        other_parts = []
        for i in range(len(parts_sorted)):
            parts = parts_sorted[i]
            area = parts_areas_sorted[i]
            cp_area = parts_cp_areas_sorted[i]
            if area / (cp_area + 1e-7) > 0.001:
                filtered_parts.append(i)
            else:
                other_parts.append(i)
    if show_info:
        print(f"保留{len(filtered_parts)}个mesh, 其他{len(other_parts)}个mesh")

    with Timer("去除面积过小区域的label"):
        face_ids_2 = face_ids.copy()
        part_num = len(cluster2.keys())
        for j in other_parts:
            parts = parts_sorted[j]
            for i in parts:
                face_ids_2[i] = -1

    with Timer("第二次修复face_ids"):
        face_ids_3 = face_ids_2.copy()
        face_ids_3 = fix_label(
            face_ids_3, adjacent_faces, mesh=mesh, show_info=show_info
        )

    if save_mid_res:
        save_mesh(
            os.path.join(save_path, "auto_mask_mesh_filtered_2.glb"),
            mesh,
            face_ids_3,
            color_map,
        )
        if show_info:
            print("保存mesh结果完成")

    with Timer("第二次计算连通区域"):
        parts_2 = get_connected_region(face_ids_3, adjacent_faces)
        parts_areas_2 = [float(np.sum(face_areas[x])) for x in parts_2]
        parts_ids_2 = [face_ids_3[x[0]] for x in parts_2]

    with Timer("添加过大的缺失part"):
        color_map_2 = copy.deepcopy(color_map)
        max_id = np.max(parts_ids_2)
        for i in range(len(parts_2)):
            _parts = parts_2[i]
            _area = parts_areas_2[i]
            _parts_id = face_ids_3[_parts[0]]
            if _area / mesh_total_area > 0.001:
                if _parts_id == -1 or _parts_id == -2:
                    parts_ids_2[i] = max_id + 1
                    max_id += 1
                    color_map_2[max_id] = np.random.rand(3) * 255
                    if show_info:
                        print(f"新增part {max_id}")
            # else:
            #     parts_ids_2[i] = -1

    with Timer("赋值新的face_ids"):
        face_ids_4 = face_ids_3.copy()
        for i in range(len(parts_2)):
            _parts = parts_2[i]
            _parts_id = parts_ids_2[i]
            for j in _parts:
                face_ids_4[j] = _parts_id
    with Timer("计算part和label的aabb"):
        ids_aabb = {}
        unique_ids = np.unique(face_ids_4)
        for i in unique_ids:
            if i < 0:
                continue
            _part_mask = face_ids_4 == i
            _faces = mesh.faces[_part_mask]
            _faces = np.reshape(_faces, (-1))
            _points = mesh.vertices[_faces]
            min_xyz = np.min(_points, axis=0)
            max_xyz = np.max(_points, axis=0)
            ids_aabb[i] = [min_xyz, max_xyz]

        parts_2_aabb = []
        for i in range(len(parts_2)):
            _parts = parts_2[i]
            _faces = mesh.faces[_parts]
            _faces = np.reshape(_faces, (-1))
            _points = mesh.vertices[_faces]
            min_xyz = np.min(_points, axis=0)
            max_xyz = np.max(_points, axis=0)
            parts_2_aabb.append([min_xyz, max_xyz])

    with Timer("计算part的邻居"):
        parts_2_neighbor = find_neighbor_part(
            parts_2, adjacent_faces, parts_2_aabb, parts_ids_2
        )
    with Timer("合并无mask区域"):
        for i in range(len(parts_2)):
            _parts = parts_2[i]
            _ids = parts_ids_2[i]
            if _ids == -1 or _ids == -2:
                _cur_aabb = parts_2_aabb[i]
                _min_aabb_increase = 1e10
                _min_id = -1
                for j in parts_2_neighbor[i]:
                    if parts_ids_2[j] == -1 or parts_ids_2[j] == -2:
                        continue
                    _tar_id = parts_ids_2[j]
                    _tar_aabb = ids_aabb[_tar_id]
                    _min_increase, _max_increase = aabb_increase(_tar_aabb, _cur_aabb)
                    _increase = max(np.max(_min_increase), np.max(_max_increase))
                    if _min_aabb_increase > _increase:
                        _min_aabb_increase = _increase
                        _min_id = _tar_id
                if _min_id >= 0:
                    parts_ids_2[i] = _min_id

    with Timer("再次赋值新的face_ids"):
        face_ids_4 = face_ids_3.copy()
        for i in range(len(parts_2)):
            _parts = parts_2[i]
            _parts_id = parts_ids_2[i]
            for j in _parts:
                face_ids_4[j] = _parts_id

    final_face_ids = face_ids_4
    if save_mid_res:
        save_mesh(
            os.path.join(save_path, "auto_mask_mesh_final.glb"),
            mesh,
            face_ids_4,
            color_map_2,
        )

    if post_process:
        parts = get_connected_region(final_face_ids, adjacent_faces)
        final_face_ids = do_no_mask_process(parts, final_face_ids)
        face_ids_5 = do_post_process(
            face_areas,
            parts,
            adjacent_faces,
            face_ids_4,
            threshold,
            show_info=show_info,
        )
        if save_mid_res:
            save_mesh(
                os.path.join(save_path, "auto_mask_mesh_final_post.glb"),
                mesh,
                face_ids_5,
                color_map_2,
            )
        final_face_ids = face_ids_5
    with Timer("计算最后的aabb"):
        aabb = get_aabb_from_face_ids(mesh, final_face_ids)
    return aabb, final_face_ids, mesh
