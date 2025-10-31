import os
import torch
import random
import trimesh
import argparse
import numpy as np

from ppp_sam.Model.auto_mask import AutoMask
from ppp_sam.Module.timer import Timer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--sonata_path", type=str, default=None, help="Sonata模型路径"
    )
    argparser.add_argument("--ckpt_path", type=str, default=None, help="模型路径")
    argparser.add_argument(
        "--mesh_path", type=str, default="assets/1.glb", help="输入网格路径"
    )
    argparser.add_argument(
        "--output_path", type=str, default="results/1", help="保存路径"
    )
    argparser.add_argument("--point_num", type=int, default=100000, help="采样点数量")
    argparser.add_argument("--prompt_num", type=int, default=400, help="提示数量")
    argparser.add_argument("--threshold", type=float, default=0.95, help="阈值")
    argparser.add_argument("--post_process", type=int, default=0, help="是否后处理")
    argparser.add_argument(
        "--save_mid_res", type=int, default=1, help="是否保存中间结果"
    )
    argparser.add_argument("--show_info", type=int, default=1, help="是否显示信息")
    argparser.add_argument(
        "--show_time_info", type=int, default=1, help="是否显示时间信息"
    )
    argparser.add_argument("--seed", type=int, default=42, help="随机种子")
    argparser.add_argument("--parallel", type=int, default=1, help="是否使用多卡")
    argparser.add_argument(
        "--prompt_bs", type=int, default=8, help="提示点推理时的batch size大小"
    )
    argparser.add_argument("--clean_mesh", type=int, default=1, help="是否清洗网格")
    args = argparser.parse_args()
    Timer.STATE = args.show_time_info

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    sonata_model_file_path = args.sonata_path
    ckpt_path = args.ckpt_path
    auto_mask = AutoMask(sonata_model_file_path, ckpt_path)
    mesh_path = args.mesh_path
    if os.path.isdir(mesh_path):
        for file in os.listdir(mesh_path):
            if not (
                file.endswith(".glb") or file.endswith(".obj") or file.endswith(".ply")
            ):
                continue
            _mesh_path = os.path.join(mesh_path, file)
            _output_path = os.path.join(output_path, file[:-4])
            os.makedirs(_output_path, exist_ok=True)
            mesh = trimesh.load(_mesh_path, force="mesh")
            set_seed(args.seed)
            aabb, face_ids, mesh = auto_mask.predict_aabb(
                mesh,
                save_path=_output_path,
                point_num=args.point_num,
                prompt_num=args.prompt_num,
                threshold=args.threshold,
                post_process=args.post_process,
                save_mid_res=args.save_mid_res,
                show_info=args.show_info,
                seed=args.seed,
                is_parallel=args.parallel,
                clean_mesh_flag=args.clean_mesh,
                prompt_bs=args.prompt_bs,
            )
    else:
        mesh = trimesh.load(mesh_path, force="mesh")
        set_seed(args.seed)
        aabb, face_ids, mesh = auto_mask.predict_aabb(
            mesh,
            save_path=output_path,
            point_num=args.point_num,
            prompt_num=args.prompt_num,
            threshold=args.threshold,
            post_process=args.post_process,
            save_mid_res=args.save_mid_res,
            show_info=args.show_info,
            seed=args.seed,
            is_parallel=args.parallel,
            clean_mesh_flag=args.clean_mesh,
            prompt_bs=args.prompt_bs,
        )

    ###############################################
    ## 可以通过以下代码保存返回的结果
    ## You can save the returned result by the following code
    ################# save result #################
    # color_map = {}
    # unique_ids = np.unique(face_ids)
    # for i in unique_ids:
    #     if i == -1:
    #         continue
    #     part_color = np.random.rand(3) * 255
    #     color_map[i] = part_color
    # face_colors = []
    # for i in face_ids:
    #     if i == -1:
    #         face_colors.append([0, 0, 0])
    #     else:
    #         face_colors.append(color_map[i])
    # face_colors = np.array(face_colors).astype(np.uint8)
    # mesh_save = mesh.copy()
    # mesh_save.visual.face_colors = face_colors
    # mesh_save.export(os.path.join(output_path, 'auto_mask_mesh.glb'))
    # scene_mesh = trimesh.Scene()
    # scene_mesh.add_geometry(mesh_save)
    # for i in range(len(aabb)):
    #     min_xyz, max_xyz = aabb[i]
    #     center = (min_xyz + max_xyz) / 2
    #     size = max_xyz - min_xyz
    #     box = trimesh.path.creation.box_outline()
    #     box.vertices *= size
    #     box.vertices += center
    #     scene_mesh.add_geometry(box)
    # scene_mesh.export(os.path.join(output_path, 'auto_mask_aabb.glb'))
    ################# save result #################

"""
python auto_mask.py --parallel 0 
python auto_mask.py --ckpt_path ../weights/last.ckpt --mesh_path assets/1.glb --output_path results/1 --parallel 0 
python auto_mask.py --ckpt_path ../weights/last.ckpt --mesh_path assets --output_path results/all 
"""
