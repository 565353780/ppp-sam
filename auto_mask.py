import os
import trimesh

from ppp_sam.Method.utils import set_seed


if __name__ == "__main__":
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
