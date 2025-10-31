import torch

from ppp_sam.Model.p3_sam import P3SAM
from ppp_sam.Method.kernel import mesh_sam


class AutoMask:
    def __init__(
        self,
        sonata_model_file_path,
        ckpt_path,
        point_num=100000,
        prompt_num=400,
        threshold=0.95,
        post_process=True,
    ):
        """
        ckpt_path: str, 模型路径
        point_num: int, 采样点数量
        prompt_num: int, 提示数量
        threshold: float, 阈值
        post_process: bool, 是否后处理
        """
        self.model = P3SAM(sonata_model_file_path)
        self.model.load_state_dict(ckpt_path)
        self.model.eval()
        self.model_parallel = torch.nn.DataParallel(self.model)
        self.model.cuda()
        self.model_parallel.cuda()
        self.point_num = point_num
        self.prompt_num = prompt_num
        self.threshold = threshold
        self.post_process = post_process

    def predict_aabb(
        self,
        mesh,
        point_num=None,
        prompt_num=None,
        threshold=None,
        post_process=None,
        save_path=None,
        save_mid_res=False,
        show_info=True,
        clean_mesh_flag=True,
        seed=42,
        is_parallel=True,
        prompt_bs=32,
    ):
        """
        Parameters:
            mesh: trimesh.Trimesh, 输入网格
            point_num: int, 采样点数量
            prompt_num: int, 提示数量
            threshold: float, 阈值
            post_process: bool, 是否后处理
        Returns:
            aabb: np.ndarray, 包围盒
            face_ids: np.ndarray, 面id
        """
        point_num = point_num if point_num is not None else self.point_num
        prompt_num = prompt_num if prompt_num is not None else self.prompt_num
        threshold = threshold if threshold is not None else self.threshold
        post_process = post_process if post_process is not None else self.post_process
        return mesh_sam(
            [self.model, self.model_parallel if is_parallel else self.model],
            mesh,
            save_path=save_path,
            point_num=point_num,
            prompt_num=prompt_num,
            threshold=threshold,
            post_process=post_process,
            show_info=show_info,
            save_mid_res=save_mid_res,
            clean_mesh_flag=clean_mesh_flag,
            seed=seed,
            prompt_bs=prompt_bs,
        )
