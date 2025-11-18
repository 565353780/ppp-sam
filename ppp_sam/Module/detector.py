import os
import torch
import trimesh
from typing import Union

from ppp_sam.Model.auto_mask import AutoMask
from ppp_sam.Method.utils import set_seed
from ppp_sam.Module.timer import Timer


class Detector(object):
    def __init__(
        self,
        sonata_model_file_path: Union[str, None] = None,
        model_file_path: Union[str, None] = None,
    ) -> None:
        self.point_num = 100000
        self.prompt_num = 400
        self.threshold = 0.95
        self.post_process = False
        self.prompt_bs = 8

        self.show_info = True
        self.show_time_info = True
        self.save_mid_res = True
        self.seed = 42
        self.parallel = True
        self.clean_mesh = True

        Timer.STATE = self.show_time_info

        if sonata_model_file_path is not None and model_file_path is not None:
            self.loadModel(sonata_model_file_path, model_file_path)
        return

    def loadModel(
        self,
        sonata_model_file_path: str,
        model_file_path: str,
    ) -> bool:
        if not os.path.exists(sonata_model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t sonata model file not exist!")
            print("\t sonata_model_file_path:", sonata_model_file_path)
            return False

        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        self.auto_mask = AutoMask(sonata_model_file_path, model_file_path)
        return True

    def detect(self, image) -> torch.Tensor:
        # Perform detection using the model
        results = self.model.predict(image)
        return results

    def detectMeshFile(
        self,
        mesh_file_path: str,
        save_folder_path: str,
        point_num: int = 100000,
        prompt_num: int = 400,
        threshold: float = 0.95,
        post_process: bool = False,
        prompt_bs: int = 8,
    ) -> Union[torch.Tensor, None]:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][Detector::detectMeshFile]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path:", mesh_file_path)
            return None

        # Load the mesh file (implementation depends on the specific use case)
        mesh_data = self.loadMesh(mesh_file_path)

        # Perform detection on the loaded mesh data
        results = self.detect(mesh_data)
        return results
