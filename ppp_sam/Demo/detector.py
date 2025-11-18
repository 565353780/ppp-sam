from ppp_sam.Module.detector import Detector


def demo():
    sonata_model_file_path = "/home/chli/chLi/Model/Sonata/sonata.pth"
    model_file_path = "/home/chli/chLi/Model/P3SAM/p3sam.safetensors"
    point_num = 100000
    prompt_num = 400
    threshold = 0.95
    post_process = False
    prompt_bs = 8

    mesh_file_path = "/home/chli/chLi/Dataset/ABC/unzipped/obj/00000002/00000002_1ffb81a71e5b402e966b9341_trimesh_001.obj"
    save_folder_path = "./output/test_abc_00000002/"

    detector = Detector(sonata_model_file_path, model_file_path)
    detector.detectMeshFile(
        mesh_file_path,
        save_folder_path,
        point_num,
        prompt_num,
        threshold,
        post_process,
        prompt_bs,
    )
    return True
