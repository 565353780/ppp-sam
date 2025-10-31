import torch
import torch.nn as nn

from ppp_sam.Model.model import build_P3SAM, load_state_dict


class P3SAM(nn.Module):
    def __init__(
        self,
        sonata_model_file_path: str,
    ):
        super().__init__()
        build_P3SAM(self, sonata_model_file_path)

    def load_state_dict(
        self,
        ckpt_path=None,
        state_dict=None,
        strict=True,
        assign=False,
        ignore_seg_mlp=False,
        ignore_seg_s2_mlp=False,
        ignore_iou_mlp=False,
    ):
        load_state_dict(
            self,
            ckpt_path=ckpt_path,
            state_dict=state_dict,
            strict=strict,
            assign=assign,
            ignore_seg_mlp=ignore_seg_mlp,
            ignore_seg_s2_mlp=ignore_seg_s2_mlp,
            ignore_iou_mlp=ignore_iou_mlp,
        )

    def forward(self, feats, points, point_prompt, iter=1):
        """
        feats: [K, N, 512]
        points: [K, N, 3]
        point_prompt: [K, N, 3]
        """
        # print(feats.shape, points.shape, point_prompt.shape)
        point_num = points.shape[1]
        feats = feats.transpose(0, 1)  # [N, K, 512]
        points = points.transpose(0, 1)  # [N, K, 3]
        point_prompt = point_prompt.transpose(0, 1)  # [N, K, 3]
        feats_seg = torch.cat([feats, points, point_prompt], dim=-1)  # [N, K, 512+3+3]

        # 预测mask stage-1
        pred_mask_1 = self.seg_mlp_1(feats_seg).squeeze(-1)  # [N, K]
        pred_mask_2 = self.seg_mlp_2(feats_seg).squeeze(-1)  # [N, K]
        pred_mask_3 = self.seg_mlp_3(feats_seg).squeeze(-1)  # [N, K]
        pred_mask = torch.stack(
            [pred_mask_1, pred_mask_2, pred_mask_3], dim=-1
        )  # [N, K, 3]

        for _ in range(iter):
            # 预测mask stage-2
            feats_seg_2 = torch.cat([feats_seg, pred_mask], dim=-1)  # [N, K, 512+3+3+3]
            feats_seg_global = self.seg_s2_mlp_g(feats_seg_2)  # [N, K, 512]
            feats_seg_global = torch.max(feats_seg_global, dim=0).values  # [K, 512]
            feats_seg_global = feats_seg_global.unsqueeze(0).repeat(
                point_num, 1, 1
            )  # [N, K, 512]
            feats_seg_3 = torch.cat(
                [feats_seg_global, feats_seg_2], dim=-1
            )  # [N, K, 512+3+3+3+512]
            pred_mask_s2_1 = self.seg_s2_mlp_1(feats_seg_3).squeeze(-1)  # [N, K]
            pred_mask_s2_2 = self.seg_s2_mlp_2(feats_seg_3).squeeze(-1)  # [N, K]
            pred_mask_s2_3 = self.seg_s2_mlp_3(feats_seg_3).squeeze(-1)  # [N, K]
            pred_mask_s2 = torch.stack(
                [pred_mask_s2_1, pred_mask_s2_2, pred_mask_s2_3], dim=-1
            )  # [N,, K 3]
            pred_mask = pred_mask_s2

        mask_1 = torch.sigmoid(pred_mask_s2_1).to(dtype=torch.float32)  # [N, K]
        mask_2 = torch.sigmoid(pred_mask_s2_2).to(dtype=torch.float32)  # [N, K]
        mask_3 = torch.sigmoid(pred_mask_s2_3).to(dtype=torch.float32)  # [N, K]

        feats_iou = torch.cat(
            [feats_seg_global, feats_seg, pred_mask_s2], dim=-1
        )  # [N, K, 512+3+3+3+512]
        feats_iou = self.iou_mlp(feats_iou)  # [N, K, 512]
        feats_iou = torch.max(feats_iou, dim=0).values  # [K, 512]
        pred_iou = self.iou_mlp_out(feats_iou)  # [K, 3]
        pred_iou = torch.sigmoid(pred_iou).to(dtype=torch.float32)  # [K, 3]

        mask_1 = mask_1.transpose(0, 1)  # [K, N]
        mask_2 = mask_2.transpose(0, 1)  # [K, N]
        mask_3 = mask_3.transpose(0, 1)  # [K, N]

        return mask_1, mask_2, mask_3, pred_iou
