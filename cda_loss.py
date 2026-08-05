import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsensusKnowledgeDirectionalAlignmentLoss(nn.Module):
    """CDA Loss from Eqs. (7)--(11) of the DGNet paper."""

    def __init__(self, device="cuda", ratio=0.8, model_name="ViT-B/32"):
        super().__init__()
        try:
            import clip as openai_clip
        except ImportError as error:
            raise ImportError(
                "OpenAI CLIP is required for CDA Loss. Install the source from "
                "https://github.com/openai/CLIP (see README.md)."
            ) from error
        self.ratio = ratio
        self.model, _ = openai_clip.load(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self._interpolate_ms_pos_embed(target_img_size=256)

        text_src = [
            "an infrared image with complex background clutter and small bright thermal targets in the scene"
        ]
        text_tgt = [
            "an infrared image where the small thermal targets remain bright against a dimmed background"
        ]

        with torch.no_grad():
            t_src = openai_clip.tokenize(text_src).to(device)
            t_tgt = openai_clip.tokenize(text_tgt).to(device)

            t_src = self.model.encode_text(t_src).float()
            t_tgt = self.model.encode_text(t_tgt).float()

            t_src = t_src / t_src.norm(dim=-1, keepdim=True)
            t_tgt = t_tgt / t_tgt.norm(dim=-1, keepdim=True)

            text_direction = t_tgt - t_src
            self.text_direction = text_direction / text_direction.norm(
                dim=-1, keepdim=True
            )

    def _interpolate_ms_pos_embed(self, target_img_size=256):
        """Interpolate ViT-B/32 positional embeddings for 256-pixel inputs."""
        patch_size = 32
        target_grid = target_img_size // patch_size
        pos_embed = self.model.visual.positional_embedding
        dim = pos_embed.shape[-1]
        cls_pos = pos_embed[:1, :]
        grid_pos = pos_embed[1:, :]
        old_grid = int(grid_pos.shape[0] ** 0.5)
        grid_pos = grid_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
        grid_pos = F.interpolate(
            grid_pos,
            size=(target_grid, target_grid),
            mode="bicubic", # bilinear
            align_corners=False,
        )
        grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(-1, dim)
        new_pos_embed = torch.cat([cls_pos, grid_pos], dim=0)
        self.model.visual.positional_embedding = nn.Parameter(
            new_pos_embed, requires_grad=False
        )
        print(
            f"[Init] CLIP Positional Embedding interpolated to {target_grid}x{target_grid} for {target_img_size}px input."
        )

    def clip_normalize(self, x):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(
            1, 3, 1, 1
        )
        return (x - mean) / std

    def safe_normalize(self, x, eps=1e-6):
        norm = x.norm(dim=-1, keepdim=True)
        return x / (norm + eps)

    def forward(self, pred, mask, orig_img):
        """Compute CDA Loss for tensors shaped [B, 1, H, W]."""
        pred = pred.clamp(0, 1)
        mask = mask.clamp(0, 1).float()
        orig_img_3c = orig_img.repeat(1, 3, 1, 1)
        img_src_in = self.clip_normalize(orig_img_3c)
        img_pred_in = self.clip_normalize(
            (self.ratio * pred + 1 - self.ratio) * orig_img_3c
        )
        img_gt = self.clip_normalize((self.ratio * mask + 1 - self.ratio) * orig_img_3c)
        with torch.amp.autocast("cuda", enabled=False):
            V_pred = self.model.encode_image(img_pred_in.float()).float()
            with torch.no_grad():
                V_src = self.model.encode_image(img_src_in.float()).float()
                V_gt = self.model.encode_image(img_gt.float()).float()
        V_src = self.safe_normalize(V_src)
        V_pred = self.safe_normalize(V_pred)
        V_gt = self.safe_normalize(V_gt)

        delta_V = V_pred - V_src
        delta_V = self.safe_normalize(delta_V)
        cos_sim_dir = (delta_V * self.text_direction).sum(dim=-1)
        loss_cd_unscaled = (1.0 - cos_sim_dir).mean()

        visual_alignment = (V_pred * V_gt).sum(dim=-1)
        loss_ca_unscaled = (1.0 - visual_alignment).mean()

        # Eq. (9): L_CD = 1/2(1-cos); Eq. (10): L_CA = 1/2(1-cos);
        # Eq. (11): L_CDA = 1/2 L_CD + 1/2 L_CA.
        return 0.25 * (loss_cd_unscaled + loss_ca_unscaled)
