"""GLD-TConv training-side auxiliary heads and losses (NOT deployed).

This module is attached to the trainer via ``DetectionModel.gld_aux`` only when
``gld_lambda > 0``. It is removed before ONNX/edge export and never appears in
the inference graph. Its responsibilities:

  1) ``ROILanguageHead``      : projects per-instance ROI features into the CLIP
                                text space and aligns with frozen geometry-
                                language prompt embeddings.
  2) ``MaskBoundaryHead``     : 1x1 conv mask logits + Sobel boundary logits on
                                a chosen GLD feature tap, supervised by SAM2
                                pseudo masks.
  3) ``compute_spatial_rigidity_loss`` : per-cell rigidity SmoothL1 against a
                                rendered spatial rigidity map.
  4) ``compute_trapezoid_consistency_loss`` : penalizes elasticity drift between
                                the operator output mask and a fitted trapezoid
                                support derived from SAM masks.
  5) ``compute_image_rigidity_loss`` : the original v1 image-level scalar loss,
                                kept for ablation.

All five terms are summed inside :class:`GldAuxModule.compute` and returned as
both a scalar tensor (for backprop) and a small dict of per-term values for
logging.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _safe_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """SmoothL1 with optional per-element weight mask. Returns 0 when weight sums to 0."""
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_zeros(())
    diff = F.smooth_l1_loss(pred, target, reduction="none")
    if weight is None:
        return diff.mean()
    weight = weight.to(diff.dtype)
    denom = weight.sum().clamp_min(1.0)
    return (diff * weight).sum() / denom


def iter_gld_layers(model: nn.Module) -> Iterable[nn.Module]:
    """Yield every GLDTrapezoidConv-flagged layer of ``model`` in forward order."""
    for module in model.modules():
        if getattr(module, "GLD_FEATURE_TAP", False):
            yield module


# ---------------------------------------------------------------------------
# 1) ROI-language alignment head
# ---------------------------------------------------------------------------

class ROILanguageHead(nn.Module):
    """Project ROI features to CLIP text-space and contrast against frozen prompts.

    The ROI feature comes from the *output* of a chosen GLDTrapezoidConv layer
    (typically the deepest one, ~P4 resolution). ``roi_align`` is applied with
    GT bounding boxes, then a small MLP projects to ``text_dim`` (default 512
    for ViT-B/32). The loss is 1 - cos(<pred, frozen_prompt_embedding>),
    averaged over all valid instances in the batch.

    The frozen text embeddings are loaded from disk by the trainer and passed
    in via ``self.set_text_bank(...)`` exactly once.
    """

    def __init__(self, in_channels: int, text_dim: int = 512, roi_size: int = 7, hidden: int = 256) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.text_dim = int(text_dim)
        self.roi_size = int(roi_size)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(hidden, text_dim, bias=True),
        )
        self.register_buffer("text_bank", torch.zeros(1, text_dim), persistent=False)
        self._has_bank = False

    def set_text_bank(self, bank: torch.Tensor) -> None:
        """Register the frozen prompt embedding matrix (N_prompts, text_dim)."""
        bank = bank.detach().float()
        bank = bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # use register_buffer-like assignment so .to(device) tracks it
        self.text_bank = bank
        self._has_bank = True

    def forward(
        self,
        feat: torch.Tensor,
        rois: torch.Tensor,
        prompt_rows: torch.Tensor,
        feature_stride: float,
    ) -> torch.Tensor:
        """Compute the ROI-language alignment loss.

        Args:
            feat: (B, C, H, W) feature map output by the chosen GLD layer.
            rois: (N, 5) tensor with columns [batch_idx, x1, y1, x2, y2] in
                input-image pixel coordinates.
            prompt_rows: (N,) long tensor of prompt-bank row indices.
            feature_stride: ratio of input image size to feature map size,
                e.g. 16.0 for P4 at 320 input.
        """
        if not self._has_bank or rois.numel() == 0:
            return feat.new_zeros(())
        # torchvision's roi_align expects feature-space coordinates if
        # spatial_scale = 1/stride. We use that variant.
        try:
            from torchvision.ops import roi_align
        except Exception:
            return feat.new_zeros(())
        pooled = roi_align(
            feat,
            boxes=rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0 / float(feature_stride),
            sampling_ratio=2,
            aligned=True,
        )                                                               # (N, C, r, r)
        emb = self.proj(pooled)                                          # (N, text_dim)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        bank = self.text_bank.to(emb.device, emb.dtype)
        valid = (prompt_rows >= 0) & (prompt_rows < bank.shape[0])
        if not bool(valid.any()):
            return feat.new_zeros(())
        rows = prompt_rows[valid]
        target = bank[rows]                                               # (N_valid, text_dim)
        target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos = (emb[valid] * target).sum(dim=-1)                           # (N_valid,)
        return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# 2) Mask + boundary auxiliary head
# ---------------------------------------------------------------------------

class MaskBoundaryHead(nn.Module):
    """Tiny 1x1-conv mask head with a Sobel boundary auxiliary.

    Operates on the *output* of the first GLD layer (P3-ish resolution,
    typically 40x40 at 320 input). Two 1x1 convs produce mask and boundary
    logits; both are upsampled bilinearly to the supervision resolution before
    BCE.

    The supervision resolution defaults to the GLD feature map (40x40); the
    trainer can render SAM masks at exactly that resolution to keep memory
    overhead negligible.
    """

    def __init__(self, in_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(hidden, 1, kernel_size=1)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    @torch.no_grad()
    def _boundary_target(self, mask_target: torch.Tensor) -> torch.Tensor:
        sx = self.sobel_x.to(device=mask_target.device, dtype=mask_target.dtype)
        sy = self.sobel_y.to(device=mask_target.device, dtype=mask_target.dtype)
        gx = F.conv2d(mask_target, sx, padding=1)
        gy = F.conv2d(mask_target, sy, padding=1)
        edge = (gx.abs() + gy.abs()).clamp(0.0, 1.0)
        return (edge > 0.1).to(mask_target.dtype)

    def forward(
        self,
        feat: torch.Tensor,
        mask_target: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mask BCE loss and boundary BCE loss.

        Args:
            feat: (B, C, h, w) GLD feature output.
            mask_target: (B, 1, H_t, W_t) binary union-of-instances mask at the
                supervision resolution.
            target_valid: (B,) bool tensor marking which images have a usable
                mask target.
        """
        body = self.body(feat)
        mask_logit = self.mask_head(body)
        boundary_logit = self.boundary_head(body)
        # match supervision resolution
        if mask_logit.shape[-2:] != mask_target.shape[-2:]:
            mask_logit = F.interpolate(mask_logit, size=mask_target.shape[-2:], mode="bilinear", align_corners=False)
            boundary_logit = F.interpolate(
                boundary_logit, size=mask_target.shape[-2:], mode="bilinear", align_corners=False
            )
        if not bool(target_valid.any()):
            zero = feat.new_zeros(())
            return zero, zero
        idx = torch.nonzero(target_valid, as_tuple=False).flatten()
        m_pred = mask_logit[idx]
        b_pred = boundary_logit[idx]
        m_tgt = mask_target[idx].to(feat.dtype)
        b_tgt = self._boundary_target(m_tgt)
        loss_mask = F.binary_cross_entropy_with_logits(m_pred, m_tgt)
        loss_boundary = F.binary_cross_entropy_with_logits(b_pred, b_tgt)
        return loss_mask, loss_boundary


# ---------------------------------------------------------------------------
# 3) spatial rigidity / 4) trapezoid consistency / 5) image rigidity losses
# ---------------------------------------------------------------------------

def compute_image_rigidity_loss(layers: Sequence[nn.Module], target: torch.Tensor) -> torch.Tensor:
    """Image-level scalar rigidity loss (kept for ablation against v1)."""
    target = target.view(-1)
    valid = torch.isfinite(target) & target.ge(0.0)
    if not bool(valid.any()):
        return target.new_zeros(())
    preds = []
    for m in layers:
        rig = m.last_rigidity
        if rig is None or rig.shape[0] != target.numel():
            continue
        preds.append(rig.float().mean(dim=tuple(range(1, rig.ndim))))
    if not preds:
        return target.new_zeros(())
    pred = torch.stack(preds, dim=0).mean(dim=0)
    return _safe_smooth_l1(pred[valid], target[valid].to(pred.dtype))


def compute_spatial_rigidity_loss(
    layers: Sequence[nn.Module],
    spatial_target: torch.Tensor,
    target_weight: torch.Tensor,
) -> torch.Tensor:
    """Per-cell rigidity SmoothL1.

    ``spatial_target`` is rendered by the trainer at the *highest-resolution*
    GLD rigidity grid (e.g. 10x10 at window=4 + stride=4 over 40x40). For
    deeper GLD layers we average-pool the target down to that layer's grid.

    ``target_weight`` is 1 inside instance bboxes and 0 outside, so the loss
    only penalizes deviation where geometry evidence exists.
    """
    if spatial_target.numel() == 0:
        return spatial_target.new_zeros(())
    losses = []
    B = int(spatial_target.shape[0])
    for m in layers:
        rig = m.last_rigidity
        if rig is None or rig.shape[0] != B:
            continue
        # downsample target to this layer's spatial grid
        h, w = rig.shape[-2:]
        tgt = F.adaptive_avg_pool2d(spatial_target.to(rig.dtype), (h, w))
        wgt = F.adaptive_avg_pool2d(target_weight.to(rig.dtype), (h, w))
        losses.append(_safe_smooth_l1(rig, tgt, weight=wgt))
    if not losses:
        return spatial_target.new_zeros(())
    return torch.stack(losses).mean()


def compute_trapezoid_consistency_loss(
    layers: Sequence[nn.Module],
    spatial_target: torch.Tensor,
    target_weight: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Penalize *unsupervised* spatial collapse of rigidity.

    Without supervision the controller can output a constant per image and
    still satisfy the image-level loss. We add a low-weight regularizer that
    rewards rigidity ALIGNMENT with the per-cell target:

        loss = weighted L1(rigidity - sg(target)).detach() across cells where
        target_weight > 0  AND  spatial_var(target) > eps

    The condition prevents the loss from collapsing to zero when the target is
    spatially uniform (e.g. one tiny vessel rendered as a single rigidity cell)
    and complements ``compute_spatial_rigidity_loss`` by running on a pure
    detach()-of-target signal so the gradient path is decoupled from main
    rigidity supervision.
    """
    if spatial_target.numel() == 0:
        return spatial_target.new_zeros(())
    var = spatial_target.flatten(2).var(dim=-1)                          # (B, 1)
    active = (var > eps).flatten()                                        # (B,)
    if not bool(active.any()):
        return spatial_target.new_zeros(())
    losses = []
    B = int(spatial_target.shape[0])
    for m in layers:
        rig = m.last_rigidity
        if rig is None or rig.shape[0] != B:
            continue
        h, w = rig.shape[-2:]
        tgt = F.adaptive_avg_pool2d(spatial_target.to(rig.dtype), (h, w)).detach()
        wgt = F.adaptive_avg_pool2d(target_weight.to(rig.dtype), (h, w))
        # per-image weighted L1, then keep only "active" images
        diff = (rig - tgt).abs() * wgt
        denom = wgt.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_img = diff.sum(dim=(1, 2, 3)) / denom                        # (B,)
        if not bool(active.any()):
            continue
        losses.append(per_img[active].mean())
    if not losses:
        return spatial_target.new_zeros(())
    return torch.stack(losses).mean()


def _supcon_loss(feat: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss. feat: (N, D) L2-normed, labels: (N,) long."""
    N = feat.shape[0]
    if N < 2:
        return feat.new_zeros(())
    eye = torch.eye(N, dtype=torch.bool, device=feat.device)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    if not pos.any():
        return feat.new_zeros(())
    sim = torch.mm(feat, feat.T) / temperature                     # (N, N)
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    log_denom = ((sim - sim_max).exp() * ~eye).sum(dim=1).log() + sim_max.squeeze()
    return -((sim.diagonal() * 0 + (sim - log_denom.unsqueeze(1)))[pos]).mean()


# ---------------------------------------------------------------------------
# top-level wrapper attached to DetectionModel during training
# ---------------------------------------------------------------------------

class GldAuxModule(nn.Module):
    """Container for all GLD-TConv training-side losses.

    Attached to ``DetectionModel`` as ``model.gld_aux`` only when
    ``args.gld_lambda > 0``. The forward pass is invoked by ``DetectionModel.
    loss(...)`` AFTER ``self.forward(batch['img'])``, so the GLD layers' scratch
    buffers are already populated.
    """

    # ordering of per-term losses for trainer logging
    loss_names: Tuple[str, ...] = ("gld_img", "gld_spa", "gld_con", "gld_mask", "gld_bnd", "gld_lan", "gld_idc")

    def __init__(
        self,
        roi_in_channels: int,
        text_dim: int = 512,
        mask_in_channels: int = 0,
        weights: Optional[dict] = None,
        roi_feature_stride: float = 16.0,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        weights = weights or {}
        self.w_image = float(weights.get("image", 1.0))
        self.w_spatial = float(weights.get("spatial", 1.0))
        self.w_consistency = float(weights.get("consistency", 0.5))
        self.w_mask = float(weights.get("mask", 0.5))
        self.w_boundary = float(weights.get("boundary", 0.5))
        self.w_lang = float(weights.get("language", 1.0))
        self.w_contrast = float(weights.get("contrast", 0.0))
        self.gain = float(gain)
        self.roi_feature_stride = float(roi_feature_stride)

        self.roi_head = ROILanguageHead(in_channels=roi_in_channels, text_dim=text_dim)
        self.mask_head: Optional[MaskBoundaryHead] = (
            MaskBoundaryHead(in_channels=mask_in_channels) if mask_in_channels > 0 else None
        )

    # ----- callable entry point (used by DetectionModel._gld_auxiliary_loss) ---
    def forward(self, model: nn.Module, batch: dict) -> Tuple[torch.Tensor, dict]:
        layers = list(iter_gld_layers(model))
        return self.compute(layers, batch, self.roi_feature_stride)

    # ----- text bank loading ---------------------------------------------
    def set_text_bank(self, bank: torch.Tensor) -> None:
        self.roi_head.set_text_bank(bank)

    # ----- forward --------------------------------------------------------
    def compute(
        self,
        layers: Sequence[nn.Module],
        batch: dict,
        feature_stride_for_roi: float,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the aggregated auxiliary loss and a per-term log dict."""
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device)
        terms = {
            "gld_img": zero,
            "gld_spa": zero,
            "gld_con": zero,
            "gld_mask": zero,
            "gld_bnd": zero,
            "gld_lan": zero,
            "gld_idc": zero,
        }

        layers = list(layers)
        if not layers:
            return zero, terms

        # image-level rigidity (always available when target exists)
        img_target = batch.get("gld_rigidity")
        if isinstance(img_target, torch.Tensor) and self.w_image > 0:
            terms["gld_img"] = compute_image_rigidity_loss(layers, img_target.to(device).float())

        # spatial rigidity + trapezoid consistency
        spa_target = batch.get("gld_spatial_target")
        spa_weight = batch.get("gld_spatial_weight")
        if (
            isinstance(spa_target, torch.Tensor)
            and isinstance(spa_weight, torch.Tensor)
            and (self.w_spatial > 0 or self.w_consistency > 0)
        ):
            spa_target = spa_target.to(device).float()
            spa_weight = spa_weight.to(device).float()
            if self.w_spatial > 0:
                terms["gld_spa"] = compute_spatial_rigidity_loss(layers, spa_target, spa_weight)
            if self.w_consistency > 0:
                terms["gld_con"] = compute_trapezoid_consistency_loss(layers, spa_target, spa_weight)

        # mask + boundary
        mask_target = batch.get("gld_mask_target")
        mask_valid = batch.get("gld_mask_valid")
        if (
            self.mask_head is not None
            and isinstance(mask_target, torch.Tensor)
            and isinstance(mask_valid, torch.Tensor)
            and (self.w_mask > 0 or self.w_boundary > 0)
        ):
            feat = layers[0].last_output
            if isinstance(feat, torch.Tensor):
                lm, lb = self.mask_head(feat, mask_target.to(device).float(), mask_valid.to(device).bool())
                terms["gld_mask"] = lm
                terms["gld_bnd"] = lb

        # ROI-language alignment
        rois = batch.get("gld_rois")           # (N, 5) [batch_idx, x1, y1, x2, y2] in input pixels
        prompt_rows = batch.get("gld_prompt_rows")   # (N,) long
        if (
            isinstance(rois, torch.Tensor)
            and isinstance(prompt_rows, torch.Tensor)
            and self.w_lang > 0
        ):
            feat = layers[-1].last_output
            if isinstance(feat, torch.Tensor):
                terms["gld_lan"] = self.roi_head(
                    feat,
                    rois.to(device).float(),
                    prompt_rows.to(device).long(),
                    feature_stride_for_roi,
                )

        # IDC: instance discriminative contrastive on deep ROI features
        track_ids = batch.get("gld_track_ids")
        if (
            isinstance(rois, torch.Tensor)
            and isinstance(track_ids, torch.Tensor)
            and self.w_contrast > 0
        ):
            feat = layers[-1].last_output
            if isinstance(feat, torch.Tensor):
                try:
                    from torchvision.ops import roi_align
                    pooled = roi_align(
                        feat, boxes=rois.to(device).float(),
                        output_size=(4, 4),
                        spatial_scale=1.0 / float(feature_stride_for_roi),
                        sampling_ratio=2, aligned=True,
                    ).mean(dim=(-2, -1))                               # (N, C)
                    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    valid = track_ids.to(device) >= 0
                    if valid.sum() >= 2:
                        terms["gld_idc"] = _supcon_loss(pooled[valid], track_ids.to(device)[valid])
                except Exception:
                    pass

        total = self.gain * (
            self.w_image * terms["gld_img"]
            + self.w_spatial * terms["gld_spa"]
            + self.w_consistency * terms["gld_con"]
            + self.w_mask * terms["gld_mask"]
            + self.w_boundary * terms["gld_bnd"]
            + self.w_lang * terms["gld_lan"]
            + self.w_contrast * terms["gld_idc"]
        )
        return total, terms


__all__ = [
    "GldAuxModule",
    "ROILanguageHead",
    "MaskBoundaryHead",
    "iter_gld_layers",
    "compute_image_rigidity_loss",
    "compute_spatial_rigidity_loss",
    "compute_trapezoid_consistency_loss",
]
