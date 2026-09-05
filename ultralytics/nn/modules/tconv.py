"""TConv and GLD-TConv operators (deployment-side).

The v2 GLDTrapezoidConv keeps the deployable inference graph identical in spirit
to the conference TConv operator, while introducing a hard-gated elasticity
controller that cannot be bypassed by the affine parameter branch.

Auxiliary training-side heads (mask / boundary / ROI-language / spatial
rigidity / trapezoid consistency) are NOT placed in this module; they are
attached to the trainer and removed prior to export so that the deployed
operator only contains: 1) one stride-k 6-channel param conv, 2) one stride-k
1-channel rigidity conv, 3) the existing offset->mask 1x1 conv, 4) the trailing
3x3 conv. Hi3403V100 / NPU friendly: the only non-standard ops are sigmoid,
softplus, tanh, sin/cos, pixel_shuffle and bilinear/nearest interpolate, all of
which are supported by the original conference deployment toolchain.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _env_float(name: str, default: float) -> float:
    """Read a float hyper-parameter from the environment (sweep knob)."""
    val = os.environ.get(name)
    if val is None or val == "":
        return float(default)
    try:
        return float(val)
    except ValueError:
        return float(default)


# ---------------------------------------------------------------------------
# trapezoid template point grids
# ---------------------------------------------------------------------------
# Each row of templates_pts<k> contains k*k normalized sampling points on the
# canonical [-1, 1]^2 trapezoid manifold (top width 2, bottom width 1). The v2
# templates fix the top-row right-most x bug present in v1 (formerly -1.0).

_TEMPLATE_PTS_2 = [
    [-1.0,  1.0], [ 1.0,  1.0],
    [-0.5, -1.0], [ 0.5, -1.0],
]

_TEMPLATE_PTS_3 = [
    [-1.00,  1.0], [0.00,  1.0], [1.00,  1.0],
    [-0.75,  0.0], [0.00,  0.0], [0.75,  0.0],
    [-0.50, -1.0], [0.00, -1.0], [0.50, -1.0],
]

# y in {1, 1/3, -1/3, -1} -> half-widths (1.0, 5/6, 2/3, 1/2)
_TEMPLATE_PTS_4 = [
    [-1.0,  1.0], [-1.0 / 3,  1.0], [1.0 / 3,  1.0], [1.0,  1.0],
    [-5.0 / 6,  1.0 / 3], [-5.0 / 18,  1.0 / 3], [5.0 / 18, 1.0 / 3], [5.0 / 6, 1.0 / 3],
    [-2.0 / 3, -1.0 / 3], [-2.0 / 9, -1.0 / 3], [2.0 / 9, -1.0 / 3], [2.0 / 3, -1.0 / 3],
    [-1.0 / 2, -1.0], [-1.0 / 6, -1.0], [1.0 / 6, -1.0], [1.0 / 2, -1.0],
]

# y in {1, 1/2, 0, -1/2, -1} -> half-widths (1.0, 7/8, 3/4, 5/8, 1/2)
_TEMPLATE_PTS_5 = [
    [-1.0,  1.0], [-1.0 / 2,  1.0], [0.0,  1.0], [1.0 / 2,  1.0], [1.0,  1.0],
    [-7.0 / 8,  1.0 / 2], [-7.0 / 16,  1.0 / 2], [0.0,  1.0 / 2], [7.0 / 16,  1.0 / 2], [7.0 / 8,  1.0 / 2],
    [-3.0 / 4,  0.0], [-3.0 / 8,  0.0], [0.0,  0.0], [3.0 / 8,  0.0], [3.0 / 4,  0.0],
    [-5.0 / 8, -1.0 / 2], [-5.0 / 16, -1.0 / 2], [0.0, -1.0 / 2], [5.0 / 16, -1.0 / 2], [5.0 / 8, -1.0 / 2],
    [-1.0 / 2, -1.0], [-1.0 / 4, -1.0], [0.0, -1.0], [1.0 / 4, -1.0], [1.0 / 2, -1.0],
]


def _make_template(k: int) -> torch.Tensor:
    if k == 2:
        return torch.tensor(_TEMPLATE_PTS_2, dtype=torch.float32)
    if k == 3:
        return torch.tensor(_TEMPLATE_PTS_3, dtype=torch.float32)
    if k == 4:
        return torch.tensor(_TEMPLATE_PTS_4, dtype=torch.float32)
    if k == 5:
        return torch.tensor(_TEMPLATE_PTS_5, dtype=torch.float32)
    raise ValueError(f"Unsupported trapezoid window_size={k}.")


class TrapezoidConv(nn.Module):
    """Conference-version trapezoid convolution. Kept for ablation A."""

    def __init__(self, in_channels, out_channels, window_size=2, s=2):
        super().__init__()
        self.K = window_size
        k = window_size
        self.s = s
        self.register_buffer("template_pts2", torch.tensor(_TEMPLATE_PTS_2, dtype=torch.float32))
        self.register_buffer("template_pts3", torch.tensor(_TEMPLATE_PTS_3, dtype=torch.float32))
        self.register_buffer("template_pts4", torch.tensor(_TEMPLATE_PTS_4, dtype=torch.float32))
        self.register_buffer("template_pts5", torch.tensor(_TEMPLATE_PTS_5, dtype=torch.float32))

        self.param_conv = nn.Conv2d(in_channels, 6, kernel_size=k, stride=k, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.act = nn.Hardswish(inplace=True)

        self.offset2mask = nn.Conv2d(4 * k * k, k * k, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(k * k)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.s, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.K
        device, dtype = x.device, x.dtype

        params = self.act(self.bn1(self.param_conv(x))).permute(0, 2, 3, 1).contiguous()

        Sx = F.softplus(params[..., 0]) + 1e-3
        Sy = F.softplus(params[..., 1]) + 1e-3
        angle = params[..., 2]
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        shear = torch.tanh(params[..., 3]) * 0.5
        Tx = torch.tanh(params[..., 4]) * 0.5
        Ty = torch.tanh(params[..., 5]) * 0.5

        a11 = Sx * cos
        a12 = cos * Sx * shear + sin * Sy
        a21 = -Sx * sin
        a22 = -sin * Sx * shear + cos * Sy

        if k == 2:
            pts = self.template_pts2.to(device, dtype)
        elif k == 3:
            pts = self.template_pts3.to(device, dtype)
        elif k == 4:
            pts = self.template_pts4.to(device, dtype)
        else:
            pts = self.template_pts5.to(device, dtype)
        x_pts, y_pts = pts[:, 0], pts[:, 1]

        x_trans = a11.unsqueeze(-1) * x_pts + a12.unsqueeze(-1) * y_pts + Tx.unsqueeze(-1)
        y_trans = a21.unsqueeze(-1) * x_pts + a22.unsqueeze(-1) * y_pts + Ty.unsqueeze(-1)
        offsets = torch.cat(
            [torch.abs(x_trans), torch.abs(y_trans), 1 - torch.abs(x_trans), 1 - torch.abs(y_trans)], dim=-1
        )

        mask_in = offsets.permute(0, 3, 1, 2).contiguous()
        mask = self.act(self.bn2(self.offset2mask(mask_in)))
        mask = F.pixel_shuffle(mask, upscale_factor=k)
        if (mask.shape[2] != H) or (mask.shape[3] != W):
            mask = F.interpolate(mask, size=(H, W), mode="nearest")
        out = self.act(self.silu(self.conv(x * mask)))
        return out


class GLDTrapezoidConv(nn.Module):
    """Geometry-Language Distilled Trapezoid Convolution (v2, deployable).

    Key differences vs. v1:
      * Hard-gated shear/translation. The trapezoid bound is multiplied by
        elasticity directly, so when elasticity -> 0 the operator collapses to
        the rigid trapezoid; when elasticity -> 1 the operator recovers the
        full conference TConv bound. This blocks the v1 degeneracy where the
        param conv could re-scale logits to bypass the elasticity gate.
      * Single-channel rigidity conv with NO BatchNorm and a learnable bias
        (initialized so initial rigidity ~ 0.5). BN(1) + sigmoid was
        gradient-starved.
      * Stashes per-cell rigidity AND the conv input/output features under
        ``last_*`` attributes (only when training and grad is enabled). The
        trainer reads these through ``module.iter_gld_layers`` for the
        spatial-rigidity, mask, boundary and ROI-language auxiliary losses.
    """

    GLD_FEATURE_TAP = True  # marker for trainer-side discovery

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        window_size: int = 2,
        s: int = 2,
        max_shear: float = 0.5,
        max_trans: float = 0.5,
        rigidity_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.K = int(window_size)
        k = self.K
        self.s = int(s)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        # ---- sweep knobs (env overrides; defaults preserve prior behaviour) ----
        # GLD_RIGIDITY_INIT : initial rigidity. Lower => starts closer to the
        #   conference TConv deformation range (elasticity ~ 1). Old runs used 0.5,
        #   which halved the deformable range and crippled the operator-only model.
        # GLD_MAX_SHEAR / GLD_MAX_TRANS : deformation bound when fully elastic.
        # GLD_TAIL : "gld" (SiLU(BN(conv)), current) or "tconv"
        #   (Hardswish(SiLU(conv)), exactly the conference tail) to isolate the
        #   tail-architecture confound.
        rigidity_init = _env_float("GLD_RIGIDITY_INIT", rigidity_init)
        self.max_shear = _env_float("GLD_MAX_SHEAR", max_shear)
        self.max_trans = _env_float("GLD_MAX_TRANS", max_trans)
        self.tail = os.environ.get("GLD_TAIL", "gld").strip().lower() or "gld"

        self.register_buffer("template_pts2", torch.tensor(_TEMPLATE_PTS_2, dtype=torch.float32))
        self.register_buffer("template_pts3", torch.tensor(_TEMPLATE_PTS_3, dtype=torch.float32))
        self.register_buffer("template_pts4", torch.tensor(_TEMPLATE_PTS_4, dtype=torch.float32))
        self.register_buffer("template_pts5", torch.tensor(_TEMPLATE_PTS_5, dtype=torch.float32))

        self.param_conv = nn.Conv2d(in_channels, 6, kernel_size=k, stride=k, padding=0)
        self.bn1 = nn.BatchNorm2d(6)

        # Rigidity controller: a single Conv2d with bias, no BN. Initial bias
        # is set so sigmoid(bias) ~= rigidity_init.
        self.rigidity_conv = nn.Conv2d(in_channels, 1, kernel_size=k, stride=k, padding=0, bias=True)
        nn.init.zeros_(self.rigidity_conv.weight)
        init_logit = float(torch.logit(torch.tensor(rigidity_init).clamp(1e-4, 1 - 1e-4)).item())
        nn.init.constant_(self.rigidity_conv.bias, init_logit)

        self.act = nn.Hardswish(inplace=True)

        self.offset2mask = nn.Conv2d(4 * k * k, k * k, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(k * k)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.s, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

        # ---- Residual + LayerScale (zero-init) -> IDENTITY at initialisation ----
        # GLD_RESIDUAL (default on): when in==out and stride==1, wrap the operator
        # as out = x + gamma * tconv(x) with gamma zero-initialised. At init the
        # layer is a pure identity, so warm-started backbone features (base r18,
        # val R=0.70) flow through UNCHANGED instead of being corrupted by a
        # randomly-initialised inserted layer (which collapsed recall to ~0.50).
        # GLD geometry is then learned gradually as gamma grows during training.
        # This resolves the detection-recall bottleneck without touching GLD loss.
        _res_env = os.environ.get("GLD_RESIDUAL", "1").strip().lower()
        _res_on = _res_env not in ("0", "false", "no", "off")
        # Identity residual: in==out & s==1 (e.g. idx6 128->128, P3 path).
        self.use_residual = _res_on and (self.in_channels == self.out_channels) and (self.s == 1)
        # Projection residual: in!=out & s==1 (e.g. idx8 128->256, P4/P5 path).
        # GLD_PROJ_SHORTCUT (default on) adds a 1x1 conv shortcut initialised to
        # CHANNEL-TILE the input (out[i] = in[i % in_ch]) so at init the layer
        # passes a faithful (repeated) copy of the warm-started features into the
        # next layer instead of random garbage. gamma zero-init keeps tconv off
        # at step 0, so P4/P5 features are preserved just like idx6's P3 path.
        _proj_env = os.environ.get("GLD_PROJ_SHORTCUT", "1").strip().lower()
        _proj_on = _proj_env not in ("0", "false", "no", "off")
        self.use_proj = (not self.use_residual) and _proj_on and (self.s == 1)

        if self.use_residual:
            self.gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
            self.proj = None
        elif self.use_proj:
            self.gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
            # 1x1 channel-tile projection (in->out), zero bias
            self.proj = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False)
            with torch.no_grad():
                self.proj.weight.zero_()
                for i in range(self.out_channels):
                    self.proj.weight[i, i % self.in_channels, 0, 0] = 1.0
        else:
            self.gamma = None
            self.proj = None

        # Training-time scratch buffers (never serialized, never deepcopied).
        self.last_rigidity = None       # (B, 1, h, w), h=H/k, w=W/k
        self.last_elasticity = None     # (B, 1, h, w)
        self.last_input = None          # (B, C_in, H, W)
        self.last_output = None         # (B, C_out, H/s, W/s)

    # ------------------------------------------------------------------
    # serialization helpers
    # ------------------------------------------------------------------
    def __getstate__(self):
        # ``deepcopy`` (used by Ultralytics EMA and checkpointing) refuses to
        # copy non-leaf tensors with grad. Strip all training scratch buffers.
        state = self.__dict__.copy()
        for k in ("last_rigidity", "last_elasticity", "last_input", "last_output"):
            state[k] = None
        return state

    def export_strip(self) -> "GLDTrapezoidConv":
        """Drop all training scratch state. Call before export for safety."""
        self.last_rigidity = None
        self.last_elasticity = None
        self.last_input = None
        self.last_output = None
        return self

    # ------------------------------------------------------------------
    # template selection
    # ------------------------------------------------------------------
    def _template_points(self, k: int, device, dtype) -> torch.Tensor:
        if k == 2:
            return self.template_pts2.to(device, dtype)
        if k == 3:
            return self.template_pts3.to(device, dtype)
        if k == 4:
            return self.template_pts4.to(device, dtype)
        if k == 5:
            return self.template_pts5.to(device, dtype)
        raise ValueError(f"Unsupported GLDTrapezoidConv window_size={k}.")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.K
        device, dtype = x.device, x.dtype

        params = self.act(self.bn1(self.param_conv(x)))                      # (B, 6, H/k, W/k)
        rigidity_logit = self.rigidity_conv(x)                               # (B, 1, H/k, W/k)
        rigidity = torch.sigmoid(rigidity_logit)
        elasticity = 1.0 - rigidity

        params_p = params.permute(0, 2, 3, 1).contiguous()                   # (B, h, w, 6)
        elasticity_p = elasticity.permute(0, 2, 3, 1).contiguous().squeeze(-1)  # (B, h, w)

        Sx = F.softplus(params_p[..., 0]) + 1e-3
        Sy = F.softplus(params_p[..., 1]) + 1e-3
        angle = params_p[..., 2]
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        # *** Hard-gated bound ***: shear/trans = tanh(p) * elasticity * max_*
        # When elasticity -> 0 the cell is rigid; when elasticity -> 1 the
        # cell recovers the conference TConv range.
        shear = torch.tanh(params_p[..., 3]) * elasticity_p * self.max_shear
        Tx = torch.tanh(params_p[..., 4]) * elasticity_p * self.max_trans
        Ty = torch.tanh(params_p[..., 5]) * elasticity_p * self.max_trans

        a11 = Sx * cos
        a12 = cos * Sx * shear + sin * Sy
        a21 = -Sx * sin
        a22 = -sin * Sx * shear + cos * Sy

        pts = self._template_points(k, device, dtype)
        x_pts, y_pts = pts[:, 0], pts[:, 1]

        x_trans = a11.unsqueeze(-1) * x_pts + a12.unsqueeze(-1) * y_pts + Tx.unsqueeze(-1)
        y_trans = a21.unsqueeze(-1) * x_pts + a22.unsqueeze(-1) * y_pts + Ty.unsqueeze(-1)
        offsets = torch.cat(
            [torch.abs(x_trans), torch.abs(y_trans), 1 - torch.abs(x_trans), 1 - torch.abs(y_trans)], dim=-1
        )

        mask_in = offsets.permute(0, 3, 1, 2).contiguous()
        mask = self.act(self.bn2(self.offset2mask(mask_in)))
        mask = F.pixel_shuffle(mask, upscale_factor=k)
        if (mask.shape[2] != H) or (mask.shape[3] != W):
            mask = F.interpolate(mask, size=(H, W), mode="nearest")
        gated = x * mask
        if self.tail == "tconv":
            # exactly the conference TConv tail (no BN; Hardswish ∘ SiLU)
            out = self.act(self.silu(self.conv(gated)))
        else:
            out = self.silu(self.bn(self.conv(gated)))

        # Residual / projection shortcut with zero-init LayerScale:
        #   idx6 (in==out): out = x + gamma * tconv(x)          [identity at init]
        #   idx8 (in!=out): out = proj(x) + gamma * tconv(x)    [channel-tile at init]
        # gamma zero-init → at step 0 the geometry branch is off and the
        # warm-started features flow through (identity or faithful channel-tile),
        # preserving base r18 recall. gamma grows during training as GLD helps.
        #
        # getattr() with defaults keeps this BACKWARD-COMPATIBLE: checkpoints
        # pickled by an older tconv.py (no use_proj/proj attrs) deserialize into
        # instances lacking these fields; getattr falls back to the pre-residual
        # behaviour (plain tconv output) instead of raising AttributeError.
        _gamma = getattr(self, "gamma", None)
        if getattr(self, "use_residual", False) and _gamma is not None:
            out = x + _gamma * out
        elif getattr(self, "use_proj", False) and _gamma is not None:
            out = getattr(self, "proj")(x) + _gamma * out

        # Training-only scratch buffers (no graph leak in eval/export).
        if self.training and torch.is_grad_enabled():
            self.last_rigidity = rigidity
            self.last_elasticity = elasticity
            self.last_input = x
            self.last_output = out
        else:
            self.last_rigidity = None
            self.last_elasticity = None
            self.last_input = None
            self.last_output = None

        return out


__all__ = ["TrapezoidConv", "GLDTrapezoidConv"]
