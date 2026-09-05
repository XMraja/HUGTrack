"""kd_anchor.py — Detection-Anchored Geometry Distillation (DAGD).

Core innovation (resolves the "trains worse every epoch" law observed across
v2/v3/v4/slow_gamma/gld_only):

  Every GLD training run degrades detection because the geometry loss gradient
  and the detection loss gradient CONFLICT on the shared backbone — training GLD
  pulls the warm-started base-r18 features (the global detection optimum) away.
  Freezing helps but that is "relying entirely on the pretrained weights".

  DAGD keeps a FROZEN self-teacher: a deep copy of the student captured with all
  residual/projection gammas forced to ZERO, i.e. the pure base-r18 detector.
  A distillation loss anchors the student's DETECTION HEAD OUTPUT to the teacher:

      L = L_det(student) + gld_lambda * L_geometry + kd_lambda * KD(student, teacher)

  KD is applied on the raw Detect outputs (P3/P4/P5 logits), NOT on intermediate
  features. So the student must produce the SAME detections as base r18 (recall
  preserved), while gamma is free to encode geometry in the detection head's
  NULL SPACE (high-dim features -> few detection outputs => ample room). The
  geometry branch can therefore keep improving association WITHOUT moving the
  detection output — the gradient conflict is removed by construction.

Deployability: the teacher is training-only, stripped before export exactly like
gld_aux. Inference graph is unchanged.
"""
from __future__ import annotations
import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_all_gammas(model: nn.Module) -> int:
    """Force every GLDTrapezoidConv residual/projection gamma to 0 so the copy
    behaves as the pure warm-started base-r18 detector. Returns count zeroed."""
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("gamma") or ".gamma" in name:
                p.zero_()
                n += 1
    return n


class KDAnchor:
    """Holds a frozen self-teacher (gamma=0 => base r18) and computes a KD loss
    that anchors the student's detection outputs to it.

    IMPORTANT: this is a PLAIN object, NOT an nn.Module. That is deliberate — if
    it were a submodule of the student, the teacher's weights would land in the
    student's state_dict (2x checkpoint size) and get deep-copied by EMA. As a
    plain attribute (model.kd_anchor = KDAnchor(...)) the teacher stays out of
    the nn graph, state_dict, optimizer, and export entirely.

    Call ``kd_loss(student_preds, img)`` inside DetectionModel.loss().
    """

    def __init__(self, student: nn.Module, kd_lambda: float = 1.0):
        self.kd_lambda = float(kd_lambda)
        # deep copy the student, zero its gammas -> pure base-r18 detector
        teacher = copy.deepcopy(student)
        for attr in ("gld_aux", "kd_anchor"):
            if hasattr(teacher, attr):
                try:
                    delattr(teacher, attr)
                except Exception:
                    setattr(teacher, attr, None)
        n = _zero_all_gammas(teacher)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher
        self._n_gamma = n

    def __deepcopy__(self, memo):
        """Ultralytics EMA does copy.deepcopy(model). The EMA copy is only used
        for validation / final weights and never computes KD loss, so it does
        NOT need the (heavy) teacher. Return a lightweight copy with teacher=None
        to avoid duplicating ~7.4M teacher params and copying its scratch state.
        """
        light = KDAnchor.__new__(KDAnchor)
        light.kd_lambda = 0.0     # KD disabled on the EMA copy
        light.teacher = None
        light._n_gamma = getattr(self, "_n_gamma", 0)
        memo[id(self)] = light
        return light

    @torch.no_grad()
    def _teacher_forward(self, img: torch.Tensor):
        """Return the teacher's RAW Detect head maps (list of P3/P4/P5 tensors).

        In eval mode ultralytics Detect returns (decoded, raw_list); in train
        mode it returns raw_list directly. We keep the teacher in eval (stable
        BN) and extract the raw list from the tuple so the format matches the
        student's train-mode output.
        """
        # device safety: model.to() does NOT move the teacher (plain attribute,
        # not an nn submodule), so align it to the current batch device here.
        try:
            t_dev = next(self.teacher.parameters()).device
            if t_dev != img.device:
                self.teacher.to(img.device)
        except StopIteration:
            pass
        self.teacher.eval()
        out = self.teacher(img)
        return self._extract_raw(out)

    @staticmethod
    def _extract_raw(out):
        """Normalise a Detect output (train list OR eval (decoded, raw) tuple)
        into the raw feature-map list used for KD."""
        if isinstance(out, (list, tuple)):
            # eval-mode tuple: (decoded_tensor, raw_list)
            if len(out) == 2 and isinstance(out[1], (list, tuple)):
                return list(out[1])
            # train-mode: already a list of raw maps
            return list(out)
        return [out]

    def kd_loss(self, student_preds, img: torch.Tensor) -> torch.Tensor:
        """MSE between student and teacher raw Detect outputs (P3/P4/P5)."""
        if self.teacher is None or self.kd_lambda <= 0:
            return torch.zeros((), device=img.device)
        s_list = self._extract_raw(student_preds)
        t_list = self._teacher_forward(img)

        if len(s_list) != len(t_list):
            m = min(len(s_list), len(t_list))
            s_list, t_list = s_list[:m], t_list[:m]

        loss = torch.zeros((), device=img.device)
        cnt = 0
        for s, t in zip(s_list, t_list):
            if isinstance(s, torch.Tensor) and isinstance(t, torch.Tensor) and s.shape == t.shape:
                loss = loss + F.mse_loss(s, t.detach())
                cnt += 1
        if cnt > 0:
            loss = loss / cnt
        return loss
