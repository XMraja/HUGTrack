# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""GLD-TConv journal v2 detection trainer.

Adds the following over the original Ultralytics DetectionTrainer:

  * Loads frame-level instance prompts (``train_instances_with_prompts.jsonl``)
    in addition to the v1 image-level rigidity targets, and renders three
    optional per-batch tensors used by ``GldAuxModule``:
        - ``gld_rigidity``        (B,)             image-level scalar
        - ``gld_spatial_target``  (B, 1, hg, wg)   per-cell rigidity map
        - ``gld_spatial_weight``  (B, 1, hg, wg)   bbox-occupancy weight
        - ``gld_mask_target``     (B, 1, hg, wg)   union mask (bbox fill)
        - ``gld_mask_valid``      (B,) bool        which images have a target
        - ``gld_rois``            (N, 5) float     [b_idx, x1, y1, x2, y2] in
                                                  input-image pixel coords
        - ``gld_prompt_rows``     (N,) long        text-bank row indices
  * Reports a hit-rate the first batch and aborts with a clear error if the
    hit-rate is below ``--gld-min-hit-rate`` (default 0.3).
  * Loads the frozen CLIP text bank from ``args.gld_text_bank`` and pushes it
    into ``model.gld_aux.set_text_bank(...)`` once per process.
  * Constructs ``model.gld_aux`` exactly when ``args.gld_lambda > 0``; otherwise
    the deployment graph is unchanged.

The deployable inference graph is NOT touched: ``model.gld_aux`` is mounted on
``DetectionModel`` (not on the YAML-parsed backbone module list), and an
``export_strip`` hook removes it before ONNX/edge export.
"""

from __future__ import annotations

import math
import random
import json
import re
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.gld_aux import GldAuxModule, iter_gld_layers
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """Detection trainer with the GLD-TConv journal v2 auxiliary supervision."""

    # ------------------------------------------------------------------
    # dataset
    # ------------------------------------------------------------------
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    # ------------------------------------------------------------------
    # GLD: side-channel data loading
    # ------------------------------------------------------------------
    def _gld_enabled(self) -> bool:
        return float(getattr(self.args, "gld_lambda", 0.0) or 0.0) > 0.0

    def _gld_grid_size(self) -> int:
        """Spatial-target grid size. Picks H/k/k for window=4 + stride=4 by default."""
        return int(getattr(self.args, "gld_grid_size", 0) or max(1, int(self.args.imgsz // 32)))

    def _gld_load_image_targets(self):
        if hasattr(self, "_gld_image_targets"):
            return self._gld_image_targets
        self._gld_image_targets = {}
        path = getattr(self.args, "gld_target", None)
        if not path:
            return self._gld_image_targets
        path = Path(str(path)).expanduser()
        if not path.exists():
            LOGGER.warning(f"[GLD] image-level target file not found: {path}. Skipping image-level loss.")
            return self._gld_image_targets
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                key = (str(item.get("split", "")), str(item.get("seq", "")), int(item.get("frame", -1)))
                self._gld_image_targets[key] = {
                    "rigidity": float(item.get("rigidity_target", -1.0)),
                    "prompt_rows": list(item.get("prompt_rows", [])),
                }
        LOGGER.info(f"[GLD] loaded {len(self._gld_image_targets)} image-level targets from {path}")
        return self._gld_image_targets

    def _gld_load_instance_index(self):
        """Load per-instance bbox + prompt_row + rigidity, grouped by (split, seq, frame)."""
        if hasattr(self, "_gld_instance_index"):
            return self._gld_instance_index
        self._gld_instance_index = defaultdict(list)
        path = getattr(self.args, "gld_instance_target", None)
        if not path:
            return self._gld_instance_index
        path = Path(str(path)).expanduser()
        if not path.exists():
            LOGGER.warning(f"[GLD] instance target file not found: {path}. ROI/spatial losses disabled.")
            return self._gld_instance_index
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                key = (str(item.get("split", "")), str(item.get("seq", "")), int(item.get("frame", -1)))
                bbox = item.get("bbox") or item.get("prompt_bbox")
                if not bbox or len(bbox) != 4:
                    continue
                rec = {
                    "bbox": [float(v) for v in bbox],          # xywh in original-image px
                    "img_w": int(item.get("image_width", 0)),
                    "img_h": int(item.get("image_height", 0)),
                    "rigidity": float(item.get("rigidity_target", 0.5)),
                    "prompt_row": int(item.get("prompt_row", -1)),
                    "track_id": int(item.get("track_id", -1)),
                }
                self._gld_instance_index[key].append(rec)
        LOGGER.info(
            f"[GLD] loaded {sum(len(v) for v in self._gld_instance_index.values())} instances "
            f"across {len(self._gld_instance_index)} frames from {path}"
        )
        return self._gld_instance_index

    @staticmethod
    def _gld_key_from_im_file(im_file):
        path = Path(str(im_file))
        parts = path.parts
        seq, seq_idx = None, -1
        for i, part in enumerate(parts):
            if part.startswith("geovesselmot-"):
                seq, seq_idx = part, i
        if seq is None:
            m = re.search(r"(geovesselmot-\d+)", path.stem)
            if m is not None:
                seq = m.group(1)
        split = None
        for i in range(seq_idx - 1, -1, -1):
            if parts[i] in {"train", "val", "test"}:
                split = parts[i]
                break
        if split is None:
            for part in parts:
                if part in {"train", "val", "test"}:
                    split = part
                    break
        match = re.search(r"(\d+)$", path.stem)
        if seq is None or split is None or match is None:
            return None
        return split, seq, int(match.group(1))

    # ------------------------------------------------------------------
    # GLD: batch tensor construction
    # ------------------------------------------------------------------
    def _add_gld_to_batch(self, batch):
        if not self._gld_enabled():
            return batch

        img_targets = self._gld_load_image_targets()
        ins_index = self._gld_load_instance_index()
        H = int(batch["img"].shape[2])
        W = int(batch["img"].shape[3])
        B = int(batch["img"].shape[0])
        G = self._gld_grid_size()

        device = self.device
        dtype = torch.float32

        rigidity = torch.full((B,), -1.0, device=device, dtype=dtype)
        spatial_target = torch.zeros((B, 1, G, G), device=device, dtype=dtype)
        spatial_weight = torch.zeros((B, 1, G, G), device=device, dtype=dtype)
        mask_target = torch.zeros((B, 1, G, G), device=device, dtype=dtype)
        mask_valid = torch.zeros((B,), device=device, dtype=torch.bool)

        rois = []
        prompt_rows = []
        track_ids = []
        n_hit = 0

        for b, im_file in enumerate(batch.get("im_file", [])):
            key = self._gld_key_from_im_file(im_file)
            if key is None:
                continue
            n_hit += int((key in img_targets) or (key in ins_index))
            rec = img_targets.get(key)
            if rec is not None and rec["rigidity"] >= 0.0:
                rigidity[b] = float(rec["rigidity"])

            instances = ins_index.get(key, [])
            if not instances:
                continue
            mask_valid[b] = True
            for ins in instances:
                bw, bh = ins["img_w"], ins["img_h"]
                if bw <= 0 or bh <= 0:
                    continue
                x, y, w, h = ins["bbox"]
                # original-image px -> letterboxed (H, W) px (assume square 320 input
                # produced by Ultralytics LetterBox: scale = min(W/bw, H/bh), pad)
                scale = min(W / bw, H / bh)
                pad_x = (W - bw * scale) * 0.5
                pad_y = (H - bh * scale) * 0.5
                x1 = x * scale + pad_x
                y1 = y * scale + pad_y
                x2 = (x + w) * scale + pad_x
                y2 = (y + h) * scale + pad_y
                # clamp
                x1 = max(0.0, min(W - 1.0, x1))
                y1 = max(0.0, min(H - 1.0, y1))
                x2 = max(0.0, min(W - 1.0, x2))
                y2 = max(0.0, min(H - 1.0, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                # render onto the spatial-target grid
                gx1 = int(math.floor(x1 / W * G))
                gy1 = int(math.floor(y1 / H * G))
                gx2 = int(math.ceil(x2 / W * G))
                gy2 = int(math.ceil(y2 / H * G))
                gx1 = max(0, min(G - 1, gx1))
                gy1 = max(0, min(G - 1, gy1))
                gx2 = max(gx1 + 1, min(G, gx2))
                gy2 = max(gy1 + 1, min(G, gy2))

                rig = float(ins["rigidity"])
                # paint rigidity, but compose multiple instances by *weighted* mean
                roi_area = (gx2 - gx1) * (gy2 - gy1)
                if roi_area <= 0:
                    continue
                w_now = spatial_weight[b, 0, gy1:gy2, gx1:gx2]
                t_now = spatial_target[b, 0, gy1:gy2, gx1:gx2]
                spatial_target[b, 0, gy1:gy2, gx1:gx2] = (t_now * w_now + rig) / (w_now + 1.0)
                spatial_weight[b, 0, gy1:gy2, gx1:gx2] = w_now + 1.0
                mask_target[b, 0, gy1:gy2, gx1:gx2] = 1.0

                rois.append([float(b), x1, y1, x2, y2])
                prompt_rows.append(int(ins["prompt_row"]))
                track_ids.append(int(ins.get("track_id", -1)))

        # 方案2: adjacent-frame ROIs for cross-frame IDC positives
        # Use frame F-1's bbox coords on frame F's feature map (valid approximation
        # for slow-moving vessels; frames are consecutive ~25fps UAV footage).
        if getattr(self.args, "gld_pair_frames", False):
            for b, im_file in enumerate(batch.get("im_file", [])):
                key = self._gld_key_from_im_file(im_file)
                if key is None:
                    continue
                split, seq, frame = key
                for delta in (-1, 1):
                    for ins in ins_index.get((split, seq, frame + delta), []):
                        tid = int(ins.get("track_id", -1))
                        if tid < 0:
                            continue
                        bw, bh = ins["img_w"], ins["img_h"]
                        if bw <= 0 or bh <= 0:
                            continue
                        x, y, w, h = ins["bbox"]
                        scale = min(W / bw, H / bh)
                        pad_x = (W - bw * scale) * 0.5
                        pad_y = (H - bh * scale) * 0.5
                        x1 = max(0.0, min(W - 1.0, x * scale + pad_x))
                        y1 = max(0.0, min(H - 1.0, y * scale + pad_y))
                        x2 = max(0.0, min(W - 1.0, (x + w) * scale + pad_x))
                        y2 = max(0.0, min(H - 1.0, (y + h) * scale + pad_y))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        rois.append([float(b), x1, y1, x2, y2])
                        prompt_rows.append(int(ins.get("prompt_row", -1)))
                        track_ids.append(tid)

        # normalize spatial_weight to {0, 1} for the loss helper (multi-instance overlap
        # already averaged into spatial_target by weighted mean)
        spatial_weight = (spatial_weight > 0).to(dtype)

        batch["gld_rigidity"] = rigidity
        batch["gld_spatial_target"] = spatial_target
        batch["gld_spatial_weight"] = spatial_weight
        batch["gld_mask_target"] = mask_target
        batch["gld_mask_valid"] = mask_valid
        if rois:
            batch["gld_rois"] = torch.tensor(rois, device=device, dtype=dtype)
            batch["gld_prompt_rows"] = torch.tensor(prompt_rows, device=device, dtype=torch.long)
            batch["gld_track_ids"] = torch.tensor(track_ids, device=device, dtype=torch.long)

        # --- diagnostics: report match rate the first time we touch a batch ---
        if not getattr(self, "_gld_first_batch_logged", False):
            self._gld_first_batch_logged = True
            min_hit = float(getattr(self.args, "gld_min_hit_rate", 0.3) or 0.0)
            ratio = n_hit / max(1, B)
            LOGGER.info(f"[GLD] first-batch hit-rate = {n_hit}/{B} = {ratio:.2f}")
            if ratio < min_hit:
                raise RuntimeError(
                    f"[GLD] hit-rate {ratio:.2f} < gld_min_hit_rate {min_hit:.2f}. "
                    f"Check that frame indices in {self.args.gld_target} / "
                    f"{self.args.gld_instance_target} match the image filenames "
                    f"in your detection dataset."
                )
        return batch

    # ------------------------------------------------------------------
    # standard preprocess + GLD injection
    # ------------------------------------------------------------------
    def _gld_update_warmup(self):
        """方案3: linear lambda warmup. Called from preprocess_batch (the base
        trainer dispatches epoch hooks via run_callbacks, not method override,
        so we update gain here where the call is guaranteed)."""
        aux = getattr(getattr(self, "model", None), "gld_aux", None)
        if aux is None:
            return
        warmup = int(getattr(self.args, "gld_lambda_warmup", 0) or 0)
        target = float(getattr(self.args, "gld_lambda", 0.0) or 0.0)
        ep = int(getattr(self, "epoch", 0) or 0)
        aux.gain = target * (ep + 1) / warmup if (warmup > 0 and ep < warmup) else target

    def preprocess_batch(self, batch):
        self._gld_update_warmup()
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        batch = self._add_gld_to_batch(batch)
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args
        self._setup_gld_module()

    # ------------------------------------------------------------------
    # GLD aux module construction + frozen text-bank loading
    # ------------------------------------------------------------------
    def _setup_gld_module(self):
        if not self._gld_enabled():
            return
        layers = list(iter_gld_layers(self.model))
        if not layers:
            LOGGER.warning("[GLD] gld_lambda > 0 but no GLDTrapezoidConv layers found. Auxiliary loss disabled.")
            return

        # Use the deepest GLD layer's output channels for ROI head input,
        # the shallowest layer's output channels for mask head input.
        roi_in = int(layers[-1].out_channels)
        mask_in = int(layers[0].out_channels) if float(getattr(self.args, "gld_w_mask", 0.5) or 0.0) > 0 else 0

        weights = dict(
            image=float(getattr(self.args, "gld_w_image", 1.0) or 0.0),
            spatial=float(getattr(self.args, "gld_w_spatial", 1.0) or 0.0),
            consistency=float(getattr(self.args, "gld_w_consistency", 0.5) or 0.0),
            mask=float(getattr(self.args, "gld_w_mask", 0.5) or 0.0),
            boundary=float(getattr(self.args, "gld_w_boundary", 0.5) or 0.0),
            language=float(getattr(self.args, "gld_w_language", 1.0) or 0.0),
            contrast=float(getattr(self.args, "gld_w_contrast", 0.0) or 0.0),
        )
        text_dim = int(getattr(self.args, "gld_text_dim", 512) or 512)
        roi_stride = float(getattr(self.args, "gld_roi_stride", 16.0) or 16.0)
        gain = float(getattr(self.args, "gld_lambda", 0.0) or 0.0)

        aux = GldAuxModule(
            roi_in_channels=roi_in,
            text_dim=text_dim,
            mask_in_channels=mask_in,
            weights=weights,
            roi_feature_stride=roi_stride,
            gain=gain,
        ).to(self.device)
        self.model.gld_aux = aux

        # frozen text bank
        bank_path = getattr(self.args, "gld_text_bank", None)
        if bank_path:
            bank_path = Path(str(bank_path)).expanduser()
            if bank_path.exists():
                bank = torch.from_numpy(np.load(str(bank_path))).float().to(self.device)
                aux.set_text_bank(bank)
                LOGGER.info(f"[GLD] loaded text-bank with shape {tuple(bank.shape)} from {bank_path}")
            else:
                LOGGER.warning(f"[GLD] text-bank file not found: {bank_path}. ROI-language loss disabled.")

        LOGGER.info(
            f"[GLD] auxiliary module attached: roi_in={roi_in} mask_in={mask_in} "
            f"text_dim={text_dim} weights={weights} gain={gain}"
        )

    # ------------------------------------------------------------------
    # validators / loss labels
    # ------------------------------------------------------------------
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        if self._gld_enabled():
            loss_names.extend(GldAuxModule.loss_names)
        self.loss_names = tuple(loss_names)
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch", "GPU_mem", *self.loss_names, "Instances", "Size",
        )

    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot)

    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return super().auto_batch(max_num_obj)
