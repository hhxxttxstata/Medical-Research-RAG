import os

_omp_threads = os.environ.get("OMP_NUM_THREADS", "").strip()
os.environ["OMP_NUM_THREADS"] = _omp_threads if _omp_threads.isdigit() else "8"
_mkl_threads = os.environ.get("MKL_NUM_THREADS", "").strip()
os.environ["MKL_NUM_THREADS"] = _mkl_threads if _mkl_threads.isdigit() else "8"

import argparse
import copy
import glob
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import binary_dilation, binary_fill_holes, label, zoom
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augmentations import get_overfit_transforms, get_train_transforms, get_val_transforms
from config_attention import Config
from get_slice_key import KeySliceLabeler

from models.resnet25d_attention import resnet25d_attention


class CTPreprocessor:
    def __init__(self, config):
        pp = config.preprocess
        self.hu_min = pp["hu_min"]
        self.hu_max = pp["hu_max"]
        self.slice_resolution = pp["slice_resolution"]
        self.use_roi_crop = pp["use_roi_crop"]
        self.roi_padding = pp["roi_padding"]
        self.keep_original_depth = getattr(config, "keep_original_depth", False)
        self.target_hw = getattr(config, "target_hw", (256, 256))
        self.use_multi_window = getattr(config, "use_multi_window", False)
        self.roi_mode = getattr(config, "roi_mode", "none")
        self.roi_body_threshold = getattr(config, "roi_body_threshold", -500)
        self.roi_fill_value = getattr(config, "roi_body_fill_value", -1000)
        self.roi_dilation = getattr(config, "roi_body_dilation", 3)
        if not self.keep_original_depth:
            self.model_input_size = config.input_size

    def _hu_normalize(self, volume):
        volume = np.clip(volume, self.hu_min, self.hu_max)
        volume = (volume - self.hu_min) / (self.hu_max - self.hu_min)
        return volume

    def _resample_hw(self, volume, target_h, target_w):
        zoom_factors = (1.0, target_h / volume.shape[1], target_w / volume.shape[2])
        return zoom(volume, zoom_factors, order=1)

    def _resample_volume(self, volume, target_shape):
        zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
        return zoom(volume, zoom_factors, order=1)

    def _compute_body_mask(self, volume_hu):
        body = volume_hu > self.roi_body_threshold
        d = volume_hu.shape[0]
        for i in range(d):
            sl = body[i]
            lbl, n = label(sl)
            if n == 0:
                body[i] = False
                continue
            sizes = np.bincount(lbl.ravel())
            if len(sizes) <= 1:
                body[i] = False
                continue
            sizes[0] = 0
            largest = sizes.argmax()
            body[i] = lbl == largest
            body[i] = binary_fill_holes(body[i])
        if self.roi_dilation > 0:
            struct = np.ones((1, self.roi_dilation, self.roi_dilation), dtype=bool)
            body = binary_dilation(body, structure=struct, iterations=1)
        return body

    def _apply_roi_mask(self, volume_hu):
        body_mask = self._compute_body_mask(volume_hu)
        roi_volume = volume_hu.copy()
        roi_volume[~body_mask] = self.roi_fill_value
        return roi_volume, body_mask

    def __call__(self, volume_3d):
        volume_raw = volume_3d.astype(np.float32)

        if self.roi_mode == "body":
            volume_raw, _ = self._apply_roi_mask(volume_raw)

        if self.use_multi_window:
            th, tw = self.target_hw
            if volume_raw.shape[1] != th or volume_raw.shape[2] != tw:
                volume_raw = self._resample_hw(volume_raw, th, tw)
            return volume_raw

        volume = self._hu_normalize(volume_raw)

        if self.keep_original_depth:
            th, tw = self.target_hw
            if volume.shape[1] != th or volume.shape[2] != tw:
                volume = self._resample_hw(volume, th, tw)
        else:
            if tuple(volume.shape) != tuple(self.model_input_size):
                volume = self._resample_volume(volume, self.model_input_size)

        return volume.astype(np.float32)


class PEDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, is_train=True, slice_csv=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.preprocessor = CTPreprocessor(Config)
        self.slab_depth = Config.slice_thickness
        self.slab_half = self.slab_depth // 2
        self.num_slabs_train = Config.num_slabs_train
        self.eval_stride = Config.eval_stride
        self.is_train = is_train

        self.filter_empty_slices = getattr(Config, "filter_empty_slices", False)
        self.chest_body_hu_threshold = getattr(Config, "chest_body_hu_threshold", -600)
        self.chest_lung_hu_low = getattr(Config, "chest_lung_hu_low", -1000)
        self.chest_lung_hu_high = getattr(Config, "chest_lung_hu_high", -300)
        self.chest_body_ratio_threshold = getattr(Config, "chest_body_ratio_threshold", 0.05)
        self.chest_lung_ratio_threshold = getattr(Config, "chest_lung_ratio_threshold", 0.02)
        self.chest_segment_margin = getattr(Config, "chest_segment_margin", 8)
        self.chest_segment_margin_upper = getattr(Config, "chest_segment_margin_upper", 20)
        self.chest_segment_margin_lower = getattr(Config, "chest_segment_margin_lower", 8)
        self.chest_min_filtered_slices = getattr(Config, "chest_min_filtered_slices", 16)
        self.body_threshold = getattr(Config, "body_threshold", -800)
        self.lung_threshold_low = getattr(Config, "lung_threshold_low", -1000)
        self.lung_threshold_high = getattr(Config, "lung_threshold_high", -300)
        self.enhance_threshold = getattr(Config, "enhance_threshold", 100)
        self.body_ratio_threshold = getattr(Config, "body_ratio_threshold", 0.05)
        self.lung_ratio_threshold = getattr(Config, "lung_ratio_threshold", 0.03)
        self.enhance_ratio_threshold = getattr(Config, "enhance_ratio_threshold", 0.005)
        self.min_valid_slices_per_slab = getattr(Config, "min_valid_slices_per_slab", 1)
        self.min_candidate_slabs = getattr(Config, "min_candidate_slabs", 16)

        pp = Config.preprocess
        self.hu_min = pp["hu_min"]
        self.hu_max = pp["hu_max"]

        self.use_multi_window = getattr(Config, "use_multi_window", False)
        if self.use_multi_window:
            self.windows = getattr(Config, "windows", [(-1000, 600)])
            self.num_windows = len(self.windows)
            self.model_in_channels = self.slab_depth * self.num_windows
        else:
            self.windows = None
            self.num_windows = 1
            self.model_in_channels = self.slab_depth

        if is_train and slice_csv is not None:
            self.key_labeler = KeySliceLabeler(
                slice_csv=slice_csv, case_id_width=10, save_clean_path="slice_labels_clean.csv"
            )
        else:
            self.key_labeler = None

    def _hu_to_norm(self, hu_val):
        return (hu_val - self.hu_min) / (self.hu_max - self.hu_min)

    def _apply_multi_window(self, slab):
        channels = []
        for w_min, w_max in self.windows:
            windowed = np.clip(slab, w_min, w_max)
            windowed = (windowed - w_min) / (w_max - w_min)
            channels.append(windowed.astype(np.float32))
        return np.concatenate(channels, axis=0)

    def __len__(self):
        return len(self.file_paths)

    def _to_dhw(self, arr):
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {arr.shape}")

        if arr.shape[0] >= 128 and arr.shape[1] >= 128 and arr.shape[2] < arr.shape[0]:
            arr = np.transpose(arr, (2, 0, 1))
        return arr

    def _maybe_pad_volume(self, volume):
        depth = volume.shape[0]
        if depth < self.slab_depth:
            pad_needed = self.slab_depth - depth
            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")
        return volume

    @staticmethod
    def _find_continuous_segments(bool_array):
        segments = []
        in_segment = False
        start = 0
        for i, val in enumerate(bool_array):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                segments.append((start, i - 1))
                in_segment = False
        if in_segment:
            segments.append((start, len(bool_array) - 1))
        return segments

    def _filter_slices(self, volume_hu, key_slices=None):
        depth = volume_hu.shape[0]

        body_area = (volume_hu > self.chest_body_hu_threshold).mean(axis=(1, 2))
        lung_area = ((volume_hu > self.chest_lung_hu_low) & (volume_hu < self.chest_lung_hu_high)).mean(axis=(1, 2))

        valid_lung = lung_area > self.chest_lung_ratio_threshold
        lung_indices = np.where(valid_lung)[0]

        fallback_used = False
        filter_method = "lung_range"

        if len(lung_indices) >= self.chest_min_filtered_slices:
            lung_first = int(lung_indices[0])
            lung_last = int(lung_indices[-1])
            seg_start = max(0, lung_first - self.chest_segment_margin_upper)
            seg_end = min(depth - 1, lung_last + self.chest_segment_margin_lower)
        else:
            valid_body = body_area > self.chest_body_ratio_threshold
            segments = self._find_continuous_segments(valid_body)
            if segments:
                filter_method = "body_ratio"
                main_seg = max(segments, key=lambda s: s[1] - s[0])
                seg_start = max(0, main_seg[0] - self.chest_segment_margin)
                seg_end = min(depth - 1, main_seg[1] + self.chest_segment_margin)
            else:
                filter_method = "fallback_full"
                fallback_used = True
                seg_start = 0
                seg_end = depth - 1

        filtered = list(range(seg_start, seg_end + 1))

        if key_slices:
            for ks in key_slices:
                if ks not in filtered:
                    for z in range(max(0, ks - 3), min(depth, ks + 4)):
                        if z not in filtered:
                            filtered.append(z)
            filtered = sorted(set(filtered))

        if len(filtered) < self.chest_min_filtered_slices:
            filtered = list(range(depth))
            fallback_used = True
            filter_method = "fallback_too_few"

        lung_range = [int(lung_indices[0]), int(lung_indices[-1])] if len(lung_indices) > 0 else None

        return filtered, {
            "filtered_range": [min(filtered), max(filtered)],
            "lung_indices_count": len(lung_indices),
            "lung_range": lung_range,
            "filter_method": filter_method,
            "main_segment_width": seg_end - seg_start + 1,
            "fallback_used": fallback_used,
        }

    def _get_candidate_slab_starts(self, filtered_indices):
        if not filtered_indices:
            return [], []

        segments = []
        seg_start = filtered_indices[0]
        for i in range(1, len(filtered_indices)):
            if filtered_indices[i] != filtered_indices[i - 1] + 1:
                segments.append({"start": seg_start, "end": filtered_indices[i - 1]})
                seg_start = filtered_indices[i]
        segments.append({"start": seg_start, "end": filtered_indices[-1]})

        candidates = []
        for seg in segments:
            seg_start, seg_end = seg["start"], seg["end"]
            max_start = seg_end - self.slab_depth + 1
            if seg_start <= max_start:
                for z in range(seg_start, max_start + 1):
                    candidates.append(z)

        return candidates, segments

    def _extract_slabs_train(self, volume, key_slices=None):
        depth_original = volume.shape[0]
        volume = self._maybe_pad_volume(volume)
        if depth_original < self.slab_depth:
            pad_before = (self.slab_depth - depth_original) // 2
        else:
            pad_before = 0

        if self.filter_empty_slices:
            filter_key_slices = None
            if key_slices:
                filter_key_slices = [int(z) + pad_before for z in key_slices]
            filtered_indices, filter_diag = self._filter_slices(volume, key_slices=filter_key_slices)
        else:
            filtered_indices = list(range(volume.shape[0]))
            filter_diag = {
                "filtered_range": [0, volume.shape[0] - 1],
                "lung_indices_count": volume.shape[0],
                "lung_range": [0, volume.shape[0] - 1],
                "filter_method": "disabled",
                "main_segment_width": volume.shape[0],
                "fallback_used": False,
            }

        after_filter = len(filtered_indices)
        candidate_starts, segments = self._get_candidate_slab_starts(filtered_indices)
        num_candidates = len(candidate_starts)
        num_samples = self.num_slabs_train

        pad_used = False
        fallback_used = filter_diag.get("fallback_used", False)

        key_starts_selected = []
        key_slices_covered = None
        if key_slices:
            adjusted_key_slices = [int(z) + pad_before for z in key_slices]
            key_slices_covered = True
            for ks in adjusted_key_slices:
                covering = [s for s in candidate_starts if s <= ks <= s + self.slab_depth - 1]
                if covering:
                    ideal = max(min(ks - self.slab_half, max(candidate_starts)), min(candidate_starts))
                    best = min(covering, key=lambda s: abs(s - ideal))
                    key_starts_selected.append(best)
                else:
                    key_slices_covered = False
                    pad_depth = volume.shape[0]
                    for z in range(max(0, ks - 3), min(pad_depth, ks + 4)):
                        if z not in filtered_indices:
                            filtered_indices.append(z)
                    filtered_indices = sorted(set(filtered_indices))
                    candidate_starts, segments = self._get_candidate_slab_starts(filtered_indices)
                    num_candidates = len(candidate_starts)
                    after_filter = len(filtered_indices)
                    covering_retry = [s for s in candidate_starts if s <= ks <= s + self.slab_depth - 1]
                    if covering_retry:
                        ideal = max(min(ks - self.slab_half, max(candidate_starts)), min(candidate_starts))
                        best = min(covering_retry, key=lambda s: abs(s - ideal))
                        key_starts_selected.append(best)
                        key_slices_covered = True
                    else:
                        print(
                            f"  WARNING: key slice {ks - pad_before} (adjusted={ks}) still not coverable for case with {len(filtered_indices)} filtered slices"
                        )

            key_starts_selected = sorted(set(key_starts_selected))

        num_key = len(key_starts_selected)

        if num_key >= num_samples:
            indices = np.linspace(0, num_key - 1, num_samples, dtype=int)
            selected = [key_starts_selected[i] for i in indices]
        else:
            need_extra = num_samples - num_key
            remaining = [s for s in candidate_starts if s not in key_starts_selected]
            if len(remaining) >= need_extra:
                extra = np.random.choice(remaining, size=need_extra, replace=False).tolist()
            elif len(remaining) > 0:
                extra = np.random.choice(remaining, size=need_extra, replace=True).tolist()
                pad_used = True
            else:
                extra = []
                if len(key_starts_selected) < num_samples:
                    repeats_needed = num_samples - len(key_starts_selected)
                    extra = np.random.choice(
                        key_starts_selected if key_starts_selected else candidate_starts,
                        size=repeats_needed,
                        replace=True,
                    ).tolist()
                    pad_used = True
            selected = sorted(key_starts_selected + extra)

        if key_slices and key_slices_covered:
            for ks_orig in key_slices:
                ks = int(ks_orig) + pad_before
                assert any(s <= ks <= s + self.slab_depth - 1 for s in selected), (
                    f"Key slice {ks_orig} not covered by selected slabs"
                )

        slabs = []
        for start in selected:
            slab = volume[start : start + self.slab_depth, :, :]
            slabs.append(slab)

        diagnostics = {
            "before_filter_slices": volume.shape[0],
            "after_filter_slices": after_filter,
            "filtered_range": filter_diag.get("filtered_range", [0, volume.shape[0] - 1]),
            "candidate_slabs": num_candidates,
            "used_slabs_train": len(slabs),
            "key_slices": key_slices if key_slices else [],
            "key_slices_covered": key_slices_covered,
            "fallback_used": fallback_used,
            "pad_used": pad_used,
            "key_centers_included": num_key,
            "selected_centers": [int(s) + self.slab_half - pad_before for s in selected],
            "pad_before": pad_before,
        }
        return slabs, diagnostics

    def _extract_slabs_eval(self, volume):
        depth_original = volume.shape[0]
        volume = self._maybe_pad_volume(volume)

        if self.filter_empty_slices:
            filtered_indices, filter_diag = self._filter_slices(volume, key_slices=None)
        else:
            filtered_indices = list(range(volume.shape[0]))
            filter_diag = {
                "filtered_range": [0, volume.shape[0] - 1],
                "lung_indices_count": volume.shape[0],
                "lung_range": [0, volume.shape[0] - 1],
                "filter_method": "disabled",
                "main_segment_width": volume.shape[0],
                "fallback_used": False,
            }

        after_filter = len(filtered_indices)
        candidate_starts, segments = self._get_candidate_slab_starts(filtered_indices)
        num_candidates = len(candidate_starts)

        num_eval_slabs = getattr(Config, "num_slabs_eval", 48)
        fallback_used = filter_diag.get("fallback_used", False)
        pad_used = False

        n = len(candidate_starts)
        if n > 1 and n > num_eval_slabs:
            indices = np.linspace(0, n - 1, num_eval_slabs, dtype=int)
            selected = [candidate_starts[i] for i in indices]
        elif n > 0:
            selected = candidate_starts
            if n < num_eval_slabs:
                extra_needed = num_eval_slabs - n
                repeats = np.random.choice(candidate_starts, size=extra_needed, replace=True).tolist()
                selected = selected + repeats
                pad_used = True
        else:
            selected = [0]

        slabs = []
        for start in selected:
            slab = volume[start : start + self.slab_depth, :, :]
            slabs.append(slab)

        diagnostics = {
            "before_filter_slices": volume.shape[0],
            "after_filter_slices": after_filter,
            "filtered_range": filter_diag.get("filtered_range", [0, volume.shape[0] - 1]),
            "candidate_slabs": num_candidates,
            "used_slabs_eval": len(slabs),
            "fallback_used": fallback_used,
            "pad_used": pad_used,
            "selected_centers": [int(s) + self.slab_half for s in selected],
            "selected_slab_starts": [int(s) for s in selected],
        }
        return slabs, diagnostics

    def load_and_preprocess_ct(self, file_path):
        try:
            nii = nib.load(file_path)
            ct_data = nii.get_fdata().astype(np.float32)
            ct_data = self._to_dhw(ct_data)
            ct_data = self.preprocessor(ct_data)
            return ct_data
        except Exception as exc:
            print(f"Error loading or processing {file_path}: {exc}")
            return None

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        volume = self.load_and_preprocess_ct(file_path)

        if volume is None:
            return None

        label_tensor = torch.tensor(label, dtype=torch.long)

        filename = os.path.basename(file_path)
        case_id = filename.replace("_enhance.nii.gz", "")

        if self.is_train and self.key_labeler is not None:
            key_slices = self.key_labeler.key_slices(case_id)
        else:
            key_slices = None

        if self.is_train:
            slabs, diagnostics = self._extract_slabs_train(volume, key_slices=key_slices)
        else:
            slabs, diagnostics = self._extract_slabs_eval(volume)

        if idx < 5:
            mode = "train" if self.is_train else "eval"
            raw_min = float(volume.min())
            raw_max = float(volume.max())
            depth_val = diagnostics.get("before_filter_slices", volume.shape[0])
            filtered_range = diagnostics.get("filtered_range", [0, volume.shape[0] - 1])
            key_slices_val = diagnostics.get("key_slices", None)
            key_slices_covered = diagnostics.get("key_slices_covered", None)
            fallback_used = diagnostics.get("fallback_used", False)
            pad_used = diagnostics.get("pad_used", False)
            log_parts = [
                f"case_id: {case_id}",
                f"depth={depth_val}",
                f"raw_HU=[{raw_min:.0f},{raw_max:.0f}]",
                f"before_filter_slices: {diagnostics['before_filter_slices']}",
                f"after_filter_slices: {diagnostics['after_filter_slices']}",
                f"filtered_range={filtered_range}",
                f"filter_method: {diagnostics.get('filter_method', 'N/A')}",
                f"lung_range: {diagnostics.get('lung_range', 'N/A')}",
                f"candidate_slabs: {diagnostics['candidate_slabs']}",
                f"used_slabs_{mode}: {len(slabs)}",
            ]
            if key_slices_val is not None:
                log_parts.append(f"key_slices={key_slices_val}")
                log_parts.append(f"key_slices_covered={key_slices_covered}")
            else:
                log_parts.append("key_slices=[]")
                log_parts.append("key_slices_covered=None")
            log_parts.append(f"fallback_used: {fallback_used}")
            log_parts.append(f"pad_used: {pad_used}")
            print(", ".join(log_parts))

        slab_tensors = []
        first_windowed = True
        for slab in slabs:
            if self.use_multi_window:
                slab = self._apply_multi_window(slab)
                if first_windowed and idx < 5:
                    print(f"  -> windowed slab min/max: [{slab.min():.4f}, {slab.max():.4f}]")
                    first_windowed = False
            if self.transform:
                slab = self.transform(slab)
            slab_tensors.append(torch.from_numpy(slab))

        if len(slab_tensors) == 1:
            slabs_tensor = slab_tensors[0].unsqueeze(0)
        else:
            slabs_tensor = torch.stack(slab_tensors, dim=0)

        selected_centers = diagnostics.get("selected_centers", [])

        slice_targets = []
        slice_mask = []
        slice_weights = []
        if self.key_labeler is not None and selected_centers:
            targets, mask, weights = self.key_labeler.supervision(
                case_id=case_id,
                selected_indices=selected_centers,
                case_label=label,
                positive_tolerance=self.slab_half,
            )
            slice_targets = targets.tolist()
            slice_mask = mask.tolist()
            slice_weights = weights.tolist()

        meta = {
            "file_path": file_path,
            "filename": filename,
            "case_id": case_id,
            "depth": diagnostics.get("before_filter_slices", int(volume.shape[0])),
            "candidate_slabs": diagnostics.get("candidate_slabs", len(slabs)),
            "used_slabs_eval": len(slabs),
            "before_filter_slices": diagnostics.get("before_filter_slices", 0),
            "after_filter_slices": diagnostics.get("after_filter_slices", 0),
            "filtered_range": diagnostics.get("filtered_range", [0, int(volume.shape[0]) - 1]),
            "fallback_used": diagnostics.get("fallback_used", False),
            "pad_used": diagnostics.get("pad_used", False),
            "key_slices": diagnostics.get("key_slices", []),
            "key_slices_covered": diagnostics.get("key_slices_covered", None),
            "key_centers_included": diagnostics.get("key_centers_included", 0),
            "selected_centers": selected_centers,
            "selected_slab_starts": diagnostics.get("selected_slab_starts", []),
            "slice_targets": slice_targets,
            "slice_mask": slice_mask,
            "slice_weights": slice_weights,
        }

        return slabs_tensor, label_tensor, meta


def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None

    inputs, labels, metas = zip(*batch)
    batch_inputs = torch.stack(inputs, dim=0)
    batch_labels = torch.stack(labels)
    return batch_inputs, batch_labels, list(metas)


def _extract_case_id_from_path(file_path):
    filename = os.path.basename(file_path)
    return filename.replace("_enhance.nii.gz", "")


def _load_annotated_case_ids(csv_path):
    if csv_path is None or not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path, dtype={"case_id": str})
    case_ids = set(df["case_id"].astype(str).str.zfill(10).unique())
    return case_ids


def load_data():
    print("Loading data...")
    normal_dir = os.path.join(Config.data_root, Config.normal_dir)
    pe_dir = os.path.join(Config.data_root, Config.pe_dir)

    print("Looking for data in:")
    print(f"  Normal: {normal_dir}")
    print(f"  PE: {pe_dir}")

    normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.nii.gz")))
    pe_files = sorted(glob.glob(os.path.join(pe_dir, "*.nii.gz")))

    print(f"Found {len(normal_files)} normal cases")
    print(f"Found {len(pe_files)} PE cases")

    if len(normal_files) == 0 or len(pe_files) == 0:
        raise ValueError(f"No data files found! Please check paths:\n  Normal: {normal_dir}\n  PE: {pe_dir}")

    all_files = normal_files + pe_files
    all_labels = [0] * len(normal_files) + [1] * len(pe_files)
    return all_files, all_labels


def split_data(all_files, all_labels, annotated_case_ids=None):
    annotated_case_ids = annotated_case_ids or set()

    if getattr(Config, "overfit_test_samples", None) is not None:
        print(f"Running overfit split with {Config.overfit_test_samples} samples.")
        normal_indices = [i for i, label in enumerate(all_labels) if label == 0]
        pe_indices = [i for i, label in enumerate(all_labels) if label == 1]
        num_per_class = Config.overfit_test_samples // 2

        if len(normal_indices) < num_per_class or len(pe_indices) < num_per_class:
            raise ValueError(f"Not enough samples for overfit test. Need {num_per_class} per class.")

        selected_indices = normal_indices[:num_per_class] + pe_indices[:num_per_class]
        all_files = [all_files[i] for i in selected_indices]
        all_labels = [all_labels[i] for i in selected_indices]

        test_size_overfit = 0.2 if len(all_files) > 1 else 0.0
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            all_files,
            all_labels,
            test_size=test_size_overfit,
            stratify=all_labels,
            random_state=42,
        )
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files,
            train_val_labels,
            test_size=0.2,
            stratify=train_val_labels,
            random_state=42,
        )

        return {
            "train": (train_files, train_labels),
            "val": (val_files, val_labels),
            "test": (test_files, test_labels),
        }

    annotated_indices = []
    unannotated_indices = []
    for i, fp in enumerate(all_files):
        cid = _extract_case_id_from_path(fp)
        if cid in annotated_case_ids:
            annotated_indices.append(i)
        else:
            unannotated_indices.append(i)

    annotated_files = [all_files[i] for i in annotated_indices]
    annotated_labels = [all_labels[i] for i in annotated_indices]
    unannotated_files = [all_files[i] for i in unannotated_indices]
    unannotated_labels = [all_labels[i] for i in unannotated_indices]

    num_total = len(all_files)
    num_annotated = len(annotated_files)
    num_unannotated = len(unannotated_files)

    print("\nData split (8:1:1, annotated→train only):")
    print(f"  Total cases: {num_total}")
    print(
        f"  Annotated cases (forced → train): {num_annotated}  "
        f"(labels: {sum(annotated_labels)} pos / {num_annotated - sum(annotated_labels)} neg)"
    )
    print(
        f"  Unannotated cases (split normally): {num_unannotated}  "
        f"(labels: {sum(unannotated_labels)} pos / {num_unannotated - sum(unannotated_labels)} neg)"
    )

    if num_unannotated < 3:
        raise ValueError(f"Not enough unannotated cases ({num_unannotated}) for 3-way split.")

    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        unannotated_files,
        unannotated_labels,
        test_size=0.1,
        stratify=unannotated_labels,
        random_state=42,
    )

    val_ratio = 1 / 9
    train_files_ua, val_files, train_labels_ua, val_labels = train_test_split(
        train_val_files,
        train_val_labels,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=42,
    )

    train_files = train_files_ua + annotated_files
    train_labels = train_labels_ua + annotated_labels

    def _count_labels(files, labels):
        pos = sum(labels)
        neg = len(labels) - pos
        return f"{len(labels)} ({pos} pos / {neg} neg)"

    print("  After split:")
    print(
        f"    train: {_count_labels(train_files, train_labels)} "
        f"[unannotated {_count_labels(train_files_ua, train_labels_ua)} + annotated {num_annotated}]"
    )
    print(f"    val:   {_count_labels(val_files, val_labels)}")
    print(f"    test:  {_count_labels(test_files, test_labels)}")

    annotated_in_non_train = []
    for fp in val_files + test_files:
        cid = _extract_case_id_from_path(fp)
        if cid in annotated_case_ids:
            annotated_in_non_train.append(cid)
    if annotated_in_non_train:
        raise RuntimeError(f"BUG: annotated cases leaked to val/test: {annotated_in_non_train}")

    return {
        "train": (train_files, train_labels),
        "val": (val_files, val_labels),
        "test": (test_files, test_labels),
    }


def build_dataloader(file_paths, labels, is_train=True, overfit_mode=False, experiment="A", slice_csv=None):
    if overfit_mode:
        transform = get_overfit_transforms()
    else:
        transform = get_train_transforms(experiment=experiment) if is_train else get_val_transforms()
    dataset = PEDataset(file_paths, labels, transform=transform, is_train=is_train, slice_csv=slice_csv)
    batch_size = Config.batch_size if is_train else 1
    num_workers = Config.num_workers if is_train else min(4, Config.num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if is_train else 2,
        persistent_workers=is_train,
        collate_fn=collate_fn_skip_none,
    )


def build_model(device):
    num_windows = len(Config.windows) if getattr(Config, "use_multi_window", False) else 1
    input_channels = Config.slice_thickness * num_windows
    model = resnet25d_attention(
        num_classes=2,
        input_channels=input_channels,
        pretrained=Config.use_pretrained,
        pretrained_path=Config.pretrained_path,
        dropout=Config.dropout,
        use_middle_slice_only=Config.use_middle_slice_only,
        use_topk_branch=getattr(Config, "use_topk_branch", True),
        topk=getattr(Config, "topk", 5),
        dual_mil_alpha=getattr(Config, "dual_mil_alpha", 0.5),
    )
    model = model.to(device)
    return model


def forward_case_logits(model, inputs, return_aux=False):
    out = model(inputs)
    if return_aux:
        return out
    return out["bag_logit"]


def compute_metrics(labels, preds, probs, losses=None):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)

    if len(np.unique(labels)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(labels, probs)

    results = {
        "loss": float(np.mean(losses)) if losses else None,
        "auc": float(auc),
        "acc": float(acc),
        "sen": float(sensitivity),
        "spe": float(specificity),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "num_cases": int(len(labels)),
    }
    return results


def find_best_threshold_youden(targets, probs):
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    if len(np.unique(targets)) < 2:
        return 0.5
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_youden = -1.0
    for t in thresholds:
        preds = (probs > t).astype(int)
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = sen + spe - 1.0
        if youden > best_youden:
            best_youden = youden
            best_thresh = t
    return float(best_thresh)


def run_sanity_check(name, targets, probs):
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    if len(np.unique(targets)) < 2:
        auc_val = 0.5
        auc_inv = 0.5
    else:
        auc_val = roc_auc_score(targets, probs)
        auc_inv = roc_auc_score(targets, 1.0 - probs)
    pe_mask = targets == 1
    norm_mask = targets == 0
    print(f"\n[{name}]")
    print(f"  AUC:          {auc_val:.4f}")
    print(f"  AUC_inverted: {auc_inv:.4f}")
    print(f"  PE prob mean:    {probs[pe_mask].mean():.4f}" if pe_mask.any() else "  PE prob mean:    N/A")
    print(f"  Normal prob mean:{probs[norm_mask].mean():.4f}" if norm_mask.any() else "  Normal prob mean: N/A")
    print(f"  PE prob median:  {np.median(probs[pe_mask]):.4f}" if pe_mask.any() else "  PE prob median:  N/A")
    print(f"  Normal prob median:{np.median(probs[norm_mask]):.4f}" if norm_mask.any() else "  Normal prob median: N/A")
    if auc_val < 0.5 and auc_inv > auc_val:
        print("  *** WARNING: PE/Normal direction may be inverted! ***")
    if pe_mask.any() and norm_mask.any():
        pe_m = probs[pe_mask].mean()
        norm_m = probs[norm_mask].mean()
        if pe_m < norm_m:
            print(f"  *** WARNING: PE prob mean ({pe_m:.4f}) < Normal prob mean ({norm_m:.4f}) ***")


def _gather_slice_supervision(metas, device, num_slabs):
    slice_targets_list = []
    slice_mask_list = []
    slice_weights_list = []
    for m in metas:
        st = m.get("slice_targets")
        sm = m.get("slice_mask")
        sw = m.get("slice_weights")
        if st and len(st) == num_slabs:
            slice_targets_list.append(torch.tensor(st, dtype=torch.float32, device=device))
            slice_mask_list.append(torch.tensor(sm, dtype=torch.float32, device=device))
            slice_weights_list.append(torch.tensor(sw, dtype=torch.float32, device=device))
        else:
            slice_targets_list.append(torch.zeros(num_slabs, dtype=torch.float32, device=device))
            slice_mask_list.append(torch.zeros(num_slabs, dtype=torch.float32, device=device))
            slice_weights_list.append(torch.ones(num_slabs, dtype=torch.float32, device=device))
    slice_targets = torch.stack(slice_targets_list, dim=0)
    slice_mask = torch.stack(slice_mask_list, dim=0)
    slice_weights = torch.stack(slice_weights_list, dim=0)
    return slice_targets, slice_mask, slice_weights


def train_epoch(model, dataloader, device, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    total_case_loss = 0.0
    total_slice_loss = 0.0
    num_batches = 0
    all_targets = []
    all_probs = []
    valid_samples_in_epoch = 0

    for inputs, targets, metas in dataloader:
        if inputs is None or targets is None:
            continue

        valid_samples_in_epoch += targets.size(0)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if Config.use_amp:
            with autocast():
                out = model(inputs)
                bag_logit = out["bag_logit"]
                instance_logits = out["instance_logits"]

                case_loss = criterion(bag_logit, targets.float())

                slice_targets, slice_mask, slice_weights = _gather_slice_supervision(
                    metas, device, instance_logits.size(1)
                )

                slice_loss_raw = nn.functional.binary_cross_entropy_with_logits(
                    instance_logits, slice_targets, reduction="none"
                )
                valid_weight = slice_mask * slice_weights
                norm = valid_weight.sum().clamp_min(1.0)
                slice_loss = (slice_loss_raw * valid_weight).sum() / norm

                loss = case_loss + Config.lambda_slice * slice_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(inputs)
            bag_logit = out["bag_logit"]
            instance_logits = out["instance_logits"]

            case_loss = criterion(bag_logit, targets.float())

            slice_targets, slice_mask, slice_weights = _gather_slice_supervision(metas, device, instance_logits.size(1))

            slice_loss_raw = nn.functional.binary_cross_entropy_with_logits(
                instance_logits, slice_targets, reduction="none"
            )
            valid_weight = slice_mask * slice_weights
            norm = valid_weight.sum().clamp_min(1.0)
            slice_loss = (slice_loss_raw * valid_weight).sum() / norm

            loss = case_loss + Config.lambda_slice * slice_loss

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_case_loss += case_loss.item()
        total_slice_loss += slice_loss.item()
        num_batches += 1

        with torch.no_grad():
            probs = torch.sigmoid(bag_logit.detach())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    if num_batches == 0:
        return 0.0, all_targets, all_probs, valid_samples_in_epoch

    avg_loss = total_loss / num_batches
    avg_case_loss = total_case_loss / num_batches
    avg_slice_loss = total_slice_loss / num_batches
    print(f"  Train loss: {avg_loss:.4f} (case={avg_case_loss:.4f}, slice={avg_slice_loss:.4f})")
    return avg_loss, all_targets, all_probs, valid_samples_in_epoch


def evaluate(model, dataloader, device, criterion):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    all_logits = []
    losses = []
    case_losses = []
    all_metas = []
    all_attention_weights = []
    valid_samples_in_epoch = 0

    with torch.no_grad():
        for inputs, targets, metas in dataloader:
            if inputs is None or targets is None:
                continue

            valid_samples_in_epoch += targets.size(0)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if Config.use_amp:
                with autocast():
                    out_dict = forward_case_logits(model, inputs, return_aux=True)
                    bag_logit = out_dict["bag_logit"]
                    loss = criterion(bag_logit, targets.float())
            else:
                out_dict = forward_case_logits(model, inputs, return_aux=True)
                bag_logit = out_dict["bag_logit"]
                loss = criterion(bag_logit, targets.float())

            probs = torch.sigmoid(bag_logit)
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            predicted = (probs > 0.5).long()

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(bag_logit.cpu().numpy())
            losses.append(loss.item())
            case_losses.extend([loss.item()] * len(targets))
            all_metas.extend(metas)

            attn = out_dict.get("attention_weights", None)
            if attn is not None:
                all_attention_weights.extend(attn.cpu().numpy())
            else:
                all_attention_weights.extend([np.array([]) for _ in range(targets.size(0))])

    return (
        all_preds,
        all_targets,
        np.array(all_probs),
        np.array(all_logits),
        losses,
        case_losses,
        all_metas,
        valid_samples_in_epoch,
        all_attention_weights,
    )


def main():
    parser = argparse.ArgumentParser(description="Train ResNet25d Attention model")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use_topk", action="store_true", default=None, help="Enable top-k branch (overrides config)")
    parser.add_argument(
        "--experiment", type=str, default=None, help="Overfit control experiment name (A/B/C/D/D_roi_body)"
    )
    parser.add_argument(
        "--roi_mode", type=str, default=None, choices=["none", "body"], help="ROI mode for preprocessing"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--slice-csv", type=str, default=None, help="Path to key-slice annotation CSV (e.g. slice_labels.csv)"
    )
    args = parser.parse_args()

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = args.output_dir or Config.save_dir
    Config.save_dir = os.path.join(base_save_dir, timestamp)
    Config.model_save_path = os.path.join(Config.save_dir, "best_resnet25d_attention.pth")
    os.makedirs(Config.save_dir, exist_ok=True)
    print(f"Output directory: {Config.save_dir}")

    slice_csv = args.slice_csv or getattr(Config, "slice_label_csv", None)
    if slice_csv is not None:
        print(f"Slice supervision enabled: {slice_csv}")
    else:
        print("Slice supervision disabled (no slice CSV provided)")

    experiment = args.experiment or getattr(Config, "experiment", "A")
    print(f"\n========== Experiment {experiment} ==========")
    print("  A = lower lr + dropout + weight_decay + early stopping")
    print("  B = A + intensity + mild spatial augmentation")
    print("  C = B + freeze encoder / reduce head capacity")
    print("  D_roi_body = body ROI mask + A training strategy")

    if args.roi_mode:
        Config.roi_mode = args.roi_mode
        print(f"  CLI override: roi_mode = {Config.roi_mode}")

    warmup_epochs = getattr(Config, "warmup_epochs", 3)
    min_delta_auc = getattr(Config, "min_delta_auc", 0.005)
    freeze_encoder_blocks = getattr(Config, "freeze_encoder_blocks", 0)
    freeze_epochs = getattr(Config, "freeze_epochs", 0)

    if experiment == "C":
        freeze_encoder_blocks = 1
        freeze_epochs = 5
        Config.dropout = 0.5
        print(
            f"  Experiment C overrides: freeze_blocks={freeze_encoder_blocks}, "
            f"freeze_epochs={freeze_epochs}, dropout={Config.dropout}"
        )

    if experiment == "B":
        Config.use_spatial_aug = True
        print("  Experiment B override: spatial augmentation enabled")

    if experiment in ("B", "C"):
        Config.use_intensity_aug = True
        print(f"  Experiment {experiment} override: intensity augmentation enabled")

    if experiment == "D":
        Config.use_anatomy_roi = True
        Config.use_intensity_aug = True
        print("  Experiment D override: anatomy ROI crop enabled")

    if experiment == "D_roi_body":
        Config.roi_mode = "body"
        Config.use_intensity_aug = True
        Config.patience_epochs = 8
        Config.min_delta_auc = 0.005
        base_save_dir = "/root/autodl-tmp/xzt/2.5D-Attention/results"
        Config.save_dir = args.output_dir or os.path.join(base_save_dir, "D_roi_body")
        Config.model_save_path = os.path.join(Config.save_dir, "best.pth")
        os.makedirs(Config.save_dir, exist_ok=True)
        print(f"  Experiment D_roi_body: roi_mode=body, output={Config.save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Model type: resnet25d_attention")

    all_files, all_labels = load_data()

    debug_overfit = getattr(Config, "debug_overfit", False)
    overfit_mode = False

    if debug_overfit:
        overfit_mode = True
        num_per = getattr(Config, "debug_overfit_cases_per_class", 8)
        normal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]
        pe_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]
        if len(normal_indices) < num_per or len(pe_indices) < num_per:
            raise ValueError(f"Need {num_per} per class, got {len(normal_indices)} normal, {len(pe_indices)} PE")
        selected = normal_indices[:num_per] + pe_indices[:num_per]
        overfit_files = [all_files[i] for i in selected]
        overfit_labels = [all_labels[i] for i in selected]
        print("\n*** DEBUG OVERFIT MODE ***")
        print(f"  Cases: {num_per} normal + {num_per} PE = {len(overfit_files)} total")
        print("  No train/val/test split — all cases used for both train and eval")
        Config.num_epochs = 150
        Config.patience_epochs = 150
        Config.num_slabs_train = 64
        Config.batch_size = 2
        Config.num_workers = 4
        Config.dropout = 0.2
        print(
            f"  Overrides: epochs={Config.num_epochs}, patience={Config.patience_epochs}, "
            f"num_slabs={Config.num_slabs_train}, bs={Config.batch_size}, dropout={Config.dropout}"
        )

        splits = {
            "train": (overfit_files, overfit_labels),
            "val": (overfit_files, overfit_labels),
            "test": (overfit_files, overfit_labels),
        }
    else:
        annotated_case_ids = _load_annotated_case_ids(slice_csv)
        splits = split_data(all_files, all_labels, annotated_case_ids=annotated_case_ids)

    train_loader = build_dataloader(
        splits["train"][0],
        splits["train"][1],
        is_train=True,
        overfit_mode=overfit_mode,
        experiment=experiment,
        slice_csv=slice_csv,
    )
    val_loader = build_dataloader(
        splits["val"][0],
        splits["val"][1],
        is_train=False,
        overfit_mode=overfit_mode,
        experiment=experiment,
        slice_csv=slice_csv,
    )
    test_loader = build_dataloader(
        splits["test"][0],
        splits["test"][1],
        is_train=False,
        overfit_mode=overfit_mode,
        experiment=experiment,
        slice_csv=slice_csv,
    )

    if args.use_topk is not None:
        Config.use_topk_branch = args.use_topk
        print(f"CLI override: use_topk_branch = {Config.use_topk_branch}")
    effective_use_topk = getattr(Config, "use_topk_branch", False)
    if not effective_use_topk:
        Config.dual_mil_alpha = 1.0
        print("Top-k disabled → forcing dual_mil_alpha = 1.0 (gated-attention only)")

    model = build_model(device)
    num_windows = len(Config.windows) if getattr(Config, "use_multi_window", False) else 1
    input_channels = Config.slice_thickness * num_windows
    print("\nModel: resnet25d_attention")
    print(f"  use_multi_window: {getattr(Config, 'use_multi_window', False)}")
    print(f"  slice_thickness: {Config.slice_thickness}")
    print(f"  num_windows: {num_windows}")
    print(f"  input_channels: {input_channels}")
    print(f"  model conv1 in_channels: {model.encoder.conv1.in_channels}")
    print(f"  dropout: {Config.dropout}")
    print(f"  keep_original_depth: {getattr(Config, 'keep_original_depth', False)}")
    print(f"  target_hw: {getattr(Config, 'target_hw', (256, 256))}")
    print(f"  num_slabs_train: {Config.num_slabs_train}")
    print(f"  eval_stride: {Config.eval_stride}")
    print(f"  train_random_sample: {getattr(Config, 'train_random_sample', True)}")
    print(f"  eval_sliding_window: {getattr(Config, 'eval_sliding_window', True)}")
    print(f"  filter_empty_slices: {getattr(Config, 'filter_empty_slices', False)}")
    print(f"  min_candidate_slabs: {getattr(Config, 'min_candidate_slabs', 16)}")
    print(f"  min_valid_slices_per_slab: {getattr(Config, 'min_valid_slices_per_slab', 1)}")
    print(f"  use_topk_branch: {getattr(Config, 'use_topk_branch', True)}")
    print(f"  topk: {getattr(Config, 'topk', 5)}")
    print(f"  dual_mil_alpha: {getattr(Config, 'dual_mil_alpha', 0.5)}")
    mode_name = "Gated-Attention + Top-K" if getattr(Config, "use_topk_branch", False) else "Gated-Attention ONLY"
    print(f"  MIL mode: {mode_name}")
    if slice_csv is not None:
        print(f"  slice_supervision: ENABLED (csv={slice_csv}, lambda={getattr(Config, 'lambda_slice', 0.3)})")
    else:
        print("  slice_supervision: DISABLED")
    if getattr(Config, "roi_mode", "none") != "none":
        print(
            f"  ROI mode: {Config.roi_mode} (threshold={getattr(Config, 'roi_body_threshold', -500)}, "
            f"dilation={getattr(Config, 'roi_body_dilation', 3)}, fill={getattr(Config, 'roi_body_fill_value', -1000)})"
        )

    print("\nDual-MIL shape diagnostic (dry run)...")
    model.eval()
    with torch.no_grad():
        for inp, _, _meta in train_loader:
            if inp is None:
                continue
            dummy_batch = inp[:1].to(device)
            out = forward_case_logits(model, dummy_batch, return_aux=True)
            print(f"  bag_logit shape:        {out['bag_logit'].shape}")
            print(f"  instance_logits shape:  {out['instance_logits'].shape}")
            print(f"  attention_weights shape:{out['attention_weights'].shape}")
            print(f"  topk_indices shape:     {out['topk_indices'].shape}")
            attn_sum = out["attention_weights"].sum(dim=1)
            print(f"  attention_weights.sum(dim=1): {attn_sum.cpu().numpy()} (should be ~1.0)")
            break
    model.train()
    print()

    runtime_config = {
        "debug_overfit": getattr(Config, "debug_overfit", False),
        "keep_original_depth": getattr(Config, "keep_original_depth", False),
        "target_hw": getattr(Config, "target_hw", (256, 256)),
        "use_topk_branch": getattr(Config, "use_topk_branch", False),
        "dual_mil_alpha": getattr(Config, "dual_mil_alpha", 0.5),
        "num_slabs_train": Config.num_slabs_train,
        "eval_stride": Config.eval_stride,
        "windows": getattr(Config, "windows", [(-1000, 600)]),
        "use_multi_window": getattr(Config, "use_multi_window", False),
        "dropout": Config.dropout,
        "batch_size": Config.batch_size,
        "num_epochs": Config.num_epochs,
        "encoder_lr": Config.encoder_lr,
        "head_lr": Config.head_lr,
        "weight_decay": Config.weight_decay,
        "patience_epochs": Config.patience_epochs,
        "slice_thickness": Config.slice_thickness,
        "filter_empty_slices": getattr(Config, "filter_empty_slices", False),
        "model_type": Config.model_type,
        "use_pretrained": Config.use_pretrained,
        "model_in_channels": input_channels,
        "roi_mode": getattr(Config, "roi_mode", "none"),
    }
    os.makedirs(Config.save_dir, exist_ok=True)
    import json

    with open(os.path.join(Config.save_dir, "config_runtime.json"), "w") as jf:
        json.dump(runtime_config, jf, indent=2, default=str)
    print(f"Runtime config saved to {Config.save_dir}/config_runtime.json")

    criterion = nn.BCEWithLogitsLoss()

    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    encoder_lr = Config.encoder_lr
    head_lr = Config.head_lr
    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=Config.weight_decay,
    )
    print(f"Optimizer: encoder_lr={encoder_lr}, head_lr={head_lr}, wd={Config.weight_decay}")

    scaler = GradScaler() if Config.use_amp else None

    from torch.optim.lr_scheduler import LambdaLR

    def _warmup_cosine(step, warmup_steps, total_steps):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    total_steps = Config.num_epochs
    warmup_steps = warmup_epochs
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: _warmup_cosine(s, warmup_steps, total_steps))
    print(f"Scheduler: CosineAnnealingLR with {warmup_steps}-epoch warmup, max_epochs={total_steps}")

    if freeze_encoder_blocks > 0 and not overfit_mode:
        block_names = ["layer1", "layer2"]
        for bn in block_names[:freeze_encoder_blocks]:
            for name, param in model.encoder.named_parameters():
                if name.startswith(bn):
                    param.requires_grad = False
        print(f"Frozen encoder blocks: {block_names[:freeze_encoder_blocks]}")

    start_epoch = 0
    best_val_auc = 0.0
    best_val_probs = None
    best_val_targets = None
    best_val_metas = None
    best_val_case_losses = None
    best_val_logits = None
    best_val_threshold = 0.5
    best_epoch = 0
    patience = Config.patience_epochs
    epochs_without_improvement = 0

    best_checkpoint_path = os.path.join(Config.save_dir, f"best_{experiment}.pth")
    metrics_csv_path = os.path.join(Config.save_dir, f"overfit_control_experiment_{experiment}_metrics.csv")

    import csv as csv_mod

    csv_file = open(metrics_csv_path, "w", newline="")
    csv_writer = csv_mod.writer(csv_file)
    csv_writer.writerow(
        [
            "epoch",
            "train_loss",
            "train_auc",
            "train_acc",
            "val_loss",
            "val_auc",
            "val_auc_inverted",
            "val_acc_05",
            "train_pos_prob_mean",
            "train_neg_prob_mean",
            "val_pos_prob_mean",
            "val_neg_prob_mean",
            "lr_encoder",
            "lr_head",
            "best_val_auc",
            "early_stop_counter",
        ]
    )
    csv_file.flush()

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_auc = checkpoint.get("best_val_auc", 0.0)
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

    os.makedirs(Config.save_dir, exist_ok=True)

    for epoch in range(start_epoch, Config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")

        train_loss, train_targets, train_probs, _train_samples = train_epoch(
            model, train_loader, device, criterion, optimizer, scaler
        )
        train_probs_arr = np.asarray(train_probs)
        train_targets_arr = np.asarray(train_targets)
        train_auc = roc_auc_score(train_targets_arr, train_probs_arr) if len(np.unique(train_targets_arr)) > 1 else 0.5

        (
            val_preds,
            val_targets,
            val_probs,
            val_logits,
            val_losses,
            val_case_losses,
            val_metas,
            _val_samples,
            val_attention_weights,
        ) = evaluate(model, val_loader, device, criterion)
        val_targets_arr = np.asarray(val_targets)
        val_probs_arr = np.asarray(val_probs)

        val_auc = roc_auc_score(val_targets_arr, val_probs_arr) if len(np.unique(val_targets_arr)) > 1 else 0.5
        val_auc_inv = (
            roc_auc_score(val_targets_arr, 1.0 - val_probs_arr) if len(np.unique(val_targets_arr)) > 1 else 0.5
        )
        val_acc_05 = accuracy_score(val_targets_arr, val_probs_arr >= 0.5)
        val_cm = confusion_matrix(val_targets_arr, val_probs_arr >= 0.5, labels=[0, 1])

        val_pe_mask = val_targets_arr == 1
        val_norm_mask = val_targets_arr == 0
        val_pe_mean = float(val_probs_arr[val_pe_mask].mean()) if val_pe_mask.any() else 0.0
        val_norm_mean = float(val_probs_arr[val_norm_mask].mean()) if val_norm_mask.any() else 0.0

        train_pe_mask = train_targets_arr == 1
        train_norm_mask = train_targets_arr == 0
        train_pe_mean = float(train_probs_arr[train_pe_mask].mean()) if train_pe_mask.any() else 0.0
        train_norm_mean = float(train_probs_arr[train_norm_mask].mean()) if train_norm_mask.any() else 0.0

        current_lr_encoder = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"  Val Loss: {np.mean(val_losses):.4f}, Val AUC: {val_auc:.4f}, Val Acc@0.5: {val_acc_05:.4f}")

        n_val = len(val_targets_arr)
        n_pos = int(val_pe_mask.sum())
        n_neg = int(val_norm_mask.sum())
        print("\n  --- Val Sanity Check ---")
        print(f"  n_val={n_val}, n_pos={n_pos}, n_neg={n_neg}")
        print(f"  y_true unique/counts: {np.unique(val_targets_arr, return_counts=True)}")
        print(f"  prob min={val_probs_arr.min():.6f}, max={val_probs_arr.max():.6f}, mean={val_probs_arr.mean():.6f}")
        print(f"  PE prob mean:    {val_pe_mean:.4f}")
        print(f"  Normal prob mean:{val_norm_mean:.4f}")
        print(f"  Val AUC:          {val_auc:.4f}")
        print(f"  Val AUC inverted: {val_auc_inv:.4f}")
        print(f"  Val Acc@0.5:      {val_acc_05:.4f}")
        print(f"  Confusion Matrix: TN={val_cm[0, 0]}, FP={val_cm[0, 1]}, FN={val_cm[1, 0]}, TP={val_cm[1, 1]}")

        if val_auc_inv > val_auc + 0.05:
            print(
                f"  *** WARNING: Inverted AUC ({val_auc_inv:.4f}) >> AUC ({val_auc:.4f}) — prob direction may be flipped! ***"
            )

        if val_pe_mean < val_norm_mean:
            print(
                f"  *** WARNING: PE prob mean ({val_pe_mean:.4f}) < Normal prob mean ({val_norm_mean:.4f}) — labels may be swapped! ***"
            )

        print(f"  Train PE prob mean:    {train_pe_mean:.4f}")
        print(f"  Train Normal prob mean:{train_norm_mean:.4f}")
        print(f"  lr_encoder: {current_lr_encoder:.2e}, lr_head: {current_lr_head:.2e}")

        print("\n  --- First 20 validation samples ---")
        print(f"  {'filename':<45s} {'true':>5s} {'prob':>8s} {'pred_0.5':>8s} {'loss':>8s}")
        for i in range(min(20, n_val)):
            fn = val_metas[i].get("filename", "?") if i < len(val_metas) else "?"
            tl = int(val_targets_arr[i])
            pb = float(val_probs_arr[i])
            pd = int(pb >= 0.5)
            cl = float(val_case_losses[i]) if i < len(val_case_losses) else 0.0
            print(f"  {fn:<45s} {tl:>5d} {pb:>8.4f} {pd:>8d} {cl:>8.4f}")

        sorted_idx = np.argsort(val_probs_arr)[::-1]
        print("\n  --- Top 10 highest prob ---")
        print(f"  {'filename':<45s} {'true':>5s} {'prob':>8s}")
        for i in sorted_idx[:10]:
            fn = val_metas[i].get("filename", "?") if i < len(val_metas) else "?"
            print(f"  {fn:<45s} {int(val_targets_arr[i]):>5d} {float(val_probs_arr[i]):>8.4f}")

        print("\n  --- Top 10 lowest prob ---")
        print(f"  {'filename':<45s} {'true':>5s} {'prob':>8s}")
        for i in sorted_idx[-10:][::-1]:
            fn = val_metas[i].get("filename", "?") if i < len(val_metas) else "?"
            print(f"  {fn:<45s} {int(val_targets_arr[i]):>5d} {float(val_probs_arr[i]):>8.4f}")

        debug_csv_path = os.path.join(Config.save_dir, f"val_debug_predictions_epoch_{epoch + 1}.csv")
        with open(debug_csv_path, "w", newline="") as csvf:
            import csv as csv_mod

            writer = csv_mod.writer(csvf)
            writer.writerow(
                [
                    "split",
                    "epoch",
                    "case_index",
                    "case_id",
                    "filename",
                    "path",
                    "true_label",
                    "logit",
                    "prob",
                    "prob_inverted",
                    "pred_0.5",
                    "loss",
                    "depth",
                    "candidate_slabs",
                    "used_slabs_eval",
                    "selected_slab_starts",
                    "selected_slab_centers",
                    "attention_weights",
                    "top_attention_slices",
                    "top_attention_scores",
                ]
            )
            for i in range(n_val):
                meta = val_metas[i] if i < len(val_metas) else {}
                attn_weights = val_attention_weights[i] if i < len(val_attention_weights) else np.array([])
                selected_starts = meta.get("selected_slab_starts", [])
                selected_centers = meta.get("selected_centers", [])
                num_slabs = (
                    len(attn_weights) if isinstance(attn_weights, np.ndarray) else meta.get("used_slabs_eval", 0)
                )

                top_k = min(5, len(attn_weights) if isinstance(attn_weights, np.ndarray) else 0)
                if isinstance(attn_weights, np.ndarray) and len(attn_weights) > 0:
                    attn_sorted_idx = np.argsort(attn_weights)[::-1][:top_k]
                    top_slices = ",".join(
                        str(selected_centers[j]) if j < len(selected_centers) else f"idx{j}" for j in attn_sorted_idx
                    )
                    top_scores = ",".join(f"{attn_weights[j]:.6f}" for j in attn_sorted_idx)
                else:
                    top_slices = ""
                    top_scores = ""

                attn_str = (
                    ",".join(f"{w:.6f}" for w in attn_weights)
                    if isinstance(attn_weights, np.ndarray) and len(attn_weights) > 0
                    else ""
                )
                starts_str = ",".join(str(s) for s in selected_starts) if selected_starts else ""
                centers_str = ",".join(str(c) for c in selected_centers) if selected_centers else ""

                writer.writerow(
                    [
                        "val",
                        epoch + 1,
                        i,
                        meta.get("case_id", i),
                        meta.get("filename", "?"),
                        meta.get("file_path", "?"),
                        int(val_targets_arr[i]),
                        float(val_logits[i]) if i < len(val_logits) else 0.0,
                        float(val_probs_arr[i]),
                        float(1.0 - val_probs_arr[i]),
                        int(val_probs_arr[i] >= 0.5),
                        float(val_case_losses[i]) if i < len(val_case_losses) else 0.0,
                        meta.get("depth", 0),
                        meta.get("candidate_slabs", 0),
                        meta.get("used_slabs_eval", 0),
                        starts_str,
                        centers_str,
                        attn_str,
                        top_slices,
                        top_scores,
                    ]
                )
        print(f"  Val debug CSV saved to {debug_csv_path}")

        scheduler.step()

        train_acc_05 = accuracy_score(train_targets_arr, train_probs_arr >= 0.5)
        csv_writer.writerow(
            [
                epoch + 1,
                f"{train_loss:.6f}",
                f"{train_auc:.6f}",
                f"{train_acc_05:.6f}",
                f"{np.mean(val_losses):.6f}",
                f"{val_auc:.6f}",
                f"{val_auc_inv:.6f}",
                f"{val_acc_05:.6f}",
                f"{train_pe_mean:.6f}",
                f"{train_norm_mean:.6f}",
                f"{val_pe_mean:.6f}",
                f"{val_norm_mean:.6f}",
                f"{current_lr_encoder:.2e}",
                f"{current_lr_head:.2e}",
                f"{best_val_auc:.6f}",
                epochs_without_improvement,
            ]
        )
        csv_file.flush()

        improved = (val_auc - best_val_auc) >= min_delta_auc

        if improved:
            best_val_auc = val_auc
            best_val_probs = val_probs_arr.copy()
            best_val_targets = val_targets_arr.copy()
            best_val_metas = val_metas.copy() if val_metas else []
            best_val_case_losses = val_case_losses.copy() if val_case_losses else []
            best_val_logits = val_logits.copy() if val_logits is not None else []
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "best_val_auc": best_val_auc,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                best_checkpoint_path,
            )
            print(f"  Saved best model (AUC: {best_val_auc:.4f}, improved by {val_auc - best_val_auc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No significant improvement for {epochs_without_improvement} epoch(s) (min_delta={min_delta_auc})")

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement")
                break

        if freeze_epochs > 0 and epoch + 1 == freeze_epochs:
            for param in model.encoder.parameters():
                param.requires_grad = True
            print(f"\n--- All encoder layers unfrozen at epoch {epoch + 1} ---")

    csv_file.close()
    print(f"\nTraining complete. Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    print(f"Metrics saved to {metrics_csv_path}")

    if best_val_probs is not None and best_val_targets is not None:
        best_val_threshold = find_best_threshold_youden(best_val_targets, best_val_probs)
    print(f"Best Val Threshold (Youden): {best_val_threshold:.4f}")

    print(f"\n--- Loading best model from disk (epoch {best_epoch}) ---")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded best checkpoint: {best_checkpoint_path}")
    else:
        print(f"  WARNING: Best checkpoint not found at {best_checkpoint_path}, using current model")

    print("\nCollecting predictions on train/val/test...")
    _tp, train_targets, train_probs, _tl, _tl2, _tl3, _, _ts, _ = evaluate(model, train_loader, device, criterion)
    _vp, val_targets, val_probs, _vl, _vl2, val_case_losses, val_metas, _vs, val_attention_weights = evaluate(
        model, val_loader, device, criterion
    )
    _test_preds, test_targets, test_probs, _test_logits, test_losses, test_case_losses, test_metas, _test_samples, _ = (
        evaluate(model, test_loader, device, criterion)
    )

    train_targets_arr = np.asarray(train_targets)
    train_probs_arr = np.asarray(train_probs)
    val_targets_arr = np.asarray(val_targets)
    val_probs_arr = np.asarray(val_probs)
    test_targets_arr = np.asarray(test_targets)
    test_probs_arr = np.asarray(test_probs)

    train_auc = roc_auc_score(train_targets_arr, train_probs_arr) if len(np.unique(train_targets_arr)) > 1 else 0.5
    val_auc = roc_auc_score(val_targets_arr, val_probs_arr) if len(np.unique(val_targets_arr)) > 1 else 0.5
    test_auc = roc_auc_score(test_targets_arr, test_probs_arr) if len(np.unique(test_targets_arr)) > 1 else 0.5

    print("\n========== Label-Probability Sanity Check ==========")
    run_sanity_check("Train", train_targets_arr, train_probs_arr)
    run_sanity_check("Val", val_targets_arr, val_probs_arr)
    run_sanity_check("Test", test_targets_arr, test_probs_arr)

    print(f"\n========== Final Results ({experiment}) ==========")
    print(f"  Best Val AUC:        {best_val_auc:.4f}")
    print(f"  Best Val Epoch:      {best_epoch}")
    print(f"  Train AUC:           {train_auc:.4f}")
    print(f"  Val   AUC:           {val_auc:.4f}")
    print(f"  Test  AUC:           {test_auc:.4f}")
    gap = train_auc - val_auc
    print(f"  Train-Val Gap:       {gap:.4f}")

    val_predictions_path = os.path.join(Config.save_dir, f"experiment_{experiment}_val_predictions.csv")
    with open(val_predictions_path, "w", newline="") as csvf:
        writer = csv_mod.writer(csvf)
        writer.writerow(
            [
                "case_id",
                "filename",
                "true_label",
                "prob",
                "pred_0.5",
                "loss",
                "split",
                "depth",
                "candidate_slabs",
                "used_slabs_eval",
            ]
        )
        for i in range(len(val_targets_arr)):
            meta = val_metas[i] if i < len(val_metas) else {}
            writer.writerow(
                [
                    meta.get("case_id", i),
                    meta.get("filename", "?"),
                    int(val_targets_arr[i]),
                    float(val_probs_arr[i]),
                    int(val_probs_arr[i] >= 0.5),
                    float(val_case_losses[i]) if i < len(val_case_losses) else 0.0,
                    "val",
                    meta.get("depth", 0),
                    meta.get("candidate_slabs", 0),
                    meta.get("used_slabs_eval", 0),
                ]
            )
    print(f"  Val predictions saved to {val_predictions_path}")

    sorted_idx_val = np.argsort(val_probs_arr)[::-1]
    print("\n  --- Top 10 highest prob (val) ---")
    print(f"  {'filename':<45s} {'true':>5s} {'prob':>8s}")
    for i in sorted_idx_val[:10]:
        fn = val_metas[i].get("filename", "?") if i < len(val_metas) else "?"
        print(f"  {fn:<45s} {int(val_targets_arr[i]):>5d} {float(val_probs_arr[i]):>8.4f}")

    print("\n  --- Top 10 lowest prob (val) ---")
    for i in sorted_idx_val[-10:][::-1]:
        fn = val_metas[i].get("filename", "?") if i < len(val_metas) else "?"
        print(f"  {fn:<45s} {int(val_targets_arr[i]):>5d} {float(val_probs_arr[i]):>8.4f}")

    fp_mask = (val_targets_arr == 0) & (val_probs_arr >= 0.5)
    fn_mask = (val_targets_arr == 1) & (val_probs_arr < 0.5)
    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())
    print("\n  --- Error Analysis (val) ---")
    print(f"  False Positives: {fp_count}")
    print(f"  False Negatives: {fn_count}")

    if fp_count > 0:
        fp_idx = np.where(fp_mask)[0]
        fp_sorted = fp_idx[np.argsort(val_probs_arr[fp_idx])[::-1]]
        print("\n  --- False Positives with highest prob ---")
        print(f"  {'filename':<45s} {'prob':>8s}")
        for i in fp_sorted[: min(10, fp_count)]:
            fn = val_metas[i].get("filename", "?") if i < len(val_metas) else "?"
            print(f"  {fn:<45s} {float(val_probs_arr[i]):>8.4f}")

    if fn_count > 0:
        fn_idx = np.where(fn_mask)[0]
        fn_sorted = fn_idx[np.argsort(val_probs_arr[fn_idx])]
        print("\n  --- False Negatives with lowest prob ---")
        print(f"  {'filename':<45s} {'prob':>8s}")
        for i in fn_sorted[: min(10, fn_count)]:
            fn = val_metas[i].get("filename", "?") if i < len(val_metas) else "?"
            print(f"  {fn:<45s} {float(val_probs_arr[i]):>8.4f}")

    print("\n=====================================================")
    print(f" Experiment {experiment} complete.")
    print("=====================================================")


if __name__ == "__main__":
    main()
