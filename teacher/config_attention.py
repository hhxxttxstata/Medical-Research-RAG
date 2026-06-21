class Config:
    # Input volume after preprocessing: (D, H, W) — D kept original, only H/W resized
    input_size = (96, 256, 256)  # depth is ignored when keep_original_depth=True
    keep_original_depth = True
    target_hw = (256, 256)

    # Training
    batch_size = 4
    num_workers = 8
    num_epochs = 40
    encoder_lr = 1e-5
    head_lr = 5e-5
    learning_rate = head_lr  # kept for backward compat, not used directly
    weight_decay = 1e-4
    use_amp = True
    overfit_test_samples = None
    patience_epochs = 5
    min_delta_auc = 0.02

    # Overfit debug
    debug_overfit = False
    debug_overfit_cases_per_class = 8

    # Overfit control experiment
    experiment = "A"  # "A" | "B" | "C" | "D" | "D_roi_body"
    warmup_epochs = 3
    use_deterministic_val = True
    freeze_encoder_blocks = 0  # 0=none, 1=freeze layer1, 2=freeze layer1+layer2
    freeze_epochs = 0  # freeze encoder for first N epochs (0=disabled)

    # Anatomy ROI
    roi_mode = "none"  # "none" | "body"
    roi_body_threshold = -500  # HU threshold for body mask
    roi_body_fill_value = -1000  # HU fill value outside body mask
    roi_body_dilation = 3  # dilation pixels for body mask
    roi_margin_ratio = 0.05  # bbox expansion ratio

    # Augmentation (used when experiment >= "B")
    use_intensity_aug = True
    use_spatial_aug = False  # mild spatial in experiment B

    # 2.5D instance extraction
    slice_thickness = 3  # 3 adjacent slices as one 2.5D instance

    # --- Train slab sampling ---
    num_slabs_train = 48  # number of randomly sampled slabs per case
    num_slabs_eval = 48  # number of evenly-spaced deterministic slabs per case in val/test
    train_random_sample = True

    # --- Eval slab sampling (sliding window) ---
    eval_sliding_window = True
    eval_stride = 2  # sliding window stride during validation/test

    # --- Chest slice filtering (HU-based, operates on raw HU before normalization) ---
    filter_empty_slices = True
    chest_body_hu_threshold = -600  # HU: above this => body/soft tissue
    chest_lung_hu_low = -1000  # HU: lung window low
    chest_lung_hu_high = -300  # HU: lung window high
    chest_body_ratio_threshold = 0.05  # min body-ratio for chest slice
    chest_lung_ratio_threshold = 0.02  # min lung-ratio for chest slice
    chest_segment_margin = 8  # slices to expand around the main contiguous chest segment
    chest_segment_margin_upper = 20  # extra slices above (cranial) lung to include mediastinum/PA
    chest_segment_margin_lower = 8  # extra slices below (caudal) lung to include lung bases
    chest_min_filtered_slices = 16  # fallback: if fewer slices survive filtering, use all
    # Legacy filter params (kept for compatibility)
    body_threshold = -800
    lung_threshold_low = -1000
    lung_threshold_high = -300
    enhance_threshold = 100
    body_ratio_threshold = 0.05
    lung_ratio_threshold = 0.03
    enhance_ratio_threshold = 0.005
    min_valid_slices_per_slab = 1
    min_candidate_slabs = 16

    # --- Multi-window preprocessing ---
    use_multi_window = True
    # Each window is (hu_min, hu_max); slabs are normalized per-window and concatenated as channels
    windows = [
        (-1000, 400),  # lung window
        (-160, 240),  # mediastinal window
        (-100, 700),  # CTA / vessel window
    ]

    # Slice-level supervision
    lambda_slice = 0.3  # weight for slice-level BCE loss

    # Dual-MIL model
    model_type = "resnet25d_attention"
    use_pretrained = True
    pretrained_path = "/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    dropout = 0.4
    use_middle_slice_only = False
    base_channels = 16
    use_topk_branch = False
    topk = 5
    dual_mil_alpha = 1.0  # blend weight: alpha*attention + (1-alpha)*topk (ignored when use_topk_branch=False)

    # Kept for compatibility with older 3D/slab branches in train.py
    slab_depth = 16
    slab_overlap = 4

    # Data paths
    data_root = "/root/autodl-tmp/xzt/dataset"
    normal_dir = "normal/enhance"
    pe_dir = "pe/enhance"
    slice_label_csv = "/root/autodl-tmp/xzt/slice_labels.csv"  # key-slice annotation CSV

    # Output paths
    save_dir = "/root/autodl-tmp/xzt/mymodel/results_attention"
    model_save_path = "/root/autodl-tmp/xzt/mymodel/results_attention/best_resnet25d_attention.pth"

    preprocess = {
        "hu_min": -1000,
        "hu_max": 700,
        "slice_resolution": 96,
        "use_roi_crop": False,
        "roi_padding": 30,
    }
