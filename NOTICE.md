# NOTICE

depthvlm-finetune — Copyright (c) 2026 Jihyuck Hong.

The MIT License (see LICENSE) covers ONLY the code written for this repository: `run.sh`, `h200/`, `eval/`, `prep/`, `train/`, `envs/`.
It does not cover the third-party material below.

## 1. DepthVLM (Hanxun Yu et al.) — Apache License 2.0
- `third_party/DepthVLM/` is a copy of https://github.com/hanxunyu/DepthVLM at commit `5d1472d3b983fb7bf8bec2e9adf23bd0d69ed860` (2026-07-22), including `train/` (`train.py`, `train-stage1.sh`, `train-stage2.sh`). `assets/`, `examples/` and `.DS_Store` were not copied. No file was modified.
- `train/train_ft.py` runs the official `train/train.py` unchanged; it only adds one entry (`kitti`: 0.001–80 m) to the depth-range table `utils.datasets.DATASET_CONFIGS` before the official code reads it. `train/train_ft.sh` passes the official stage-2 arguments; the differences are listed in its header and in `docs/PROTOCOL.md`.
- `eval/predict.py` re-implements the official inference call of `eval/eval.py` (same prompt, chat template, `process_vision_info`, single forward pass); to the extent that code follows the original, it remains under Apache-2.0.
- The DepthVLM-4B checkpoint (https://huggingface.co/JonnyYu828/DepthVLM-4B, revision `2b2d02fcfe0c89c8aa7d541055e5a078930077c9`, Apache-2.0) is downloaded at run time and is not stored here. Fine-tuned weights are not stored here.
- Citation: Yu et al., *Unlocking Dense Metric Depth Estimation in VLMs*, arXiv:2605.15876.

## 2. BTS (Jin Han Lee et al.) — GPL-3.0
- The standard split lists (`train_test_inputs/{nyudepthv2,eigen}_{train,test}_files_with_gt.txt`) are downloaded by `prep/fetch_splits.sh` from https://github.com/cleinc/bts at commit `5e3406b35d1497b2e55d2dd600524d1f4efacaed` and checked by SHA256; they are not stored in this repository.
- `prep/build_nyu.py` reproduces the behaviour of BTS `utils/extract_official_train_test_set_from_mat.py` (7-pixel black border, `rawDepths` × 1000 as uint16) to build the NYU test set, and `eval/std_eval.py` reproduces the masks, clipping and `compute_errors` of BTS `pytorch/bts_eval.py`. These are re-implementations written for this repository; no BTS source file is included.
- The NYU training set is the `sync.zip` archive distributed by the BTS authors (Google Drive id `1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP`), derived from NYU Depth V2 raw data.
- Citation: Lee et al., *From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation*, arXiv:1907.10326.

## 3. Metric3D — BSD 2-Clause, fetched at run time, not included
- `prep/fetch_ext.sh` downloads https://github.com/YvanYin/Metric3D @ `eb5b6fac0dc155e4e52f576e304fbf11655ff339` into `ext/` (git-ignored); the weights `metric_depth_vit_large_800k.pth` are downloaded by its `hubconf.py`. It is used only to check the scoring code (`eval/predict.py --model metric3d`), which re-implements the test-time pre-processing of `training/mono` (NYU and KITTI datasets).
- Citation: Hu et al., *Metric3D v2*, arXiv:2404.15506.

## 4. Datasets — not redistributed in this repository
- NYU Depth V2 (Silberman et al., ECCV 2012): `nyu_depth_v2_labeled.mat` and `splits.mat` from the official page; research use.
- KITTI raw data and the annotated depth maps of the KITTI depth benchmark (Geiger et al., IJRR 2013; Uhrig et al., 3DV 2017): CC BY-NC-SA 3.0.
- Images and depth maps are rebuilt locally from the official sources (`prep/`) and are delivered privately to the compute server for noncommercial research use. If a license above is uncertain, check the original dataset page; this file does not override any dataset's terms.

## 5. Comparison numbers
- The comparison tables follow Table V (NYU) and Table VI (KITTI Eigen split) of Piccinelli et al., *UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler*, arXiv:2502.20110. Numbers of other methods are quoted from that paper.

## 6. Python packages
PyTorch (BSD-3), transformers, accelerate, trl, peft, datasets and huggingface_hub (Apache-2.0), qwen-vl-utils (Apache-2.0), flash-attn (BSD-3), pandas, numpy, scipy, OpenCV, Pillow, h5py (BSD/MIT/Apache/HPND).
