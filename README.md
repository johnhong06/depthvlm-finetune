# depthvlm-finetune

Fine-tune **DepthVLM-4B** ([Yu et al., 2026](https://arxiv.org/abs/2605.15876)) on **NYU Depth V2** and on the **KITTI Eigen split**, and evaluate it with the same protocol as Table V (NYU) and Table VI (KITTI) of [UniDepthV2](https://arxiv.org/abs/2502.20110):
δ1 / δ2 / δ3, AbsRel, RMS and Log10 (NYU) or RMSlog (KITTI). The zero-shot DepthVLM-4B is measured with the same protocol for reference.

In those tables only the evaluation protocol is shared; every method uses its own fine-tuning recipe. This repository therefore follows the standard evaluation exactly
and keeps DepthVLM's official stage-2 training recipe, changing only what a single GPU and a ~23k-image dataset require. Every change is listed in [`docs/PROTOCOL.md`](docs/PROTOCOL.md) (Korean).

## Protocol in short

| | NYU (Table V) | KITTI (Table VI) |
|:--|:--|:--|
| Test set | official 654 images, GT = `rawDepths` (BTS extraction) | Eigen test, 652 images with annotated depth |
| Train set | BTS `sync` list, 24,231 pairs | Eigen train list, 23,158 images |
| Model input | 640×480 | KB crop (bottom 352 rows, centre 1216 columns) |
| Evaluated pixels | 0.001 < GT < 10 m, Eigen crop | 0.001 < GT < 80 m, Garg crop |
| Averaging | per image, then mean (as BTS / UniDepth) | same |

- Epochs are selected on a validation split held out from the training scenes/drives (5 %, seed 0); the test set is scored once, after selection.
- The scorer (`eval/std_eval.py`) is checked against published Metric3Dv2 numbers before any training result is read.

## Layout

```
prep/        fetch_splits.sh, fetch_kitti.sh, fetch_nyu_sync.sh, fetch_ext.sh   download sources (pinned)
             build_nyu.py, build_kitti.py, make_jsonl.py                        rebuild + verify data, DepthVLM jsonl
             pack.py                                                            zip bundles for the GPU server
eval/        predict.py (DepthVLM official inference path; Metric3Dv2 for checks), std_eval.py (BTS rules), select_epoch.py
train/       train_ft.py (runs the official train.py, adds the KITTI depth range), train_ft.sh (stage-2 arguments)
h200/        unpack.py (verify SHA256 and extract the zip bundles)
run.sh       server entry point: env | smoke | zeroshot | nyu | kitti | all
third_party/DepthVLM   official code at commit 5d1472d (unmodified)
envs/        pinned Python environments
```

## Reproduce

```bash
# data (local): download, rebuild, verify, pack
bash prep/fetch_splits.sh && bash prep/fetch_kitti.sh && bash prep/fetch_nyu_sync.sh
python prep/build_nyu.py --sync ~/data/nyuv2_bts/sync.zip --mat ~/data/nyuv2 --splits ~/data/dvft/splits --out ~/data/dvft/data
python prep/build_kitti.py --kitti ~/data/kitti --splits ~/data/dvft/splits --out ~/data/dvft/data
python prep/make_jsonl.py --ds nyu --data ~/data/dvft/data --splits ~/data/dvft/splits
python prep/make_jsonl.py --ds kitti --data ~/data/dvft/data --splits ~/data/dvft/splits
python prep/pack.py --data ~/data/dvft/data --splits ~/data/dvft/splits --out ~/data/h200_staging/dvft

# GPU server (one H200): the zip bundles go under /app/data
bash run.sh env && bash run.sh smoke && bash run.sh nyu && bash run.sh kitti
```

`nyu_depth_v2_labeled.mat` and `splits.mat` come from the [NYU Depth V2 page](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html); KITTI raw drives and `data_depth_annotated.zip` from the [KITTI depth benchmark](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).
Data, predictions and weights are not stored in this repository.

## Results

To be filled after the server runs (see `NOTES.md`).

## License

Code written for this repository: MIT (see `LICENSE`). Third-party code, models and datasets keep their own licenses — see [`NOTICE.md`](NOTICE.md).
