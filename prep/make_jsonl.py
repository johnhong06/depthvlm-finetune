"""DepthVLM 학습용 jsonl (공식 curate_datasets/create_data_pixel_level_*.py 와 같은 필드) + 검증 목록.
  nyu  : nyudepthv2_train_files_with_gt.txt 에서 검증용 장면을 떼어 낸다 — 장면 이름(끝 글자 a–z 를 뗀 것)을 시드 0 으로 섞은 순서대로,
         학습 장수의 5 % 를 넘을 때까지. 목록 nyu/lists/{train_ft,val_all,val}.txt (val = val_all 에서 고르게 400 장).
  kitti: prep/build_kitti.py 가 만든 kitti/lists/train_ft.txt (이미 KB crop 으로 자른 train_kb/ 경로).
  canonical_size = round(원본 크기 × 1000 / fx) (공식 식), prompt·solution 은 공식 문장. 경로는 데이터 루트 기준 (학습 때 --image_folder = --depth_root = 루트).
  파일 이름에 데이터셋 키(nyuv2 / kitti)가 들어가야 utils.datasets._match_dataset_config 가 깊이 범위를 고른다.
사용: python prep/make_jsonl.py --ds nyu --data ~/data/dvft/data --splits ~/data/dvft/splits
"""
import argparse
import collections
import hashlib
import json
import os
import re

import numpy as np

PROMPT = "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image."
SOLUTION = "OK, I will estimate the depth map of this image."
HOLD, NVAL = 0.05, 400


def read_list(p):
    return [l.split() for l in open(p) if l.strip()]


def record(img, dep, scale, w, h, fx):
    return {"image": img, "depth_path": dep, "depth_scale": scale, "original_rgb_size": [w, h], "original_depth_size": [w, h],
            "original_fx": round(fx, 2), "canonical_fx": 1000.0, "canonical_size": [round(w * 1000.0 / fx), round(h * 1000.0 / fx)],
            "prompt": PROMPT, "solution": SOLUTION}


def nyu(data, splits):
    tr = read_list(os.path.join(splits, "nyudepthv2_train_files_with_gt.txt"))
    base = lambda r: re.sub(r"[a-z]$", "", r.strip("/").split("/")[0])  # noqa: E731
    cnt = collections.Counter(base(r) for r, _, _ in tr)
    order = [sorted(cnt)[i] for i in np.random.default_rng(0).permutation(len(cnt))]
    hold, n = [], 0
    for s in order:
        if n >= HOLD * len(tr):
            break
        hold.append(s)
        n += cnt[s]
    val_all = [x for x in tr if base(x[0]) in hold]
    train = [x for x in tr if base(x[0]) not in hold]
    val = [val_all[i] for i in np.linspace(0, len(val_all) - 1, NVAL).round().astype(int)]
    os.makedirs(os.path.join(data, "nyu", "lists"), exist_ok=True)
    for k, L in dict(train_ft=train, val_all=val_all, val=val).items():
        open(os.path.join(data, "nyu", "lists", f"{k}.txt"), "w").write("".join(" ".join(x) + "\n" for x in L))
    recs = [record(f"nyu/train/{r.lstrip('/')}", f"nyu/train/{d.lstrip('/')}", 1000.0, 640, 480, float(f)) for r, d, f in train]
    return recs, dict(train_ft=len(train), val_all=len(val_all), val=len(val), val_scenes=hold, scenes_total=len(cnt))


def kitti(data):
    train = read_list(os.path.join(data, "kitti", "lists", "train_ft.txt"))
    recs = [record(f"kitti/train_kb/{r}", f"kitti/train_kb_gt/{d}", 256.0, 1216, 352, float(f)) for r, d, f in train]
    return recs, dict(train_ft=len(train))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["nyu", "kitti"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    a = ap.parse_args()
    data, splits = os.path.expanduser(a.data), os.path.expanduser(a.splits)
    recs, info = nyu(data, splits) if a.ds == "nyu" else kitti(data)
    missing = [r["image"] for r in recs if not (os.path.exists(os.path.join(data, r["image"])) and os.path.exists(os.path.join(data, r["depth_path"])))]
    assert not missing, f"없는 파일 {len(missing)}: {missing[:3]}"
    name = {"nyu": "nyuv2", "kitti": "kitti"}[a.ds]
    out = os.path.join(data, a.ds, "jsonl", f"{name}_train_ft.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    txt = "".join(json.dumps(r) + "\n" for r in recs)
    open(out, "w").write(txt)
    info.update(jsonl=out, records=len(recs), sha256=hashlib.sha256(txt.encode()).hexdigest()[:16],
                canonical_sizes=dict(collections.Counter(str(r["canonical_size"]) for r in recs)))
    json.dump(info, open(os.path.join(data, a.ds, "jsonl", "info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
