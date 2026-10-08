"""NYU Depth V2 표준 데이터 → <out>/nyu/{train,test}/ (경로 = BTS 목록 그대로) + 검증.
  train: BTS sync.zip 에서 nyudepthv2_train_files_with_gt.txt 의 24,231 쌍만 꺼낸다 (RGB jpg, Kinect 깊이 png = mm).
  test : 공식 labeled.mat 654 장을 BTS 추출 스크립트(utils/extract_official_train_test_set_from_mat.py)와 똑같이 만든다 —
         RGB 는 테두리 7 px 를 검게 칠한 jpg(cv2 기본 품질), 깊이는 rawDepths × 1000 을 uint16 으로 자른 png. 이름 = sceneTypes/rgb_<0 부터 번호>.
검증: 목록과 파일이 정확히 일치, 모든 RGB 480×640×3·깊이 480×640 uint16 이 읽힘, 깊이 범위, 테스트 장면이 학습 목록에 없음,
      (선택) 테스트 깊이 = vlm-depth-rmse 의 nyuv2_official rawDepths npy (1 mm 안).
사용: python prep/build_nyu.py --sync ~/data/nyuv2_bts/sync.zip --mat ~/data/nyuv2 --splits ~/data/dvft/splits --out ~/data/dvft/data [--official ~/data/nyuv2_official]
"""
import argparse
import json
import os
import re
import zipfile

import cv2
import h5py
import numpy as np
import scipy.io as sio


def read_list(p):
    return [l.split() for l in open(p) if l.strip()]


def check_pair(rgb_p, dep_p, scale=1000.0):
    rgb = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
    dep = cv2.imread(dep_p, cv2.IMREAD_UNCHANGED)
    assert rgb is not None and rgb.shape == (480, 640, 3), rgb_p
    assert dep is not None and dep.dtype == np.uint16 and dep.shape == (480, 640), dep_p
    d = dep[dep > 0] / scale
    return dict(valid=float((dep > 0).mean()), dmin=float(d.min()) if d.size else 0.0, dmax=float(d.max()) if d.size else 0.0)


def build_train(a, out):
    L = read_list(os.path.join(a.splits, "nyudepthv2_train_files_with_gt.txt"))
    z = zipfile.ZipFile(a.sync)
    for rgb, dep, _ in L:
        for p in (rgb, dep):
            dst = os.path.join(out, p.lstrip("/"))
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open("sync" + p) as s, open(dst + ".tmp", "wb") as d:  # zipfile 이 끝까지 읽으며 CRC 를 검사한다
                    d.write(s.read())
                os.replace(dst + ".tmp", dst)
    st = [check_pair(os.path.join(out, r.lstrip("/")), os.path.join(out, d.lstrip("/"))) for r, d, _ in L]
    scenes = sorted({r.split("/")[1] for r, _, _ in L})
    return L, st, scenes


def build_test(a, out):
    f = h5py.File(os.path.join(a.mat, "nyu_depth_v2_labeled.mat"), "r")
    sp = sio.loadmat(os.path.join(a.mat, "splits.mat"))
    test = {int(x) for x in sp["testNdxs"].ravel()}
    scenes = ["".join(chr(c) for c in f[r][()].ravel()) for r in f["sceneTypes"][0]]
    raw = f["rawDepths"]
    made = []
    for i in range(f["images"].shape[0]):
        if i + 1 not in test:
            continue
        folder = os.path.join(out, scenes[i])
        os.makedirs(folder, exist_ok=True)
        depth_raw = raw[i, :, :].T
        cv2.imwrite(os.path.join(folder, "sync_depth_%05d.png" % i), (depth_raw * 1000.0).astype(np.uint16))
        image = f["images"][i].T[:, :, ::-1]
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        black[7:474, 7:632, :] = image[7:474, 7:632, :]
        cv2.imwrite(os.path.join(folder, "rgb_%05d.jpg" % i), black)
        made.append((f"{scenes[i]}/rgb_{i:05d}.jpg", f"{scenes[i]}/sync_depth_{i:05d}.png", i + 1))
    names = ["".join(chr(c) for c in f[r][()].ravel()) for r in f["scenes"][0]]
    return made, {re.sub(r"[a-z]$", "", names[i - 1]) for i in test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", required=True)
    ap.add_argument("--mat", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--official", default="")
    a = ap.parse_args()
    a = argparse.Namespace(**{k: os.path.expanduser(v) for k, v in vars(a).items()})
    root = os.path.join(a.out, "nyu")
    rep = {}

    L, st, scenes = build_train(a, os.path.join(root, "train"))
    rep["train"] = dict(pairs=len(L), folders=len(scenes), focal=sorted({x[2] for x in L}),
                        valid_mean=round(float(np.mean([s["valid"] for s in st])), 4),
                        depth_min=round(min(s["dmin"] for s in st if s["dmin"] > 0), 4), depth_max=round(max(s["dmax"] for s in st), 3))

    made, test_scenes = build_test(a, os.path.join(root, "test"))
    T = read_list(os.path.join(a.splits, "nyudepthv2_test_files_with_gt.txt"))
    want, got = {(r, d) for r, d, _ in T}, {(r, d) for r, d, _ in made}
    assert want == got, f"테스트 목록 불일치: 목록에만 {len(want - got)}, 만든 것에만 {len(got - want)}"
    st = [check_pair(os.path.join(root, "test", r), os.path.join(root, "test", d)) for r, d, _ in made]
    rep["test"] = dict(images=len(made), list_match=True, valid_mean=round(float(np.mean([s["valid"] for s in st])), 4),
                       depth_max=round(max(s["dmax"] for s in st), 3))
    base = {re.sub(r"[a-z]$", "", s) for s in scenes}
    rep["scene_overlap_train_test"] = sorted(base & test_scenes)   # 장면 이름(끝 글자 a–z 제거) 기준

    if a.official:  # vlm-depth-rmse prep/nyu_official.py 의 rawDepths npy (labeled.mat 에서 직접) 와 대조
        diff = []
        for r, d, k in made:
            ref = np.load(os.path.join(a.official, f"{k:04d}_rawdepth.npy"))
            png = cv2.imread(os.path.join(root, "test", d), cv2.IMREAD_UNCHANGED) / 1000.0
            diff.append(float(np.abs(png - ref).max()))
        rep["test_vs_official_rawdepth_max_abs_m"] = round(max(diff), 6)
    rep["cv2"] = cv2.__version__
    json.dump(rep, open(os.path.join(root, "build_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
