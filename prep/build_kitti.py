"""KITTI Eigen 표준 데이터 → <out>/kitti/ + 검증. 경로 = BTS 목록 그대로 (1 열 = raw 기준 RGB, 2 열 = annotated depth 기준 정답).
  test : eigen_test_files_with_gt.txt 의 697 장 중 정답 있는 652 장 — RGB·정답 원본 바이트 그대로 → full/, full_gt/
  val  : 학습 주행 중 떼어 낸 주행(시드 0 으로 섞은 주행 순서대로 학습 장수의 5 % 가 넘을 때까지)의 장들 — 원본 그대로 → full/, full_gt/
  train: 나머지 학습 장 — RGB·정답을 KB crop(아래 352 줄, 가운데 1216 칸)으로 잘라 무손실 png 로 → train_kb/, train_kb_gt/
         (DepthVLM 학습 코드는 자르기를 하지 않으므로 미리 자른다. 테스트·검증은 eval/predict.py 가 같은 상자로 자른다)
  목록: lists/{train_ft,val_all,val,test}.txt (BTS 형식), val = val_all 에서 고르게 400 장.
검증: 목록의 모든 장이 zip 에 있음, RGB 크기 = 정답 크기, 정답 uint16, 목록 초점 = calib P_rect_02 fx, 학습·검증·테스트 주행이 서로 겹치지 않음.
사용: python prep/build_kitti.py --kitti ~/data/kitti --splits ~/data/dvft/splits --out ~/data/dvft/data [--workers 8]
"""
import argparse
import collections
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

KB = (352, 1216)
HOLD, NVAL = 0.05, 400


def read_list(p):
    return [l.split() for l in open(p) if l.strip()]


def drive(r):
    return r.split("/")[1]


def calib_fx(kitti, date):
    with zipfile.ZipFile(os.path.join(kitti, "raw", f"{date}_calib.zip")) as z:
        for l in z.read(f"{date}/calib_cam_to_cam.txt").decode().splitlines():
            if l.startswith("P_rect_02:"):
                return float(l.split()[1])


def gt_index(kitti):
    """annotated depth zip 안 이름 → (train|val)/<2 열 경로>."""
    z = zipfile.ZipFile(os.path.join(kitti, "data_depth_annotated.zip"))
    idx = {}
    for n in z.namelist():
        if n.endswith(".png") and "/proj_depth/groundtruth/image_02/" in n:
            sp, rest = n.split("/", 1)
            idx[rest] = n
    return idx


def work(job):
    """한 주행: [(목록 줄, 종류)] → 꺼내기·자르기·검사. 반환 = 장마다 (rgb, gt, 종류, H, W, 유효 비율, 최대 깊이)."""
    kitti, out, dname, items, gidx = job
    zr = zipfile.ZipFile(os.path.join(kitti, "raw", f"{dname}.zip"))
    zg = zipfile.ZipFile(os.path.join(kitti, "data_depth_annotated.zip"))
    res = []
    for (r, d, f), kind in items:
        rb, gb = zr.read(r), zg.read(gidx[d])          # zipfile 이 CRC 를 검사한다
        img = cv2.imdecode(np.frombuffer(rb, np.uint8), cv2.IMREAD_COLOR)
        gt = cv2.imdecode(np.frombuffer(gb, np.uint8), cv2.IMREAD_UNCHANGED)
        assert img is not None and gt is not None and gt.dtype == np.uint16 and img.shape[:2] == gt.shape, (r, d)
        H, W = gt.shape
        if kind == "train":
            top, left = H - KB[0], (W - KB[1]) // 2
            pr, pg = os.path.join(out, "train_kb", r), os.path.join(out, "train_kb_gt", d)
            for p, a in ((pr, img[top:, left:left + KB[1]]), (pg, gt[top:, left:left + KB[1]])):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                assert cv2.imwrite(p, a), p
        else:
            for p, b in ((os.path.join(out, "full", r), rb), (os.path.join(out, "full_gt", d), gb)):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "wb").write(b)
        v = gt > 0
        res.append((r, d, kind, H, W, float(v.mean()), float(gt.max()) / 256.0))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    kitti, out = os.path.expanduser(a.kitti), os.path.join(os.path.expanduser(a.out), "kitti")
    tr = read_list(os.path.join(os.path.expanduser(a.splits), "eigen_train_files_with_gt.txt"))
    te = [x for x in read_list(os.path.join(os.path.expanduser(a.splits), "eigen_test_files_with_gt.txt")) if x[1] != "None"]

    # 검증용 주행: 학습 주행을 시드 0 으로 섞은 순서대로 학습 장수의 5 % 를 넘을 때까지
    cnt = collections.Counter(drive(r) for r, _, _ in tr)
    order = [sorted(cnt)[i] for i in np.random.default_rng(0).permutation(len(cnt))]
    hold, n = [], 0
    for dn in order:
        if n >= HOLD * len(tr):
            break
        hold.append(dn)
        n += cnt[dn]
    val_all = [x for x in tr if drive(x[0]) in hold]
    train = [x for x in tr if drive(x[0]) not in hold]
    val = [val_all[i] for i in np.linspace(0, len(val_all) - 1, NVAL).round().astype(int)]
    lists = dict(train_ft=train, val_all=val_all, val=val, test=te)
    os.makedirs(os.path.join(out, "lists"), exist_ok=True)
    for k, L in lists.items():
        open(os.path.join(out, "lists", f"{k}.txt"), "w").write("".join(" ".join(x) + "\n" for x in L))

    gidx = gt_index(kitti)
    missing = [d for _, d, _ in train + val_all + te if d not in gidx]
    assert not missing, f"annotated depth 에 없는 정답 {len(missing)}: {missing[:3]}"
    jobs = collections.defaultdict(list)
    for x in train:
        jobs[drive(x[0])].append((x, "train"))
    for x in val_all + te:
        jobs[drive(x[0])].append((x, "full"))
    with Pool(a.workers) as p:
        res = [r for rs in p.imap_unordered(work, [(kitti, out, dn, it, gidx) for dn, it in sorted(jobs.items())]) for r in rs]

    fx_bad = []
    for date in sorted({x[0].split("/")[0] for x in tr + te}):
        fx = calib_fx(kitti, date)
        fx_bad += [(x[0], x[2], fx) for x in tr + te if x[0].startswith(date) and abs(float(x[2]) - fx) > 1e-3]
    dtr, dte = {drive(x[0]) for x in train}, {drive(x[0]) for x in te}
    by = collections.defaultdict(list)
    for r in res:
        by[r[2]].append(r)
    rep = dict(train_frames=len(train), train_drives=len(dtr), val_drives=hold, val_all_frames=len(val_all), val_frames=len(val),
               test_frames=len(te), test_drives=len(dte), overlap_train_test=sorted(dtr & dte), overlap_val_test=sorted(set(hold) & dte),
               focal_mismatch=fx_bad[:5], n_focal_mismatch=len(fx_bad),
               sizes=dict(collections.Counter(f"{r[4]}x{r[3]}" for r in res)),
               valid_mean={k: round(float(np.mean([r[5] for r in v])), 4) for k, v in by.items()},
               depth_max={k: round(max(r[6] for r in v), 2) for k, v in by.items()}, processed=len(res))
    assert len(res) == len(train) + len(val_all) + len(te) and not fx_bad and not rep["overlap_train_test"] and not rep["overlap_val_test"]
    json.dump(rep, open(os.path.join(out, "build_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
