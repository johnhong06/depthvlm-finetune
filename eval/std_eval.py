"""표준 채점 — BTS 공식 평가 코드(pytorch/bts_eval.py, 커밋 5e3406b)와 같은 규칙. UniDepthV2 표 V(NYU)·표 VI(KITTI) 의 방식.
  NYU  : 정답 png ÷ 1000, 1e-3 < 정답 < 10 m, Eigen crop [45:471, 41:601], 예측(정답 크기)을 [1e-3, 10] 으로 자름.
  KITTI: 정답 png ÷ 256, 1e-3 < 정답 < 80 m, Garg crop, 예측(정답 크기, KB crop 밖은 0)을 [1e-3, 80] 으로 자름.
  지표는 이미지마다 계산한 뒤 평균 (표의 방식, 주 결과). pooled(전체 픽셀 한꺼번에)는 보조.
  --protocol metric3d: Metric3D 공식 평가 코드(training/mono, 검증용) 그대로 — KITTI Eigen crop, 정답 0.1–80 m (경계 포함, clip_depth), NYU 0.1–10 m,
                      예측은 자르지 않음(do_test.py 가 clip_range 를 넘기지 않는다, |예측| + 1e-10). 논문 KITTI 표의 'RMS_log' 칸은 이 코드의 silog 다 (NOTES F-4).
예측 = <pred>/<id>.npy (float16/32, 정답과 같은 크기). id 는 records() 참고.
사용: python eval/std_eval.py --ds nyu --data ~/data/dvft/data --splits ~/data/dvft/splits --pred preds/m3d/nyu --out results/m3d_nyu [--protocol metric3d]
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

RULES = {  # (정답 배율, 최소, 최대)
    ("nyu", "bts"): (1000.0, 1e-3, 10.0), ("kitti", "bts"): (256.0, 1e-3, 80.0),
    ("nyu", "metric3d"): (1000.0, 0.1, 10.0), ("kitti", "metric3d"): (256.0, 0.1, 80.0),
}
NAMES = ["d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog"]


def records(ds, data, splits, lst=None, split="test"):
    """BTS 형식 목록(rgb 정답 초점) → [(id, rgb 경로, 정답 경로, fx)]. 기본은 공식 테스트 목록. KITTI 는 정답 없는 45 장(None)을 뺀다.
    split = val 이면 학습에서 떼어 낸 검증 목록 — NYU 는 학습 폴더(nyu/train)에서 읽고, KITTI 는 테스트와 같이 원본 크기(kitti/full)에서 읽는다."""
    if ds == "nyu":
        lst = lst or os.path.join(splits, "nyudepthv2_test_files_with_gt.txt")
        img = gt = os.path.join(data, "nyu", split if split == "test" else "train")
    else:
        lst = lst or os.path.join(splits, "eigen_test_files_with_gt.txt")
        img, gt = os.path.join(data, "kitti", "full"), os.path.join(data, "kitti", "full_gt")
    out = []
    for line in open(lst):
        if not line.strip():
            continue
        r, d, f = line.split()
        if d == "None":
            continue
        rid = (r.strip("/").replace("/", "__").rsplit(".", 1)[0] if ds == "nyu"
               else f"{r.split('/')[1]}__{os.path.splitext(os.path.basename(r))[0]}")
        out.append((rid, os.path.join(img, r.lstrip("/")), os.path.join(gt, d.lstrip("/")), float(f)))
    return out


def eval_mask(ds, protocol, h, w):
    m = np.zeros((h, w), bool)
    if ds == "nyu":
        m[45:471, 41:601] = True
    elif protocol == "bts":   # Garg crop
        m[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
    else:                     # Eigen crop (KITTI) — Metric3D kitti_dataset.process_depth
        m[int(0.3324324 * h):int(0.91351351 * h), int(0.0359477 * w):int(0.96405229 * w)] = True
    return m


def compute_errors(gt, pred):
    """bts_eval.compute_errors 그대로 (float64)."""
    thresh = np.maximum(gt / pred, pred / gt)
    d1, d2, d3 = [(thresh < 1.25 ** k).mean() for k in (1, 2, 3)]
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    log10 = np.mean(np.abs(np.log10(pred) - np.log10(gt)))
    return dict(zip(NAMES, [d1, d2, d3, abs_rel, sq_rel, rmse, rmse_log, log10, silog]))


def score(ds, recs, pred_dir, protocol="bts"):
    scale, lo, hi = RULES[(ds, protocol)]
    rows, pool = [], []
    for rid, _, gp, _ in recs:
        gt = cv2.imread(gp, cv2.IMREAD_UNCHANGED).astype(np.float64) / scale
        pred = np.load(os.path.join(pred_dir, rid + ".npy")).astype(np.float64)
        assert pred.shape == gt.shape, (rid, pred.shape, gt.shape)
        if protocol == "bts":   # bts_eval.py: 예측을 [최소, 최대] 로 자르고, 정답은 경계 제외
            pred[np.isinf(pred)] = hi
            pred[np.isnan(pred)] = lo
            pred = np.clip(pred, lo, hi)
            valid = (gt > lo) & (gt < hi)
        else:                   # Metric3D do_test.py: 예측 안 자름, 정답은 clip_depth 로 [0.1, 최대] 밖만 무효
            pred = np.abs(pred) + 1e-10
            valid = (gt >= lo) & (gt <= hi)
        valid &= eval_mask(ds, protocol, *gt.shape)
        g, p = gt[valid], pred[valid]
        rows.append(dict(id=rid, n=int(valid.sum()), **compute_errors(g, p)))
        lg = np.log(p) - np.log(g)
        th = np.maximum(g / p, p / g)
        pool.append([valid.sum(), *[(th < 1.25 ** k).sum() for k in (1, 2, 3)], (np.abs(g - p) / g).sum(), ((g - p) ** 2).sum(),
                     (lg ** 2).sum(), np.abs(lg / np.log(10)).sum()])
    df = pd.DataFrame(rows)
    s = np.sum(pool, axis=0)
    pooled = dict(d1=s[1] / s[0], d2=s[2] / s[0], d3=s[3] / s[0], abs_rel=s[4] / s[0], rmse=np.sqrt(s[5] / s[0]),
                  rmse_log=np.sqrt(s[6] / s[0]), log10=s[7] / s[0])
    return df, pooled


def table_row(ds, m):
    """표 V·VI 표기: δ 는 %, AbsRel 은 ×100, RMS 는 m, 마지막 칸은 NYU Log10 / KITTI RMSlog."""
    last = ("Log10", m["log10"], 3) if ds == "nyu" else ("RMSlog", m["rmse_log"], 3)
    return (f"| δ1 {100 * m['d1']:.1f} | δ2 {100 * m['d2']:.1f} | δ3 {100 * m['d3']:.1f} | A.Rel {100 * m['abs_rel']:.2f} "
            f"| RMS {m['rmse']:.3f} | {last[0]} {last[1]:.{last[2]}f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["nyu", "kitti"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", default=None, help="BTS 형식 목록 (기본: 공식 테스트)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--protocol", default="bts", choices=["bts", "metric3d"])
    a = ap.parse_args()
    recs = records(a.ds, os.path.expanduser(a.data), os.path.expanduser(a.splits), a.list, a.split)
    df, pooled = score(a.ds, recs, a.pred, a.protocol)
    mean = {k: float(df[k].mean()) for k in NAMES}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out + "_per_image.csv", index=False)
    res = dict(ds=a.ds, split=a.split, protocol=a.protocol, images=len(df), pixels=int(df.n.sum()), per_image_mean=mean,
               pooled={k: float(v) for k, v in pooled.items()}, table=table_row(a.ds, mean))
    json.dump(res, open(a.out + ".json", "w"), indent=1, ensure_ascii=False)
    print(f"[std_eval] {a.ds} {a.protocol} 이미지 {len(df)} 픽셀 {df.n.sum():,} → {a.out}.json")
    print("  이미지별 평균:", res["table"], f"SILog {mean['silog']:.2f}")
    print("  pooled      :", table_row(a.ds, pooled))


if __name__ == "__main__":
    main()
