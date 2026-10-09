"""파인튜닝 전후 예측 비교 (표준 규칙의 채점 픽셀, NOTES F-13·F-14). 같은 사진끼리:
  ① 배율·모양: d = ln 예측 − ln 정답, 사진마다 배율 = 평균 d, 모양 = d 의 표준편차(= SILog/100), 배율 비중 = Σ(평균 d)² / Σ 평균 d²,
     사진마다 배율을 정답으로 맞췄을 때(상한)의 δ1 / A.Rel / RMS.
  ② 경계·내부: 정답에서 이웃(상하좌우) 깊이 비 > 1.1 인 픽셀을 3 px 넓힌 것이 경계 (vlm-depth-rmse 와 같은 정의). 픽셀을 모아(pooled) δ1 / A.Rel / RMS.
  ③ 거리 구간: NYU 0–2·2–4·4–6·6–10 m, KITTI 0–10·10–20·20–40·40–80 m (pooled).
  ④ 토큰 격자: log 예측을 행 방향으로 33 px 이동평균만큼 빼고 FFT — 토큰 주기 주파수의 세기 ÷ 주변 주파수 중앙값 (1 이면 격자 없음).
     토큰 주기는 정답 해상도 기준 NYU 640/39 = 16.4 px (입력 1233 → 1248 = 토큰 39 개), KITTI 는 KB crop 1216 칸 / 토큰 수(입력 너비에 따라 53–54).
  ⑤ 사진별: A.Rel 이 좋아진 사진 비율.
예측 = float16/32 npy (eval/predict.py 형식). 사용: python eval/breakdown.py --ds nyu --data ~/data/dvft/data --splits ~/data/dvft/splits --pred 이름=폴더 ...
"""
import argparse
import os

import cv2
import numpy as np

from std_eval import RULES, compute_errors, eval_mask, records

BINS = {"nyu": [0, 2, 4, 6, 10], "kitti": [0, 10, 20, 40, 80]}


def pooled(g, p):
    t = np.maximum(g / p, p / g)
    return np.array([(t < 1.25).mean(), (np.abs(g - p) / g).mean(), np.sqrt(((g - p) ** 2).mean())])


def boundary(g, valid):
    r = np.ones_like(g)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = g[: g.shape[0] - dy, : g.shape[1] - dx], g[dy:, dx:]
        ok = (a > 0) & (b > 0)
        q = np.where(ok, np.maximum(a, b) / np.maximum(np.minimum(a, b), 1e-6), 1.0)
        r[: g.shape[0] - dy, : g.shape[1] - dx] = np.maximum(r[: g.shape[0] - dy, : g.shape[1] - dx], q)
        r[dy:, dx:] = np.maximum(r[dy:, dx:], q)
    seed = (r > 1.1).astype(np.uint8)
    return cv2.dilate(seed, np.ones((7, 7), np.uint8)).astype(bool) & valid


def grid_ratio(logp, period, x0=0, x1=None):
    x = logp[:, x0:x1]
    hp = x - cv2.blur(x, (33, 1))
    P = (np.abs(np.fft.rfft(hp, axis=1)) ** 2).mean(axis=0)
    k = x.shape[1] / period
    kt = int(round(k))
    nb = [i for i in range(kt - 8, kt + 9) if abs(i - kt) > 1 and 0 < i < len(P)]
    return P[kt - 1: kt + 2].max() / np.median(P[nb])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["nyu", "kitti"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="이름=예측 폴더 (첫 번째가 기준)")
    a = ap.parse_args()
    scale, lo, hi = RULES[(a.ds, "bts")]
    recs = records(a.ds, os.path.expanduser(a.data), os.path.expanduser(a.splits))
    preds = dict(x.split("=", 1) for x in a.pred)
    acc = {n: dict(per=[], sc2=[], ms=[], shp=[], ora=[], cat={}, grid=[]) for n in preds}
    for rid, _, gp, _ in recs:
        g = cv2.imread(gp, cv2.IMREAD_UNCHANGED).astype(np.float64) / scale
        valid = (g > lo) & (g < hi) & eval_mask(a.ds, "bts", *g.shape)
        bd = boundary(g, valid) if a.ds == "nyu" else None
        cats = {"전체": valid}
        if bd is not None:
            cats["경계"], cats["내부"] = bd, valid & ~bd
        e = BINS[a.ds]
        for lo_, hi_ in zip(e[:-1], e[1:]):
            cats[f"{lo_}–{hi_} m"] = valid & (g >= lo_) & (g < hi_)
        for n, d in preds.items():
            p = np.clip(np.load(os.path.join(d, rid + ".npy")).astype(np.float64), lo, hi)
            gg, pp = g[valid], p[valid]
            dd = np.log(pp) - np.log(gg)
            A = acc[n]
            A["per"].append(compute_errors(gg, pp)["abs_rel"])
            A["sc2"].append(dd.mean() ** 2); A["ms"].append((dd ** 2).mean()); A["shp"].append(dd.std())
            A["ora"].append(pooled(gg, np.clip(pp * np.median(gg / pp), lo, hi)))
            for c, m in cats.items():
                if m.any():
                    A["cat"].setdefault(c, []).append((g[m], p[m]))
            if a.ds == "nyu":
                A["grid"].append(grid_ratio(np.log(p), 640 / 39))
            else:
                H, W = g.shape
                left = (W - 1216) // 2
                ntok = round(round(1216 * 1000 / 721.5377) / 32)   # 정답 해상도 근사 (초점 차이는 무시)
                A["grid"].append(grid_ratio(np.log(p[H - 352:, :]), 1216 / ntok, left, left + 1216))
    names = list(preds)
    print(f"[breakdown] {a.ds} 사진 {len(recs)} 장 — pooled δ1 / A.Rel / RMS (δ·A.Rel 은 %)")
    for c in acc[names[0]]["cat"]:
        row = []
        for n in names:
            g = np.concatenate([x[0] for x in acc[n]["cat"][c]]); p = np.concatenate([x[1] for x in acc[n]["cat"][c]])
            m = pooled(g, p)
            row.append(f"{n} {100 * m[0]:.1f} / {100 * m[1]:.2f} / {m[2]:.3f}")
        share = 100 * len(np.concatenate([x[0] for x in acc[names[0]]["cat"][c]])) / len(np.concatenate([x[0] for x in acc[names[0]]["cat"]["전체"]]))
        print(f"  {c:8s} ({share:4.1f} % 픽셀) | " + " | ".join(row))
    for n in names:
        A = acc[n]
        o = np.mean(A["ora"], axis=0)
        print(f"  {n}: 배율 비중 {100 * np.sum(A['sc2']) / np.sum(A['ms']):.0f} %, 평균 |배율| {np.mean(np.sqrt(A['sc2'])):.3f}, 모양 {np.mean(A['shp']):.3f}, "
              f"사진마다 배율 맞춤(상한, 사진별 평균) {100 * o[0]:.1f} / {100 * o[1]:.2f} / {o[2]:.3f}, 토큰 격자 세기 중앙 {np.median(A['grid']):.2f}")
    base = np.array(acc[names[0]]["per"])
    for n in names[1:]:
        cur = np.array(acc[n]["per"])
        print(f"  {n} vs {names[0]}: A.Rel 이 좋아진 사진 {100 * np.mean(cur < base):.0f} %, 사진별 A.Rel 변화 중앙 {100 * np.median(cur - base):+.2f}")


if __name__ == "__main__":
    main()
