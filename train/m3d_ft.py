"""Metric3Dv2 ViT-L 을 NYU 에 파인튜닝해 SILog 를 잰다 (NOTES D-8). 공개 가중치는 zero-shot 뿐이라 (Metric3D 이슈 #144) 직접 파인튜닝하고,
테스트 δ1·AbsRel·RMSE 가 README 의 NYU 파인튜닝 값 (ViT-L 98.9 / 0.047 / 0.183) 에 닿는지로 재현을 확인한다 (판정 기준 = PASS, 결과 전에 고정).

공식 설정 (Metric3D training/mono/configs/RAFTDecoder/vit.raft5.giant2.nyu.py · vit.raft5.large.kitti.py, datasets/nyu_dataset.py) 에서 따른 것:
  AdamW 인코더 5e-7 · 디코더 1e-5 (betas 0.9/0.999, wd 0, eps 1e-10), 선형 워밍업 20 걸음 (비율 1e-6) 뒤 poly 0.9 (최소 1e-8), 기울기 자르기 5 (AMP 경로),
  사진 테두리 6 px 검게, 정답은 Eigen crop [45:471, 41:601] 안만, canonical 라벨 (깊이 × 1000/fx, 초점 → 1000),
  RandomResize 0.85–1.15 (p 0.5) → 616×1064 무작위 자르기 (모자라면 채움: 사진 0, 정답 무효), 좌우 뒤집기 p 0.4,
  깊이 손실 = 공식 손실 파일 그대로: GRUSequence (1.0, γ 0.9, stereo 0) + HDNRandom (0.5, 10) + HDSNRandom (0.5, 20, batch_limit 4) + VNL (1.0, 0.2).
바꾼 것: 법선 정답 (GeoNet NYU-normal) 이 없어 NormalBranch·DeNoConsistency·PWNPlanes 는 뺌, 색 변형은 회색 0.1 · 밝기/대비 0.1 만 (날씨·흐림·압축·가장자리 가리기 뺌),
  배치 16 · 6,000 걸음 (논문 부록: 배치 16, 6K 걸음, 약 2K 에 수렴), bf16 autocast, 어텐션은 SDPA (같은 계산 — 0 걸음 예측으로 확인).
검증: 학습에서 떼어 낸 400 장 (nyu/lists/val.txt) 을 평가와 같은 전처리 (eval/predict.py metric3d) 로 VAL_EVERY 걸음마다 → AbsRel 이 가장 낮은 걸음으로 테스트 654 장 한 번.
0 걸음 (zero-shot) 도 검증·테스트를 재서 기존 zero-shot 측정 (Metric3D 규칙 97.5 / 6.28 / 0.251) 재현을 확인한다.
사용: python train/m3d_ft.py --data <데이터> --splits <데이터/splits> --ext ext --out <결과 폴더> --work <임시 폴더> [--steps 6000 --bs 16 --mb 4]
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
import predict  # noqa: E402
from std_eval import RULES, eval_mask, records, score  # noqa: E402

cv2.setNumThreads(0)   # 데이터 작업자마다 OpenCV 스레드가 겹치지 않게
CROP = (616, 1064)
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
README = dict(d1=0.989, abs_rel=0.047, rmse=0.183)        # Metric3D README NYU 파인튜닝 ViT-L
PASS = dict(d1=0.984, abs_rel=0.052, rmse=0.203)          # 재현 판정 (NOTES D-8, 결과 전 고정): δ1 −0.5 %p, AbsRel·RMSE +10 % 이내


class NYUTrain(torch.utils.data.Dataset):
    """BTS sync 학습 쌍 → (정규화 사진 3×616×1064, canonical 정답 1×616×1064 (무효 −1 = 공식 ignore_label), canonical intrinsics 3×3)."""

    def __init__(self, recs):
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        _, rp, gp, _ = self.recs[i]
        rgb = cv2.imread(rp)[:, :, ::-1].astype(np.float32)
        h, w = rgb.shape[:2]
        assert (h, w) == (480, 640), (rp, rgb.shape)
        new = np.zeros_like(rgb)
        new[6:-6, 6:-6] = rgb[6:-6, 6:-6]                          # 공식 nyu_dataset: 테두리 6 px 검게
        rgb = new
        d0 = cv2.imread(gp, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        dep = np.zeros_like(d0)
        dep[45:471, 41:601] = d0[45:471, 41:601]                  # 공식 process_depth: Eigen crop 안만
        if random.random() < 0.1:                                  # 회색 (공식 to_gray 0.1)
            rgb = np.repeat((rgb @ np.array([0.299, 0.587, 0.114], np.float32))[..., None], 3, 2)
        if random.random() < 0.1:                                  # 밝기·대비 (공식 distortion 0.1 의 일부)
            rgb = np.clip(rgb * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)
        fx, fy, cx, cy = predict.M3D_NYU_K
        if random.random() < 0.5:                                  # RandomResize 0.85–1.15 (p 0.5), 정답은 최근접
            r = random.uniform(0.85, 1.15)
            nh, nw = int(round(h * r)), int(round(w * r))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            dep = cv2.resize(dep, (nw, nh), interpolation=cv2.INTER_NEAREST)
            fx, fy, cx, cy = fx * nw / w, fy * nh / h, cx * nw / w, cy * nh / h
            h, w = nh, nw
        s = 1000.0 / fx                                            # LabelScaleCononical: 깊이 × 1000/fx, 초점 × 같은 비율
        dep, fx, fy = dep * s, fx * s, fy * s
        H, W = CROP                                                # 616×1064 무작위 자르기 — 모자란 쪽은 캔버스 안 무작위 자리에 놓기
        y0, oy = (random.randint(0, h - H), 0) if h >= H else (0, random.randint(0, H - h))
        x0, ox = (random.randint(0, w - W), 0) if w >= W else (0, random.randint(0, W - w))
        ch, cw = min(h, H), min(w, W)
        img = np.zeros((H, W, 3), np.float32)
        lab = np.full((H, W), -1.0, np.float32)                  # 공식 RandomCrop ignore_label −1 (0 이면 신뢰도 손실이 0 나누기로 NaN → 매번 빠짐)
        img[oy:oy + ch, ox:ox + cw] = rgb[y0:y0 + ch, x0:x0 + cw]
        lab[oy:oy + ch, ox:ox + cw] = np.where(dep[y0:y0 + ch, x0:x0 + cw] > 0, dep[y0:y0 + ch, x0:x0 + cw], -1.0)
        cx, cy = cx - x0 + ox, cy - y0 + oy
        if random.random() < 0.4:                                  # 좌우 뒤집기 (공식 p 0.4)
            img, lab, cx = img[:, ::-1], lab[:, ::-1], W - 1 - cx
        x = (img - MEAN) / STD
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.from_numpy(np.ascontiguousarray(lab))[None],
                torch.from_numpy(K))


def load_loss(ext, fname, cls):
    """공식 손실 파일 (torch·numpy 만 씀) 을 패키지 __init__ 없이 불러온다."""
    p = os.path.join(ext, "Metric3D", "training", "mono", "model", "losses", fname)
    spec = importlib.util.spec_from_file_location("m3d_loss_" + cls, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, cls)


def patch_sdpa():
    """ViT 어텐션 (xFormers 없을 때의 q·kᵀ softmax) 을 같은 계산의 SDPA 로 — 3,300 토큰 학습의 메모리 때문. 기본 scale = head_dim^-0.5 = 원래 self.scale."""
    V = sys.modules["mono.model.backbones.ViT_DINO_reg"]
    orig = V.Attention.forward

    def fwd(self, x, attn_bias=None):
        if self.tuning_mode == "ssf" or attn_bias is not None:
            return orig(self, x, attn_bias)
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        return self.proj_drop(self.proj(y.transpose(1, 2).reshape(B, N, C)))
    V.Attention.forward = fwd


def predict_set(net, ext, recs, out_dir):
    net.eval()
    f = predict.metric3d(ext, model=net)
    os.makedirs(out_dir, exist_ok=True)
    for rid, rp, gp, fx in recs:
        np.save(os.path.join(out_dir, rid + ".npy"), f("nyu", rp, fx).astype(np.float32))
    net.train()


def evaluate(recs, pred_dir, protocol):
    """std_eval 과 같은 규칙의 사진별 평균 지표 + 전역 스케일 오차 (사진별 평균 log 오차의 RMS ×100)·평균 log 오차 ×100."""
    df, pooled = score("nyu", recs, pred_dir, protocol)
    sc, lo, hi = RULES[("nyu", protocol)]
    mus = []
    for rid, _, gp, _ in recs:
        g = cv2.imread(gp, cv2.IMREAD_UNCHANGED).astype(np.float64) / sc
        p = np.load(os.path.join(pred_dir, rid + ".npy")).astype(np.float64)
        if protocol == "bts":
            p = np.clip(np.nan_to_num(p, nan=lo, posinf=hi), lo, hi)
            v = (g > lo) & (g < hi)
        else:
            p = np.abs(p) + 1e-10
            v = (g >= lo) & (g <= hi)
        v &= eval_mask("nyu", protocol, *g.shape)
        mus.append(np.mean(np.log(p[v]) - np.log(g[v])))
    mus = np.array(mus)
    m = df.mean(numeric_only=True)
    return dict(n=len(recs), d1=m.d1, abs_rel=m.abs_rel, rmse=m.rmse, silog=m.silog, rmse_log=m.rmse_log, log10=m.log10,
                global_scale=100 * float(np.sqrt((mus ** 2).mean())), mean_log_err=100 * float(mus.mean()), pooled=pooled)


def fmt(r):
    return (f"δ1 {100 * r['d1']:.2f} AbsRel {r['abs_rel']:.4f} RMSE {r['rmse']:.4f} SILog {r['silog']:.2f} "
            f"전역 스케일 오차 {r['global_scale']:.2f} (평균 log 오차 {r['mean_log_err']:+.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--ext", default=os.path.join(ROOT, "ext"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16, help="걸음당 사진 수 (= mb × 누적)")
    ap.add_argument("--mb", type=int, default=4, help="한 번에 올리는 사진 수 — 메모리가 모자라면 첫 걸음에서 반으로 줄이고 누적을 늘린다")
    ap.add_argument("--lr_enc", type=float, default=5e-7)
    ap.add_argument("--lr_dec", type=float, default=1e-5)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--n", type=int, default=0, help="학습 사진 수 제한 (스모크)")
    ap.add_argument("--val_n", type=int, default=0)
    ap.add_argument("--test_n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--max_hours", type=float, default=8.0, help="학습 시간 한도 — 넘으면 그때까지의 최고 걸음으로 테스트")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.work, exist_ok=True)
    D, SP = os.path.expanduser(a.data), os.path.expanduser(a.splits)
    tr = records("nyu", D, SP, os.path.join(D, "nyu", "lists", "train_ft.txt"), "val")[: a.n or None]   # split='val' → nyu/train 폴더에서 읽음
    va = records("nyu", D, SP, os.path.join(D, "nyu", "lists", "val.txt"), "val")[: a.val_n or None]
    te = records("nyu", D, SP)[: a.test_n or None]
    assert not ({r[0] for r in tr} & {r[0] for r in va}), "학습·검증 겹침"
    print(f"[m3d_ft] 학습 {len(tr)} · 검증 {len(va)} · 테스트 {len(te)} 장, {a.steps} 걸음 × {a.bs} 장 (한 번에 {a.mb})", flush=True)

    net = torch.hub.load(os.path.join(a.ext, "Metric3D"), "metric3d_vit_large", pretrain=True, source="local").cuda()
    patch_sdpa()
    L = dict(gru=load_loss(a.ext, "GRUSequenceLoss.py", "GRUSequenceLoss")(loss_weight=1.0, loss_gamma=0.9, stereo_sup=0.0),
             hdn=load_loss(a.ext, "HDNL_random.py", "HDNRandomLoss")(loss_weight=0.5, random_num=10),
             hdsn=load_loss(a.ext, "HDSNL_random.py", "HDSNRandomLoss")(loss_weight=0.5, random_num=20, batch_limit=4),
             vnl=load_loss(a.ext, "VNL.py", "VNLoss")(sample_ratio=0.2, loss_weight=1.0))
    res = dict(args=vars(a), readme=README, pass_rule=PASS, val=[])

    # 0 걸음 = zero-shot: 기존 측정 재현 확인 (테스트) + 검증 기준점
    predict_set(net, a.ext, te, os.path.join(a.work, "pred_test_step0"))
    res["test_step0"] = {p: evaluate(te, os.path.join(a.work, "pred_test_step0"), p) for p in ("metric3d", "bts")}
    print(f"[m3d_ft] 0 걸음 테스트 (Metric3D 규칙) {fmt(res['test_step0']['metric3d'])}", flush=True)

    def validate(step):
        d = os.path.join(a.work, f"pred_val_{step}")
        predict_set(net, a.ext, va, d)
        r = evaluate(va, d, "metric3d")
        r["step"] = step
        res["val"].append({k: v for k, v in r.items() if k != "pooled"})
        print(f"[m3d_ft] 검증 {step} 걸음: {fmt(r)}", flush=True)
        return r["abs_rel"]

    best = (validate(0), 0)
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, os.path.join(a.work, "best.pt"))
    enc, dec = list(net.depth_model.encoder.parameters()), list(net.depth_model.decoder.parameters())
    opt = torch.optim.AdamW([dict(params=enc, lr=a.lr_enc), dict(params=dec, lr=a.lr_dec)], betas=(0.9, 0.999), weight_decay=0.0, eps=1e-10)
    params = enc + dec

    def set_lr(step):  # mmcv poly (power 0.9, min 1e-8) + 선형 워밍업 20 걸음 (ratio 1e-6)
        for g, b in zip(opt.param_groups, (a.lr_enc, a.lr_dec)):
            lr = (b - 1e-8) * (1 - step / a.steps) ** 0.9 + 1e-8
            if step < 20:
                lr *= 1 - (1 - step / 20) * (1 - 1e-6)
            g["lr"] = lr

    ds = NYUTrain(tr)

    def loader(mb):
        return torch.utils.data.DataLoader(ds, batch_size=mb, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                                           persistent_workers=a.workers > 0)

    mb = min(a.mb, a.bs)
    dl, it = loader(mb), None
    net.train()
    t0, tlog, acc, first = time.time(), time.time(), {}, True
    step = 0
    while step < a.steps:
        accum = a.bs // mb
        set_lr(step)
        opt.zero_grad(set_to_none=True)
        parts = {}
        try:
            for _ in range(accum):
                try:
                    x, y, K = next(it)
                except (StopIteration, TypeError):
                    it = iter(dl)
                    x, y, K = next(it)
                x, y, K = x.cuda(non_blocking=True), y.cuda(non_blocking=True), K.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = net.depth_model(input=x)
                preds = [p.float() for p in out["predictions_list"]]
                confs = [c.float() for c in out["confidence_list"]]
                m = y > 1e-3
                kw = dict(dataset=["NYU"] * x.shape[0], data_type=["lidar"] * x.shape[0])
                ls = dict(gru=L["gru"](preds, y, y, confidence_list=confs, mask=m, **kw), hdn=L["hdn"](preds[-1], y, mask=m, **kw),
                          hdsn=L["hdsn"](preds[-1], y, mask=m, sem_mask=None, **kw), vnl=L["vnl"](preds[-1], y, m, K, **kw))
                loss = sum(ls.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"손실이 유한하지 않음: { {k: float(v) for k, v in ls.items()} }")
                (loss / accum).backward()
                for k, v in ls.items():
                    parts[k] = parts.get(k, 0.0) + float(v) / accum
        except torch.cuda.OutOfMemoryError:
            if not first or mb == 1:
                raise
            opt.zero_grad(set_to_none=True)
            out = preds = confs = loss = ls = None
            torch.cuda.empty_cache()
            mb //= 2
            print(f"[m3d_ft] 메모리 부족 → 한 번에 {mb} 장 × 누적 {a.bs // mb}", flush=True)
            dl, it = loader(mb), None
            continue
        gn = torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        step += 1
        first = False
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v
        if step % a.log_every == 0 or step == a.steps:
            n = a.log_every if step % a.log_every == 0 else step % a.log_every
            print(f"[m3d_ft] 걸음 {step}/{a.steps} 손실 {sum(acc.values()) / n:.4f} ("
                  + " ".join(f"{k} {v / n:.4f}" for k, v in acc.items())
                  + f") 기울기 {float(gn):.2f} lr {opt.param_groups[0]['lr']:.2e}/{opt.param_groups[1]['lr']:.2e} "
                  f"{(time.time() - tlog) / n:.2f} s/걸음 GPU 최대 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            acc, tlog = {}, time.time()
        over = time.time() - t0 > a.max_hours * 3600
        if step % a.val_every == 0 or step == a.steps or over:
            v = validate(step)
            if v < best[0]:
                best = (v, step)
                torch.save({k: t.detach().cpu() for k, t in net.state_dict().items()}, os.path.join(a.work, "best.pt"))
            json.dump(res, open(os.path.join(a.out, "m3d_ft.json"), "w"), indent=1, default=float)
        if over:
            print(f"[m3d_ft] 시간 한도 {a.max_hours} h — {step} 걸음에서 멈춤", flush=True)
            break
    res["train_hours"] = (time.time() - t0) / 3600
    res["best_step"] = best[1]
    print(f"[m3d_ft] 학습 끝 {res['train_hours']:.2f} h, 검증 AbsRel 최저 {best[0]:.4f} @ {best[1]} 걸음", flush=True)

    net.load_state_dict(torch.load(os.path.join(a.work, "best.pt"), map_location="cuda"))
    predict_set(net, a.ext, te, os.path.join(a.work, "pred_test_best"))
    res["test_best"] = {p: evaluate(te, os.path.join(a.work, "pred_test_best"), p) for p in ("metric3d", "bts")}
    t = res["test_best"]["metric3d"]
    ok = {k: (t[k] >= PASS[k]) if k == "d1" else (t[k] <= PASS[k]) for k in PASS}
    res["pass"] = ok
    json.dump(res, open(os.path.join(a.out, "m3d_ft.json"), "w"), indent=1, default=float)
    z = res["test_step0"]["metric3d"]
    print(f"[m3d_ft] 테스트 {len(te)} 장 (Metric3D 규칙) zero-shot  {fmt(z)}", flush=True)
    print(f"[m3d_ft] 테스트 {len(te)} 장 (Metric3D 규칙) 파인튜닝 {fmt(t)}", flush=True)
    print(f"[m3d_ft] 테스트 (BTS 규칙) zero-shot {fmt(res['test_step0']['bts'])} | 파인튜닝 {fmt(res['test_best']['bts'])}", flush=True)
    print(f"[m3d_ft] 재현 판정 (README δ1 98.9 · AbsRel 0.047 · RMSE 0.183, 기준 δ1 ≥ 98.4 · AbsRel ≤ 0.052 · RMSE ≤ 0.203): "
          + " · ".join(f"{k} {'통과' if v else '미달'}" for k, v in ok.items())
          + f" → {'재현 성공 — SILog 를 파인튜닝 Metric3Dv2 값으로 씀' if all(ok.values()) else '재현 미달 — SILog 는 참고로만'}", flush=True)


if __name__ == "__main__":
    main()
