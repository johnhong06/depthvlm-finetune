"""표준 테스트(또는 검증) 이미지마다 깊이 예측 → <out>/<id>.npy (float32, 정답과 같은 크기 — 채점이 정확하도록 반올림하지 않는다). 채점은 eval/std_eval.py.
모델:
  depthvlm  DepthVLM 공식 추론 경로 (eval/eval.py 와 같은 프롬프트·채팅 틀·process_vision_info·forward 한 번의 depth_pred, bf16).
            입력 크기 = 원본 × 1000 / fx (공식 canonical, 가로·세로 모두 fx 기준). KITTI 는 KB crop(아래 352 줄, 가운데 1216 칸)만 넣고
            예측을 원래 크기의 같은 자리에 붙인다 (밖은 0 → 채점에서 Garg crop 밖이라 안 쓰임). NYU 는 640×480 전체.
  metric3d  (채점 검증 전용) Metric3Dv2 ViT-L 을 Metric3D 공식 평가 전처리로: NYU = 테두리 6 px 검게, KITTI = 열 43:1197 만,
            비율 유지로 616×1064 안에 맞추고 검은 패딩, canonical(초점 1000) 깊이 × (fx'+fy')/2/1000.
사용: python eval/predict.py --model depthvlm --weights <DepthVLM 폴더 또는 HF id@rev> --ds kitti --data ~/data/dvft/data --splits ~/data/dvft/splits --out preds/zs/kitti
"""
import argparse
import importlib
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

from std_eval import records

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = (352, 1216)
M3D_NYU_K = [518.8579, 519.4691, 325.58245, 253.73617]  # Metric3D 의 NYU 테스트 intrinsics (vlm-depth-rmse m3d_nyu.py 와 같음)


def kb_box(h, w):
    top, left = h - KB[0], (w - KB[1]) // 2
    return top, left


def depthvlm(weights, processor):
    sys.path.insert(0, os.path.join(ROOT, "third_party/DepthVLM"))
    from model import Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from utils.datasets import DEFAULT_DEPTH_PROMPT
    try:
        importlib.import_module("flash_attn")
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    pid, prev = (processor or weights).split("@") if "@" in (processor or weights) else (processor or weights, None)
    wid, wrev = weights.split("@") if "@" in weights else (weights, None)
    proc = AutoProcessor.from_pretrained(pid, revision=prev)
    proc.tokenizer.padding_side = "left"
    m = Qwen3VLForConditionalGeneration.from_pretrained(wid, revision=wrev, torch_dtype=torch.bfloat16,
                                                        attn_implementation=attn, device_map="cuda:0").eval()
    print(f"[depthvlm] {weights} attention {attn}", flush=True)

    @torch.no_grad()
    def f(ds, path, fx):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        if ds == "kitti":
            top, left = kb_box(H, W)
            img = img.crop((left, top, left + KB[1], top + KB[0]))
        w, h = img.size
        img = img.resize((round(w * 1000.0 / fx), round(h * 1000.0 / fx)), Image.BILINEAR)  # = utils.datasets._load_and_resize_image
        msg = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
               {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DEFAULT_DEPTH_PROMPT}]}]
        inp = proc(text=[proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)], images=[process_vision_info(msg)[0]],
                   padding=True, return_tensors="pt").to("cuda")
        p = m(input_ids=inp.input_ids, attention_mask=inp.attention_mask, pixel_values=inp.get("pixel_values"),
              image_grid_thw=inp.get("image_grid_thw")).depth_pred[0].float().cpu().numpy()
        while p.ndim > 2:
            p = p[0]
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        if ds == "kitti":
            full = np.zeros((H, W), np.float32)
            full[top:top + KB[0], left:left + KB[1]] = p
            return full
        return p
    return f


def metric3d(ext):
    m = torch.hub.load(os.path.join(ext, "Metric3D"), "metric3d_vit_large", pretrain=True, source="local").cuda().eval()
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    @torch.no_grad()
    def f(ds, path, fx):
        rgb = cv2.imread(path)[:, :, ::-1]
        H, W = rgb.shape[:2]
        if ds == "nyu":
            x0, k = 0, M3D_NYU_K
            new = np.zeros_like(rgb)
            new[6:-6, 6:-6] = rgb[6:-6, 6:-6]   # Metric3D nyu_dataset.py: 테두리 6 px 검게
        else:
            x0, k = 43, [fx, fx]                 # Metric3D kitti_dataset.get_data_for_test: rgb[:, 43:1197]
            new = rgb[:, 43:1197]
        h, w = new.shape[:2]
        s = min(616 / h, 1064 / w)
        x = cv2.resize(np.ascontiguousarray(new), (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        ph, pw = 616 - x.shape[0], 1064 - x.shape[1]
        x = cv2.copyMakeBorder(x, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x = torch.div(torch.from_numpy(x.transpose((2, 0, 1))).float() - mean, std)[None].cuda()
        p = m.inference({"input": x})[0].squeeze()
        p = p[ph // 2: 616 - (ph - ph // 2), pw // 2: 1064 - (pw - pw // 2)]
        p = torch.nn.functional.interpolate(p[None, None], (h, w), mode="bilinear").squeeze()
        p = (p * (k[0] * s + k[1] * s) / 2 / 1000.0).cpu().numpy()
        full = np.zeros((H, W), np.float32)
        full[:, x0:x0 + w] = p
        return full
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["depthvlm", "metric3d"])
    ap.add_argument("--weights", default="JonnyYu828/DepthVLM-4B@2b2d02fcfe0c89c8aa7d541055e5a078930077c9")
    ap.add_argument("--processor", default=None, help="프로세서 폴더 (학습 체크포인트에는 이미지 프로세서 설정이 없어 기본 가중치 폴더를 준다)")
    ap.add_argument("--ext", default=os.path.join(ROOT, "ext"))
    ap.add_argument("--ds", required=True, choices=["nyu", "kitti"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--list", default=None)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    recs = records(a.ds, os.path.expanduser(a.data), os.path.expanduser(a.splits), a.list, a.split)[: a.limit or None]
    f = depthvlm(a.weights, a.processor) if a.model == "depthvlm" else metric3d(a.ext)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for i, (rid, rp, gp, fx) in enumerate(recs):
        dst = os.path.join(a.out, rid + ".npy")
        if os.path.exists(dst):
            continue
        p = f(a.ds, rp, fx)
        gh, gw = cv2.imread(gp, cv2.IMREAD_UNCHANGED).shape
        assert p.shape == (gh, gw), (rid, p.shape, (gh, gw))
        np.save(dst, p.astype(np.float32))
        if i % 100 == 0:
            print(f"[predict] {a.model} {a.ds} {i + 1}/{len(recs)} {(time.time() - t0) / (i + 1):.2f} s/장", flush=True)
    print(f"[predict] {a.model} {a.ds} 끝 {len(recs)} 장 → {a.out}", flush=True)


if __name__ == "__main__":
    main()
