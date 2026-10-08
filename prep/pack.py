"""H200 전달용 zip 묶음 → <out>/dvft_<묶음>_<NN>.zip (무압축 ZIP, 조각마다 2 GB 이하 — 담는 파일 합 1,900 MiB 까지) + dvft_SHA256SUMS.txt + dvft_manifest.json.
zip 안 경로 = 데이터 루트(~/data/dvft/data) 기준 상대 경로 그대로 → H200 에서 한 폴더에 풀면 로컬과 같은 구조가 된다.
  meta        : 분할 목록 4 개(splits/), nyu·kitti 의 lists/·jsonl/·build_report.json
  nyu_test    : nyu/test/ (공식 테스트 654 장, BTS 형식)
  nyu_train   : nyu/train/ 중 BTS 학습 목록의 24,231 쌍 (검증용으로 떼어 낸 장면 포함 — 학습 jsonl 에서만 빠진다)
  kitti_eval  : kitti/full/, kitti/full_gt/ (테스트 652 장 + 검증 주행의 장, 원본 크기)
  kitti_train : kitti/train_kb/, kitti/train_kb_gt/ (KB crop 으로 자른 학습 장)
DepthVLM-4B 가중치는 넣지 않는다 (H200 이 HF 에서 고정 리비전으로 받는다 — vlm-depth-rmse Track B 에서 확인된 방식).
사용: python prep/pack.py --data ~/data/dvft/data --splits ~/data/dvft/splits --out ~/data/h200_staging/dvft
"""
import argparse
import hashlib
import json
import os
import zipfile

LIMIT = 1900 * 2 ** 20


def files_under(root, rel):
    out = []
    for dp, _, fn in os.walk(os.path.join(root, rel)):
        out += [os.path.relpath(os.path.join(dp, f), root) for f in fn]
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, splits, out = (os.path.expanduser(x) for x in (a.data, a.splits, a.out))
    nyu_train = sorted({p.strip("/") for l in open(os.path.join(splits, "nyudepthv2_train_files_with_gt.txt")) if l.strip()
                        for p in [f"nyu/train/{l.split()[0].lstrip('/')}", f"nyu/train/{l.split()[1].lstrip('/')}"]})
    groups = {
        "meta": [(os.path.join(splits, f), f"splits/{f}") for f in sorted(os.listdir(splits))]
                + [(os.path.join(data, p), p) for ds in ("nyu", "kitti") for sub in ("lists", "jsonl") for p in files_under(data, f"{ds}/{sub}")]
                + [(os.path.join(data, f"{ds}/build_report.json"), f"{ds}/build_report.json") for ds in ("nyu", "kitti")],
        "nyu_test": [(os.path.join(data, p), p) for p in files_under(data, "nyu/test")],
        "nyu_train": [(os.path.join(data, p), p) for p in nyu_train],
        "kitti_eval": [(os.path.join(data, p), p) for p in files_under(data, "kitti/full") + files_under(data, "kitti/full_gt")],
        "kitti_train": [(os.path.join(data, p), p) for p in files_under(data, "kitti/train_kb") + files_under(data, "kitti/train_kb_gt")],
    }
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.startswith("dvft_"):
            os.remove(os.path.join(out, f))
    man, sums = {}, []
    for g, items in groups.items():
        missing = [s for s, _ in items if not os.path.exists(s)]
        assert not missing, f"{g}: 없는 파일 {len(missing)}: {missing[:3]}"
        part, size, z, k = [], 0, None, -1
        for src, arc in items:
            n = os.path.getsize(src)
            if z is None or size + n > LIMIT:
                if z:
                    z.close()
                k += 1
                name = f"dvft_{g}_{k:02d}.zip"
                z, size = zipfile.ZipFile(os.path.join(out, name), "w", zipfile.ZIP_STORED, allowZip64=True), 0
                part.append([name, 0, 0])
            z.write(src, arc)
            size += n
            part[-1][1] += 1
            part[-1][2] += n
        z.close()
        man[g] = [dict(zip=p[0], files=p[1], bytes=p[2]) for p in part]
        for p in part:
            h = hashlib.sha256()
            with open(os.path.join(out, p[0]), "rb") as f:
                while c := f.read(16 << 20):
                    h.update(c)
            sums.append(f"{h.hexdigest()}  {p[0]}")
        print(f"[pack] {g}: 파일 {sum(p[1] for p in part):,} 개, {sum(p[2] for p in part) / 2**30:.2f} GiB, zip {len(part)} 개", flush=True)
    open(os.path.join(out, "dvft_SHA256SUMS.txt"), "w").write("\n".join(sums) + "\n")
    json.dump(man, open(os.path.join(out, "dvft_manifest.json"), "w"), indent=1)
    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.startswith("dvft_"))
    print(f"[pack] 끝: zip {len(sums)} 개, 합 {total / 2**30:.2f} GiB → {out}")


if __name__ == "__main__":
    main()
