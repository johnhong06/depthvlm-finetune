"""에폭 고르기 — 에폭마다 검증 세트(학습에서 떼어 낸 장면·주행 400 장) 채점 결과(std_eval .json) 중 이미지별 평균 AbsRel 이 가장 낮은 것.
같으면 나중 에폭. 테스트 세트는 고른 뒤 한 번만 잰다 (NOTES D-3).
사용: python eval/select_epoch.py <out.json> <체크포인트 폴더:결과.json> ...
"""
import json
import sys

out, pairs = sys.argv[1], [a.split(":", 1) for a in sys.argv[2:]]
rows = []
for ck, res in pairs:
    r = json.load(open(res))
    rows.append(dict(checkpoint=ck, step=int(ck.rstrip("/").rsplit("-", 1)[-1]), abs_rel=r["per_image_mean"]["abs_rel"], table=r["table"]))
best = min(rows, key=lambda r: (r["abs_rel"], -r["step"]))
json.dump(dict(rule="검증 세트 이미지별 평균 AbsRel 최소, 같으면 나중 에폭", candidates=rows, selected=best), open(out, "w"), indent=1, ensure_ascii=False)
for r in rows:
    print(f"[select] {'*' if r is best else ' '} {r['checkpoint']} AbsRel {100 * r['abs_rel']:.2f} {r['table']}")
print(best["checkpoint"])
