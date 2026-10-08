# depthvlm-finetune — 프로젝트 규칙

상위 `Jihyuck/CLAUDE.md` 의 공통 규칙이 그대로 적용된다. 진행 상황·결정 근거는 `NOTES.md`, 실제 적용한 설정 전부는 `docs/PROTOCOL.md`.
vlm-depth-rmse(측정 전용)에서 나온 질문 "DepthVLM 을 NYU·KITTI 에 파인튜닝하면 UniDepthV2 표 V·VI 에서 어디쯤인가"를 잰다. 학습이 들어가 저장소를 따로 둔다.

## 목표

DepthVLM-4B 를 NYU Depth V2 와 KITTI Eigen 분할에 각각 파인튜닝하고, UniDepthV2 표 V(NYU: δ1·δ2·δ3, A.Rel, RMS, Log10)·
표 VI(KITTI: δ1·δ2·δ3, A.Rel, RMS, RMSlog)와 같은 규칙으로 잰다. 파인튜닝 전(zero-shot) 값도 같은 규칙으로 함께 낸다.

## 반드시 지킬 규칙 (2026-10-08 사용자가 승인한 계획 — 바꿔야 하면 진행하지 말고 먼저 보고)

1. 평가 = BTS 공식 평가 규칙 (`docs/PROTOCOL.md` 2 절): NYU 654 장·rawDepths·Eigen crop·0.001–10 m / KITTI 652 장·annotated depth·Garg crop·0.001–80 m.
2. 지표는 이미지마다 계산한 뒤 평균 (표의 방식). pooled 는 보조. (vlm-depth-rmse 의 주 지표 pooled 와 다르다 — 표와 비교하려면 이 방식이어야 한다)
3. 테스트 세트는 학습에도, 에폭·설정 고르기에도 쓰지 않는다. 고르기는 학습에서 떼어 낸 검증 세트로만 하고, 테스트는 고른 뒤 한 번만 잰다.
4. 학습은 공식 2 단계 설정을 그대로 쓰고, 바꾼 것은 전부 `docs/PROTOCOL.md` 3 절과 NOTES 결정 기록에 적는다.
5. 데이터는 공식 원본에서 다시 만들고, 쓰기 전에 검증한다 (목록과 파일 일치, 크기·형식·범위, 장면/주행 겹침, 초점). 검증 결과는 `build_report.json` 과 NOTES.
6. 채점 코드는 학습 결과를 보기 전에 공개 수치 재현으로 검증한다 (NYU·KITTI 모두 Metric3Dv2).

## 작업 원칙

- 공개 코드를 최대한 그대로 쓴다: DepthVLM 은 `third_party/DepthVLM/`(고정 커밋 5d1472d, 수정 없음), Metric3D 는 `prep/fetch_ext.sh` 가 `ext/` 에 고정 커밋으로.
- 수치가 논문과 크게 다르면 버그부터 의심하고 원인을 확인한 뒤 진행한다.
- H200 에 줄 데이터는 zip 묶음(`prep/pack.py`, 무압축·2 GB 이하 조각 + SHA256)으로 만들고, 이미 서버에 있는 것은 다시 보내지 않는다. 저장소 푸시는 사용자가 직접 한다.
- 모델 가중치·데이터·예측은 저장소에 넣지 않는다 (`~/data/dvft`, `~/checkpoints`).

## 실행

```bash
# 로컬: 데이터 받기·만들기·검증 → zip 묶음
bash prep/fetch_splits.sh && bash prep/fetch_kitti.sh && bash prep/fetch_nyu_sync.sh
python prep/build_nyu.py --sync ~/data/nyuv2_bts/sync.zip --mat ~/data/nyuv2 --splits ~/data/dvft/splits --out ~/data/dvft/data --official ~/data/nyuv2_official
python prep/build_kitti.py --kitti ~/data/kitti --splits ~/data/dvft/splits --out ~/data/dvft/data
python prep/make_jsonl.py --ds nyu --data ~/data/dvft/data --splits ~/data/dvft/splits    # kitti 도 같은 식
python prep/pack.py --data ~/data/dvft/data --splits ~/data/dvft/splits --out ~/data/h200_staging/dvft
# 로컬: 채점 검증 (Metric3Dv2, ~/venv/metric3d)
bash prep/fetch_ext.sh && cd eval && python predict.py --model metric3d --ds kitti ... && python std_eval.py --ds kitti --protocol metric3d ...
# H200
bash run.sh env | smoke | zeroshot | nyu | kitti | all
```

환경: 로컬 `~/venv/depthvlm_train` (envs/depthvlm.txt, uv `--index-strategy unsafe-best-match`), `~/venv/metric3d` (검증용). H200 = run.sh 가 envs/depthvlm.txt 로 만든다.
