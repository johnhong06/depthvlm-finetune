# 실험 설정 전체 (depthvlm-finetune)

DepthVLM-4B 를 NYU Depth V2 와 KITTI Eigen 분할에 각각 파인튜닝하고, UniDepthV2 논문 표 V(NYU)·표 VI(KITTI)와 같은 규칙으로 잰다.
표에서 모델끼리 같은 것은 평가 규칙뿐이고 학습 방식은 모델마다 다르다. UniDepthV2 도 파인튜닝 설정은 "표준 관행대로"라고만 적었다.
그래서 평가는 표준 규칙을 그대로 따르고, DepthVLM 은 공식 학습 설정을 가능한 한 그대로 쓴다. 바꾼 것은 모두 아래에 적는다.

## 1. 데이터

| | NYU (표 V) | KITTI (표 VI) |
|:--|:--|:--|
| 테스트 | 공식 테스트 654 장 (`nyu_depth_v2_labeled.mat` + `splits.mat`) | Eigen 테스트 697 장 중 정답이 있는 652 장 |
| 테스트 정답 | rawDepths (Kinect 원측정, 빈 곳 = 0) × 1000 → uint16 png (BTS 추출과 같음) | 공식 annotated depth (png ÷ 256) |
| 학습 | BTS `sync` 24,231 쌍 (NYU raw 의 학습 장면) | Eigen 학습 23,158 장 (주행 33 개, 왼쪽 컬러 카메라 `image_02`) |
| 학습 정답 | Kinect 깊이 png (mm, 빈 곳 = 0) | annotated depth png ÷ 256 |
| 검증 (에폭 고르기) | 학습 장면 이름(끝 글자 a–z 뗀 것)을 시드 0 으로 섞어 5 % 가 넘을 때까지 떼어 냄 | 학습 주행을 시드 0 으로 섞어 5 % 가 넘을 때까지 떼어 냄 |
| 분할 목록 | BTS 저장소 `train_test_inputs/` (커밋 5e3406b, SHA256 고정) | 같음 |

- 테스트 RGB 는 BTS 추출 스크립트와 똑같이 만든다: 가장자리 7 px 를 검게 칠하고 cv2 jpg(기본 품질)로 저장. 서버에서 다시 만들면 jpg 바이트가 달라질 수 있어 만든 파일을 그대로 보낸다.
- 검증용으로 떼어 낸 장은 학습 jsonl 에서만 빠진다. 테스트 세트는 에폭을 고를 때 쓰지 않는다.
- NYU 표준 학습 목록에는 겹침이 하나 있다: 테스트 11 장이 나온 `bookstore_0001` 과 같은 서점의 다른 녹화 구간(d–j, 1,383 장)이 학습 목록에 있다. 표의 다른 모델도 같은 목록을 썼다고 보고 그대로 두며, 결과에 각주로 적는다.
- KITTI depth selection 검증 세트(1,000 장)는 쓰지 않는다. 주행 13 개 중 7 개가 Eigen 테스트 주행이다.

## 2. 평가 (BTS 공식 평가 코드 `pytorch/bts_eval.py` 와 같은 규칙)

| | NYU | KITTI |
|:--|:--|:--|
| 모델 입력 | 640×480 전체 | KB crop: 아래 352 줄, 가운데 1216 칸. 예측을 원래 크기의 같은 자리에 붙인다 |
| 채점 픽셀 | 0.001 m < 정답 < 10 m, Eigen crop `[45:471, 41:601]` | 0.001 m < 정답 < 80 m, Garg crop (행 0.4081–0.9919 H, 열 0.0359–0.9641 W) |
| 예측 처리 | 정답 크기로 bilinear → [0.001, 10] 으로 자름 (inf → 최대, nan → 최소) | 같음, [0.001, 80] |
| 지표 | δ1·δ2·δ3, AbsRel, RMS, Log10 | δ1·δ2·δ3, AbsRel, RMS, RMSlog |

- 지표는 이미지마다 계산한 뒤 평균한다 (BTS·UniDepth 코드와 같음, 표의 방식). pooled(전체 픽셀 한꺼번에)는 보조로 함께 낸다.
- 식: δk = max(예측/정답, 정답/예측) < 1.25^k 인 비율, AbsRel = 평균 |예측 − 정답| / 정답, RMS = √평균(예측 − 정답)²,
  Log10 = 평균 |log10 예측 − log10 정답|, RMSlog = √평균(ln 예측 − ln 정답)². 표기: δ 는 %, AbsRel 은 ×100.
- 테스트 때 좌우 뒤집기 평균(TTA)은 쓰지 않는다 (UniDepthV2 도 평가에서 쓰지 않음).
- DepthVLM 공식 `eval/eval.py` 의 채점은 쓰지 않는다 (정답을 초점 1000 크기로 바꿔 채점해 표준과 다르다). 추론만 공식 경로를 따른다:
  입력을 원본 × 1000 / fx 크기로 bilinear, 공식 프롬프트·채팅 틀, `process_vision_info`, forward 한 번의 `depth_pred`, bf16.
- 예측은 float32 npy 로 저장해 채점한다. 결과 zip 에는 float16 사본을 넣는다 (분석용).

### 채점 검증 (학습 전)
- NYU: Metric3Dv2 ViT-L 을 Metric3D 공식 평가 전처리로 돌려 논문 zero-shot 값과 비교 — 통과 (NOTES F-5).
- KITTI: 같은 모델로 Metric3D 논문 zero-shot 값과 비교 — 통과 (NOTES F-4). Metric3D 의 KITTI 평가 코드는 표준과 셋이 다르다:
  Eigen crop(0.3324–0.9135 H, 입력은 열 43:1197), 예측을 자르지 않음, 논문 표의 'RMS_log' 칸 = silog. 검증은 이 규칙(`--protocol metric3d`)으로 하고, 주 결과는 표준(Garg crop).

## 3. 학습 (공식 `train/train-stage2.sh` 기준)

| 항목 | 공식 2 단계 | 이 실험 |
|:--|:--|:--|
| 시작 가중치 | 1 단계 결과 | 공개 DepthVLM-4B (HF 리비전 2b2d02f) |
| 학습하는 부분 | LLM + DPT 깊이 헤드 + lm_head (비전 인코더 고정) | 같음 |
| 손실 | SILog(λ 0.5, 이미지마다) + 답 문장("OK, I will estimate the depth map of this image.") LM 손실, 가중치 1.0 | 같음 |
| 학습률 | 2e-5, cosine, warmup 5 %, grad clip 1.0 | 같음 |
| 전체 배치 | 640 (GPU 80 장 × 8) | 64 (GPU 1 장, 장치 배치 8 × 누적 8) |
| 에폭 | 1 | 3, 에폭마다 가중치 저장 → 검증 세트 AbsRel 최소 에폭 |
| 정밀도 | FSDP + bf16 혼합 (accelerate 가 FSDP 가중치를 fp32 로 올림) | fp32 로 불러 bf16 autocast (FSDP 없이 같은 구성) |
| 데이터 증강 | 없음 | 없음 |
| 그 밖 | max_seq_length 4096, gradient checkpointing, flash_attention_2, seed 42 | 같음 + dataloader 작업자 4 |

- 데이터셋 깊이 범위: NYU 는 공식 표의 `nyuv2` (0.005–10 m), KITTI 는 공식 표에 없어 0.001–80 m 를 더한다 (`train/train_ft.py`).
- KITTI 학습 이미지는 KB crop 으로 미리 잘라 저장한다 (공식 학습 코드에 자르기가 없다). 입력 크기 = 1216×352 × 1000 / fx.
- 왜 FSDP 를 빼는가: GPU 1 장에서는 FSDP 가 NO_SHARD 로 바뀌고, accelerate 의 fp32 승격과 충돌해 첫 걸음에서 멈춘다 (NOTES F-3).
- 왜 TRL 0.19.1 인가: 공식 스크립트의 `--torch_dtype`·`--max_seq_length` 를 둘 다 받는 마지막 판 (NOTES F-2). transformers 는 체크포인트의 5.2.0.

## 4. 결과 표 형식

표 V·VI 의 열 그대로: NYU = δ1·δ2·δ3 (%), A.Rel (×100), RMS, Log10 / KITTI = δ1·δ2·δ3 (%), A.Rel (×100), RMS, RMSlog.
DepthVLM 은 두 줄: 파인튜닝 전 (zero-shot) / 파인튜닝 후. 각주: 모델 크기 (4B, 표의 다른 모델은 0.5B 이하), 학습 설정 (3 절), NYU 학습 목록의 bookstore 겹침,
표 VI 의 Metric3Dv2 행은 Metric3D 코드 기준(Eigen crop, 예측 안 자름, RMSlog 칸 = SILog)일 가능성이 큼 (NOTES F-4).
