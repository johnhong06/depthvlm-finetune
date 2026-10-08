# depthvlm-finetune — 진행 기록

기록 규칙: 기능 하나·실험 하나마다 체크리스트와 실행 로그를 갱신한다. 결정은 "결정 기록"(D-번호), 확인이 필요한 발견은 F-번호.
시작 배경: vlm-depth-rmse 에서 사용자가 "DepthVLM 을 NYU·KITTI 에 파인튜닝해 UniDepthV2 표 V·VI 와 같은 지표로 비교할 수 있나"를 물었고,
계획(데이터 받기 → 채점 맞추기 → H200 실험용 저장소·zip)을 승인했다 (2026-10-08).

## 체크리스트

### 데이터 (로컬)
- [x] 분할 목록 4 개 (BTS 커밋 5e3406b, SHA256 고정) — `prep/fetch_splits.sh`
- [x] NYU 학습 BTS sync.zip (5.9 GiB, rclone — F-1) → 24,231 쌍 꺼내기·검증, 테스트 654 장 BTS 방식 재현 (`prep/build_nyu.py`, 실행 로그 11:20)
- [x] NYU 학습 jsonl 22,923 장 + 검증 장면 13 개 1,308 장 (400 장 고르게) — `prep/make_jsonl.py`
- [x] KITTI raw 주행 61 개 (로컬 30 + 새로 31 = 55.7 GB) + annotated depth (14.2 GB) — 크기 전부 서버 Content-Length 와 일치
- [x] KITTI 꺼내기·검증 (`prep/build_kitti.py`) → jsonl 21,922 장 (실행 로그 12:11)
- [x] zip 묶음 (`prep/pack.py`) → `~/data/h200_staging/dvft/` 17 개 25.55 GiB + 관리자 안내문 `README_ADMIN.txt` — `h200/unpack.py` 로 전부 풀어 원본과 바이트 대조 일치, 드라이브식 중첩 zip 도 확인 (실행 로그 12:28)

### 채점 (로컬)
- [x] `eval/std_eval.py` (BTS 규칙) + `eval/predict.py` (DepthVLM 공식 추론 경로, Metric3Dv2 검증용)
- [x] NYU 검증: Metric3Dv2 = 논문 zero-shot 값 (F-5)
- [x] KITTI 검증: Metric3Dv2 를 Metric3D 공식 평가 코드 규칙으로 → 논문 zero-shot 6 개 값 모두 일치 (F-4)
- [x] DepthVLM-4B zero-shot NYU 654 장 (로컬, sdpa) — F-6
- [x] DepthVLM-4B zero-shot KITTI 652 장 (로컬, sdpa) — F-8

### 학습 코드
- [x] 환경 `envs/depthvlm.txt` (transformers 5.2.0, trl 0.19.1 — F-2), 로컬 `~/venv/depthvlm_train`
- [x] `train/train_ft.py`·`train_ft.sh` (공식 2 단계 인자) — 로컬 기능 점검 통과 (깊이 헤드만 4 걸음, F-3·F-7)
- [x] `eval/select_epoch.py`, `run.sh` (env·smoke·zeroshot·nyu·kitti·all), `h200/unpack.py` — KITTI 검증 채점·에폭 고르기(같으면 나중 에폭)·KITTI 학습 데이터 읽기 로컬 확인 (12:31)
- [x] pyflakes·shellcheck 깨끗, 커밋할 파일에 개인 경로·토큰 없음
- [x] 커밋 (푸시는 사용자)

### H200 (사용자 제출)
- [ ] (사용자) 저장소 푸시, 드라이브에 `~/data/h200_staging/dvft/` 올리기(요청하면 rclone 으로 올림) → 관리자에게 전달 → `/app/data` 아래
- [ ] `bash run.sh env` → `bash run.sh smoke` (속도·메모리 확인 후 본 실험 시간 추정)
- [ ] `bash run.sh nyu`, `bash run.sh kitti` → 결과 표 (README·docs)

## 결정 기록

### D-1 (2026-10-08) 평가 = BTS 공식 평가 규칙, 이미지별 평균
- 표 V·VI 의 모델 대부분이 BTS 평가 코드 계열을 쓰고, UniDepth 코드도 같은 규칙(이미지별 지표 → 평균, NYU Eigen crop, KITTI Garg crop)이다.
- NYU 정답 = labeled.mat 의 rawDepths (BTS 추출 스크립트가 `h5_file['rawDepths']` 를 쓴다). vlm-depth-rmse 에서 Metric3Dv2 재현이 rawDepths 로만 맞았다 (0.253 vs 0.251, depths 는 0.345).
- KITTI 정답 = annotated depth, 정답 있는 652 장. 상한 80 m, 하한 0.001 m (UniDepth 는 0.05 m 지만 KITTI 정답에 그보다 가까운 값이 사실상 없다).
- 다른 선택지: pooled 주 결과(vlm-depth-rmse 방식) — 표 수치와 비교할 수 없어 보조로만.

### D-2 (2026-10-08) 학습 = 공식 2 단계 설정 + 꼭 필요한 변경만
- 바꾼 것: 시작 가중치(공개 DepthVLM-4B), GPU 1 장, 전체 배치 640 → 64 (2 만 장에 640 이면 에폭당 38 걸음), 에폭 1 → 3, 에폭마다 가중치만 저장,
  dataloader 작업자 4, FSDP → fp32 로 불러 bf16 autocast (F-3), torchrun 없이 실행 (F-3). 손실·학습률·warmup·고정 부분·증강 없음은 공식 그대로.
- 다른 선택지: 깊이 헤드만 (1 단계식) — 싸지만 성능이 낮을 가능성이 커 주 실험에서 뺌. LoRA — 공식 방식이 아님.

### D-3 (2026-10-08) 에폭 = 검증 세트로 고른다, 테스트는 한 번
- 학습 장면(NYU)·주행(KITTI)을 시드 0 으로 섞어 5 % 가 넘을 때까지 떼어 내 검증 세트로 둔다 (학습 jsonl 에서 빠짐). 검증 400 장에서 이미지별 평균 AbsRel 이 가장 낮은 에폭, 같으면 나중 에폭.
- 다른 선택지: 학습 세트 전부 + 고정 에폭 (학습 데이터 5 % 더 씀, 고르기 없음) / BTS 처럼 학습 중 테스트 세트로 고르기 (기각 — 테스트 누출).

### D-4 (2026-10-08) KITTI 입력 = KB crop
- BTS·UniDepth 표준. 학습 이미지는 미리 잘라 무손실 png 로 저장 (공식 학습 코드에 자르기가 없음), 테스트·검증은 원본을 두고 예측 때 같은 상자로 잘라 붙인다.

### D-5 (2026-10-08) H200 전달 = zip 묶음 (사용자 요청), 서버에 있는 것은 다시 보내지 않음
- 무압축 zip, 조각 2 GB 이하(담는 파일 합 1,900 MiB 이하) + dvft_SHA256SUMS.txt. 관리자가 드라이브 폴더를 통째로 받아 zip 안에 zip 이 들어 있어도 `h200/unpack.py` 가 찾는다.
- DepthVLM-4B 가중치는 보내지 않는다 — vlm-depth-rmse Track B 처럼 서버가 HF 고정 리비전(2b2d02f)으로 받는다 (서버에서 확인된 방식). 드라이브의 vdr_depthvlm4b 팩도 같은 가중치.
- NYU 테스트는 서버의 vdr_nyu_official(labeled.mat 에서 뽑은 png·npy)이 아니라 BTS 형식(jpg)으로 새로 보낸다 — jpg 바이트가 로컬 검증본과 같아야 해서 (0.2 GB).

### D-6 (2026-10-08) NYU 학습 목록의 bookstore 겹침은 그대로
- 테스트 11 장(`bookstore_0001`)과 같은 서점의 다른 녹화 구간(d–j, 1,383 장)이 BTS 학습 목록에 있다. 표의 다른 모델도 같은 목록이라 조건을 맞추려고 그대로 두고 각주로 적는다.

## 발견

- **F-1 (10-08) BTS sync.zip 공개 링크를 gdown 이 못 받음** ("Cannot retrieve the public link", 조회 한도). 인증된 rclone(`gdrive:`)의 `backend copyid` 로 같은 파일 ID 를 받음 (5.9 GiB, 4 분).
- **F-2 (10-08) TRL 판**: 공식 train-stage*.sh 는 `--torch_dtype`(ModelConfig)·`--max_seq_length`(SFTConfig) 를 넘긴다. 둘 다 받는 판은 0.19.1 이하
  (0.20 부터 max_seq_length 없음, 0.27 부터 torch_dtype 없음 — 휠 9 개 대조). 0.19.1 이 transformers 5.2.0 과 함께 import·인자 해석 정상 → 고정.
- **F-3 (10-08) GPU 1 장 실행 문제 둘**: ① FSDP full_shard 가 1 장에서 NO_SHARD 로 바뀌고 accelerate 의 fp32 승격과 충돌 —
  `RuntimeError: Cannot writeback when the parameter shape changes` (비전 patch_embed). 공식(80 장)에서는 FULL_SHARD 라 생기지 않는 문제.
  → fp32 로 불러 bf16 autocast (공식과 같은 'fp32 주 가중치 + bf16 계산'). bf16 로만 불러 FSDP 없이 돌리면 주 가중치가 bf16 이 되어 공식과 달라진다.
  ② torchrun 1 프로세스에서 NCCL `unhandled cuda error` → torchrun 없이 바로 실행.
- **F-4 (10-08) Metric3D 의 KITTI 평가 규칙은 표준과 셋이 다르다 — 그 규칙으로 재면 논문 zero-shot 값 6 개가 모두 맞는다 (KITTI 채점 검증 통과).**
  ① Eigen crop (`kitti_dataset.py` process_depth: 0.3324–0.9135 H), 표준(BTS·UniDepth)은 Garg crop. 입력은 열 43:1197 만.
  ② 예측을 자르지 않는다 (`do_test.py` 가 clip_range 를 넘기지 않음). 정답은 0.1–80 m 밖만 무효 (clip_depth).
  ③ 논문 KITTI 표의 'RMS_log' 칸 = 이 코드의 silog (√(평균 d² − (평균 d)²)), 진짜 RMSlog 가 아니다.
  Metric3Dv2 ViT-L, 652 장, 이 규칙: δ1 97.4 / δ2 99.5 / δ3 99.9 / A.Rel 5.19 / RMS 2.511 / SILog 7.37 (RMSlog 0.082)
  — 논문(arXiv 2404.15506) ZS 행 0.974 / 0.995 / 0.999 / 0.052 / 2.511 / 0.074. (Metric3D README 의 주석 처리된 표는 δ1 을 0.985 로 적어 논문과 다르다)
  같은 예측을 표준 규칙으로: 97.9 / 99.6 / 99.9 / 4.82 / 2.184 / RMSlog 0.075. 예측을 자르면(처음 구현) RMS 2.395 로 어긋났다.
  → 표 VI 의 Metric3Dv2 행(Metric3D 저장소의 파인튜닝 수치와 같음)은 Eigen crop·자르지 않은 예측·RMSlog 칸 = SILog 일 가능성이 크다 — 결과 각주.
- **F-5 (10-08) NYU 채점 검증 통과**: Metric3Dv2 ViT-L (Metric3D 공식 평가 전처리) 를 새 테스트 세트(BTS 형식 654 장)로,
  Metric3D 규칙(예측 안 자름, 0.1–10 m): δ1 97.5 / δ2 99.4 / δ3 99.8 / A.Rel 6.28 / RMS 0.251 / Log10 0.027 — 논문 zero-shot 0.975 / 0.994 / 0.998 / 0.063 / 0.251 / 0.028.
  표준(BTS) 규칙: 97.6 / 99.4 / 99.8 / 6.27 / 0.246 / 0.027.
- **F-6 (10-08) DepthVLM-4B zero-shot NYU (표준 규칙, 로컬 sdpa, 654 장)**: δ1 93.4 / δ2 99.0 / δ3 99.8 / A.Rel 8.93 / RMS 0.353 / Log10 0.039 (pooled RMS 0.409).
  최종 표에는 H200(flash_attention_2) 값을 쓴다.
- **F-8 (10-08) DepthVLM-4B zero-shot KITTI (표준 규칙, 로컬 sdpa, 652 장)**: δ1 91.5 / δ2 98.5 / δ3 99.7 / A.Rel 8.88 / RMS 3.461 / RMSlog 0.130 (SILog 12.05, pooled RMS 3.631).
- **F-7 (10-08) 로컬 학습 기능 점검**: 깊이 헤드만(1 단계식, lr 3.5e-4) NYU 16 장 × 1 에폭(4 걸음) — 데이터 읽기(이미지 토큰 1,102 = 38×29, 정답 925×1233),
  손실(SILog 0.44), 체크포인트 저장(fp32 19.4 GB)·불러오기·검증 예측·채점까지 정상. 같은 검증 20 장에서 원래 모델 δ1 90.2 / A.Rel 9.09 (예측 코드 정상),
  4 걸음 모델은 44.2 / 33.0 (높은 lr 로 헤드만 흔든 기능 점검이라 의미 없음). 2 단계(전체) 메모리·속도는 로컬 32 GB 로 못 재 H200 스모크에서 잰다.

## 실행 로그

- 2026-10-08 10:55 분할 목록 4 개 받기·SHA256 고정. KITTI 받기 시작 (기존 38 개 중 Eigen 에 쓰는 30 개는 크기 대조로 확인, 새로 31 개).
- 2026-10-08 11:05 NYU sync.zip 받음 (rclone, 6,353,519,817 바이트): 항목 73,077 (jpg 36,396 + png 36,396), BTS 목록 24,231 쌍 모두 있음, 초점 518.8579 하나.
- 2026-10-08 11:20 `prep/build_nyu.py` (1 분 52 초): 학습 24,231 쌍·폴더 284 개, 유효 픽셀 67.6 %, 깊이 0.713–10.0 m / 테스트 654 장 = BTS 목록과 정확히 일치,
  유효 69.3 %, 깊이 ≤ 10.0 m, rawDepths npy(vlm-depth-rmse nyuv2_official)와 최대 차 0.001 m (mm 자르기). 장면 겹침 bookstore_0001 (D-6). cv2 5.0.0.
- 2026-10-08 11:22 `prep/make_jsonl.py --ds nyu`: 학습 22,923 장 (canonical 1233×925 하나), 검증 장면 13 개 1,308 장 → 400 장. jsonl sha256 4687fcd95f5a4ca5.
- 2026-10-08 11:25 NYU 채점 검증 (Metric3Dv2, 0.56 s/장) → F-5.
- 2026-10-08 11:28–11:40 학습 환경·기능 점검 → F-2·F-3·F-7.
- 2026-10-08 11:45 DepthVLM-4B zero-shot NYU 654 장 (로컬) → F-6.
- 2026-10-08 11:51 KITTI 마지막 주행·annotated depth: 한 줄 받기가 느려(약 5 MB/s) annotated depth 는 4 갈래 범위 요청으로 받아 이어 붙임 (14,241,086,697 바이트 = 서버 크기),
  fetch_kitti.sh 재실행으로 주행 61 개 + annotated 모두 크기 일치 확인. (fetch_kitti.sh 만으로도 받을 수 있다 — 느릴 뿐)
- 2026-10-08 12:11 `prep/build_kitti.py` (작업자 8): 학습 21,922 장(주행 27) / 검증 주행 6 개 1,236 장 → 400 장 / 테스트 652 장(주행 28), 처리 23,810 장.
  주행 겹침 학습–테스트·검증–테스트 없음, 목록 초점 = calib P_rect_02 (어긋남 0), 이미지 크기 5 종(1242×375 등), 정답 유효 15–16 %, 정답 최대 90.4 m (채점은 80 m 까지).
  `make_jsonl.py --ds kitti`: 21,922 장, canonical 4 종(1685×488 / 1692×490 / 1720×498 / 1693×490), sha256 a8f3b2a26480ca60 (두 번 돌려 같음).
- 2026-10-08 12:20 KITTI 채점 검증 (Metric3Dv2) → F-4. 검증 규칙을 Metric3D 코드와 똑같이 고침 (예측 안 자름) → NYU 도 RMS 0.251 로 논문과 같아짐 (F-5).
- 2026-10-08 12:25 DepthVLM-4B zero-shot KITTI 652 장 (로컬) → F-8.
- 2026-10-08 12:23 `prep/pack.py`: meta 17 파일 / nyu_test 1,308 / nyu_train 48,462 (zip 3) / kitti_eval 3,776 / kitti_train 43,844 (zip 11) = zip 17 개, 25.55 GiB.
- 2026-10-08 12:28 `h200/unpack.py` 로 17 개 전부 풀기 (24 초, SHA256 통과) → `diff -rq` 로 원본과 97,407 파일 바이트 일치. 드라이브 폴더 zip 처럼 감싼 경우(zip 안의 zip)도 풀림 확인.
- 2026-10-08 12:31 로컬 점검: KITTI 검증 5 장 예측·채점, select_epoch (가짜 결과 3 개 → 동점이면 나중 에폭), KITTI 학습 데이터 읽기 (깊이 헤드만 2 걸음:
  깊이 범위 0.001–80 m, 입력 1685×488 = 이미지 토큰 780, 정답 유효 7–20 %, 손실 0.147).
