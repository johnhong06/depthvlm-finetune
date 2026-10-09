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
- [x] 드라이브 `gdrive:h200_dvft` 업로드 (20 파일 = zip 17 + SHA256SUMS + manifest + README_ADMIN, 25.55 GiB) — 드라이브 SHA256 20/20·MD5 일치 (실행 로그 13:25)
- [ ] (사용자) 저장소 푸시 → 관리자에게 드라이브 `h200_dvft` 전달 → `/app/data` 아래
- [x] `bash run.sh env` 통과 (이슈 johnhong06_888, commit b925eee, 6 분) — 실행 로그 10-08 14:16
- [x] `bash run.sh smoke` 통과 (이슈 johnhong06_889, commit 8c1cbb8, 13 분) — 실행 로그 10-08 14:24. 본 실험 추정 nyu ≈ 5.5 h · kitti ≈ 4.8 h · all ≈ 10 h. NYU 스모크 학습 뒤 하락 → F-9
- [x] (사용자 결정 → 점검 없이 바로 `nyu` 제출) 바로 `all` vs 먼저 운영 설정 점검 `bash run.sh nyu EPOCHS=1 N=4000` (약 1 h, 검증 400 장으로만 판단 — 규칙 3) — F-9
- [x] `bash run.sh nyu` (이슈 johnhong06_890, commit 8c1cbb8, 장치 배치 8, 3 h 53 m) — F-12. 결과 zip `dvft_nyu_1008_0613.zip` (322M) 은 아직 못 받음 (콘솔 로그로 기록)
- [ ] `bash run.sh kitti LBS=4` (D-7: 입력 크기 4 종, 스모크 99.5 %) → 결과 표 (README·docs)

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

### D-7 (2026-10-08) 본 실험 장치 배치 — NYU 는 8 로 제출됨(사용자), KITTI 는 LBS = 4 권장 (전체 배치 64 그대로, 누적 16)
- 스모크(LBS 8)의 GPU 최대 사용이 NYU 139,681 MiB·KITTI 143,087 MiB (H200 NVL 143,771 MiB 의 97 %·99.5 %) — 4–5 시간 돌리다 메모리 부족으로 멈출 위험.
  KITTI 스모크 400 장에 가장 큰 입력(1720×498)이 150 장 들어 있어 최대 크기는 지나갔지만 여유가 1 GB 미만이다.
- 전체 배치가 같아 기울기(사진별 SILog 평균, 답 문장 길이 같음)는 같다. 시간은 5–10 % 늘 것으로 추정. 다른 선택지: LBS 8 그대로 두고 메모리 부족이 나면 LBS=4 (F-9 실행 로그의 의견),
  NYU 만 LBS 8 (입력 크기가 하나라 안정적일 수 있음). 4–5 시간 뒤에 멈추면 그만큼 잃으므로 처음부터 4 를 제안 — 두 실험 조건도 같아진다.

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
- **F-9 (10-08) 스모크에서 NYU 만 학습 뒤 나빠짐 (KITTI 는 좋아짐) — 스모크 설정 탓일 가능성이 크지만 본 실험 전에 확인 필요**
  · NYU 원래 → 스모크 학습(400 장 × 1 에폭): 검증 20 장 δ1 90.2 → 86.4, A.Rel 9.08 → 12.45, SILog 9.87 → 13.17 / 테스트 20 장 91.3 → 71.7, 10.02 → 16.71, 11.11 → 16.00.
    KITTI: 검증 20 장 δ1 88.1 → 88.1, A.Rel 10.11 → 9.50, SILog 14.99 → 13.77 / 테스트 20 장 97.4 → 98.4, 8.01 → 6.21, 8.00 → 7.47.
  · 배제한 것: 좁은 장면 과적합(앞 400 장 = 장면 여러 개, 가장 많은 것이 dining_room_0031 11 장), 학습 GT 배율 문제(원래 모델이 학습 출처 검증 20 장에서 δ1 90.2 = 테스트와 비슷),
    서버 경로 차이(원래 모델 검증 20 장 H200 90.2 / 9.08 = 로컬 F-7 90.2 / 9.09).
  · 스모크 설정이 본 실험과 다름: 장치 배치 8 × 누적 1 = 전체 배치 8 (본 실험 64, 공식 640) 에 공식 lr 2e-5, warmup 5 % = 2.5 걸음, 50 걸음 전체 모델(4.45 B) 갱신.
    공식보다 80 배 작은 배치에 같은 lr 이라 거칠다. SILog 도 나빠졌으므로 배율만의 문제는 아니다. 이미지별 원인은 스모크 zip(60K)의 이미지별 csv 가 있어야 본다.
  · 본 실험 판단은 검증 세트로만 한다(규칙 3). 테스트 20 장 하락폭은 설정 고르기에 쓰지 않는다.
  · 만약 운영 설정(배치 64)에서도 검증이 나빠지면 lr 을 바꾸는 것은 D-2 변경 → 사용자 결정 (후보: 배치 비례 2e-6, 제곱근 비례 6.3e-6).

- **F-10 (10-09) 멈춤 감지가 빠져 있었다**: vlm-depth-rmse·depthlm-improve 의 run.sh 에는 '40 분 동안 진행이 없으면 끄고 결과 zip 을 남기는' 장치가 있었는데
  이 저장소 run.sh 에 옮기지 않았다 → `bash run.sh kitti LBS=4` 가 사용자 확인 시점 14 시간째 진행 중 (예상 5–5.5 h). 학습 중 10 분마다 찍는 `[진행]` 줄은 학습이 멈춰도
  계속 찍히므로 그 안의 loss·epoch 가 바뀌는지로 판단해야 한다. → run.sh 에 run_train 추가: 학습을 setsid 로 따로 띄워 로그(진행 막대가 걸음마다 늘림)가
  STALL_MIN(기본 40)분 동안 그대로면 프로세스 묶음을 끄고 다음 단계(채점·zip)로 넘어간다. 예측·채점은 한 번에 STEP_MIN(기본 90)분 제한 (timeout).
  로컬 가짜 명령 시험: 멈춤 → 종료 124·자식까지 정리, 정상 → 0, 실패 → 원래 종료 코드 3.

- **F-11 (10-09) kitti 작업 14 시간 — 우리 코드 흐름으로는 설명되지 않는다** (서버 로그를 받기 전 점검)
  · 예상: `kitti LBS=4` 작업 전체 5–5.5 h (스모크 속도 0.235 s/장 × 65,766 장 ≈ 4.3 h + LBS 4 로 5–10 % + 준비·채점 약 1 h).
  · 누적 손실 정규화 정상: DepthVLM 의 Qwen3VLForConditionalGeneration 에 `accepts_loss_kwargs = False` → transformers 5.2 Trainer 가 손실을 누적 걸음 수로 나눈다
    (TRL 0.19.1 은 이 값을 바꾸지 않음). 누적 8·16 이어도 기울기는 64 장 평균, 기록되는 손실도 정상.
  · 로컬 전체 흐름 시험 (`STAGE=1 bash run.sh kitti N=16 EPOCHS=2 LBS=2 GBS=4`, 미리 만든 환경·데이터·가중치 링크): 603 초에 원래 모델 테스트·검증 → 학습 →
    체크포인트 2 개 검증 → 에폭 고르기 → 테스트 → zip(597 MB) 까지 끝나고 바로 종료. 원래 모델 테스트 = F-8 과 같은 값, 체크포인트는 끝에 지워짐.
    (첫 시도는 로컬 data/splits 링크가 없어 테스트 예측만 실패 — 서버는 meta zip 이 data/splits 에 풀어 스모크에서도 썼다)
  · 옛 run.sh(지금 서버에서 도는 판)의 끝: 진행 줄 루프를 끈 뒤에도 그 sleep 이 출력 파이프를 잡아 종료가 최대 PROGRESS_SEC(10 분) 늦다 (시험: 5 초 작업이 60 초 뒤 종료).
    14 h 의 원인은 아님. 새 run.sh 는 진행 확인을 학습 감시 안에서 해 해당 없음.
  · 남는 후보: 늦게 시작(대기) / 준비 단계(pip 설치·HF 받기 — 출력이 없어 멈춰도 조용) / 학습 중 멈춤([진행] 줄의 loss·epoch 가 그대로) / 끝났는데 상태 미반영.
    콘솔 로그의 [setup] 시각(UTC)과 마지막 줄로 가린다. 옛 run.sh 는 어디서 멈춰도 스스로 끝나지 않는다 (F-10).
  · 조치: 준비 단계에도 시간 제한 (패키지 설치 60 분, flash-attn 30 분, 가중치 받기 60 분, zip 풀기 90 분) — 로컬 `run.sh env` 로 확인.

- **F-12 (10-09) NYU 파인튜닝 결과 (H200, 이슈 890, 표준 규칙 이미지별 평균)**
  · 테스트 654 장: 파인튜닝 전 δ1 93.4 / δ2 99.0 / δ3 99.8 / A.Rel 8.92 / RMS 0.353 / Log10 0.039 / SILog 10.28 (pooled RMS 0.410)
    → 파인튜닝 후(3 에폭 체크포인트) 94.3 / 99.3 / 99.9 / 8.39 / 0.302 / 0.036 / SILog 9.17 (pooled RMS 0.333). 로컬 sdpa zero-shot(F-6)과 A.Rel 만 0.01 다름.
  · 검증 400 장 (에폭 고르기, D-3): 파인튜닝 전 91.8 / A.Rel 9.48 / RMS 0.383 → 1 에폭 86.3 / 11.73 / 0.469 (나빠짐) → 2 에폭 93.7 / 9.01 / 0.338 → 3 에폭 93.9 / 8.86 / 0.338 (고름).
    1 에폭(학습률이 가장 높은 구간)에서 나빠졌다가 학습률이 줄며 회복·개선 — 스모크의 하락(F-9)도 이 초반 구간이었던 것으로 보인다.
  · 표 V 에서: 학습만 한 4 개(BTS·AdaBins·NeWCRFs·iDisc)는 모든 지표에서 앞서고, 사전학습 후 파인튜닝한 4 개 중 ZoeDepth(95.2 / 7.70 / 0.278) 보다는 δ3 만 앞선다.
    Metric3Dv2·DAv2·UniDepthV2(δ1 98.4–98.9, A.Rel 4.68–5.60, RMS 0.180–0.206) 와는 차이가 크다. δ3 99.9 는 새 2 위 (README 표 밑줄 다시 매김).
  · 학습: 3 에폭 1,077 걸음, train_runtime 3:22:36 (5.657 장/s — 스모크 3.85 장/s 보다 빠름: 누적 8 이라 최적화 걸음이 1/8), 손실 0.123 → 0.049, grad_norm 0.03–0.8.
    GPU 메모리 89–141 GB (08:46 에 141,165 MiB — 장치 배치 8 이 한계 근처였음, D-7), 07:56 의 GPU 5 % 는 1 에폭 체크포인트 저장 중.
  · 작업 시간: 준비 23 분(환경·가중치 2 분·zip 풀기) + zero-shot 테스트·검증 약 4 분 + 학습 3 h 23 m + 검증 3 개·테스트 약 8 분 = 3 h 53 m.

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
- 2026-10-08 12:43–13:23 `rclone copy ~/data/h200_staging/dvft gdrive:h200_dvft` (전송 4 개, 청크 128M) — 종료 0. 속도가 3–60 MiB/s 로 출렁였고
  rclone 이 일부를 다시 보내 전송 합계가 38.5 GiB (파일 합 25.55 GiB) — 오류 줄은 없음. 13:25 `rclone hashsum sha256` 20/20 = 로컬, `rclone check` 0 differences / 20 matching.
- 2026-10-08 14:10–14:16 KST (서버 05:10–05:16 UTC) H200 `bash run.sh env` (이슈 johnhong06_888, commit b925eee) — 통과. conda 없음 → uv 로 환경 생성(예상대로, vlm-depth-rmse F-6),
  torch 2.7.1+cu128 · transformers 5.2.0 · trl 0.19.1, H200 NVL 140 GiB, flash_attention_2. DepthVLM-4B HF 리비전 2b2d02f 받기 2 분, 불러오기 정상.
  WORK = /app/scratch/dvft_work (여유 455 GB) — /app/data 는 쓰기 불가라 작업 사이에 남지 않는 곳 → 작업마다 환경·zip 풀기를 다시 한다 (작업당 10–15 분 더).
  가중치 크기 표시 '12K' 는 HF 캐시의 심볼릭 링크만 센 것 → run.sh 의 du 를 -L 로 고침 (표시만, 동작 무관).
- 2026-10-08 06:13–10:06 UTC (15:13–19:06 KST) H200 `bash run.sh nyu` (이슈 johnhong06_890, commit 8c1cbb8, 장치 배치 8) — 정상 종료 → F-12. 사용자가 콘솔 로그를 붙여 줌 (10-09).
  README NYU 표: DepthVLM zero-shot 행을 H200 값(A.Rel 8.92)으로, 파인튜닝 행 채움, δ3 2 위(밑줄)를 DepthVLM 99.9 로 다시 매김 (zero-shot 행은 순위에서 뺌).
- 2026-10-09 14:00 README 표에 DepthVLM-4B zero-shot 행 추가 (로컬 sdpa 측정 F-6·F-8, ‡ 각주: H200 값이 나오면 바꾼다). 파인튜닝 행은 아직 비움 (서버 결과 zip 없음).
- 2026-10-08 14:00 README 를 한국어로 교체 (사용자 지시: vlm-depth-rmse README 와 같은 형식). 결과 표 = UniDepthV2 표 V·VI 수치·굵게/밑줄 그대로 + 모델명 옆 학회·연도
  (arXiv 원문 comments·journal-ref 로 확인: BTS arXiv 2019, AdaBins CVPR 2021, NeWCRFs CVPR 2022, iDisc CVPR 2023, ZoeDepth arXiv 2023, Metric3Dv2 TPAMI 2024,
  Depth Anything V2 NeurIPS 2024, UniDepthV2 arXiv 2025, DepthVLM arXiv 2026), DepthVLM-4B 칸은 비움. humanize-korean 윤문(light, 변경률 0.2 %, 게이트 통과, 표·헤딩 바이트 동일).
- 2026-10-08 12:31 로컬 점검: KITTI 검증 5 장 예측·채점, select_epoch (가짜 결과 3 개 → 동점이면 나중 에폭), KITTI 학습 데이터 읽기 (깊이 헤드만 2 걸음:
  깊이 범위 0.001–80 m, 입력 1685×488 = 이미지 토큰 780, 정답 유효 7–20 %, 손실 0.147).
- 2026-10-08 14:24–14:38 KST (서버 05:24–05:38 UTC) H200 `bash run.sh smoke` (이슈 johnhong06_889, commit 8c1cbb8) — 정상 종료, 결과 zip 60K. 사용자가 콘솔 로그를 붙여 줌.
  · 준비 6 분: conda 없음 → uv (예상대로), torch 2.7.1+cu128 · transformers 5.2.0 · trl 0.19.1, flash_attention_2, 가중치 HF 2b2d02f 받기 1 분 50 초 (9.1G), zip 17 개 SHA256 통과.
    데이터 요약 = 로컬과 같음: NYU 학습 24,231 쌍·초점 518.8579·테스트 654 장 목록 일치·유효 0.6926, KITTI 학습 21,922·검증 400·테스트 652, jsonl sha256 nyu 4687fcd95f5a4ca5 · kitti a8f3b2a26480ca60.
  · 학습(400 장 × 1 에폭, 장치 배치 8, 50 걸음, 학습 파라미터 4,446,570,177 / 4,861,917,889): NYU 103.8 초(3.85 장/s, 손실 0.215 = 깊이 0.2115 + LM 4.5e-6),
    KITTI 94.0 초(4.26 장/s, 손실 0.184). GPU 최대(nvidia-smi 15 초 간격, 할당기 캐시 포함) NYU 139,681 MiB · KITTI 143,087 MiB — 140 GiB 의 거의 끝.
    KITTI 앞 400 장에 canonical 4 종(가장 큰 1720×498 포함)이 다 있어 본 실험 최악 입력이 이미 들어간 측정. 본 실험도 장치 배치 8 이라 같은 수준 예상, OOM 이면 LBS=4 (전체 배치 64 유지, 최적화 같음).
  · 체크포인트 37G = 에폭 저장 19 GB + 마지막 저장 19 GB (예상대로). 본 실험은 데이터셋당 19 × 3 + 19 ≈ 76 GB, 끝나면 지움 (WORK 여유 436G).
  · 예측 0.85–1.08 초/장. 채점(20 장, 이미지별 평균) → F-9. 원래 모델 NYU 검증 20 장 90.2 / 9.08 = 로컬 F-7.
  · 본 실험 시간 추정(스모크 속도로 위쪽 어림 — 누적 8 이면 최적화 걸음이 1/8 이라 더 빠를 수 있음): NYU 학습 22,923 × 3 / 3.85 ≈ 5.0 h, KITTI 21,922 × 3 / 4.26 ≈ 4.3 h,
    검증 400 장 × 3 에폭 ≈ 17 분·테스트 ≈ 10 분·원래 모델 ≈ 16 분 (데이터셋마다) → nyu ≈ 5.5 h, kitti ≈ 4.8 h, all ≈ 10 h.

