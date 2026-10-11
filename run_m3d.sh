#!/usr/bin/env bash
# H200 진입점 — Metric3Dv2 ViT-L 을 NYU 에 파인튜닝해 SILog 를 잰다 (NOTES D-8, train/m3d_ft.py). 데이터·결과 규칙은 run.sh 와 같고 DepthVLM 환경·가중치는 쓰지 않는다.
#   bash run_m3d.sh smoke   # 학습 64 장 · 20 걸음 (배치 4) · 검증/테스트 20 장 → 코드 경로·속도·메모리 확인 (약 15 분)
#   bash run_m3d.sh nyu     # 본 실행: 6,000 걸음 × 배치 16 → 500 걸음마다 검증 400 장 → AbsRel 최저 걸음으로 테스트 654 장 (+ 0 걸음 zero-shot)
# KEY=값: STEPS(6000) BS(16) MB(4) VAL_EVERY(500) MAX_HOURS(8) STALL_MIN(30) DATA_SRC  — 로컬 시험용: PY(이미 있는 파이썬) D(풀린 데이터 폴더) SP(분할 목록 폴더, 기본 D/splits)
# 받는 것 (토큰 없음): Metric3D 코드 = GitHub 고정 커밋 (prep/fetch_ext.sh), 가중치 = Hugging Face JUGGHM/Metric3D 의 metric_depth_vit_large_800k.pth.
# 데이터: 관리자가 /app/data 에 둔 dvft_*.zip (run.sh 와 같은 묶음, 이미 서버에 있음). 결과: /app/output/dvft/dvft_m3d_<모드>_<시각>.zip
set -uo pipefail
cd "$(dirname "$0")" || exit 1
MODE=${1:-smoke}; shift || true
case $MODE in smoke|nyu) ;; *) echo "!!! MODE 는 smoke|nyu"; exit 1;; esac
for a in "$@"; do case $a in STEPS=*|BS=*|MB=*|VAL_EVERY=*|MAX_HOURS=*|STALL_MIN=*|DATA_SRC=*|PY=*|D=*|SP=*) export "${a%%=*}=${a#*=}";; *) echo "!!! 모르는 인자: $a"; exit 1;; esac; done
OUT=$([ -d /app/output ] && echo /app/output/dvft || echo "$PWD/results"); mkdir -p "$OUT"
w() { mkdir -p "$1" 2>/dev/null && touch "$1/.w" 2>/dev/null && rm -f "$1/.w"; }
WORK=${WORK:-$(w /app/data/dvft_work && echo /app/data/dvft_work || { w /app/scratch/dvft_work && echo /app/scratch/dvft_work || echo /tmp/dvft_work; })}
mkdir -p "$WORK"; export HF_HOME=${HF_HOME:-$WORK/hf} TORCH_HOME=${TORCH_HOME:-$WORK/torch} PIP_CACHE_DIR=${PIP_CACHE_DIR:-$WORK/pip}
TAG=m3d_${MODE}_$(date +%m%d_%H%M); R=$OUT/$TAG; mkdir -p "$R"; LOG=$R/run.log
exec > >(tee -a "$LOG") 2>&1; TEE=$!
trap 'exec >&- 2>&-; wait $TEE' EXIT   # run.sh 와 같은 이유 (H200 bash 5.1 의 wait)
echo "[setup] $(date '+%F %T') MODE=$MODE OUT=$R WORK=$WORK commit=$(git rev-parse --short HEAD 2>/dev/null) $(df -h "$WORK" | awk 'NR==2 {print "WORK 여유 " $4}')"

# --- 환경: envs/metric3d.txt (로컬 Metric3Dv2 추론에서 검증된 것) — conda-forge python 3.12, 실패하면 uv ---
CONDA=$(command -v conda || echo /opt/conda/bin/conda)
mkenv() {
  local E=$WORK/envs/metric3d
  if [ ! -f "$E/.ok" ]; then {
    echo "[env] metric3d 생성"; rm -rf "$E"
    "$CONDA" create -y -q -p "$E" --override-channels -c conda-forge python=3.12 >/dev/null \
      || { echo "!!! [env] conda 실패 → uv"; pip install -q uv && uv venv -q -p 3.12 "$E" && uv pip install -q -p "$E/bin/python" pip; } || return 1
    timeout -k 60 60m "$E/bin/pip" install -q -r envs/metric3d.txt || { echo "!!! [env] 패키지 설치 실패 또는 60분 초과"; return 1; }
    touch "$E/.ok"
  } >&2; fi
  "$E/bin/python" -c "import torch, mmcv, mmengine, timm; p = torch.cuda.get_device_properties(0); print(f'[env] torch {torch.__version__} mmcv {mmcv.__version__} | {p.name} {p.total_memory / 2**30:.0f} GiB')" >&2 || return 1
  echo "$E/bin/python"
}
if [ -n "${PY:-}" ]; then echo "[env] 주어진 파이썬 $PY"; else PY=$(mkenv) || exit 1; fi
timeout -k 60 30m bash prep/fetch_ext.sh || { echo "!!! [ext] Metric3D 코드 받기 실패 또는 30분 초과"; exit 1; }
timeout -k 60 60m $PY -c "import torch; torch.hub.load_state_dict_from_url('https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth', map_location='cpu'); print('[weights] Metric3Dv2 ViT-L 받음')" \
  || { echo "!!! [weights] Metric3Dv2 ViT-L 받기 실패 또는 60분 초과"; exit 1; }

# --- 데이터 (run.sh 와 같은 묶음 — 이미 풀려 있으면 건너뜀) ---
if [ -z "${D:-}" ]; then D=$WORK/data; timeout -k 60 90m python3 h200/unpack.py "$D" || { echo "!!! [unpack] 실패 또는 90분 초과"; exit 1; }; fi
SP=${SP:-$D/splits}
echo "[data] nyu 학습 $(grep -c . "$D/nyu/lists/train_ft.txt") · 검증 $(grep -c . "$D/nyu/lists/val.txt") · 테스트 $(grep -c . "$SP/nyudepthv2_test_files_with_gt.txt") 장"

# --- 학습: 따로 띄워 PROGRESS 마다 마지막 걸음 줄·GPU 를 찍고, 로그가 STALL_MIN 분 그대로면 끈다 (run.sh run_train 과 같은 방식) ---
STALL=$(( ${STALL_MIN:-30} * 60 )); SETSID=$(command -v setsid || true)
if [ "$MODE" = smoke ]; then ARGS=(--n 64 --steps 20 --bs 4 --mb "${MB:-4}" --val_every 10 --log_every 5 --val_n 20 --test_n 20)
else ARGS=(--steps "${STEPS:-6000}" --bs "${BS:-16}" --mb "${MB:-4}" --val_every "${VAL_EVERY:-500}" --max_hours "${MAX_HOURS:-8}"); fi
$SETSID $PY -u train/m3d_ft.py --data "$D" --splits "$SP" --ext ext --out "$R" --work "$WORK/m3d_$TAG" "${ARGS[@]}" > "$R/train.log" 2>&1 & tp=$!
t=0; idle=0; last=-1
while kill -0 "$tp" 2>/dev/null; do
  sleep 10; t=$((t + 10))
  sig=$(stat -c %s "$R/train.log" 2>/dev/null || echo 0)
  if [ "$sig" = "$last" ]; then idle=$((idle + 10)); else idle=0; last=$sig; fi
  if [ $((t % 600)) -eq 0 ]; then
    echo "[진행 $(date '+%T')] $(grep -E '^\[m3d_ft\] (걸음|검증)' "$R/train.log" | tail -n 1 | cut -c1-220) | GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
  fi
  if [ "$idle" -ge "$STALL" ]; then
    echo "[중단] 학습 로그가 $((STALL / 60))분 동안 그대로 — 멈춘 것으로 보고 끈다. 로그 끝:"; tail -n 5 "$R/train.log"
    kill -TERM -- "-$tp" 2>/dev/null; kill -TERM "$tp" 2>/dev/null; sleep 30; kill -KILL -- "-$tp" 2>/dev/null; kill -KILL "$tp" 2>/dev/null
    break
  fi
done
wait "$tp"; rc=$?
grep -E "^\[m3d_ft\]|Traceback|Error" "$R/train.log" | grep -v "^\[m3d_ft\] 걸음" | cut -c1-400
grep -E "^\[m3d_ft\] 걸음" "$R/train.log" | tail -n 3 | cut -c1-300
echo "[m3d] 학습 종료 코드 $rc"

# 테스트 예측 (float16) 을 결과에 넣는다 — 로컬에서 SILog·전역 스케일 분해를 다시 할 수 있게
for s in step0 best; do
  P=$WORK/m3d_$TAG/pred_test_$s
  [ -d "$P" ] && mkdir -p "$R/preds_f16/m3d_${s}_nyu_test" && $PY -c "import glob, numpy as np, os, sys; [np.save(os.path.join(sys.argv[2], os.path.basename(f)), np.load(f).astype(np.float16)) for f in glob.glob(sys.argv[1] + '/*.npy')]" "$P" "$R/preds_f16/m3d_${s}_nyu_test"
done
rm -rf "$WORK/m3d_$TAG"   # best.pt (1.6 GB)·검증 예측은 남기지 않는다
cd "$OUT" && python3 -m zipfile -c "dvft_$TAG.zip" "$TAG" && echo "[done] $(date '+%T') 결과: $OUT/dvft_$TAG.zip ($(du -h "dvft_$TAG.zip" | cut -f1)) — 관리자에게 이 zip 을 요청"
