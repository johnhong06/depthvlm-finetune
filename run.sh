#!/usr/bin/env bash
# H200 진입점 (gpu-request 이슈의 실행 명령 한 줄). DepthVLM-4B 를 NYU·KITTI 에 파인튜닝하고 UniDepthV2 표 V·VI 와 같은 규칙으로 잰다.
#   bash run.sh env                 # 데이터 없이: 환경 생성(+flash-attn), DepthVLM-4B 가중치(HF 고정 리비전) 받기·불러오기, GPU·디스크 확인
#   bash run.sh smoke               # 데이터까지: 데이터셋마다 400 장 × 1 에폭(장치 배치 LBS=8 이면 50 걸음) 학습 → 속도·메모리, 검증·테스트 20 장 채점 (원래 모델과 함께)
#   bash run.sh zeroshot            # 파인튜닝 전 DepthVLM-4B: NYU 654 장·KITTI 652 장 테스트 + 검증 400 장씩
#   bash run.sh nyu | kitti | all   # 본 실험: 학습(3 에폭) → 에폭마다 검증 세트 채점 → AbsRel 이 가장 낮은 에폭으로 테스트 채점 (+ 원래 모델 테스트)
# KEY=값 인자: EPOCHS(3) LBS(8) GBS(64) N(0 = 전부) KEEP_WEIGHTS(0, 1 이면 고른 체크포인트를 bf16 으로 zip 에 넣음 ≈ 9.7 GB) DATA_SRC PROGRESS_SEC(600)
#   STALL_MIN(40: 학습 로그가 이만큼 그대로면 멈춘 것으로 보고 끈다) STEP_MIN(90: 예측·채점 한 번의 시간 제한) — 어느 경우든 결과 zip(로그 포함)은 만든다.
#   준비 단계도 시간 제한: 패키지 설치 60분, flash-attn 30분, 가중치 받기 60분, zip 풀기 90분 (넘으면 이유를 찍고 끝낸다)
# 데이터: 관리자가 /app/data 아래에 둔 dvft_*.zip + dvft_SHA256SUMS.txt (prep/pack.py) → WORK/data. 결과: /app/output/dvft/dvft_<모드>_<시각>.zip
set -uo pipefail
cd "$(dirname "$0")" || exit 1
MODE=${1:-smoke}; shift || true
case $MODE in env|smoke|zeroshot|nyu|kitti|all) ;; *) echo "!!! MODE 는 env|smoke|zeroshot|nyu|kitti|all"; exit 1;; esac
for a in "$@"; do case $a in EPOCHS=*|LBS=*|GBS=*|N=*|KEEP_WEIGHTS=*|DATA_SRC=*|PROGRESS_SEC=*|STALL_MIN=*|STEP_MIN=*) export "${a%%=*}=${a#*=}";; *) echo "!!! 모르는 인자: $a"; exit 1;; esac; done
OUT=$([ -d /app/output ] && echo /app/output/dvft || echo "$PWD/results"); mkdir -p "$OUT"
w() { mkdir -p "$1" 2>/dev/null && touch "$1/.w" 2>/dev/null && rm -f "$1/.w"; }
WORK=${WORK:-$(w /app/data/dvft_work && echo /app/data/dvft_work || { w /app/scratch/dvft_work && echo /app/scratch/dvft_work || echo /tmp/dvft_work; })}
mkdir -p "$WORK"; export HF_HOME=$WORK/hf TORCH_HOME=$WORK/torch PIP_CACHE_DIR=$WORK/pip
TAG=${MODE}_$(date +%m%d_%H%M); R=$OUT/$TAG; mkdir -p "$R"; LOG=$R/run.log
exec > >(tee -a "$LOG") 2>&1; TEE=$!
trap 'exec >&- 2>&-; wait $TEE' EXIT   # H200 의 bash 5.1 은 인자 없는 wait 가 tee 까지 기다려 멈춘다 (vlm-depth-rmse NOTES F-8) → wait 에는 PID 를 준다
echo "[setup] $(date '+%F %T') MODE=$MODE OUT=$R WORK=$WORK commit=$(git rev-parse --short HEAD 2>/dev/null) $(df -h "$WORK" | awk 'NR==2 {print "WORK 여유 " $4}')"
D=$WORK/data; SP=$D/splits

# --- 환경 (vlm-depth-rmse run.sh 와 같은 방식: conda-forge python 3.12, 실패하면 uv + pip) ---
CONDA=$(command -v conda || echo /opt/conda/bin/conda)
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
mkenv() {
  local E=$WORK/envs/depthvlm
  if [ ! -f "$E/.ok" ]; then {
    echo "[env] depthvlm 생성"; rm -rf "$E"
    "$CONDA" create -y -q -p "$E" --override-channels -c conda-forge python=3.12 >/dev/null \
      || { echo "!!! [env] conda 실패 → uv"; pip install -q uv && uv venv -q -p 3.12 "$E" && uv pip install -q -p "$E/bin/python" pip; } || return 1
    timeout -k 60 60m "$E/bin/pip" install -q -r envs/depthvlm.txt || { echo "!!! [env] 패키지 설치 실패 또는 60분 초과"; return 1; }
    timeout -k 60 30m "$E/bin/pip" install -q "$FA" || echo "!!! [env] flash-attn 설치 실패 → sdpa (둘 다 exact attention, 공식은 flash_attention_2)"
    touch "$E/.ok"
  } >&2; fi
  "$E/bin/python" -c "import torch, transformers, trl; p = torch.cuda.get_device_properties(0); print(f'[env] torch {torch.__version__} transformers {transformers.__version__} trl {trl.__version__} | {p.name} {p.total_memory / 2**30:.0f} GiB')" >&2 || return 1
  echo "$E/bin/python"
}
PY=$(mkenv) || exit 1
ATTN=$($PY -c "import flash_attn; print('flash_attention_2')" 2>/dev/null || echo sdpa); echo "[env] attention $ATTN"
# DepthVLM-4B 가중치: HF 고정 리비전 (vlm-depth-rmse Track B 와 같은 리비전) → 로컬 폴더 경로
BASE=$(timeout -k 60 60m $PY -c "from huggingface_hub import snapshot_download as s; print(s('JonnyYu828/DepthVLM-4B', revision='2b2d02fcfe0c89c8aa7d541055e5a078930077c9'))" | tail -n 1) \
  || { echo "!!! [weights] DepthVLM-4B 받기 실패 또는 60분 초과"; exit 1; }
echo "[weights] DepthVLM-4B → $BASE ($(du -shL "$BASE/" | cut -f1))"

STALL=$(( ${STALL_MIN:-40} * 60 )); SETSID=$(command -v setsid || true)
run_train() {  # $1 = 학습 로그, 나머지 = 학습 명령. 따로 띄워 PROGRESS_SEC 마다 마지막 손실 줄·GPU 상태를 찍고, 로그(진행 막대가 걸음마다 늘린다)가
               # STALL_MIN 분 동안 그대로면 멈춘 것으로 보고 학습 프로세스 묶음(데이터 작업자 포함)을 끈다 — vlm-depth-rmse·depthlm-improve 와 같은 방식
  local log=$1 tp t=0 idle=0 last=-1 sig; shift
  $SETSID "$@" > "$log" 2>&1 & tp=$!
  while kill -0 "$tp" 2>/dev/null; do
    sleep 10; t=$((t + 10))
    sig=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$sig" = "$last" ]; then idle=$((idle + 10)); else idle=0; last=$sig; fi
    if [ $((t % ${PROGRESS_SEC:-600})) -eq 0 ] || [ "$idle" -ge "$STALL" ]; then
      echo "[진행 $(date '+%T')] $(grep -o "{'loss'[^}]*}" "$log" 2>/dev/null | tail -n 1) | 로그 변화 없음 $((idle / 60))분 | GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
    fi
    if [ "$idle" -ge "$STALL" ]; then
      echo "[중단] 학습 로그가 $((STALL / 60))분 동안 그대로 — 멈춘 것으로 보고 끈다. 로그 끝:"; tail -c 2000 "$log" | tr '\r' '\n' | tail -n 5
      kill -TERM -- "-$tp" 2>/dev/null; pkill -TERM -P "$tp" 2>/dev/null; kill -TERM "$tp" 2>/dev/null; sleep 30
      kill -KILL -- "-$tp" 2>/dev/null; pkill -KILL -P "$tp" 2>/dev/null; kill -KILL "$tp" 2>/dev/null
      wait "$tp" 2>/dev/null; return 124
    fi
  done
  wait "$tp"
}
predict() {  # $1 가중치 $2 ds $3 split $4 출력 폴더 [$5 목록] [$6 장수] — 한 번에 STEP_MIN 분(기본 90)을 넘으면 끊는다
  (cd eval && timeout -k 60 "${STEP_MIN:-90}m" $PY predict.py --model depthvlm --weights "$1" --processor "$BASE" --ds "$2" --split "$3" ${5:+--list "$5"} \
     --data "$D" --splits "$SP" --out "$4" --limit "${6:-0}" 2>&1 | grep -E "^\[|Traceback|Error") || echo "!!! [predict] $2 $3 실패 또는 ${STEP_MIN:-90}분 초과 ($1)"
}
score() {  # $1 ds $2 split $3 예측 폴더 $4 결과 이름 [$5 목록]
  (cd eval && timeout -k 60 "${STEP_MIN:-90}m" $PY std_eval.py --ds "$1" --split "$2" ${5:+--list "$5"} --data "$D" --splits "$SP" --pred "$3" --out "$R/$4" 2>&1 \
     | grep -E "^\[|이미지별|pooled|Traceback|Error") || echo "!!! [score] $1 $2 실패 또는 ${STEP_MIN:-90}분 초과 ($4)"
}
vlist() { echo "$D/$1/lists/val.txt"; }
head_list() { head -n "$2" "$1" > "$3"; echo "$3"; }   # 앞쪽 n 장 목록
f16() { $PY -c "import glob, numpy as np, os, sys; [np.save(os.path.join(sys.argv[2], os.path.basename(f)), np.load(f).astype(np.float16)) for f in glob.glob(sys.argv[1] + '/*.npy')]" "$1" "$2"; }

if [ "$MODE" != env ]; then
  timeout -k 60 90m python3 h200/unpack.py "$D" || { echo "!!! [unpack] 실패 또는 90분 초과"; exit 1; }
  for ds in nyu kitti; do echo "[data] $ds $(python3 -c "import json; r = json.load(open('$D/$ds/build_report.json')); print({k: r[k] for k in r if k in ('train', 'test', 'train_frames', 'val_frames', 'test_frames', 'val_drives')})")"; done
  for ds in nyu kitti; do echo "[data] $ds jsonl $(cat "$D/$ds/jsonl/info.json" | python3 -c "import json, sys; d = json.load(sys.stdin); print(d['records'], '장, sha256', d['sha256'])")"; done
fi

zeroshot() {  # 파인튜닝 전 모델: 테스트 전부 + 검증 400 장 (이미 있으면 건너뜀)
  local ds
  for ds in "$@"; do
    [ -f "$R/base_${ds}_test.json" ] && continue
    predict "$BASE" "$ds" test "$WORK/preds/base/${ds}_test"; score "$ds" test "$WORK/preds/base/${ds}_test" "base_${ds}_test"
    predict "$BASE" "$ds" val "$WORK/preds/base/${ds}_val" "$(vlist "$ds")"; score "$ds" val "$WORK/preds/base/${ds}_val" "base_${ds}_val" "$(vlist "$ds")"
    mkdir -p "$R/preds_f16/base_${ds}_test"; f16 "$WORK/preds/base/${ds}_test" "$R/preds_f16/base_${ds}_test"
  done
}
finetune() {  # $1 = ds. 학습 → 에폭 고르기 → 테스트
  local ds=$1 T=$WORK/ckpt/$1_$TAG ck pairs=() best
  mkdir -p "$T"
  run_train "$R/train_$ds.log" env PY="$PY" EPOCHS="${EPOCHS:-3}" LBS="${LBS:-8}" GBS="${GBS:-64}" N="${N:-0}" bash train/train_ft.sh "$ds" "$BASE" "$D" "$T"
  local rc=$?
  grep -E "train_ft\]|Trainable|train_runtime|'train_loss'" "$R/train_$ds.log" | cut -c1-400
  grep -o "{'loss'[^}]*}" "$R/train_$ds.log" > "$R/loss_$ds.txt"
  [ "$rc" -eq 0 ] || { echo "!!! 학습 실패 ($ds, 종료 $rc) — 로그 끝:"; tail -n 30 "$R/train_$ds.log"; return 1; }
  cp -r "$T/runs" "$R/tensorboard_$ds" 2>/dev/null
  for ck in "$T"/checkpoint-*; do
    predict "$ck" "$ds" val "$WORK/preds/$(basename "$ck")_${ds}_val_$TAG" "$(vlist "$ds")"
    score "$ds" val "$WORK/preds/$(basename "$ck")_${ds}_val_$TAG" "ft_${ds}_val_$(basename "$ck")" "$(vlist "$ds")"
    pairs+=("$ck:$R/ft_${ds}_val_$(basename "$ck").json")
  done
  best=$(cd eval && $PY select_epoch.py "$R/select_$ds.json" "${pairs[@]}" | tee /dev/stderr | tail -n 1)
  predict "$best" "$ds" test "$WORK/preds/ft_${ds}_test_$TAG"; score "$ds" test "$WORK/preds/ft_${ds}_test_$TAG" "ft_${ds}_test"
  mkdir -p "$R/preds_f16/ft_${ds}_test"; f16 "$WORK/preds/ft_${ds}_test_$TAG" "$R/preds_f16/ft_${ds}_test"
  if [ "${KEEP_WEIGHTS:-0}" = 1 ]; then
    $PY -c "import sys, torch; from safetensors.torch import load_file, save_file; sd = load_file(sys.argv[1] + '/model.safetensors'); save_file({k: v.to(torch.bfloat16) for k, v in sd.items()}, sys.argv[2])" \
      "$best" "$R/ft_${ds}_model_bf16.safetensors" && cp "$best/config.json" "$R/ft_${ds}_config.json"
  fi
  rm -rf "$T"   # fp32 체크포인트(장당 19 GB × 에폭 + 마지막 저장)는 남기지 않는다 — 다음 데이터셋 자리
}

case $MODE in
  env)
    (cd eval && $PY -c "
import sys; sys.argv = ['x']; import predict
f = predict.depthvlm('$BASE', None); print('[env] DepthVLM-4B 불러오기 정상')" 2>&1 | grep -E "^\[|Traceback|Error");;
  smoke)
    for ds in nyu kitti; do
      T=$WORK/ckpt/smoke_${ds}_$TAG
      sb=${LBS:-8}; echo "[smoke] $ds: 400 장 × 1 에폭, 배치 $sb ($((400 / sb)) 걸음)"
      PY=$PY EPOCHS=1 LBS=$sb GBS=$sb N=400 bash train/train_ft.sh "$ds" "$BASE" "$D" "$T" > "$R/train_$ds.log" 2>&1 & tp=$!
      peak=0; while kill -0 $tp 2>/dev/null; do sleep 15; m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${m:-0}" -gt "$peak" ] && peak=$m; done
      wait $tp; rc=$?
      grep -E "train_ft\]|Trainable|train_runtime|train_samples_per_second|'train_loss'|Traceback|Error" "$R/train_$ds.log" | cut -c1-300 | tail -n 12
      echo "[smoke] $ds 종료 $rc, GPU 최대 ${peak} MiB, 체크포인트 $(du -sh "$T" 2>/dev/null | cut -f1)"
      [ "$rc" -eq 0 ] || continue
      for split in val test; do
        L=$([ $split = val ] && head_list "$(vlist "$ds")" 20 "$WORK/${ds}_val20.txt" || head_list "$([ $ds = nyu ] && echo "$SP/nyudepthv2_test_files_with_gt.txt" || echo "$D/kitti/lists/test.txt")" 20 "$WORK/${ds}_test20.txt")
        predict "$T" "$ds" "$split" "$WORK/preds/smoke_${ds}_${split}_$TAG" "$L"; score "$ds" "$split" "$WORK/preds/smoke_${ds}_${split}_$TAG" "smoke_ft_${ds}_${split}20" "$L"
        predict "$BASE" "$ds" "$split" "$WORK/preds/smokebase_${ds}_${split}_$TAG" "$L"; score "$ds" "$split" "$WORK/preds/smokebase_${ds}_${split}_$TAG" "smoke_base_${ds}_${split}20" "$L"
      done
      rm -rf "$T"
    done;;
  zeroshot) zeroshot nyu kitti;;
  nyu|kitti) zeroshot "$MODE"; finetune "$MODE";;
  all) zeroshot nyu kitti; finetune nyu; finetune kitti;;
esac

cd "$OUT" && python3 -m zipfile -c "dvft_$TAG.zip" "$TAG" && echo "[done] $(date '+%T') 결과: $OUT/dvft_$TAG.zip ($(du -h "dvft_$TAG.zip" | cut -f1)) — 관리자에게 이 zip 을 요청"
