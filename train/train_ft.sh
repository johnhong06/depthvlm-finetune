#!/usr/bin/env bash
# DepthVLM-4B 파인튜닝 — 공식 train/train-stage2.sh 의 설정 그대로 (비전 인코더 고정, LLM + DPT 깊이 헤드 학습, 손실 = SILog(λ 0.5) + 답 문장 LM 손실,
# lr 2e-5 cosine, warmup 5 %, grad clip 1.0, bf16 혼합 정밀도, gradient checkpointing). 바꾸는 것: 시작 가중치(공개 DepthVLM-4B), GPU 1 장,
# 전체 배치 64·에폭 3 (데이터가 2 만 장 남짓이라), 에폭마다 가중치만 저장(검증 세트로 에폭을 고르려고), dataloader 작업자 4 (공식 1 단계 값),
# FSDP 대신 fp32 로 불러 bf16 autocast — 공식(GPU 80 장 FSDP)은 accelerate 가 FSDP 가중치를 fp32 로 올려 'fp32 주 가중치 + bf16 계산'이 되는데,
# GPU 1 장에서는 FSDP 가 NO_SHARD 로 바뀌고 이 fp32 승격과 충돌해 첫 걸음에 멈춘다 (RuntimeError: Cannot writeback when the parameter shape changes,
# 2026-10-08 로컬). fp32 로 불러 autocast 하면 같은 정밀도 구성이 된다 (bf16 로만 불러 FSDP 없이 돌리면 주 가중치가 bf16 이라 공식과 달라진다).
# GPU 1 장이라 torchrun(분산) 없이 바로 실행한다 (torchrun 1 프로세스는 DDP·NCCL 초기화만 더하고, 로컬에서 NCCL 오류로 멈췄다).
# 사용: bash train/train_ft.sh <nyu|kitti> <DepthVLM-4B 폴더> <데이터 루트> <출력 폴더>
#   환경변수: PY(파이썬), EPOCHS=3, LBS=8(장치 배치), GBS=64(전체 배치), N=0(>0 이면 jsonl 앞쪽 N 장만 — 스모크), STAGE=2(1 = 깊이 헤드만, 로컬 기능 점검용)
set -euo pipefail
DS=$1 MODEL=$2 DATA=$3 OUT=$4
PY=${PY:-python} EPOCHS=${EPOCHS:-3} LBS=${LBS:-8} GBS=${GBS:-64} N=${N:-0} STAGE=${STAGE:-2}
cd "$(dirname "$0")/.."
J=$DATA/$DS/jsonl/$([ "$DS" = nyu ] && echo nyuv2 || echo kitti)_train_ft.jsonl
if [ "$N" -gt 0 ]; then  # 스모크: 앞쪽 N 장 (파일 이름에 데이터셋 키가 남도록 같은 폴더에)
  head -n "$N" "$J" > "${J%.jsonl}_n$N.jsonl"; J=${J%.jsonl}_n$N.jsonl
fi
ACC=$(( (GBS + LBS - 1) / LBS ))
ATTN=$($PY -c "import flash_attn" 2>/dev/null && echo flash_attention_2 || echo sdpa)
FREEZE=(--freeze_vision --with_text_reply --depth_loss_weight 1.0); LR=2e-5; WARM=0.05
[ "$STAGE" = 1 ] && { FREEZE=(--freeze_mllm); LR=3.5e-4; WARM=0.04; }
echo "[train_ft] $(date '+%F %T') ds=$DS jsonl=$J ($(wc -l < "$J") 장) 시작=$MODEL 에폭=$EPOCHS 배치=$LBS×$ACC attention=$ATTN stage=$STAGE"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8} DEPTHLM_DEPTH_HEAD_TYPE=dpt
$PY train/train_ft.py \
  --model_name_or_path "$MODEL" --image_folder "$DATA" --dataset_name "$J" --depth_root "$DATA" \
  "${FREEZE[@]}" --max_seq_length 4096 --learning_rate $LR --lr_scheduler_type cosine \
  --per_device_train_batch_size "$LBS" --gradient_accumulation_steps "$ACC" --dataloader_num_workers 4 \
  --warmup_ratio $WARM --max_grad_norm 1.0 --logging_steps 10 --report_to tensorboard \
  --gradient_checkpointing true --attn_implementation "$ATTN" --num_train_epochs "$EPOCHS" \
  --log_level info --logging_strategy steps --output_dir "$OUT" \
  --save_strategy epoch --save_only_model true --save_total_limit "$EPOCHS" --eval_strategy no \
  --torch_dtype float32 --seed 42
