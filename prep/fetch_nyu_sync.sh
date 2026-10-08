#!/usr/bin/env bash
# NYU 학습 세트 = BTS 가 배포한 sync.zip (5.9 GiB; NYU raw 의 학습 장면에서 뽑은 RGB + Kinect 깊이, BTS pytorch/README.md 의 구글 드라이브 링크).
# 공개 링크는 조회 한도에 자주 걸려 gdown 이 실패한다(2026-10-08) → 인증된 rclone 원격(gdrive:)으로 같은 파일 ID 를 받는다.
# 사용: bash prep/fetch_nyu_sync.sh [out=~/data/nyuv2_bts]
set -euo pipefail
out=${1:-$HOME/data/nyuv2_bts}; mkdir -p "$out"
[ -s "$out/sync.zip" ] || rclone backend copyid gdrive: 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP "$out/" -v
ls -l "$out/sync.zip"
