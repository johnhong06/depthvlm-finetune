#!/usr/bin/env bash
# KITTI Eigen 분할에 필요한 raw 주행 zip(학습 33 + 테스트 28 개)과 공식 annotated depth(14.2 GB)를 KITTI 공개 S3 에서 받는다.
# 이미 있는 zip 은 크기가 서버 Content-Length 와 같으면 건너뛰고, 다르면 이어 받는다. 끝나면 모든 zip 의 크기를 다시 대조한다.
# 사용: bash prep/fetch_kitti.sh [kitti_root=~/data/kitti] [splits=~/data/dvft/splits]
set -uo pipefail
K=${1:-$HOME/data/kitti}; SP=${2:-$HOME/data/dvft/splits}; R=$K/raw; mkdir -p "$R"
S3=https://s3.eu-central-1.amazonaws.com/avg-kitti
drives=$(cat "$SP"/eigen_train_files_with_gt.txt "$SP"/eigen_test_files_with_gt.txt | awk 'NF {split($1, a, "/"); print a[2]}' | sort -u)
get() {  # url dst
  local want have
  want=$(curl -sI --max-time 30 "$1" | awk 'tolower($1) == "content-length:" {print $2}' | tr -d '\r')
  [ -n "$want" ] || { echo "!!! 크기 조회 실패 $1"; return 1; }
  have=$(stat -c %s "$2" 2>/dev/null || echo 0)
  [ "$have" = "$want" ] && { echo "[kitti] 있음 $(basename "$2")"; return 0; }
  curl -sfL --retry 5 --retry-delay 30 -C - -o "$2" "$1" || { echo "!!! 받기 실패 $(basename "$2")"; return 1; }
  have=$(stat -c %s "$2")
  [ "$have" = "$want" ] && echo "[kitti] 받음 $(basename "$2") $((want / 1000000)) MB" || { echo "!!! 크기 불일치 $(basename "$2") $have != $want"; return 1; }
}
export -f get
fail=0
for d in $drives; do echo "$S3/raw_data/${d%_sync}/$d.zip $R/$d.zip"; done | xargs -P 4 -n 2 bash -c 'get "$0" "$1"' || fail=1
get "$S3/data_depth_annotated.zip" "$K/data_depth_annotated.zip" || fail=1
n=$(for d in $drives; do [ -s "$R/$d.zip" ] && echo; done | wc -l)
echo "[kitti] 주행 zip $n / $(echo "$drives" | wc -w), annotated depth $([ -s "$K/data_depth_annotated.zip" ] && echo 있음 || echo 없음), 실패 $fail"
exit $fail
