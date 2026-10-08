#!/usr/bin/env bash
# 표준 분할 목록 4 개 (BTS 저장소, 고정 커밋) → <out>/ . SHA256 이 다르면 실패한다.
# NYU 학습 24,231 장(sync) / 테스트 654 장, KITTI Eigen 학습 23,158 장 / 테스트 697 장(정답 있는 652 장).
# 사용: bash prep/fetch_splits.sh [out=~/data/dvft/splits]
set -euo pipefail
out=${1:-$HOME/data/dvft/splits}; mkdir -p "$out"
B=https://raw.githubusercontent.com/cleinc/bts/5e3406b35d1497b2e55d2dd600524d1f4efacaed/train_test_inputs
while read -r sha f; do
  [ -f "$out/$f" ] || curl -sfL --retry 3 -o "$out/$f" "$B/$f"
  echo "$sha  $out/$f" | sha256sum -c --quiet || { echo "!!! $f SHA256 불일치"; exit 1; }
done <<'EOF'
3655308ee4a017094dc620402db3fc587d769cfbfa9a16c6d0f8de6dcbe109ec eigen_train_files_with_gt.txt
8d0499af9ab4c0d1cd7944d572b0e0e4412feab1e73f4003038426fde699b53a eigen_test_files_with_gt.txt
66e2beb718a58c5c946c169f6e70cc6ae166ddd19f5618266fff52ff7443e0e9 nyudepthv2_train_files_with_gt.txt
e92c191d5835a22589b843c11b80cd7235d5bd0d4136e42a6debcea4511bf906 nyudepthv2_test_files_with_gt.txt
EOF
echo "[splits] 4 개 SHA256 통과 → $out"
