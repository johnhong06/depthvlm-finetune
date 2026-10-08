#!/usr/bin/env bash
# 채점 검증(로컬)에 쓰는 Metric3D 공식 저장소를 고정 커밋으로 ext/ 에 받는다 (vlm-depth-rmse prep/fetch_ext.sh 와 같은 방식·같은 커밋). H200 컨테이너에 git 이 없을 수 있어
# GitHub 의 커밋 고정 압축본(codeload.github.com/<저장소>/tar.gz/<커밋>)을 파이썬으로 받아 푼다. 네트워크가 끊기면 30 초 쉬고 3 번까지 다시 받고,
# 그래도 실패하면 0 이 아닌 값으로 끝난다 (2026-10-02 H200 에서 Metric3D 받기가 읽기 시간 초과로 한 번 실패).
set -euo pipefail
E=${DVFT_EXT:-$(cd "$(dirname "$0")/.." && pwd)/ext}; mkdir -p "$E"
get() {  # 이름 저장소 커밋
  [ "$(cat "$E/$1/.commit" 2>/dev/null)" = "$3" ] && return 0
  local a
  for a in 1 2 3; do
  rm -rf "${E:?}/${1:?}"
  python3 - "$E/$1" "https://codeload.github.com/$2/tar.gz/$3" <<'PY' && break

import os, sys, tarfile, urllib.request
dst, url = sys.argv[1:]
with urllib.request.urlopen(url, timeout=300) as r, tarfile.open(fileobj=r, mode="r|gz") as t:
    for m in t:  # 맨 위 폴더(<저장소>-<커밋>/)를 떼고 푼다
        parts = m.name.split("/", 1)
        if len(parts) < 2 or not parts[1]:
            continue
        m.name = parts[1]
        t.extract(m, dst, filter="tar") if hasattr(tarfile, "data_filter") else t.extract(m, dst)
PY
  [ "$a" = 3 ] && { echo "!!! [ext] $1 받기 실패 3 번 ($2 @ $3)" >&2; return 1; }
  echo "[ext] $1 받기 실패 ($a/3) — 30 초 뒤 다시" >&2; sleep 30
  done
  echo "$3" > "$E/$1/.commit"
  echo "[ext] $1 @ ${3:0:7}"
}
get Metric3D YvanYin/Metric3D eb5b6fac0dc155e4e52f576e304fbf11655ff339
