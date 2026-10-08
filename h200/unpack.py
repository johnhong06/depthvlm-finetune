"""H200: /app/data (와 DATA_SRC 의 폴더들) 아래에서 dvft_*.zip 묶음과 dvft_SHA256SUMS.txt 를 찾아 SHA256 을 대조한 뒤 <dest> 에 푼다.
관리자가 드라이브 폴더를 통째로 받아 zip 안에 zip 이 들어 있어도 찾는다 (그때는 안쪽 zip 을 <dest>/.incoming 에 꺼낸 뒤 대조).
SHA256SUMS 에 적힌 zip 이 하나라도 없거나 다르면 아무것도 풀지 않고 실패한다. 다 풀면 <dest>/.done_dvft 를 남기고, 있으면 다시 풀지 않는다.
사용: python h200/unpack.py <dest> [묶음 이름 ...]   (이름 = meta nyu_test nyu_train kitti_eval kitti_train, 없으면 전부)
"""
import hashlib
import os
import shutil
import sys
import zipfile

dest, want = sys.argv[1], sys.argv[2:]
mark = os.path.join(dest, ".done_dvft" + ("_" + "_".join(sorted(want)) if want else ""))
if os.path.exists(mark):
    sys.exit(print(f"[unpack] 이미 풀려 있음 ({dest})"))
found, sums = {}, {}
for root in ["/app/data"] + [r for r in os.environ.get("DATA_SRC", "").split(":") if r]:
    for dp, dn, fn in os.walk(root, followlinks=True):
        if dp[len(root):].count(os.sep) >= 4:
            dn[:] = []
        for f in fn:
            p = os.path.join(dp, f)
            if f.startswith("dvft_") and f.endswith(".zip"):
                found.setdefault(f, (None, p))
            elif f == "dvft_SHA256SUMS.txt":
                sums.update({l.split()[1]: l.split()[0] for l in open(p) if l.strip()})
            elif f.endswith(".zip") and zipfile.is_zipfile(p):   # 드라이브 폴더 zip 안에 든 경우
                z = zipfile.ZipFile(p)
                for n in z.namelist():
                    b = os.path.basename(n)
                    if b.startswith("dvft_") and b.endswith(".zip"):
                        found.setdefault(b, (z, n))
                    elif b == "dvft_SHA256SUMS.txt":
                        sums.update({l.split()[1]: l.split()[0] for l in z.read(n).decode().splitlines() if l.strip()})
need = sorted(n for n in sums if not want or any(n.startswith(f"dvft_{w}_") for w in want))
missing = [n for n in need if n not in found]
if not sums or missing:
    sys.exit(f"!!! [unpack] SHA256SUMS {'없음' if not sums else '있음'}, 없는 zip {len(missing)} 개 {missing[:5]} — 업로드가 덜 됐거나 아직 전달되지 않았다")
os.makedirs(os.path.join(dest, ".incoming"), exist_ok=True)
print(f"[unpack] zip {len(need)} 개 → {dest}", flush=True)
for n in need:
    z, p = found[n]
    if z is not None:   # 안쪽 zip 은 꺼내서 연다 (zip 안의 zip 을 바로 열면 되감기가 느리다)
        q = os.path.join(dest, ".incoming", n)
        with z.open(p) as s, open(q, "wb") as d:
            shutil.copyfileobj(s, d, 16 << 20)
        p = q
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(16 << 20):
            h.update(c)
    if h.hexdigest() != sums[n]:
        sys.exit(f"!!! [unpack] {n}: SHA256 불일치")
    with zipfile.ZipFile(p) as zz:
        zz.extractall(dest)   # 무압축 zip 이라도 멤버마다 CRC 를 검사한다
    if z is not None:
        os.remove(p)
    print(f"[unpack] {n} 통과", flush=True)
open(mark, "w").close()
print(f"[unpack] SHA256 {len(need)} 개 통과, 풀기 완료", flush=True)
