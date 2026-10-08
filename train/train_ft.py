"""DepthVLM 공식 train/train.py 를 그대로 실행하되, 데이터셋 깊이 범위 표(utils.datasets.DATASET_CONFIGS)에 KITTI 한 줄만 더한다.
공식 표에는 KITTI 가 없어 그대로 두면 깊이 범위가 0–inf 가 된다 → 표준 평가와 같은 0.001–80 m. NYU 는 공식 값(nyuv2: 0.005–10 m) 그대로.
인자는 공식 train.py 의 것 그대로 (TrlParser). 각 jsonl 이 어떤 범위로 잡혔는지 시작할 때 찍는다.
사용: python train/train_ft.py <공식 train.py 인자...>   (train/train_ft.sh 가 부른다)
"""
import os
import runpy
import sys

DVLM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "DepthVLM")
sys.path.insert(0, DVLM)
import utils.datasets as D  # noqa: E402  (train.py 가 importlib 로 같은 모듈 객체를 쓴다)

D.DATASET_CONFIGS["kitti"] = {"min_depth": 0.001, "max_depth": 80.0}
for i, a in enumerate(sys.argv):
    if a == "--dataset_name":
        for p in sys.argv[i + 1].split(";"):
            print(f"[train_ft] {p} → 깊이 범위 {D._match_dataset_config(p)}", flush=True)
runpy.run_path(os.path.join(DVLM, "train", "train.py"), run_name="__main__")
