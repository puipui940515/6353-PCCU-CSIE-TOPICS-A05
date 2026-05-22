"""復現實驗 · baseline 數據(6麥雙間距 192kHz 窄帶相位差)

固定參數,讓你一鍵復現對比用的 baseline。
用法:
    python reproduce.py

預期結果(seed=0, 自由場理想條件):
    最終 hit_rate ≈ 0.80, mean_err ≈ 10°
    (CPU 上每樣本含合成+濾波+FFT,400 步約需數分鐘)

注意:這是「自由場幾何模型」結果,非真實聲學。
接 pyroomacoustics 後數值會下降,那才是真實難度。
"""

import subprocess
import sys

# 對比實驗的固定設定
EXPERIMENT = {
    "steps": 400,
    "batch": 32,
    "lr": 3e-4,
    "seed": 0,
}


def main() -> None:
    cmd = [sys.executable, "train.py"]
    for k, v in EXPERIMENT.items():
        cmd += [f"--{k}", str(v)]
    print("執行:", " ".join(cmd))
    print("=" * 50)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
