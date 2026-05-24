import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

# =========================================================
# RL Reward Analyzer (SAC / PPO / DDPG / TD3 通用)
# =========================================================

# ===== 直接改這裡即可 =====
CSV_PATH = "./test/ep_rew_mean.csv"  # SAC 训练日志的 CSV 文件路径

# 若不知道欄位名稱可設 None 自動搜尋
VALUE_COLUMN = None

# 是否平滑
USE_SMOOTHING = True
SMOOTH_WINDOW = 20

# rolling slope 視窗
ROLLING_WINDOW = 50

# =========================================================


def auto_find_reward_column(df):
    possible = [
        "Value",
        "value",
        "reward",
        "mean_reward",
        "ep_rew_mean",
        "rollout/ep_rew_mean",
        "eval/mean_reward",
    ]

    for c in possible:
        if c in df.columns:
            return c

    raise ValueError(
        f"找不到 reward 欄位\n目前欄位:\n{list(df.columns)}"
    )


def moving_average(data, window):
    return pd.Series(data).rolling(window).mean().values


def calculate_slope(x, y):
    mask = ~np.isnan(y)

    x = x[mask]
    y = y[mask]

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r_value ** 2,
        "p_value": p_value,
    }


# =========================================================
# 讀取 CSV
# =========================================================

csv_path = Path(CSV_PATH)

if not csv_path.exists():
    raise FileNotFoundError(f"找不到檔案: {csv_path}")

df = pd.read_csv(csv_path)

print("\n========== CSV INFO ==========")
print(f"File: {csv_path.name}")
print(f"Rows: {len(df)}")
print("\nColumns:")
print(list(df.columns))

# =========================================================
# 找 reward 欄位
# =========================================================

if VALUE_COLUMN is None:
    VALUE_COLUMN = auto_find_reward_column(df)

print(f"\nUsing reward column: {VALUE_COLUMN}")

# =========================================================
# 資料
# =========================================================

y_raw = df[VALUE_COLUMN].values.astype(float)
x = np.arange(len(y_raw))

# =========================================================
# 平滑
# =========================================================

if USE_SMOOTHING:
    y = moving_average(y_raw, SMOOTH_WINDOW)
else:
    y = y_raw

# =========================================================
# 整體斜率
# =========================================================

overall = calculate_slope(x, y)

# =========================================================
# 後半段斜率
# =========================================================

half_idx = len(x) // 2

late = calculate_slope(
    x[half_idx:],
    y[half_idx:]
)

# =========================================================
# 後30%斜率
# =========================================================

tail_idx = int(len(x) * 0.7)

tail = calculate_slope(
    x[tail_idx:],
    y[tail_idx:]
)

# =========================================================
# Rolling slope
# =========================================================

rolling_slopes = []

for i in range(ROLLING_WINDOW, len(x)):
    xs = x[i - ROLLING_WINDOW:i]
    ys = y[i - ROLLING_WINDOW:i]

    result = calculate_slope(xs, ys)

    rolling_slopes.append(result["slope"])

rolling_x = x[ROLLING_WINDOW:]

# =========================================================
# Collapse 偵測
# =========================================================

collapse_detected = False

if (
    overall["slope"] > 0
    and tail["slope"] < 0
):
    collapse_detected = True

# =========================================================
# 輸出分析
# =========================================================

print("\n========== ANALYSIS ==========")

print(f"\nOverall Slope: {overall['slope']:.6f}")
print(f"Overall R²:    {overall['r2']:.4f}")

print(f"\nLate Half Slope: {late['slope']:.6f}")
print(f"Late Half R²:    {late['r2']:.4f}")

print(f"\nLast 30% Slope: {tail['slope']:.6f}")
print(f"Last 30% R²:    {tail['r2']:.4f}")

# =========================================================
# 判斷
# =========================================================

print("\n========== INTERPRETATION ==========")

if overall["slope"] > 0:
    print("整體趨勢：成長")
else:
    print("整體趨勢：退化")

if tail["slope"] > 0:
    print("後期趨勢：仍在進步")
else:
    print("後期趨勢：開始退化")

if collapse_detected:
    print("\n⚠ 偵測到可能的 RL Collapse")
    print("代表：")
    print("- 前期學習成功")
    print("- 後期開始崩潰")
    print("- 常見於 SAC / PPO 不穩定")

# =========================================================
# 畫圖
# =========================================================

trend_line = (
    overall["intercept"]
    + overall["slope"] * x
)

plt.figure(figsize=(14, 7))

plt.plot(
    x,
    y_raw,
    alpha=0.3,
    label="Raw Reward"
)

if USE_SMOOTHING:
    plt.plot(
        x,
        y,
        linewidth=2,
        label=f"Smoothed ({SMOOTH_WINDOW})"
    )

plt.plot(
    x,
    trend_line,
    linewidth=2,
    label="Overall Trend"
)

plt.xlabel("Training Step")
plt.ylabel(VALUE_COLUMN)
plt.title("RL Reward Analysis")
plt.legend()

plt.show()

# =========================================================
# Rolling slope 圖
# =========================================================

plt.figure(figsize=(14, 5))

plt.plot(
    rolling_x,
    rolling_slopes
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Training Step")
plt.ylabel("Slope")
plt.title("Rolling Reward Slope")

plt.show()