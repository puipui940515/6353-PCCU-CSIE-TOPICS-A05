# Dobot Project 維修手冊

本手冊整理本專案常見問題、判斷方式與處理步驟。優先處理順序建議為：

1. 環境是否載入。
2. 主要資料檔是否存在。
3. MuJoCo 場景是否可編譯。
4. 感知資料/模型是否正常。
5. SAC 訓練設定是否合理。

## 1. 快速檢查指令

先執行以下基本檢查：

```bash
cd ~/dobot_project
source ~/dobot_project/setup_env.sh
python -m py_compile training/train_sac.py detect/gen_dataset.py detect/train_gpu.py
python training/test_env.py --stage 1
```

若有感知模型：

```bash
cd ~/dobot_project
python training/test_env.py \
    --stage 1 \
    --weights detect/runs/gpu_run/checkpoints/best_eval.pt
```

若要檢查資料集：

```bash
cd ~/dobot_project/detect
python dataset_inspector.py --data data/train.npz --outdir inspect_output
```

## 2. 環境問題

### 2.1 找不到 `mujoco`

症狀：

```text
ModuleNotFoundError: No module named 'mujoco'
```

處理：

```bash
cd ~/dobot_project
source .venv/bin/activate
pip install -r requirements.txt
python -c "import mujoco; print(mujoco.__version__)"
```

如果仍失敗，重新載入專案環境：

```bash
source ~/dobot_project/setup_env.sh
```

### 2.2 找不到 `rclpy` 或 ROS2

症狀：

```text
ModuleNotFoundError: No module named 'rclpy'
```

處理：

```bash
source /opt/ros/humble/setup.bash
source ~/dobot_project/setup_env.sh
```

若 `/opt/ros/humble/setup.bash` 不存在，表示本機尚未安裝 ROS 2 Humble。

### 2.3 `setup_env.sh` 提示 `.venv` 不存在

處理：

```bash
cd ~/dobot_project
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
source ~/dobot_project/setup_env.sh
```

### 2.4 終端機顯示中文亂碼

症狀：

- 註解或 README 顯示成亂碼。
- 但 `python -m py_compile ...` 可通過。

判斷：通常是終端編碼顯示問題，不一定是檔案壞掉。

處理：

```bash
python -m py_compile training/train_sac.py detect/gen_dataset.py detect/train_gpu.py
```

若語法檢查通過，可先忽略顯示亂碼。若語法檢查失敗，再回到出錯行修復字串或註解。

## 3. MuJoCo 場景問題

### 3.1 找不到 `assets/dobot/magician_scene.mjcf`

症狀：

```text
FileNotFoundError: ... assets/dobot/magician_scene.mjcf
```

原因：尚未建立 MuJoCo 場景。

處理：

```bash
cd ~/dobot_project
bash tools/prepare_dobot_assets.sh
python tools/convert_dae_to_obj.py
cd assets/dobot
python ~/dobot_project/tools/build_scene.py
```

確認：

```bash
ls ~/dobot_project/assets/dobot/magician_scene.mjcf
```

### 3.2 `build_scene.py` 找不到 URDF

症狀：

```text
找不到 ... assets/dobot/magician.urdf
```

原因：尚未執行 `prepare_dobot_assets.sh`，或 `ros2_ws/src/magician_ros2/dobot_description` 不存在。

處理：

```bash
cd ~/dobot_project
ls ros2_ws/src/magician_ros2/dobot_description
bash tools/prepare_dobot_assets.sh
```

若 source description 不存在，要先補齊 Dobot ROS2 description package。

### 3.3 Env 啟動時說 `target_zone` 或 `block_weld` 有問題

症狀：

```text
target_zone 不是 mocap body, build_scene.py 有問題
找不到 block_weld equality, build_scene.py 有問題
```

原因：`magician_scene.mjcf` 不是由目前版本的 `tools/build_scene.py` 產生，或場景檔被手動改壞。

處理：

```bash
cd ~/dobot_project/assets/dobot
python ~/dobot_project/tools/build_scene.py
cd ~/dobot_project
python training/test_env.py --stage 1
```

### 3.4 Viewer 黑畫面或場景不動

處理順序：

```bash
cd ~/dobot_project
python tools/test_scene.py --viewer
python training/test_env.py --stage 1
```

如果是在遠端或無桌面環境，先改用無 viewer 的測試方式：

```bash
python training/test_env.py --stage 1
```

## 4. 感知資料問題

### 4.1 找不到 `data/train.npz` 或 `data/eval.npz`

症狀：

```text
FileNotFoundError: data/train.npz
```

處理：

```bash
cd ~/dobot_project/detect
python gen_dataset.py --n 200000 --seed 0 --use-v2 --workers 8 --out data/train.npz
python gen_dataset.py --n 20000 --seed 999 --use-v2 --workers 8 --out data/eval.npz
```

若只要測試：

```bash
python gen_dataset.py --n 1000 --seed 0 --use-v2 --workers 4 --out data/train_smoke.npz
python gen_dataset.py --n 200 --seed 999 --use-v2 --workers 4 --out data/eval_smoke.npz
```

### 4.2 資料生成很慢

原因可能是：

- `--workers` 太少。
- pyroomacoustics 比例較高。
- 樣本數過大。

處理：

```bash
cd ~/dobot_project/detect
python gen_dataset.py --n 200000 --seed 0 --use-v2 --workers 8 --chunk-size 100 --out data/train.npz
```

如果 CPU 核心較少，把 `--workers` 降低，例如 `4` 或 `2`。

### 4.3 資料 label 分布不平均

檢查：

```bash
cd ~/dobot_project/detect
python dataset_inspector.py --data data/train.npz --outdir inspect_output
```

查看：

- `azimuth_distribution.png`
- `range_distribution.png`
- 終端輸出的每個 bin count

處理：

- 重新產生資料，換不同 `--seed`。
- 增加 `--n`。
- 檢查 `detect/config.py` 的 source range 與 domain randomization 設定。

### 4.4 Feature 全 0、NaN 或數值爆掉

檢查：

```bash
cd ~/dobot_project/detect
python dataset_inspector.py --data data/train.npz --outdir inspect_output
```

處理：

1. 查看 `feature_heatmap.png`。
2. 若特徵全 0，檢查 `detect/signal_processing.py` 的訊號合成與特徵抽取。
3. 若有 NaN，先用小資料重生確認是否穩定復現：

```bash
python gen_dataset.py --n 1000 --seed 123 --use-v2 --workers 1 --out data/debug.npz
```

4. 若只在多 worker 發生，先用 `--workers 1` 定位。

## 5. 感知模型訓練問題

### 5.1 CUDA 沒有被使用

症狀：`train_gpu.py` 顯示 device 是 `cpu`。

檢查：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

處理：

- 確認 NVIDIA driver 正常。
- 確認安裝的是 CUDA 版 PyTorch。
- 若只需先驗證流程，可接受 CPU，但要降低 `--steps` 和 `--batch`。

### 5.2 GPU 記憶體不足

症狀：

```text
CUDA out of memory
```

處理：

```bash
cd ~/dobot_project/detect
python train_gpu.py \
    --train-data data/train.npz \
    --eval-data data/eval.npz \
    --steps 5000 \
    --batch 64 \
    --tag gpu_run
```

如果仍不足，降到 `--batch 32`。

### 5.3 感知 hit rate 不上升

檢查：

```bash
cd ~/dobot_project/detect
tensorboard --logdir runs/
python dataset_inspector.py --data data/train.npz --outdir inspect_output
```

判斷重點：

- `eval/hit_rate` 是否接近隨機猜測。
- `eval/mean_err_deg` 是否長期不下降。
- `azimuth_distribution.png` 是否嚴重偏斜。
- `feature_heatmap.png` 是否無明顯訊號。

處理：

- 重新生成資料，增加 `--n`。
- 確認 train/eval 使用不同 seed。
- 確認 `train_gpu.py` 的 `--train-data`、`--eval-data` 指到正確檔案。
- 增加 `--steps`。

### 5.4 找不到 `best_eval.pt`

原因：訓練尚未跑到第一次 eval，或訓練中斷太早。

處理：

```bash
cd ~/dobot_project/detect
python train_gpu.py --steps 5000 --tag gpu_run
ls runs/gpu_run/checkpoints
```

若仍只有 `latest.pt`，可先用 `latest.pt` 做環境測試，但正式訓練 SAC 建議使用 `best_eval.pt`。

## 6. 感知模型接入 SAC 環境問題

### 6.1 `training/test_env.py --weights` 失敗

處理：

```bash
cd ~/dobot_project
ls detect/runs/gpu_run/checkpoints/best_eval.pt
python training/test_env.py --stage 1 --weights detect/runs/gpu_run/checkpoints/best_eval.pt
```

若 checkpoint 來自舊版模型，可能與目前 `detect/config.py` 的觀測維度不一致。重新訓練感知模型後再試。

### 6.2 `source_azimuth` 或 `source_range` 維度不對

原因通常是改過 `detect/config.py`：

- `n_azimuth_bins`
- `range_head.bin_edges_m`
- `n_range_bins`
- microphone 數量或 layout

處理：

1. 還原觀測契約，或同步更新環境與模型。
2. 重新生成資料。
3. 重新訓練感知模型。
4. 再執行：

```bash
cd ~/dobot_project
python training/test_env.py --stage 1 --weights detect/runs/gpu_run/checkpoints/best_eval.pt
```

### 6.3 `check_perception_in_env.py` 命中率很低

處理順序：

```bash
cd ~/dobot_project
python training/check_perception_in_env.py \
    --weights detect/runs/gpu_run/checkpoints/best_eval.pt \
    --stage 1 \
    --n 500
```

若方位或距離命中率低：

- 確認感知模型在 `detect/train_gpu.py` eval 階段表現正常。
- 確認 `detect/config.py` 的 microphone layout 與 `tools/build_scene.py` 產生的 mic site 一致。
- 重新建立 `magician_scene.mjcf`。
- 重新生成資料並重新訓練感知模型。

## 7. SAC 訓練問題

### 7.1 `train_sac.py` 找不到 config

症狀：

```text
FileNotFoundError: configs/sac_stage1.yaml
```

處理：

```bash
cd ~/dobot_project
python training/train_sac.py --config configs/sac_stage1.yaml
```

避免從 `training/` 目錄內直接用相對路徑執行，除非 config 路徑寫成絕對路徑。

### 7.2 CUDA out of memory

SAC 預設 `n_envs: 32`，可能吃掉大量 GPU/CPU 記憶體。

處理：

```bash
cd ~/dobot_project
python training/train_sac.py \
    --config configs/sac_stage1.yaml \
    --n-envs 8 \
    --total-steps 100000 \
    --tag sac_stage1_lowmem
```

若仍不足，改成：

```bash
python training/train_sac.py \
    --config configs/sac_stage1.yaml \
    --n-envs 1 \
    --total-steps 10000 \
    --tag smoke_stage1
```

### 7.3 SubprocVecEnv 卡住或沒有進度

原因可能是多進程環境初始化慢、資源不足或某個 worker 掛住。

處理：

```bash
cd ~/dobot_project
python training/train_sac.py \
    --config configs/sac_stage1.yaml \
    --n-envs 1 \
    --total-steps 10000 \
    --tag debug_stage1
```

如果 `n-envs 1` 正常，再逐步提高到 `4`、`8`、`16`。

### 7.4 Stage 2/3/4 resume 失敗

症狀：

```text
FileNotFoundError: ... best_model.zip
```

處理：

```bash
cd ~/dobot_project
find runs -path "*best_model.zip"
```

確認實際路徑後再 resume：

```bash
python training/train_sac.py \
    --config configs/sac_stage2.yaml \
    --resume runs/sac_stage1_<timestamp>/best/best_model.zip \
    --tag sac_stage2
```

### 7.5 TensorBoard 沒有看到曲線

處理：

```bash
cd ~/dobot_project
find runs -path "*tb*"
tensorboard --logdir runs/
```

若是感知模型：

```bash
cd ~/dobot_project/detect
tensorboard --logdir runs/
```

### 7.6 SAC reward 不上升

檢查：

- 是否正在訓練正確 stage。
- `configs/sac_stage*.yaml` 的 reward 權重是否被改過。
- `training/test_env.py --stage N` 是否能正常跑 100 steps。
- TensorBoard 的 eval reward 是否長期為負或震盪。

處理：

```bash
cd ~/dobot_project
python training/test_env.py --stage 1
python training/train_sac.py \
    --config configs/sac_stage1.yaml \
    --n-envs 1 \
    --total-steps 10000 \
    --tag reward_debug
```

若 Stage 1 都無法上升，優先檢查 MuJoCo 場景與 reward。不要直接跳到 Stage 3 以上。

### 7.7 Stage 3 以後方塊飛走、穿模或抖動

可能原因：

- `magician_scene.mjcf` collision 設定異常。
- 方塊初始位置或 floor 高度不合理。
- actuator gain 太大。
- reward 鼓勵過度激烈動作。

處理：

```bash
cd ~/dobot_project/assets/dobot
python ~/dobot_project/tools/build_scene.py
cd ~/dobot_project
python tools/test_scene.py --viewer
python training/test_env.py --stage 3
```

若仍異常，檢查：

- `tools/build_scene.py` 的 collision group。
- `envs/mujoco_dobot_env.py` 的方塊初始高度。
- `configs/sac_stage3.yaml` 的 `W_ACTION_SMOOTH`。

### 7.8 Eval success rate 很低

處理：

1. 先確認訓練不是太短：

```bash
cd ~/dobot_project
ls runs/<tag>_<timestamp>/eval_logs
```

2. 評估最佳模型而不是 final：

```bash
python training/eval_policy.py \
    --model runs/<tag>_<timestamp>/best/best_model.zip \
    --episodes 20 \
    --no-viewer
```

3. 若 best 與 final 差距大，用 best 接下一個 stage。

## 8. 檔案與輸出維護

### 8.1 訓練輸出太多

主要會長大的資料夾：

```text
runs/
detect/runs/
detect/data/
detect/inspect_output/
```

整理建議：

- 保留每個 stage 的 `best/best_model.zip`、`config.yaml`、`commit.txt`。
- 保留感知模型的 `best_eval.pt`、`config.json`。
- 大量中間 checkpoint 可移到備份資料夾。

### 8.2 不確定目前最佳模型是哪個

SAC：

```bash
cd ~/dobot_project
find runs -path "*best_model.zip"
```

感知模型：

```bash
cd ~/dobot_project/detect
find runs -path "*best_eval.pt"
```

### 8.3 replay buffer 是否可以跨 stage 使用

`train_sac.py` 支援：

```bash
--load-buffer <path/to/replay_buffer.pkl>
```

但跨 stage 使用 replay buffer 要小心，因為 reward 與任務定義可能改變。若不確定，建議只 resume policy，不載入舊 buffer。

## 9. 常用修復流程

### 9.1 重新建立場景

```bash
cd ~/dobot_project
bash tools/prepare_dobot_assets.sh
python tools/convert_dae_to_obj.py
cd assets/dobot
python ~/dobot_project/tools/build_scene.py
cd ~/dobot_project
python training/test_env.py --stage 1
```

### 9.2 重新建立感知資料與模型

```bash
cd ~/dobot_project/detect
python gen_dataset.py --n 200000 --seed 0 --use-v2 --workers 8 --out data/train.npz
python gen_dataset.py --n 20000 --seed 999 --use-v2 --workers 8 --out data/eval.npz
python dataset_inspector.py --data data/train.npz --outdir inspect_output
python train_gpu.py --train-data data/train.npz --eval-data data/eval.npz --steps 5000 --tag gpu_run
```

### 9.3 從最小 SAC 訓練確認流程

```bash
cd ~/dobot_project
python training/train_sac.py \
    --config configs/sac_stage1.yaml \
    --total-steps 10000 \
    --n-envs 1 \
    --tag smoke_stage1
```

### 9.4 完整健康檢查

```bash
cd ~/dobot_project
source ~/dobot_project/setup_env.sh
python -m py_compile training/train_sac.py detect/gen_dataset.py detect/train_gpu.py
python training/test_env.py --stage 1
python training/test_env.py --stage 1 --weights detect/runs/gpu_run/checkpoints/best_eval.pt
python training/check_perception_in_env.py --weights detect/runs/gpu_run/checkpoints/best_eval.pt --stage 1 --n 100
```

## 10. 問題回報時要附的資訊

遇到問題時，請保留以下資訊，方便定位：

- 執行的完整指令。
- 完整錯誤訊息。
- 使用的 config，例如 `configs/sac_stage1.yaml`。
- 使用的 checkpoint 路徑。
- `runs/<tag>_<timestamp>/config.yaml`。
- `runs/<tag>_<timestamp>/commit.txt`。
- 若是感知問題，附 `detect/inspect_output/` 圖片。
- 若是 SAC 問題，附 TensorBoard 中 reward、success rate、loss 曲線截圖。

