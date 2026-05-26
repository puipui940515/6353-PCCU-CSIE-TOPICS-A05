# Dobot Project 操作手冊

本手冊只涵蓋本專案的兩個主要操作流程：

- 資料生成：建立感知模型需要的 `.npz` 訓練/驗證資料，以及建立 MuJoCo 場景資產。
- 訓練模型：訓練聲學感知模型與 SAC 強化學習控制模型。

## 1. 專案角色分工

| 區塊 | 路徑 | 用途 |
|---|---|---|
| 感知資料生成 | `detect/gen_dataset.py` | 生成聲學定位資料集 `train.npz`、`eval.npz` |
| 感知模型訓練 | `detect/train_gpu.py` | 訓練方位角與距離分類模型 |
| 感知資料檢查 | `detect/dataset_inspector.py` | 產生資料分布、PCA、feature heatmap 圖 |
| MuJoCo 場景生成 | `tools/prepare_dobot_assets.sh`, `tools/build_scene.py` | 從 ROS2 Dobot 資產建立 MuJoCo 場景 |
| SAC 訓練 | `training/train_sac.py` | 分 stage 訓練 Dobot 控制策略 |
| SAC 評估 | `training/eval_policy.py` | 載入訓練完成的 SAC policy 做 episode 評估 |
| 設定檔 | `configs/sac_stage*.yaml` | 定義 stage、reward、SAC 參數與訓練步數 |
| 訓練輸出 | `runs/`, `detect/runs/` | 保存模型、checkpoint、TensorBoard log |

## 2. 操作前準備

每次開新的 terminal，先載入專案環境：

```bash
cd ~/dobot_project
source ~/dobot_project/setup_env.sh
```

如果尚未建立 Python 虛擬環境：

```bash
cd ~/dobot_project
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

快速確認主要訓練腳本可以被 Python 解析：

```bash
python -m py_compile training/train_sac.py detect/gen_dataset.py detect/train_gpu.py
```

## 3. 生成 MuJoCo 場景資料

這個流程會把 ROS2 Dobot description 轉成 MuJoCo 可用的資產與場景。SAC 環境會讀取：

```text
assets/dobot/magician_scene.mjcf
```

### 3.1 準備 Dobot URDF 與 mesh

```bash
cd ~/dobot_project
bash ~/dobot_project/tools/prepare_dobot_assets.sh
```

此步驟會：

- 從 `ros2_ws/src/magician_ros2/dobot_description` 複製 Dobot URDF。
- 將 `package://dobot_description/...` 路徑改成 `assets/dobot` 內部相對路徑。
- 將 `.dae` mesh 參照改成 `.obj`。

### 3.2 轉換 DAE mesh

```bash
cd ~/dobot_project
python ~/dobot_project/tools/convert_dae_to_obj.py
```

### 3.3 建立 MuJoCo 場景

```bash
cd ~/dobot_project/assets/dobot
python ~/dobot_project/tools/build_scene.py
```

完成後應產生：

```text
assets/dobot/magician_scene.mjcf
```

### 3.4 測試場景

```bash
cd ~/dobot_project
python ~/dobot_project/tools/test_scene.py --viewer
```

若 viewer 能打開，且場景內有 Dobot、地板、方塊與目標區，表示場景資料可供 SAC 使用。

## 4. 生成感知訓練資料

感知資料由 `detect/gen_dataset.py` 產生，輸出為 `.npz`，主要欄位包含：

| 欄位 | 說明 |
|---|---|
| `feats` | 聲學特徵矩陣 |
| `labels` | 方位角分類 label |
| `range_labels` | 距離分類 label |
| `sig_types` | 訊號型態紀錄 |

### 4.1 建立訓練集

```bash
cd ~/dobot_project/detect
python gen_dataset.py --n 200000 --seed 0 --use-v2 --workers 8 --out data/train.npz
```

### 4.2 建立驗證集

```bash
cd ~/dobot_project/detect
python gen_dataset.py --n 20000 --seed 999 --use-v2 --workers 8 --out data/eval.npz
```

### 4.3 小型資料集測試

若只是測試流程，先用小資料量：

```bash
cd ~/dobot_project/detect
python gen_dataset.py --n 1000 --seed 0 --use-v2 --workers 4 --out data/train_smoke.npz
python gen_dataset.py --n 200 --seed 999 --use-v2 --workers 4 --out data/eval_smoke.npz
```

## 5. 檢查感知資料

生成資料後，建議先檢查資料分布：

```bash
cd ~/dobot_project/detect
python dataset_inspector.py --data data/train.npz --outdir inspect_output
```

檢查輸出：

```text
detect/inspect_output/
```

重點查看：

- `azimuth_distribution.png`：方位角 label 是否過度集中。
- `range_distribution.png`：距離 label 是否嚴重不平衡。
- `feature_heatmap.png`：特徵是否全 0、爆量或有明顯異常。
- `pca_projection.png`：不同方位資料是否有可分性。
- `signal_type_distribution.png`：訊號型態比例是否符合預期。

## 6. 訓練感知模型

### 6.1 GPU 訓練

```bash
cd ~/dobot_project/detect
python train_gpu.py \
    --train-data data/train.npz \
    --eval-data data/eval.npz \
    --steps 5000 \
    --batch 256 \
    --lr 3e-4 \
    --tag gpu_run
```

訓練輸出會放在：

```text
detect/runs/gpu_run/
```

重要檔案：

| 檔案 | 用途 |
|---|---|
| `checkpoints/latest.pt` | 最新 checkpoint |
| `checkpoints/best_eval.pt` | 驗證集 hit rate 最佳模型 |
| `tb/` | TensorBoard log |
| `config.json` | 本次訓練參數 |

### 6.2 繼續訓練

```bash
cd ~/dobot_project/detect
python train_gpu.py \
    --resume runs/gpu_run/checkpoints/latest.pt \
    --steps 3000 \
    --tag gpu_run
```

### 6.3 查看訓練曲線

```bash
cd ~/dobot_project/detect
tensorboard --logdir runs/
```

主要觀察：

- `eval/hit_rate`：方位角命中率，越高越好。
- `eval/mean_err_deg`：平均角度誤差，越低越好。
- `eval/range_hit_rate`：距離分類命中率，應高於隨機猜測。

## 7. 檢查感知模型可否接入環境

用訓練好的 `best_eval.pt` 檢查 SAC 環境能否讀到感知模型：

```bash
cd ~/dobot_project
python training/test_env.py \
    --stage 1 \
    --weights ~/dobot_project/detect/runs/gpu_run/checkpoints/best_eval.pt
```

進一步檢查感知輸出與環境幾何是否一致：

```bash
cd ~/dobot_project
python training/check_perception_in_env.py \
    --weights detect/runs/gpu_run/checkpoints/best_eval.pt \
    --stage 1 \
    --n 500
```

## 8. 訓練 SAC 控制模型

SAC 使用 curriculum stage 訓練。設定檔位於：

```text
configs/sac_stage1.yaml
configs/sac_stage2.yaml
configs/sac_stage3.yaml
configs/sac_stage4.yaml
configs/sac_stage5.yaml
```

| Stage | 任務 | 預設步數 |
|---|---|---:|
| 1 | 末端對準方塊 | 500,000 |
| 2 | 抓取方塊，夾爪閉合 | 500,000 |
| 3 | 抓起並握住 | 1,000,000 |
| 4 | 放置到目標區 | 1,500,000 |
| 5 | 形狀泛化 | 2,000,000 |

### 8.1 訓練 Stage 1

```bash
cd ~/dobot_project
python training/train_sac.py \
    --config configs/sac_stage1.yaml \
    --tag sac_stage1
```

若要使用感知模型：

```bash
cd ~/dobot_project
python training/train_sac.py \
    --config configs/sac_stage1.yaml \
    --tag sac_stage1_perception \
    --perception-weights detect/runs/gpu_run/checkpoints/best_eval.pt
```

### 8.2 從上一個 stage 接續訓練

Stage 2 通常從 Stage 1 的 best model 接續：

```bash
cd ~/dobot_project
python training/train_sac.py \
    --config configs/sac_stage2.yaml \
    --resume runs/sac_stage1_<timestamp>/best/best_model.zip \
    --tag sac_stage2
```

Stage 3：

```bash
cd ~/dobot_project
python training/train_sac.py \
    --config configs/sac_stage3.yaml \
    --resume runs/sac_stage2_<timestamp>/best/best_model.zip \
    --tag sac_stage3
```

Stage 4：

```bash
cd ~/dobot_project
python training/train_sac.py \
    --config configs/sac_stage4.yaml \
    --resume runs/sac_stage3_<timestamp>/best/best_model.zip \
    --tag sac_stage4
```

Stage 5：

```bash
cd ~/dobot_project
python training/train_sac.py \
    --config configs/sac_stage5.yaml \
    --resume runs/sac_stage4_<timestamp>/best/best_model.zip \
    --tag sac_stage5
```

### 8.3 小步數測試

正式訓練前可先跑短版：

```bash
cd ~/dobot_project
python training/train_sac.py \
    --config configs/sac_stage1.yaml \
    --total-steps 10000 \
    --n-envs 1 \
    --tag smoke_stage1
```

## 9. SAC 訓練輸出

每次訓練會建立：

```text
runs/<tag>_<timestamp>/
```

內容通常包含：

| 路徑 | 說明 |
|---|---|
| `config.yaml` | 實際使用的訓練設定 |
| `commit.txt` | 訓練時的 git commit |
| `tb/` | TensorBoard log |
| `checkpoints/` | 中途 checkpoint |
| `best/best_model.zip` | EvalCallback 保存的最佳模型 |
| `final.zip` | 訓練結束時保存的模型 |
| `replay_buffer.pkl` | SAC replay buffer |
| `eval_logs/` | 評估紀錄 |

查看 TensorBoard：

```bash
cd ~/dobot_project
tensorboard --logdir runs/
```

## 10. 評估 SAC 模型

無 viewer 評估：

```bash
cd ~/dobot_project
python training/eval_policy.py \
    --model runs/sac_stage1_<timestamp>/best/best_model.zip \
    --episodes 20 \
    --no-viewer
```

開 viewer 評估：

```bash
cd ~/dobot_project
python training/eval_policy.py \
    --model runs/sac_stage1_<timestamp>/best/best_model.zip \
    --episodes 5
```

評估時重點看：

- success rate 是否達到該 stage 目標。
- episode reward 是否穩定上升。
- steps 是否逐漸變短。
- viewer 中是否有不合理穿模、抖動、方塊飛走等現象。

## 11. 建議標準流程

第一次完整操作建議照以下順序：

```bash
cd ~/dobot_project
source ~/dobot_project/setup_env.sh

# 1. 建立/檢查 MuJoCo 場景
bash tools/prepare_dobot_assets.sh
python tools/convert_dae_to_obj.py
cd assets/dobot
python ~/dobot_project/tools/build_scene.py
cd ~/dobot_project
python tools/test_scene.py --viewer

# 2. 生成感知資料
cd ~/dobot_project/detect
python gen_dataset.py --n 200000 --seed 0 --use-v2 --workers 8 --out data/train.npz
python gen_dataset.py --n 20000 --seed 999 --use-v2 --workers 8 --out data/eval.npz
python dataset_inspector.py --data data/train.npz --outdir inspect_output

# 3. 訓練感知模型
python train_gpu.py --train-data data/train.npz --eval-data data/eval.npz --steps 5000 --tag gpu_run

# 4. 檢查感知模型接入 SAC env
cd ~/dobot_project
python training/test_env.py --stage 1 --weights detect/runs/gpu_run/checkpoints/best_eval.pt

# 5. 訓練 SAC
python training/train_sac.py --config configs/sac_stage1.yaml --tag sac_stage1
```
