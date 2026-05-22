# 超聲方塊定位關卡 · 執行與復現說明

獨立的**定位預訓練關卡**(對應規劃中的 Stage 0.5)。
驗證:在無視覺前提下,模型能否從多通道超聲收音預測方塊方位。

⚠️ **本關卡不接 ROS 2、不碰 domain.md 主契約。** 是接 SAC 抓取之前的感知預訓練。
真機部署(Phase 2/3)等定位穩定後再說。

---

## 1. 環境

實測可跑的版本(其他相近版本應該也行):

```
Python 3.12
numpy
scipy
torch  (CPU 即可,訓練量小)
```

安裝:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy torch
```

---

## 2. 檔案職責

| 檔案 | 職責 |
|---|---|
| `config.py` | 所有仿真假設參數(麥陣列幾何、採樣率、DR 範圍、命中閾值) |
| `signal_processing.py` | 階段1 固定 DSP:隨機發聲合成 → 帶通 → 窄帶相位差特徵(無學習) |
| `model.py` | 階段2 定位網路(~28k 參數) |
| `env.py` | 關卡邏輯 + 命中驗證器 + 真值對照 |
| `train.py` | 監督訓練 |
| `reproduce.py` | 一鍵復現 baseline 數據 |

資料流:`隨機放方塊 → 合成6通道收音 → 帶通 → 相位差特徵(15維) → 網路 → 方位熱圖(72) → 命中判定`

---

## 3. 先跑各模組 smoke test(確認環境沒問題)

```bash
python signal_processing.py   # 應印出 raw (6,1920) / feature (15,)
python model.py               # 應印出 參數量 28,360
python env.py                 # 應印出 true azimuth 與命中判定
```

三個都沒報錯 → 環境 OK。

---

## 4. 復現 baseline 實驗

```bash
python reproduce.py
```

等同於 `python train.py --steps 400 --batch 32 --lr 3e-4 --seed 0`。

**預期結果**(seed=0,自由場理想條件):

```
最終 hit_rate ≈ 0.80
mean_err     ≈ 10°
```

CPU 上 400 步約數分鐘(每樣本要現合成+濾波+FFT,慢在這)。

---

## 5. 當前 baseline 的設定(對比基準)

拿去和前處理優化版對比時,記住這版的條件:

- **麥陣列**:6 麥雙間距(水平 [0,4,12,28]mm + 垂直 ±12mm)
- **採樣率**:192 kHz
- **特徵**:窄帶相位差(主頻 bin 的 sin/cos)+ 能量比,15 維
- **聲學**:⚠️ **自由場幾何延遲模型,無混響、無方塊指向性、無 pyroomacoustics**

---

## 6. 已知限制(務必知道,不是 bug)

1. **數據是理想聲學**:`signal_processing.synthesize_reception()` 是自由場占位。接真實聲學後命中率會掉。
2. **相位模糊**:靠雙間距幾何(近對 <4.3mm 半波長)解,所以陣列幾何不能亂改。
3. **chirp/pulse 精度劣於 cw**:相位差對調頻信號天生弱。
4. **僅水平方位**:垂直 2 麥(mic4/5)為日後解俯仰預留,當前未真正用上。
5. **train.py 的 `np.random.seed` 對 env 無效**:env 用獨立的 `default_rng(seed)`,復現性由它保證(仍可復現,只是那兩行是裝飾)。

---

## 7. 復現性

- `--seed` 控制 env 的 `default_rng` 與 torch 初始化
- 訓練 env seed = `--seed`,評估 env seed = `--seed + 999`(訓練/評估資料不重疊)
- 預設 `--seed 0`,不隨機

---

## 8. 怎麼改參數做你自己的實驗

- 改麥陣列/採樣率/DR → 改 `config.py`(改 `n_mics` 會自動連動特徵維度與網路)
- 改命中閾值 → `config.py` 的 `hit_threshold_deg`(預設 5°)
- 改方位解析度 → `config.py` 的 `n_azimuth_bins`(預設 72 = 5°/bin)
- 改訓練步數/批次/學習率 → `train.py` 的命令列參數
