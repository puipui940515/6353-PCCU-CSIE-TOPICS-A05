# 定位模型接 SAC 設計說明 · 「定位當眼睛」

> 本文件是**設計說明**,不是已實作代碼。涉及 domain.md §4 obs space,
> 按 Instructions.md §4:正式接入前須先更新 domain.md 再改代碼。

---

## 1. 核心概念:定位網路 = SAC 的「眼睛」

定位網路(`model.py` 的 `LocalizationNet`)訓練好後,**不丟棄、不直接輸出動作**,
而是當 SAC policy 的**感知前端(perception encoder)**:

```
超聲收音 → 階段1 DSP → 定位網路 → 方位特徵 ─┐
                                              ├─→ SAC policy → 關節動作
本體感測(joint/wrench/imu) ─────────────────┘
```

定位網路把「聽到的聲音」變成「方塊在哪個方向」,SAC 拿這個方向 + 自身狀態,決定怎麼動手臂。
這就是「定位當眼睛」:它不負責決策,只負責**看**。

---

## 2. 要接什麼:用 logits 還是 argmax?

定位網路輸出 72 維方位 logits。接 SAC 時**用 logits(或 softmax 後的分布),不要用 argmax**。

理由:
- argmax 只給單一方向,丟掉了「不確定性」資訊(模型對哪些方向有疑慮)
- logits/softmax 是連續的,SAC 能學到「模型很確定時大膽、不確定時謹慎」
- 低 SNR 時方位分布會變平(體現不確定性),這對抓取決策有用

**建議**:取 softmax(72) 當特徵,或取倒數第二層的隱藏向量(128 維)當更豐富的 embedding。

---

## 3. obs space 怎麼改(對齊 domain.md §4.1)

domain.md §4.1 現有 obs space:

```python
{
  "joint_position": Box(4,),  "joint_velocity": Box(4,),
  "tcp_pose": Box(7,),  "wrench": Box(6,),  "imu_accel": Box(3,),
  "audio": Box(N_mics, N_samples),   # ← 原始音訊
}
```

**兩種接法,二選一:**

### 接法 A:把定位特徵當新的一項 obs(建議)
保留 audio,但新增定位網路輸出當一項:
```python
{
  ...(本體感測同上),
  "audio": Box(N_mics, N_samples),
  "source_dir": Box(72,),   # ← 新增:定位網路的方位 logits/softmax
}
```
SAC 直接吃 `source_dir`,不用自己從 audio 學定位。**最省 SAC 訓練量。**

### 接法 B:audio 進去前先過定位網路(端到端可選 finetune)
obs 仍是 audio,但 SAC 的網路前端**內嵌**定位網路:
```
audio → [定位網路] → source_dir → 與本體感測 concat → SAC head
```
差別:接法 A 定位特徵是「算好的固定 obs」,接法 B 定位網路是「SAC 計算圖的一部分」,可選擇 finetune。

---

## 4. Freeze 還是 Finetune(關鍵決策)

| 策略 | 做法 | 適用 |
|---|---|---|
| **全 freeze** | 定位網路權重凍結,只訓 SAC head | Stage B 起步,最穩、最快 |
| **小 lr finetune** | 定位網路用 ×0.1 學習率跟著動 | Stage C(DR)階段,讓眼睛適應新分布 |
| **全 finetune** | 定位網路與 SAC 同學習率 | ⚠️ 不建議,會洗掉預訓練的好特徵 |

**建議路線**:Stage B 全 freeze(定位當固定的眼睛)→ Stage C(DR)解凍、小 lr finetune(讓眼睛適應隨機化後的聲學)。

實作上 freeze:
```python
for p in localization_net.parameters():
    p.requires_grad = False
```

---

## 5. 維度連動的契約風險(務必注意)

定位網路輸出維度由 `config.py` 的 `n_azimuth_bins`(72)決定。
**一旦接進 SAC 的 obs space,這個 72 就被鎖死了** —— 改它會破壞 domain.md §4 契約。

所以接 SAC 前必須先定案:
- `n_azimuth_bins`:72(5°)還是更粗/更細?
- 要不要加俯仰角(目前垂直 2 麥未用,若加 → obs 多一個 `source_elev`)?
- 用 logits(72)還是隱藏向量(128)?

**這些定了再寫進 domain.md §4.1,然後才動代碼。**

---

## 6. 落地步驟(按順序)

1. 定位網路訓到滿意(命中率穩定) → 存 `best_eval.pt`
2. 決定 §3 接法、§4 freeze 策略、§5 維度
3. **更新 domain.md §4.1**:加 `source_dir` 到 obs space(breaking change,commit 標 `[breaking]`)
4. 在 `envs/mujoco_dobot_env.py` 的 `_build_observation()` 裡:
   - 讀 audio → 階段1 DSP → 載入凍結的定位網路 → 得 source_dir → 放進 obs
5. SAC 正常訓練(BC → SAC → DR),定位網路當眼睛
6. `RealDobotEnv` 對應實作:audio 來源換成真實麥陣列,定位網路同一份權重,介面不變

---

## 7. 一個提醒

定位網路現在是在**自由場理想聲學**下訓練的(命中率 85%)。
接 SAC 前,若已接上 pyroomacoustics 真實聲學,要用**真實聲學下訓練的定位權重**,
否則「眼睛」在仿真裡看得準、一進 DR/真機就瞎。
眼睛的好壞直接決定 SAC 的上限。
