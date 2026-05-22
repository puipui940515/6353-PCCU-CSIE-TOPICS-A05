# 6 麥克風陣列改造說明 · Dobot Magician

> 設計說明,非實作代碼。涉及 domain.md §3.2 / §4.1 / §7,
> 正式改前須先更新 domain.md。
>
> 背景:定位方法需要 6 麥雙間距陣列(config.py 的 mic_layout),
> 但 Dobot Magician 原機沒有麥克風,須改造。

---

## 1. 兩個層面的「改造」(別混淆)

| 層面 | 改什麼 | 現在能做? |
|---|---|---|
| **仿真層(MJCF)** | 在模型末端加 6 個 mic site + 對應 sensor | ✅ 現在能做 |
| **真機層(硬體)** | 實體 3D 列印支架 + 裝 6 顆麥克風 + 走線 | Phase 3 才需要 |

你現在純仿真,**先做仿真層**;真機層是日後 Phase 3,這裡給設計方向。

---

## 2. 仿真層:在 build_scene.py 加麥克風

`build_scene.py` 已經有加 sensor 的模式(actuators / equality 那段)。
麥克風在 MuJoCo 裡用 **site + 不直接有 mic sensor**,做法是:

### 2.1 在末端 body 上加 6 個 site
末端 body(gripper 那節)加 6 個 site,座標 = config.py 的 `mic_layout`:
```python
# 在末端 body 下加(對齊 config.mic_layout)
end_body.add_site(name="mic0", pos=[0.000,  0.000, 0.0])
end_body.add_site(name="mic1", pos=[0.004,  0.000, 0.0])  # 近對
end_body.add_site(name="mic2", pos=[0.012,  0.000, 0.0])
end_body.add_site(name="mic3", pos=[0.028,  0.000, 0.0])  # 遠對
end_body.add_site(name="mic4", pos=[0.000,  0.012, 0.0])  # 垂直
end_body.add_site(name="mic5", pos=[0.000, -0.012, 0.0])
```

### 2.2 site 的用途
MuJoCo 本身**不模擬聲波傳播**(這是 pyroomacoustics 的事)。
site 的作用是:**提供 6 個麥克風在世界座標系的精確位置**,
讓 pyroomacoustics 知道「6 個接收點在空間哪裡」,從而算 TDOA。

資料流:
```
MuJoCo 算出 6 個 mic site 的世界座標
        ↓
餵給 pyroomacoustics(連同方塊聲源位置)
        ↓
pyroomacoustics 渲染 6 通道音訊
        ↓
階段1 DSP → 定位網路
```

**所以 site 是 MuJoCo 與 pyroomacoustics 的橋。** 這也呼應 domain.md §6.3。

---

## 3. 幾何約束(改造的硬限制)

config.py 的雙間距幾何不是隨便定的,改造時必須守住:

1. **近對間距 4mm < 半波長 4.3mm**:解相位模糊的關鍵,不能放大
2. **遠對間距 28mm**:給角分辨率,不能縮太小
3. **垂直 2 麥**:為日後解俯仰,水平任務暫不影響
4. **剛性**:6 麥相對位置必須固定。任何晃動 → 幾何變 → 相位差全錯
5. **隨末端運動**:陣列裝在末端,會跟著手臂動。**定位算的是「相對陣列」的方位**,
   要轉成世界座標需結合 TCP pose(domain.md §3.1 的 /dobot_TCP)

---

## 4. 真機層改造方向(Phase 3,先了解)

| 項目 | 設計方向 |
|---|---|
| 支架 | 3D 列印,固定在末端 gripper 上方,保證 6 麥剛性與精確間距 |
| 間距精度 | 4mm 近對要求高,支架公差需 < 0.5mm,否則相位偏 |
| 麥克風 | 超聲頻段、≥96kHz(前面討論的收音條件),品牌未定 |
| 走線 | 6 條線隨手臂運動,需理線避免拉扯;接頭固定在某一臂節 |
| 同步採集 | **最關鍵**:6 通道必須共用時鐘同步採樣(前面強調過) |
| 重量 | Dobot Magician 負載有限(payload ~500g),陣列+支架要輕 |

⚠️ **負載提醒**:Dobot Magician 工作半徑 320mm、重複精度 ±0.2mm,
但 payload 小。6 麥 + 支架 + 線材的重量若超過負載,會影響精度甚至卡住。
真機改造前要先秤重、查 payload 規格。

---

## 5. 對 domain.md 的影響(改前先更新這些)

- **§3.2**:`/microphone_array` 的 shape = (N_mics, N_samples) → 鎖 N_mics=6
- **§4.1**:obs space 的 audio Box(N_mics, N_samples) → N_mics=6, N_samples 隨 fs/win 定
- **§7**:真機端麥克風陣列選型 → 補上「6 麥雙間距、超聲頻段、同步採集」規格
- **§6.5**:新增 sim-to-real gap →「pyroomacoustics 超聲頻段模擬精度」「陣列實體間距公差」

---

## 6. 落地步驟

1. **仿真層先做**:build_scene.py 加 6 mic site → 驗證世界座標正確
2. 接 pyroomacoustics:用 site 座標當接收點,渲染 6 通道
3. 定位網路在「真實聲學」下重訓(取代現在的自由場版)
4. 真機層:等仿真跑通,再做 3D 列印支架 + 實體麥克風(Phase 3)
5. 每步都先更新 domain.md 對應章節,再改代碼(Instructions.md §4)
