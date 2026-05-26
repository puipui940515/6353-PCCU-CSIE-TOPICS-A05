"""定位網路 + Observation 的單一設定來源(single source of truth)

涵蓋範圍(對齊使用者拍板的界線):
  ✅ 聲學定位:audio / source_dr / bandpass / task / model
  ✅ Observation space 可調參數:obs(bin 邊界、各項範圍、開關)
  ❌ SAC 訓練超參數 → 留在 configs/sac_stage*.yaml(curriculum resume 機制依賴它,不搬)

設計原則:
  - 全 dataclass + type hints(對齊 Instructions.md §3)
  - 「改了只需調參」與「改了破壞契約」兩類分開,後者在 __post_init__ 加護欄
  - 一處改、全專案生效;signal_processing / model / env 都讀同一份

⚠️ 重建說明:原 config.py 未上傳,本檔由各檔案對 cfg.* 的實際用法反推重建。
   既有欄位(audio/task/source_dr/bandpass)的「名稱與結構」與原版一致;
   標 `# ⟵核對` 的數值是推定值,請對照你手邊原檔確認。

對應文件:domain.md §4.1(obs 契約)、requirement_localization_to_sac.md(設計依據)。
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ============================================================
# 既有區塊(重建自現有用法,結構不可改名 — 會破壞 signal_processing/model)
# ============================================================

@dataclass
class AudioConfig:
    """收音與陣列幾何。"""
    fs: int = 192_000                 # ⟵核對 超聲頻段,須 > 2×最高發聲頻率
    n_mics: int = 6                   # 鎖 6(雙間距陣列,docs_mic_array_mod.md)。改它破壞契約
    n_samples: int = 1024             # ⟵核對 單窗取樣數
    # 6 麥座標(取自 build_scene.py 的 mic site,單位 m,相對末端 body)
    # 近對 4mm 解相位模糊、遠對 28mm 給角分辨、垂直對為日後俯仰
    mic_layout: list = field(default_factory=lambda: [
        [0.000,  0.000, 0.0],   # mic0
        [0.004,  0.000, 0.0],   # mic1  近對(4mm < 半波長 4.3mm)
        [0.012,  0.000, 0.0],   # mic2
        [0.028,  0.000, 0.0],   # mic3  遠對
        [0.000,  0.012, 0.0],   # mic4  垂直
        [0.000, -0.012, 0.0],   # mic5  垂直
    ])


@dataclass
class BandpassConfig:
    """帶通濾波:只留超聲頻段,濾掉可聽噪聲(signal_processing.bandpass)。"""
    order: int = 4
    low_hz: float = 35_000.0          # ⟵核對
    high_hz: float = 45_000.0         # ⟵核對 須 < fs/2


@dataclass
class SourceDRConfig:
    """聲源域隨機化(signal_processing.synthesize_source/reception)。

    ⚠️ amplitude_range / snr 是距離 head 的天敵:振幅是測距的主要線索,
       這裡隨機化得越兇,source_range 在 DR 下命中率掉得越多(requirement §6 的 go/no-go)。
    """
    freq_hz_range: tuple[float, float] = (38_000.0, 42_000.0)   # ⟵核對
    amplitude_range: tuple[float, float] = (0.5, 1.0)            # ⟵核對
    signal_types: tuple[str, ...] = ("cw", "chirp", "pulse_train")
    chirp_bw_hz: float = 4_000.0                                 # ⟵核對
    snr_db_range: tuple[float, float] = (0.0, 30.0)
    audible_noise_db_range: tuple[float, float] = (-20.0, 20.0)  # ⟵核對
    # 混合渲染比例:每筆樣本以此機率走 pyroomacoustics 真實聲學,其餘走自由場(快)。
    # 0.25 = 3:1(自由場:混響)。混響佔比越低生成越快,但距離 head 的真實聲學線索越少,
    # go/no-go 可能更難過 → 距離命中率接近隨機時,提高此值是第一個該試的調整。
    pyroom_ratio: float = 0.25
    # 障礙物遮擋:以每個 mic channel 的隨機衰減近似繞射/遮蔽。
    obstacle_prob: float = 0.35
    obstacle_count_range: tuple[int, int] = (0, 3)
    obstacle_attenuation_range: tuple[float, float] = (0.25, 0.85)
    obstacle_global_attenuation_range: tuple[float, float] = (0.65, 1.0)
    # 自身移動/旋轉:資料生成時把 mic array 和聲源一起轉到世界座標,
    # label 仍用相對 array frame,用來訓練模型對本體 pose 不敏感。
    self_translation_xy_range: tuple[float, float] = (-0.08, 0.08)
    self_translation_z_range: tuple[float, float] = (-0.02, 0.08)
    self_yaw_range_deg: tuple[float, float] = (-180.0, 180.0)


@dataclass
class TaskConfig:
    """定位任務定義。"""
    n_azimuth_bins: int = 72          # 方位 5°/bin。改它破壞 obs 契約(domain §4.1)
    hit_threshold_deg: float = 10.0   # train_gpu.evaluate 判方位命中的容差
    # 工作空間範圍(env.sample_block_position 撒方塊用;相對陣列中心)
    # ⟵核對 對齊機械臂工作半徑 0.32m;距離 bin 邊界(0.08/0.16/0.24)應落在此範圍內
    workspace_r_range: tuple[float, float] = (0.05, 0.30)   # ⟵核對 陣列中心到方塊水平距離
    workspace_z_range: tuple[float, float] = (0.00, 0.18)  # 聲源高度範圍,高度 head label 用


# ============================================================
# 新增區塊:距離 head(model.py 擴充用)
# ============================================================

@dataclass
class RangeHeadConfig:
    """source_range 距離 head 設定(requirement §5)。

    bin 邊界 = 陣列中心到聲源距離(m)的切點,非等距、近密遠疏。
    n_range_bins = len(bin_edges)+1。改 bin 數會動 obs 維度 → 破壞契約。
    """
    # 4 個 bin 由 3 個切點分隔:很近 <0.08 | 近 0.08-0.16 | 遠 0.16-0.24 | 很遠 >0.24
    bin_edges_m: tuple[float, ...] = (0.08, 0.16, 0.24)
    range_ce_weight: float = 1.0      # 距離 CE 在總 loss 的權重(方位 CE 權重固定 1.0)

    @property
    def n_range_bins(self) -> int:
        return len(self.bin_edges_m) + 1


@dataclass
class HeightHeadConfig:
    """source_height 高度 head 設定。

    bin 邊界 = 聲源相對 mic array frame 的 z 高度(m)。目前模型輸出分類,
    與距離 head 一樣用 CrossEntropy 訓練,比直接回歸更穩。
    """
    bin_edges_m: tuple[float, ...] = (0.04, 0.08, 0.12, 0.16)
    height_ce_weight: float = 0.8

    @property
    def n_height_bins(self) -> int:
        return len(self.bin_edges_m) + 1


# ============================================================
# 新增區塊:Observation space 可調參數(本次需求核心)
# ============================================================

@dataclass
class ObsRanges:
    """各 obs 項的 Box 上下界。改這些「只需調參」,不破壞維度契約。

    用途:base_dobot_env.py 建 observation_space 時讀這裡,不要再 hardcode。
    機率分布項(source_*)固定 [0,1],不開放調(softmax 輸出本就如此)。
    """
    joint_position: tuple[float, float] = (-3.14159, 3.14159)
    joint_velocity: tuple[float, float] = (-10.0, 10.0)
    tcp_pose: tuple[float, float] = (-1.0, 1.0)
    gripper_state: tuple[float, float] = (0.0, 0.02)
    base_to_tcp_dist: tuple[float, float] = (0.0, 0.4)   # 工作半徑 0.32 + 餘量


@dataclass
class ObsConfig:
    """Observation space 的單一設定來源(domain.md §4.1 契約的程式對應)。

    維度鎖死區(改 = breaking change,須同步 domain.md §4 + RealDobotEnv):
      - source_azimuth 維度 = task.n_azimuth_bins
      - source_range   維度 = range_head.n_range_bins
    可自由調區(改 = 只需調參,不破壞維度):
      - ranges 內各上下界
      - 各 enable_* 開關(關掉某項 → 該 key 不進 obs;關了仍算 breaking,SAC 要重訓)
      - 聲學觀測效能參數(perception_update_every / env_pyroom_ratio,只影響速度/泛化,不動維度)
    """
    ranges: ObsRanges = field(default_factory=ObsRanges)

    # 開關:預設為「移除上帝視角後」的目標 obs(requirement §4)
    enable_proprio: bool = True          # joint_position/velocity, tcp_pose, gripper_state
    enable_base_to_tcp_dist: bool = True # 本體距離錨點(requirement §4.3)
    enable_source_azimuth: bool = True   # 聲學方位(取代 block_pose 的方向資訊)
    enable_source_range: bool = True     # 聲學遠近(go/no-go 沒過就設 False,退回只給方位)
    enable_source_height: bool = True    # 聲學高度分類(新增,舊權重無此 head 時自動不輸出)

    # ---- SAC 訓練時的聲學觀測效能參數(不影響 obs 維度,只影響速度與泛化)----
    # perception_update_every:每 K 個 env step 才真渲染一次 source_*,中間沿用緩存。
    #   pyroomacoustics 渲染是 env step 的主要瓶頸(單次 ~100ms)。靜止聲源 + 緩慢手臂
    #   運動下,相鄰 step 的 source_* 變化小,降頻幾乎不損資訊卻能快 ~K 倍。
    #   K=10 → 50Hz 控制等於 5Hz 更新定位,對靜止方塊定位足夠。設 1 = 每步都渲染(最慢)。
    perception_update_every: int = 10
    # env_pyroom_ratio:SAC 訓練時,每次「真渲染」以此機率走 pyroomacoustics、其餘走自由場(快)。
    #   獨立於 detect 的 source_dr.pyroom_ratio(訓眼睛用),兩階段比例需求不同故解耦。
    #   不可設 0:SAC 從沒見過混響,Phase 2/真機一上真聲學會掉。建議 0.1-0.2。
    env_pyroom_ratio: float = 0.15

    # 護欄:上帝視角方塊位置。預設 False = 不進 obs(本次需求)。
    # 設 True 會把 block_pose 塞回 obs,僅供 debug 對照,正式訓練務必 False。
    enable_oracle_block_pose: bool = False


# ============================================================
# 頂層:LocalizationConfig(名稱不可改 — model/signal_processing import 它)
# ============================================================

@dataclass
class LocalizationConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    bandpass: BandpassConfig = field(default_factory=BandpassConfig)
    source_dr: SourceDRConfig = field(default_factory=SourceDRConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    range_head: RangeHeadConfig = field(default_factory=RangeHeadConfig)
    height_head: HeightHeadConfig = field(default_factory=HeightHeadConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)

    def __post_init__(self) -> None:
        self._validate()

    # ---- 護欄:載入當下就擋下危險/矛盾設定,不等 runtime ----
    def _validate(self) -> None:
        a, bp, dr = self.audio, self.bandpass, self.source_dr

        # 採樣定理:帶通上限與發聲頻率須 < fs/2
        nyq = a.fs / 2
        assert bp.high_hz < nyq, f"bandpass.high_hz({bp.high_hz}) 須 < fs/2({nyq})"
        assert bp.low_hz < bp.high_hz, "bandpass.low_hz 須 < high_hz"
        assert dr.freq_hz_range[1] < nyq, f"發聲頻率上限須 < fs/2({nyq})"
        assert 0.0 <= dr.pyroom_ratio <= 1.0, \
            f"pyroom_ratio 須在 [0,1],收到 {dr.pyroom_ratio}"
        assert 0.0 <= dr.obstacle_prob <= 1.0, \
            f"obstacle_prob 須在 [0,1],收到 {dr.obstacle_prob}"

        # 陣列幾何與 n_mics 一致
        assert len(a.mic_layout) == a.n_mics, \
            f"mic_layout 有 {len(a.mic_layout)} 顆,但 n_mics={a.n_mics}"

        # 相位模糊提醒(不擋,只是物理事實):近對間距須 < 半波長
        half_wavelength = 343.0 / dr.freq_hz_range[1] / 2
        near_pair = ((a.mic_layout[1][0] - a.mic_layout[0][0]) ** 2 +
                     (a.mic_layout[1][1] - a.mic_layout[0][1]) ** 2) ** 0.5
        if near_pair > half_wavelength:
            print(f"⚠️  config 警告:近對間距 {near_pair*1000:.1f}mm > 半波長 "
                  f"{half_wavelength*1000:.1f}mm,方位會有空間混疊(signal_processing 已知限制1)")

        # 距離 bin 邊界須遞增、落在工作空間內
        edges = self.range_head.bin_edges_m
        assert list(edges) == sorted(edges), f"bin_edges_m 須遞增,收到 {edges}"
        assert all(e > 0 for e in edges), "bin_edges_m 須為正"
        assert edges[-1] < self.obs.ranges.base_to_tcp_dist[1], \
            "最遠 bin 邊界應落在 base_to_tcp_dist 上界內(否則遠近與本體錨點對不上)"

        # 契約一致性:source_range 開了,bin 數要合理
        if self.obs.enable_source_range:
            assert self.range_head.n_range_bins >= 2, "啟用 source_range 至少要 2 個 bin"
        if self.obs.enable_source_height:
            h_edges = self.height_head.bin_edges_m
            assert list(h_edges) == sorted(h_edges), f"height bin 須遞增,收到 {h_edges}"
            assert self.height_head.n_height_bins >= 2, "啟用 source_height 至少要 2 個 bin"

        # SAC env 聲學觀測效能參數護欄
        assert self.obs.perception_update_every >= 1, \
            f"perception_update_every 須 >= 1,收到 {self.obs.perception_update_every}"
        assert 0.0 <= self.obs.env_pyroom_ratio <= 1.0, \
            f"env_pyroom_ratio 須在 [0,1],收到 {self.obs.env_pyroom_ratio}"
        if self.obs.env_pyroom_ratio == 0.0:
            print("⚠️  config 警告:env_pyroom_ratio=0,SAC 訓練全程自由場、從不見混響,"
                  "Phase 2/真機一上 pyroomacoustics 可能掉。建議 ≥ 0.1")

        # 護欄:正式訓練不該開上帝視角
        if self.obs.enable_oracle_block_pose:
            print("🚨 config 警告:enable_oracle_block_pose=True,obs 含上帝視角方塊位置!"
                  "違反『無視覺』,僅供 debug,正式訓練請設 False(requirement §3)")

    # ---- 給 env 用:回傳「這份設定下 obs 該有哪些 key + shape」----
    def obs_spec(self) -> dict[str, int]:
        """回傳 obs 各項維度,base_dobot_env 照這個建 observation_space。

        把「哪些項進 obs、各多大」收斂到 config 一處決定,env 不再 hardcode。
        """
        spec: dict[str, int] = {}
        o = self.obs
        if o.enable_proprio:
            spec["joint_position"] = 4
            spec["joint_velocity"] = 4
            spec["tcp_pose"] = 7
            spec["gripper_state"] = 1
        if o.enable_base_to_tcp_dist:
            spec["base_to_tcp_dist"] = 1
        if o.enable_source_azimuth:
            spec["source_azimuth"] = self.task.n_azimuth_bins
        if o.enable_source_range:
            spec["source_range"] = self.range_head.n_range_bins
        if o.enable_source_height:
            spec["source_height"] = self.height_head.n_height_bins
        if o.enable_oracle_block_pose:
            spec["block_pose"] = 7
        return spec


# 全專案共用的預設實例(名稱不可改 — 多檔 import 它)
DEFAULT = LocalizationConfig()


if __name__ == "__main__":
    cfg = DEFAULT
    print("=== config 自檢 ===")
    print(f"n_mics            = {cfg.audio.n_mics}")
    print(f"n_azimuth_bins    = {cfg.task.n_azimuth_bins}")
    print(f"n_range_bins      = {cfg.range_head.n_range_bins}")
    print(f"n_height_bins     = {cfg.height_head.n_height_bins}")
    print(f"range bin 邊界(m) = {cfg.range_head.bin_edges_m}")
    print(f"obs 各項維度      = {cfg.obs_spec()}")
    total = sum(cfg.obs_spec().values())
    print(f"obs 攤平總維度    = {total}")
    print("✅ 護欄通過(若上方無 🚨/⚠️ 則設定健康)")
