"""定位關卡設定 · single source of truth(本關卡用)

所有「仿真假設、待硬體定案」的數值集中在這裡。
硬體選型確定後,只改這個檔,不動 env / 網路。

對應 domain.md §4.1(audio obs)、§5.3(DR 範圍)、§12(硬體待決)。
注意:本檔是「定位預訓練關卡」專用,不覆寫 domain.md 主契約的 action space。
"""

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """聲學與麥克風陣列(仿真假設,待硬體定案)。

    收音條件需求(從模型可學性反推,與品牌無關):
      - fs 192kHz:為相位精度,非為聽更高頻(40kHz @ 192k 一週期 ~4.8 採樣點)
      - 雙間距幾何:近對(<半波長 4.3mm)解相位模糊 + 遠對(2-3cm)求精度
      - ≥6 麥:水平 4 + 垂直 2,才能解俯仰與前後鏡像
      - 硬前提(不滿足方法不成立):全通道共用時鐘同步、增益固定或聯動
    """
    fs: int = 192_000             # 為相位精度
    n_mics: int = 6               # 改這裡會連動 obs shape
    win_ms: float = 10.0          # 觀測窗長度 ms
    # 雙間距幾何:水平 [0,4,12,28]mm(近對解模糊+遠對求精度)+ 垂直 2 麥
    mic_layout: tuple = (
        (0.000,  0.000, 0.0),     # mic0 參考
        (0.004,  0.000, 0.0),     # mic1 近對(<4.3mm 半波長,解模糊)
        (0.012,  0.000, 0.0),     # mic2 中基線
        (0.028,  0.000, 0.0),     # mic3 遠對(求精度)
        (0.000,  0.012, 0.0),     # mic4 垂直(俯仰維度)
        (0.000, -0.012, 0.0),     # mic5 垂直
    )

    @property
    def n_samples(self) -> int:
        return int(self.fs * self.win_ms / 1000.0)  # 192k * 0.01 = 1920


@dataclass
class SourceDR:
    """方塊發聲的 domain randomization 範圍。

    核心原則:隨機「現實不可預料的因素」,但保留超聲窄帶隔離優勢。
    可聽噪聲(< 20kHz)仍會被加入,但階段1 帶通會濾掉。
    """
    freq_hz_range: tuple = (38_000.0, 42_000.0)   # 中心頻率漂移(換能器個體差異)
    signal_types: tuple = ("cw", "chirp", "pulse_train")  # 隨機抽:連續/啁啾/脈衝串
    snr_db_range: tuple = (5.0, 30.0)             # 超聲頻段內 SNR
    amplitude_range: tuple = (0.3, 1.0)           # 發聲強度
    chirp_bw_hz: float = 4_000.0                  # chirp 帶寬(中心 ± 2k)
    audible_noise_db_range: tuple = (0.0, 40.0)   # 可聽噪聲(會被帶通濾掉,測魯棒性)


@dataclass
class BandpassConfig:
    """階段1 帶通濾波(固定 DSP,無學習)。"""
    low_hz: float = 35_000.0      # 比最低發聲頻率再低一點
    high_hz: float = 45_000.0     # 比最高發聲頻率再高一點
    order: int = 4                # Butterworth 階數


@dataclass
class TaskConfig:
    """關卡與驗證設定。"""
    # 方塊在工作空間內隨機(對齊 build_scene:半徑 ~0.32m,方塊半邊 0.02)
    workspace_r_range: tuple = (0.12, 0.30)   # 距基座水平距離
    workspace_z_range: tuple = (-0.10, 0.10)  # 高度
    # 輸出表示:方位角熱圖(先只做水平方位,最穩)
    n_azimuth_bins: int = 72                  # 360° / 72 = 5° 解析度
    # 命中判定
    hit_threshold_deg: float = 5.0            # 預測方向與真值夾角 < 5° 算命中
    seed: int = 0                             # Instructions.md §5:預設不隨機


@dataclass
class LocalizationConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    source_dr: SourceDR = field(default_factory=SourceDR)
    bandpass: BandpassConfig = field(default_factory=BandpassConfig)
    task: TaskConfig = field(default_factory=TaskConfig)


DEFAULT = LocalizationConfig()
