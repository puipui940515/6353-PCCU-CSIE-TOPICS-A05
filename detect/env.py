"""定位關卡 Env + 「發射命中」驗證器

關卡邏輯:
  1. 方塊隨機放在工作空間 → 真值方位角已知
  2. 合成多通道超聲接收 → 階段1特徵
  3. 模型預測方位 → 「朝該方位發射」
  4. 命中判定:預測方位 vs 真值方位 夾角 < threshold

兩種驗證:
  - 主指標:直接讀方塊真值算角度誤差(準、你說的備用其實該當主)
  - Demo:「發射命中」二元判定(可視化用)

⚠️ 這是獨立的定位預訓練關卡,不覆寫 domain.md §4.2 主 action space。
"""

from __future__ import annotations
import numpy as np

from config import LocalizationConfig, DEFAULT
import signal_processing as sp


def sample_block_position(cfg: LocalizationConfig, rng: np.random.Generator) -> np.ndarray:
    """在工作空間內隨機放方塊,回傳 (x, y, z)。"""
    t = cfg.task
    r = rng.uniform(*t.workspace_r_range)
    az = rng.uniform(0, 2 * np.pi)
    z = rng.uniform(*t.workspace_z_range)
    return np.array([r * np.cos(az), r * np.sin(az), z], dtype=np.float32)


def _yaw_matrix(yaw_rad: float) -> np.ndarray:
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def sample_self_pose(cfg: LocalizationConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """取樣 mic array 的世界座標位姿,回傳 (translation, rotation, yaw_deg)。"""
    dr = cfg.source_dr
    xy = rng.uniform(*dr.self_translation_xy_range, size=2)
    z = rng.uniform(*dr.self_translation_z_range)
    yaw_deg = rng.uniform(*dr.self_yaw_range_deg)
    return np.array([xy[0], xy[1], z], dtype=np.float64), _yaw_matrix(np.deg2rad(yaw_deg)), float(yaw_deg)


def sample_obstacles(cfg: LocalizationConfig, rng: np.random.Generator) -> dict:
    """以 channel attenuation 近似障礙物遮擋與材質係數。"""
    dr = cfg.source_dr
    has_obstacle = rng.random() < dr.obstacle_prob
    channel_gains = np.ones(cfg.audio.n_mics, dtype=np.float32)
    count = 0
    global_gain = 1.0
    if has_obstacle:
        lo, hi = dr.obstacle_count_range
        count = int(rng.integers(lo, hi + 1))
        global_gain = float(rng.uniform(*dr.obstacle_global_attenuation_range))
        channel_gains *= global_gain
        if count:
            blocked = rng.choice(cfg.audio.n_mics, size=min(count, cfg.audio.n_mics), replace=False)
            for idx in blocked:
                channel_gains[int(idx)] *= float(rng.uniform(*dr.obstacle_attenuation_range))
    return {
        "count": count,
        "global_gain": global_gain,
        "channel_gains": channel_gains,
    }


def true_azimuth_deg(source_xyz: np.ndarray) -> float:
    """方塊相對陣列中心的真值方位角(度,0-360)。"""
    az = np.degrees(np.arctan2(source_xyz[1], source_xyz[0]))
    return az % 360.0


def angular_error_deg(pred_deg: float, true_deg: float) -> float:
    """環形角度誤差(度),範圍 [0, 180]。"""
    d = abs(pred_deg - true_deg) % 360.0
    return min(d, 360.0 - d)


class LocalizationEnv:
    """定位關卡。一個 step = 放一次方塊 + 收音 + 出特徵。"""

    def __init__(self, cfg: LocalizationConfig = DEFAULT, seed: int = 0,
                 use_v2: bool = False, v2_opts: dict | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._last_source = None
        self._last_meta = None
        self.use_v2 = use_v2
        self.v2_opts = v2_opts or {}

    def sample(self) -> tuple[np.ndarray, float, dict]:
        """產生一個樣本。回傳 (feature, true_azimuth_deg, meta)。

        meta 內含 source_xyz 與 source_range_m(陣列中心到聲源距離),
        距離 label 由 gen_dataset 透過 range_to_bin() 從 source_range_m 算。
        """
        src_local = sample_block_position(self.cfg, self.rng)
        translation, rotation, yaw_deg = sample_self_pose(self.cfg, self.rng)
        mic_local = np.asarray(self.cfg.audio.mic_layout, dtype=np.float64)
        mic_world = mic_local @ rotation.T + translation
        src_world = src_local @ rotation.T + translation
        obstacles = sample_obstacles(self.cfg, self.rng)
        raw, meta = sp.synthesize_reception(
            self.cfg, src_world, self.rng,
            mic_world=mic_world,
            obstacle_gains=obstacles["channel_gains"],
        )
        filt = sp.bandpass(self.cfg, raw)
        if self.use_v2:
            feat = sp.extract_features_v2(self.cfg, filt, **self.v2_opts)
        else:
            feat = sp.extract_features(self.cfg, filt)
        self._last_source = src_local
        self._last_meta = meta
        true_az = true_azimuth_deg(src_local)
        meta["source_xyz"] = src_local.tolist()
        meta["source_world_xyz"] = src_world.tolist()
        meta["source_range_m"] = float(np.linalg.norm(src_local))
        meta["source_height_m"] = float(src_local[2])
        meta["self_yaw_deg"] = yaw_deg
        meta["self_translation"] = translation.tolist()
        meta["obstacle_count"] = obstacles["count"]
        meta["obstacle_global_gain"] = obstacles["global_gain"]
        meta["obstacle_channel_gains"] = obstacles["channel_gains"].tolist()
        return feat, true_az, meta

    def is_hit(self, pred_az_deg: float, true_az_deg: float) -> bool:
        """「發射命中」判定。"""
        return angular_error_deg(pred_az_deg, true_az_deg) < self.cfg.task.hit_threshold_deg


def az_to_bin(az_deg: float, n_bins: int) -> int:
    """真值方位角 → bin index(訓練 label 用)。"""
    return int((az_deg % 360.0) / (360.0 / n_bins)) % n_bins


def range_to_bin(range_m: float, bin_edges_m: tuple[float, ...]) -> int:
    """陣列到聲源距離(m)→ 距離 bin index(距離 head label 用)。

    bin_edges_m 為遞增切點(來自 config.range_head.bin_edges_m,如 (0.08,0.16,0.24))。
    n_bins = len(edges)+1。例:0.05→bin0(很近)、0.20→bin2(遠)。
    用 np.searchsorted:落在第幾個區間即 bin index。
    """
    return int(np.searchsorted(np.asarray(bin_edges_m), range_m, side="right"))


def height_to_bin(height_m: float, bin_edges_m: tuple[float, ...]) -> int:
    """聲源相對 array frame 的 z 高度(m) → 高度 bin index。"""
    return int(np.searchsorted(np.asarray(bin_edges_m), height_m, side="right"))


if __name__ == "__main__":
    env = LocalizationEnv(seed=0)
    feat, true_az, meta = env.sample()
    print(f"feature shape  = {feat.shape}")
    print(f"true azimuth   = {true_az:.1f}°")
    print(f"source xyz     = {meta['source_xyz']}")
    print(f"signal type    = {meta['type']}, f0={meta['f0']:.0f}Hz, SNR={meta['snr_db']:.1f}dB")
    # 假裝預測差 3° → 應命中
    print(f"hit (pred+3°)  = {env.is_hit(true_az + 3, true_az)}  (期望 True)")
    print(f"hit (pred+10°) = {env.is_hit(true_az + 10, true_az)}  (期望 False)")
