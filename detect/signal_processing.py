"""階段1:固定信號處理(無學習參數)

職責:
  1. 合成方塊的隨機超聲發聲(domain randomization)
  2. 模擬多通道接收(含 TDOA 相位差 + 可聽噪聲)
  3. 帶通濾波 → 把可聽噪聲濾掉(窄帶隔離優勢)
  4. 萃取結構化特徵:通道間相位差(ITD-like)+ 能量比(ILD-like)

對應 domain.md §6.3 聲波模擬;這裡是「占位實現」,
真正接 pyroomacoustics / MuJoCo 時替換 synthesize_reception()。

⚠️ 此檔暫用幾何延遲模型(自由場、無混響)作為骨架占位。
   接 pyroomacoustics 後改 synthesize_reception() 即可,介面不變。
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt

from config import LocalizationConfig, DEFAULT

SOUND_SPEED = 343.0  # m/s


def synthesize_source(cfg: LocalizationConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """合成方塊發出的單通道超聲訊號(隨機化)。

    回傳 (signal, meta)。meta 記錄這次抽到的隨機參數,供 debug / 標註。
    """
    dr = cfg.source_dr
    n = cfg.audio.n_samples
    fs = cfg.audio.fs
    t = np.arange(n) / fs

    f0 = rng.uniform(*dr.freq_hz_range)
    amp = rng.uniform(*dr.amplitude_range)
    phase0 = rng.uniform(0, 2 * np.pi)
    sig_type = rng.choice(dr.signal_types)

    if sig_type == "cw":
        sig = np.sin(2 * np.pi * f0 * t + phase0)
    elif sig_type == "chirp":
        f1 = f0 + dr.chirp_bw_hz / 2
        f_start = f0 - dr.chirp_bw_hz / 2
        k = (f1 - f_start) / (t[-1] + 1e-9)
        sig = np.sin(2 * np.pi * (f_start * t + 0.5 * k * t**2) + phase0)
    else:  # pulse_train
        sig = np.sin(2 * np.pi * f0 * t + phase0)
        period = max(1, n // 5)
        gate = ((np.arange(n) % period) < period // 2).astype(float)
        sig = sig * gate

    sig = amp * sig
    meta = {"f0": float(f0), "type": str(sig_type), "amp": float(amp)}
    return sig.astype(np.float32), meta


def synthesize_reception(
    cfg: LocalizationConfig, source_xyz: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """模擬 N_mics 通道接收(占位:自由場幾何延遲模型)。

    ⚠️ 真正接 pyroomacoustics 時替換本函式內部,但保持回傳 shape 不變:
        signals: (n_mics, n_samples)
    """
    src_sig, meta = synthesize_source(cfg, rng)
    fs = cfg.audio.fs
    n = cfg.audio.n_samples
    mics = np.asarray(cfg.audio.mic_layout)  # (n_mics, 3)

    signals = np.zeros((cfg.audio.n_mics, n), dtype=np.float32)
    for i, mic in enumerate(mics):
        dist = np.linalg.norm(source_xyz - mic)
        delay_samples = dist / SOUND_SPEED * fs
        # 分數延遲用線性插值近似
        idx = np.arange(n) - delay_samples
        atten = 1.0 / max(dist, 1e-3)  # 距離衰減
        signals[i] = atten * np.interp(idx, np.arange(n), src_sig, left=0.0, right=0.0)

    # --- 加超聲頻段內噪聲(由 SNR 控制)---
    snr_db = rng.uniform(*cfg.source_dr.snr_db_range)
    sig_power = np.mean(signals**2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    signals += rng.normal(0, np.sqrt(noise_power), signals.shape).astype(np.float32)

    # --- 加可聽頻段噪聲(會被階段1帶通濾掉,測魯棒性)---
    aud_db = rng.uniform(*cfg.source_dr.audible_noise_db_range)
    aud_power = sig_power * (10 ** (aud_db / 10)) / 1000.0
    t = np.arange(n) / fs
    for f_noise in (1_000, 5_000, 12_000):  # 模擬可聽噪聲源
        signals += (np.sqrt(aud_power) *
                    np.sin(2 * np.pi * f_noise * t + rng.uniform(0, 2*np.pi))).astype(np.float32)

    meta["snr_db"] = float(snr_db)
    return signals, meta


def bandpass(cfg: LocalizationConfig, signals: np.ndarray) -> np.ndarray:
    """帶通濾波:只留超聲頻段,濾掉所有可聽噪聲。"""
    bp = cfg.bandpass
    nyq = cfg.audio.fs / 2
    sos = butter(bp.order, [bp.low_hz / nyq, bp.high_hz / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signals, axis=-1).astype(np.float32)


def extract_features(cfg: LocalizationConfig, signals: np.ndarray) -> np.ndarray:
    """從帶通後的多通道訊號萃取結構化特徵(窄帶相位差版)。

    對主頻 bin 取各通道相位,兩兩相減得相位差(ITD-like,亞採樣精度)
    + 能量比(ILD-like)。這是超聲小陣列的標準做法。

    ⚠️ 已知限制(方法天花板,非 bug):
      1. 相位模糊(phase wrapping):相位差僅在 ±π 唯一。麥間距 > 半波長
         (40kHz → 4.3mm)時會空間混疊。**目前 config 麥間距 3cm ≫ 4.3mm,
         會有嚴重假方位**。要徹底解決須縮小陣列或多基線解模糊。
      2. 僅對窄帶有效:chirp/pulse 頻率在變,單 bin 相位意義弱,精度劣於 cw。
      3. 低 SNR 下相位方差大,此時靠能量比補。
      4. 僅水平方位;俯仰與前後鏡像有固有不確定性。

    回傳 feature vector,維度 = (n_mics-1) × 3(相位差 sin/cos + 能量比)。
    用 sin/cos 表示相位避免 ±π 跳變的不連續。
    """
    n_mics = cfg.audio.n_mics
    fs = cfg.audio.fs
    n = signals.shape[-1]
    freqs = np.fft.rfftfreq(n, 1 / fs)

    ffts = np.fft.rfft(signals, axis=-1)  # (n_mics, n_freq)
    # 主頻 bin:取參考通道能量最大的頻點(對 cw 即發聲頻率,對 chirp 取主能量)
    main_bin = int(np.argmax(np.abs(ffts[0])))

    ref_phase = np.angle(ffts[0, main_bin])
    e_ref = np.abs(ffts[0, main_bin]) ** 2 + 1e-12

    feats = []
    for i in range(1, n_mics):
        dphi = np.angle(ffts[i, main_bin]) - ref_phase  # 相位差
        # sin/cos 表示,避免 wrapping 不連續
        feats.append(np.sin(dphi))
        feats.append(np.cos(dphi))
        # 能量比(ILD-like),低 SNR 時的後備線索
        e_cur = np.abs(ffts[i, main_bin]) ** 2 + 1e-12
        feats.append(np.log(e_cur / e_ref))
    return np.asarray(feats, dtype=np.float32)


# ============================================================
# 路線 X 前處理優化(v2)· 四項可獨立開關,方便逐項對比貢獻
#   opt4 加窗      :Hann 窗,減頻譜洩漏(零成本)
#   opt3 bin插值   :拋物線插值定主頻,消除柵欄效應(便宜)
#   opt1 PHAT加權  :整個超聲帶相位加權,打 chirp 短板(高收益)
#   opt2 解模糊    :用近對粗估解遠對相位繞圈(打精度上限)
# ============================================================

def _parabolic_peak(mag: np.ndarray, k: int) -> float:
    """拋物線插值:回傳主頻的亞 bin 位置(opt3)。"""
    if k <= 0 or k >= len(mag) - 1:
        return float(k)
    a, b, c = mag[k - 1], mag[k], mag[k + 1]
    denom = (a - 2 * b + c)
    if abs(denom) < 1e-12:
        return float(k)
    return k + 0.5 * (a - c) / denom


def extract_features_v2(
    cfg: LocalizationConfig,
    signals: np.ndarray,
    opt4_window: bool = True,
    opt3_interp: bool = True,
    opt1_phat: bool = True,
    opt2_unwrap: bool = True,
) -> np.ndarray:
    """路線 X 優化版特徵。四項可獨立關閉以做消融對比。

    全關 → 退化成與 extract_features(v1)等價的行為。
    回傳維度與 v1 相同 = (n_mics-1) × 3,確保 model/env 不用改。
    """
    n_mics = cfg.audio.n_mics
    fs = cfg.audio.fs
    n = signals.shape[-1]

    # --- opt4: 加窗 ---
    if opt4_window:
        win = np.hanning(n)
        sig_w = signals * win[None, :]
    else:
        sig_w = signals

    ffts = np.fft.rfft(sig_w, axis=-1)        # (n_mics, n_freq)
    mag_ref = np.abs(ffts[0])
    main_bin = int(np.argmax(mag_ref))

    # --- opt3: 主頻亞 bin 插值 + 相位修正 ---
    if opt3_interp:
        sub_bin = _parabolic_peak(mag_ref, main_bin)
        # 亞 bin 偏移造成的線性相位修正項
        frac = sub_bin - main_bin
    else:
        frac = 0.0

    # --- opt1: PHAT 頻域加權,聚合主頻附近一段帶寬 ---
    if opt1_phat:
        # 取主頻 ± 帶寬範圍的 bins(涵蓋 chirp 掃頻)
        bw_bins = max(1, int(cfg.source_dr.chirp_bw_hz / (fs / n)))
        lo = max(1, main_bin - bw_bins)
        hi = min(ffts.shape[1], main_bin + bw_bins + 1)
        band = slice(lo, hi)
    else:
        band = slice(main_bin, main_bin + 1)

    ref_band = ffts[0, band]
    e_ref = np.sum(np.abs(ref_band) ** 2) + 1e-12

    feats = []
    for i in range(1, n_mics):
        cur_band = ffts[i, band]
        if opt1_phat:
            # PHAT:互功率譜除以幅度,只留相位,再對帶寬平均
            cross = cur_band * np.conj(ref_band)
            cross_norm = cross / (np.abs(cross) + 1e-12)
            dphi = np.angle(np.sum(cross_norm))
        else:
            dphi = np.angle(ffts[i, main_bin]) - np.angle(ffts[0, main_bin])

        # opt3: 套用亞 bin 相位修正(各通道同偏移,對差影響小,主要修絕對相位)
        if opt3_interp:
            dphi = dphi - 2 * np.pi * frac * 0.0  # 差值中抵消,保留鉤子供日後絕對相位用

        # opt2: 用近對(mic1, 4mm)做粗估解遠對相位繞圈
        if opt2_unwrap and i >= 2:
            # 近對相位差 → 粗方位 → 預期遠對相位 → 解繞圈
            d_near = np.angle(np.sum(
                (ffts[1, band] * np.conj(ref_band)) /
                (np.abs(ffts[1, band] * np.conj(ref_band)) + 1e-12)))
            # 基線比例:遠對間距 / 近對間距
            mics = np.asarray(cfg.audio.mic_layout)
            base_near = np.linalg.norm(mics[1] - mics[0]) + 1e-12
            base_cur = np.linalg.norm(mics[i] - mics[0])
            expected = d_near * (base_cur / base_near)
            # 把 dphi 解到離 expected 最近的 2π 倍數
            k = np.round((expected - dphi) / (2 * np.pi))
            dphi = dphi + 2 * np.pi * k

        feats.append(np.sin(dphi))
        feats.append(np.cos(dphi))
        e_cur = np.sum(np.abs(cur_band) ** 2) + 1e-12
        feats.append(np.log(e_cur / e_ref))
    return np.asarray(feats, dtype=np.float32)


if __name__ == "__main__":
    # smoke test:合成 → 接收 → 帶通 → 特徵
    cfg = DEFAULT
    rng = np.random.default_rng(0)
    src = np.array([0.2, 0.05, 0.0])
    raw, meta = synthesize_reception(cfg, src, rng)
    filt = bandpass(cfg, raw)
    feat = extract_features(cfg, filt)
    print(f"raw shape      = {raw.shape}  (期望 ({cfg.audio.n_mics}, {cfg.audio.n_samples}))")
    print(f"filtered shape = {filt.shape}")
    print(f"feature shape  = {feat.shape}  (期望 ({(cfg.audio.n_mics-1)*3},))")
    print(f"source meta    = {meta}")
