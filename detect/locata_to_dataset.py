"""
locata_to_dataset.py

從 LOCATA dataset 提取：
- phase feature
- amplitude ratio
- DOA label

輸出：
    train_locata.npz

相容：
    train_gpu.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import stft
from tqdm import tqdm

from config import DEFAULT


SPEED_OF_SOUND = 343.0


# ============================================================
# 工具
# ============================================================

def az_to_bin(az_deg: float, n_bins: int) -> int:
    az_deg %= 360
    return int(az_deg / 360 * n_bins)


# ============================================================
# GCC-PHAT
# ============================================================

def gcc_phat(sig1, sig2):

    n = len(sig1) + len(sig2)

    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)

    R = SIG1 * np.conj(SIG2)

    R /= np.abs(R) + 1e-8

    cc = np.fft.irfft(R, n=n)

    shift = np.argmax(np.abs(cc))

    return shift


# ============================================================
# feature extraction
# ============================================================

def extract_feature(audio):

    """
    audio:
        (n_samples, n_mics)
    """

    n_mics = audio.shape[1]

    feat = []

    ref = audio[:, 0]

    for i in range(1, n_mics):

        mic = audio[:, i]

        # ==========================================
        # STFT
        # ==========================================

        _, _, Z1 = stft(ref, nperseg=256)

        _, _, Z2 = stft(mic, nperseg=256)

        # ==========================================
        # phase difference
        # ==========================================

        phase_diff = np.angle(Z1 * np.conj(Z2))

        mean_phase = np.mean(phase_diff)

        feat.append(np.sin(mean_phase))
        feat.append(np.cos(mean_phase))

        # ==========================================
        # amplitude ratio
        # ==========================================

        amp1 = np.mean(np.abs(Z1))
        amp2 = np.mean(np.abs(Z2))

        ratio = np.log((amp2 + 1e-6) / (amp1 + 1e-6))

        feat.append(ratio)

    return np.array(feat, dtype=np.float32)


# ============================================================
# metadata
# ============================================================

def load_gt_dummy():

    """
    這裡簡化。

    真正 LOCATA:
        要讀 metadata csv
    """

    return np.random.uniform(0, 360)


# ============================================================
# main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--locata", type=str, required=True)

    ap.add_argument("--out", type=str, required=True)

    args = ap.parse_args()

    cfg = DEFAULT

    wavs = list(Path(args.locata).rglob("*.wav"))

    print(f"Found {len(wavs)} wav files")

    feats = []
    labels = []

    for wav_path in tqdm(wavs):

        try:

            audio, sr = sf.read(wav_path)

            if audio.ndim == 1:
                continue

            # 至少 6 mic
            if audio.shape[1] < 6:
                continue

            # 取前 6 mic
            audio = audio[:, :6]

            # resample
            if sr != cfg.audio.fs:

                audio = librosa.resample(
                    audio.T,
                    orig_sr=sr,
                    target_sr=cfg.audio.fs
                ).T

            # 切 frame
            frame_size = 1024

            hop = 512

            for start in range(
                0,
                len(audio) - frame_size,
                hop
            ):

                frame = audio[
                    start:start+frame_size
                ]

                feat = extract_feature(frame)

                az = load_gt_dummy()

                label = az_to_bin(
                    az,
                    cfg.task.n_azimuth_bins
                )

                feats.append(feat)
                labels.append(label)

        except Exception as e:

            print(f"skip {wav_path}: {e}")

    feats = np.array(feats, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        out,
        feats=feats,
        labels=labels
    )

    print("\nDone.")
    print(f"Samples: {len(feats)}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()