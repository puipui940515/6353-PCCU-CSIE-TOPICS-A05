"""Perception · 聲學定位感知前端(SAC 與 detect 的橋接層)

職責:把「detect/ 訓練好的定位網路」包成 env 能直接用的感知模組。
env 不需要知道 detect 內部結構,只呼叫 localize() 拿 source_azimuth / source_range。

放置位置:dobot_project/envs/perception.py

資料流(對應 domain.md §4、requirement_localization_to_sac.md):
    mic 世界座標 + 方塊聲源座標
      → signal_processing(合成接收 → 帶通 → 萃取特徵)
      → 凍結的 LocalizationNet(detect 訓練的權重)
      → softmax
      → source_azimuth(72) [+ source_range(4) 若權重含距離 head]

★ 自動能力偵測(本檔核心設計):
  detect 現在的權重只有方位 head(舊、自由場)。本層載入時會偵測權重裡
  有沒有距離 head:
    - 沒有 → 只輸出 source_azimuth,has_range=False(env 自動不放 source_range)
    - 有   → 同時輸出 source_range,has_range=True
  未來 detect 加距離 head + pyroomacoustics 重訓後,只換權重檔,本層與 env 都不用改。
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import torch

# detect/ 內部模組互相用「同目錄 import」(from config import / from signal_processing import)。
# 從 envs 外部以 detect.xxx 載入時,需讓 detect/ 也在 sys.path 上,否則內部 import 找不到。
# 這樣做不需改動 detect/ 任何訓練程式碼(維持你原本能跑的狀態)。
_DETECT_DIR = Path(__file__).resolve().parent.parent / "detect"
if str(_DETECT_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECT_DIR))

from detect.config import LocalizationConfig, DEFAULT
from detect.model import LocalizationNet, build_net
from detect.signal_processing import (
    synthesize_reception, bandpass, extract_features_v2 as extract_features)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """數值穩定 softmax,回傳 float32 機率分布(和為 1)。"""
    z = logits - np.max(logits)
    e = np.exp(z)
    return (e / (np.sum(e) + 1e-12)).astype(np.float32)


class AcousticPerception:
    """聲學定位感知前端。載入凍結的定位網路,提供 localize()。

    Args:
        cfg: LocalizationConfig(預設 DEFAULT)
        weights_path: detect 訓練好的權重檔(.pt)。None 則用隨機初始化(僅供 smoke test)
        device: "cpu" / "cuda"。定位網路很小,env 在 CPU 平行跑時建議 "cpu"
    """

    def __init__(
        self,
        cfg: LocalizationConfig = DEFAULT,
        weights_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_azimuth_bins = cfg.task.n_azimuth_bins
        self.n_range_bins = cfg.range_head.n_range_bins

        # 建網路(目前 detect.model 只有方位 head)
        self.net = build_net(cfg).to(self.device)
        self.has_range = False   # 預設無距離能力,載入時偵測

        if weights_path is not None:
            self._load_and_detect(weights_path)
        else:
            print("⚠️  AcousticPerception:未提供權重,使用隨機初始化(僅 smoke test 用)")

        # 凍結:當眼睛,不參與 SAC 反向傳播(requirement §2、docs_sac_integration §4)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

    def _load_and_detect(self, weights_path: str) -> None:
        """載入權重並偵測是否含距離 head。"""
        path = Path(weights_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"定位權重不存在: {path}")

        ckpt = torch.load(str(path), map_location=self.device)
        # train_gpu.py 存的格式是 {"model": state_dict, ...};也容許直接是 state_dict
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

        # 偵測距離 head:detect 加 head 後,state_dict 會多 range head 的 key
        self.has_range = any("range" in k for k in state.keys())

        if self.has_range:
            # 權重含距離 head → 用 with_range=True 重建網路(原本 build_net(cfg) 是單 head,
            # 不重建會導致 forward 回單 tensor、localize 的雙 head unpack 失敗)。
            self.net = build_net(self.cfg, with_range=True).to(self.device)
            self.net.load_state_dict(state, strict=True)
            print(f"✅ 載入定位權重(含距離 head): {path.name}")
        else:
            # 只有方位 head 的舊權重 → 維持單 head 網路
            self.net.load_state_dict(state, strict=True)
            print(f"✅ 載入定位權重(僅方位 head,舊版): {path.name}")
            print("   → source_range 不可用,env 將自動只給 source_azimuth")
            print("   ⚠️  此權重為自由場版,接 pyroomacoustics/DR 前精度會劣化"
                  "(docs_sac_integration §7)")

        # 重建後要重新凍結(__init__ 的凍結在重建前執行,對舊 net 無效)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def localize(
        self,
        mic_world: np.ndarray,
        source_xyz: np.ndarray,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """從一次收音推論聲源方位(與距離)。

        Args:
            mic_world: (n_mics, 3) 麥克風世界座標(env 從 mic site 取)
            source_xyz: (3,) 方塊聲源世界座標(只當 pyroomacoustics 的發聲點,
                        不是餵給 policy 的真值)
            rng: 該 env 的亂數產生器(DR 用,確保可重現)

        Returns:
            dict,一定有 "source_azimuth"(72,);
            若權重含距離 head 才有 "source_range"(4,)。
        """
        # 1) 合成多通道接收(pyroomacoustics 渲染;mic_world 為陣列當前世界座標)
        signals, _meta = synthesize_reception(
            self.cfg, np.asarray(source_xyz), rng, mic_world=np.asarray(mic_world))
        # 2) 帶通 → 特徵
        feats = extract_features(self.cfg, bandpass(self.cfg, signals))
        feat_t = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)  # (1, feat_dim)

        # 3) 推論
        out = self.net(feat_t)
        result: dict[str, np.ndarray] = {}

        if self.has_range:
            # 未來雙 head:net 回傳 (azimuth_logits, range_logits) 或含兩段的 tensor
            azimuth_logits, range_logits = out
            result["source_azimuth"] = _softmax_np(
                azimuth_logits.squeeze(0).cpu().numpy())
            result["source_range"] = _softmax_np(
                range_logits.squeeze(0).cpu().numpy())
        else:
            # 當前:單 head,只有方位
            result["source_azimuth"] = _softmax_np(out.squeeze(0).cpu().numpy())

        return result


if __name__ == "__main__":
    # smoke test:不載權重(隨機),驗證 localize 介面與輸出 shape
    print("=== AcousticPerception smoke test ===")
    cfg = DEFAULT
    percep = AcousticPerception(cfg, weights_path=None, device="cpu")
    rng = np.random.default_rng(0)
    mic_world = np.asarray(cfg.audio.mic_layout) + np.array([0.2, 0.0, 0.1])  # 假裝陣列在某處
    source = np.array([0.25, 0.05, 0.0])
    out = percep.localize(mic_world, source, rng)
    print(f"has_range       = {percep.has_range}")
    print(f"source_azimuth  shape = {out['source_azimuth'].shape}, 和 = {out['source_azimuth'].sum():.3f}")
    if "source_range" in out:
        print(f"source_range    shape = {out['source_range'].shape}, 和 = {out['source_range'].sum():.3f}")
    else:
        print("source_range    = (此權重無距離 head,不輸出)")
    print("✅ perception 介面正常")