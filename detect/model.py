"""階段2:定位神經網路(學習)

輸入:階段1 萃取的結構化特徵(相位差 + 能量比)+ 可選原始頻譜
輸出:方位角熱圖(n_azimuth_bins 維 logits)

設計原則(對齊先前討論):
  - 模型要小、可部署:目標 < 200k 參數
  - 物理特徵已算好,網路只學「怎麼用」,不學「從波形提特徵」
  - 輸出熱圖而非回歸座標 → 訓練更穩

對齊 domain.md §8(audio encoder 方向)但更輕量。
"""

from __future__ import annotations
import torch
import torch.nn as nn

from config import LocalizationConfig, DEFAULT


class LocalizationNet(nn.Module):
    """超聲方塊方位定位網路(輕量 MLP)。

    Args:
        feat_dim: 階段1特徵維度 = (n_mics-1) * 2
        n_bins: 方位角 bin 數
        hidden: 隱藏層寬度
    """

    def __init__(self, feat_dim: int, n_bins: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """回傳 logits (B, n_bins)。訓練用 CrossEntropy,推論用 argmax/softmax。"""
        return self.net(feat)

    @torch.no_grad()
    def predict_azimuth(self, feat: torch.Tensor, n_bins: int) -> torch.Tensor:
        """回傳預測方位角(度)。"""
        logits = self.forward(feat)
        bin_idx = torch.argmax(logits, dim=-1)
        return bin_idx.float() * (360.0 / n_bins)


def build_net(cfg: LocalizationConfig) -> LocalizationNet:
    feat_dim = (cfg.audio.n_mics - 1) * 3
    return LocalizationNet(feat_dim, cfg.task.n_azimuth_bins)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    cfg = DEFAULT
    net = build_net(cfg)
    feat_dim = (cfg.audio.n_mics - 1) * 3
    dummy = torch.randn(8, feat_dim)
    out = net(dummy)
    print(f"feat_dim       = {feat_dim}")
    print(f"output shape   = {tuple(out.shape)}  (期望 (8, {cfg.task.n_azimuth_bins}))")
    print(f"參數量         = {count_params(net):,}  (預算 < 200,000)")
    assert count_params(net) < 200_000, "超出參數預算!"
    print("✅ 參數量在預算內")
