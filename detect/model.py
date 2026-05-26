"""階段2:定位神經網路(學習)

輸入:階段1 萃取的結構化特徵(相位差 + 能量比)
輸出:
  - 方位 head:n_azimuth_bins 維 logits(必有)
  - 距離 head:n_range_bins 維 logits(可選,requirement §5)
  - 高度 head:n_height_bins 維 logits(可選)

設計原則(對齊先前討論):
  - 模型要小、可部署:目標 < 200k 參數
  - 物理特徵已算好,網路只學「怎麼用」,不學「從波形提特徵」
  - 輸出熱圖(分類)而非回歸座標 → 訓練更穩;距離也用分類,糊時分布自然變平

雙 head 設計(2026-05-23,requirement_localization_to_sac.md §5):
  - 共用 backbone,方位 head + 距離 head 各接一層 Linear
  - 距離 head 幾乎不增參數(backbone hidden → 4)
  - 向後相容:with_range=False 時行為與舊單 head 完全一致(forward 回單 tensor),
    舊權重可照常載入;with_range=True 時 forward 回 (azimuth_logits, range_logits)
  - state_dict 是否含 "range_head.*" key,正是 perception.py 自動偵測 has_range 的依據

對齊 domain.md §8(audio encoder 方向)但更輕量。
"""

from __future__ import annotations
import torch
import torch.nn as nn

from config import LocalizationConfig, DEFAULT


class LocalizationNet(nn.Module):
    """超聲方塊方位(+ 距離)定位網路(輕量 MLP,共用 backbone 雙 head)。

    Args:
        feat_dim: 階段1特徵維度 = (n_mics-1) * 3(相位差 sin/cos + 能量比)
        n_bins: 方位角 bin 數
        hidden: 隱藏層寬度
        n_range_bins: 距離 bin 數;None 或 0 → 不建距離 head(退回單 head 舊行為)

    forward 回傳:
        無距離 head:   azimuth_logits           (B, n_bins)
        有距離 head:   (azimuth_logits, range_logits)
                       azimuth (B, n_bins) / range (B, n_range_bins)
    """

    def __init__(
        self,
        feat_dim: int,
        n_bins: int,
        hidden: int = 128,
        n_range_bins: int | None = None,
        n_height_bins: int | None = None,
    ) -> None:
        super().__init__()
        # 共用 backbone(原本 net 的前段,去掉最後的輸出層)
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        # 方位 head(必有)
        self.azimuth_head = nn.Linear(hidden, n_bins)
        # 距離 head(可選)
        self.has_range = bool(n_range_bins)
        if self.has_range:
            self.range_head = nn.Linear(hidden, int(n_range_bins))
        self.has_height = bool(n_height_bins)
        if self.has_height:
            self.height_head = nn.Linear(hidden, int(n_height_bins))

    def forward(self, feat: torch.Tensor):
        """回傳 logits。

        無距離 head → azimuth_logits (B, n_bins)。
        有距離/高度 head → tuple,順序為 azimuth, range?, height?。
        訓練用 CrossEntropy,推論用 argmax/softmax。
        """
        h = self.backbone(feat)
        azimuth_logits = self.azimuth_head(h)
        outs = [azimuth_logits]
        if self.has_range:
            outs.append(self.range_head(h))
        if self.has_height:
            outs.append(self.height_head(h))
        if len(outs) == 1:
            return azimuth_logits
        return tuple(outs)

    @torch.no_grad()
    def predict_azimuth(self, feat: torch.Tensor, n_bins: int) -> torch.Tensor:
        """回傳預測方位角(度)。對單/雙 head 都適用(只取方位)。"""
        out = self.forward(feat)
        azimuth_logits = out[0] if isinstance(out, tuple) else out
        bin_idx = torch.argmax(azimuth_logits, dim=-1)
        return bin_idx.float() * (360.0 / n_bins)

    @torch.no_grad()
    def predict_range_bin(self, feat: torch.Tensor) -> torch.Tensor:
        """回傳預測距離 bin(僅在有距離 head 時可用)。"""
        if not self.has_range:
            raise RuntimeError("此網路無距離 head,無法 predict_range_bin")
        out = self.forward(feat)
        range_logits = out[1]
        return torch.argmax(range_logits, dim=-1)

    @torch.no_grad()
    def predict_height_bin(self, feat: torch.Tensor) -> torch.Tensor:
        """回傳預測高度 bin(僅在有高度 head 時可用)。"""
        if not self.has_height:
            raise RuntimeError("此網路無高度 head,無法 predict_height_bin")
        out = self.forward(feat)
        height_logits = out[-1]
        return torch.argmax(height_logits, dim=-1)


def build_net(
    cfg: LocalizationConfig,
    with_range: bool = False,
    with_height: bool = False,
) -> LocalizationNet:
    """建立定位網路。

    Args:
        cfg: 設定
        with_range: True 則建距離 head(維度取自 cfg.range_head.n_range_bins)。
                    預設 False = 維持舊單 head 行為(向後相容,舊權重可載)。
    """
    feat_dim = (cfg.audio.n_mics - 1) * 3
    n_range_bins = cfg.range_head.n_range_bins if with_range else None
    n_height_bins = cfg.height_head.n_height_bins if with_height else None
    return LocalizationNet(
        feat_dim, cfg.task.n_azimuth_bins,
        n_range_bins=n_range_bins,
        n_height_bins=n_height_bins,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    cfg = DEFAULT
    feat_dim = (cfg.audio.n_mics - 1) * 3
    dummy = torch.randn(8, feat_dim)

    print("=== 單 head(向後相容,舊行為)===")
    net1 = build_net(cfg, with_range=False)
    out1 = net1(dummy)
    print(f"feat_dim     = {feat_dim}")
    print(f"output       = {tuple(out1.shape)}  (期望 (8, {cfg.task.n_azimuth_bins}))")
    print(f"參數量       = {count_params(net1):,}")
    assert not isinstance(out1, tuple), "單 head 應回單 tensor"
    assert not net1.has_range
    assert count_params(net1) < 200_000

    print("\n=== 三 head(方位 + 距離 + 高度)===")
    net2 = build_net(cfg, with_range=True, with_height=True)
    az, rg, ht = net2(dummy)
    print(f"azimuth out  = {tuple(az.shape)}  (期望 (8, {cfg.task.n_azimuth_bins}))")
    print(f"range out    = {tuple(rg.shape)}  (期望 (8, {cfg.range_head.n_range_bins}))")
    print(f"height out   = {tuple(ht.shape)}  (期望 (8, {cfg.height_head.n_height_bins}))")
    print(f"參數量       = {count_params(net2):,}  (預算 < 200,000)")
    assert net2.has_range
    assert net2.has_height
    assert az.shape == (8, cfg.task.n_azimuth_bins)
    assert rg.shape == (8, cfg.range_head.n_range_bins)
    assert ht.shape == (8, cfg.height_head.n_height_bins)
    assert count_params(net2) < 200_000, "超出參數預算!"

    # 驗證 state_dict key:perception.py 靠 'range' 字串偵測 has_range
    keys = list(net2.state_dict().keys())
    has_range_key = any("range" in k for k in keys)
    has_height_key = any("height" in k for k in keys)
    print(f"\nstate_dict 含 'range' key = {has_range_key}  (perception 自動偵測依據)")
    print(f"state_dict 含 'height' key = {has_height_key}")
    assert has_range_key, "距離 head 的 key 必須含 'range',否則 perception 偵測不到"
    assert has_height_key, "高度 head 的 key 必須含 'height',否則 perception 偵測不到"
    # 單 head 不應有 range key
    assert not any("range" in k for k in net1.state_dict().keys())
    print(f"雙 head 比單 head 多 {count_params(net2)-count_params(net1)} 參數(距離 head)")
    print("✅ 參數量在預算內,雙 head 介面與 perception 契約一致")
