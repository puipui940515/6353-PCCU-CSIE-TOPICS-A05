"""自訂 SAC policy 元件

加入 LayerNorm 以穩定 SAC critic(對應 reward 曲線崩盤問題)。
不使用 Dropout(會讓 critic Q 值估計不穩)。
不使用 ResNet(網路太淺,殘差連接無顯著效果)。

對應 docs/agents/domain.md §8。
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
)


class LayerNormCombinedExtractor(CombinedExtractor):
    """Dict obs 用,先 flatten + concat,再過 LayerNorm 穩定輸入分布。

    SAC critic 對輸入尺度敏感,LayerNorm 能緩解 Q 值爆炸與崩盤。
    """

    def __init__(self, observation_space: gym.spaces.Dict, **kwargs: Any) -> None:
        super().__init__(observation_space, **kwargs)
        self.input_ln = nn.LayerNorm(self.features_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        features = super().forward(observations)
        return self.input_ln(features)


def build_mlp(
    input_dim: int,
    hidden_layers: list[int],
    activation_cls: type[nn.Module],
    use_layer_norm: bool = True,
) -> nn.Sequential:
    """產生帶 LayerNorm 的 MLP(SB3 policy_kwargs.net_arch 不支援 LN,改用這個)"""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation_cls())
        prev = h
    return nn.Sequential(*layers)
