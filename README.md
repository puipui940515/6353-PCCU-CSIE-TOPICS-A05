# Dobot Magician 聲波震動定位與抓取專題

訓練一支機械臂在**無視覺**前提下,利用聲波與震動感測完成物件定位與抓取。

- 機械臂:Dobot Magician(4 軸)
- 模擬:MuJoCo + pyroomacoustics
- RL:BC 暖機 → SAC 精修 → Domain Randomization 強化
- 介面:ROS 2 Humble(模擬與真機共用)
- 運動規劃:MoveIt2

## 文件入口

| 文件 | 用途 |
|---|---|
| [`docs/agents/domain.md`](docs/agents/domain.md) | 領域知識、架構契約、目錄結構 |
| [`docs/deployment.md`](docs/deployment.md) | 從零環境到真機部署的步驟 |
| [`Instructions.md`](Instructions.md) | 給 AI 助手的工作規約 |

## 快速啟動

```bash
# 1. 建立 venv(只做一次)
cd ~/dobot_project
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

# 2. 之後每次開 terminal
source ~/dobot_project/setup_env.sh
```

## 目錄結構

詳見 [`docs/agents/domain.md` §10](docs/agents/domain.md)。
