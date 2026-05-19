"""patch mujoco_dobot_env.py 的 reset(),修方塊掉虛空 + 加大隨機範圍

執行:
    python ~/dobot_project/tools/patch_env_reset.py
"""

from pathlib import Path
import re

env_path = Path.home() / "dobot_project" / "envs" / "mujoco_dobot_env.py"
text = env_path.read_text()

# 新版 reset 方法
new_reset = '''    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # 重置 MuJoCo 狀態
        mujoco.mj_resetData(self.model, self.data)

        # 方塊位置隨機(較大範圍,在工作空間內)
        # 工作半徑 ~32 cm,方塊放半徑 [0.15, 0.28] 環形區域,角度 ±60°
        radius = rng.uniform(0.15, 0.28)
        angle = rng.uniform(-1.05, 1.05)  # ±60° in rad
        bx = radius * float(np.cos(angle))
        by = radius * float(np.sin(angle))

        # freejoint 是 7 維:xyz(3) + quat(w,x,y,z)(4)
        # 必須完整設定,否則殘留無效 quat 會讓物理爆炸 → 方塊飛走
        addr = self._block_freejoint_addr
        self.data.qpos[addr:addr+3] = [bx, by, BLOCK_INITIAL_Z]
        self.data.qpos[addr+3:addr+7] = [1.0, 0.0, 0.0, 0.0]  # quat = identity

        # 順便把方塊 velocity 清零(freejoint 在 qvel 也有 6 維)
        # freejoint 在 qvel 的位置:從對應 dof addr 開始 6 維
        block_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "test_block_freejoint")
        vel_addr = self.model.jnt_dofadr[block_jid]
        self.data.qvel[vel_addr:vel_addr+6] = 0.0

        # 重置 ctrl 與 actuator
        self.data.ctrl[:] = 0.0

        # forward 算好 xpos / xquat
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self._last_action[:] = 0

        return self._build_obs(), {"block_xy": (bx, by)}
'''

# 用 regex 抓舊版 reset(從 "def reset" 到下一個同層方法 "    def " 為止)
pattern = re.compile(
    r'    def reset\(self, seed:.*?(?=    def step\(self)',
    re.DOTALL
)
matches = pattern.findall(text)

if not matches:
    print("❌ 找不到舊版 reset 方法")
    exit(1)

new_text = pattern.sub(new_reset + "\n", text, count=1)

if new_text == text:
    print("⚠️  沒改到任何東西")
else:
    env_path.write_text(new_text)
    print("✅ 已替換 reset() 方法")
    print(f"   檔案: {env_path}")
    print()
    print("修了兩件事:")
    print("  1. 方塊 freejoint 7 維完整重置(xyz + quat + vel)→ 不再炸開飛走")
    print("  2. 方塊位置範圍:radius ∈ [0.15, 0.28] m, angle ∈ ±60° → 更隨機")
