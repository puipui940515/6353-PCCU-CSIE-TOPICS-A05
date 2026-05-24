"""
sensor_model_demo.py

完整可執行版
--------------------------------------------------

功能:
- LocalizationNet 接入
- pyroom 聲學模擬
- 真實 mic 幾何
- 聲波可視化
- AI 自動預測方向
- 粒子發射
- 左鍵新增障礙物
- 右鍵刪除障礙物
- R 重新生成目標
- SPACE 重新發射粒子
- mic 視覺化
- mic pair 視覺化

安裝:
    pip install pygame torch numpy scipy pyroomacoustics

執行:
    python sensor_model_demo.py \
        --model runs/gpu_run/checkpoints/best_eval.pt
"""

from __future__ import annotations

import argparse
import math
import random

import numpy as np
import pygame
import pyroomacoustics as pra
import torch

from scipy.signal import chirp

from config import DEFAULT
from model import build_net


# ============================================================
# 基本設定
# ============================================================

WIDTH = 1200
HEIGHT = 900

CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float32)

BG = (18, 18, 22)

WHITE = (240, 240, 240)
RED = (255, 80, 80)
GREEN = (80, 255, 120)
BLUE = (80, 180, 255)
YELLOW = (255, 220, 120)

TARGET_SIZE = 30

PARTICLE_SPEED = 10
PARTICLE_LIFE = 120

PIXELS_PER_METER = 12000


# ============================================================
# 真實 mic layout
# ============================================================

MIC_LAYOUT = np.array([
    [0.000,  0.000],
    [0.004,  0.000],
    [0.012,  0.000],
    [0.028,  0.000],
    [0.000,  0.012],
    [0.000, -0.012],
], dtype=np.float32)

MIC_PAIRS = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
]


# ============================================================
# 工具
# ============================================================

def angle_to_vec(deg):

    rad = math.radians(deg)

    return np.array([
        math.cos(rad),
        math.sin(rad)
    ], dtype=np.float32)


def mic_to_screen(mic_xy):

    return (
        CENTER +
        mic_xy * PIXELS_PER_METER
    )


# ============================================================
# 障礙物
# ============================================================

class Obstacle:

    def __init__(self, rect):

        self.rect = rect

    def draw(self, screen):

        pygame.draw.rect(
            screen,
            (110, 110, 110),
            self.rect
        )


# ============================================================
# 聲波
# ============================================================

class SoundWave:

    def __init__(self, pos):

        self.pos = np.array(pos, dtype=np.float32)

        self.radius = 1

        self.life = 80

    def update(self):

        self.radius += 7

        self.life -= 1

    def draw(self, screen):

        if self.life <= 0:
            return

        pygame.draw.circle(
            screen,
            (255, 100, 100),
            self.pos.astype(int),
            int(self.radius),
            2
        )


# ============================================================
# 粒子
# ============================================================

class Particle:

    def __init__(self, pos, direction):

        self.pos = np.array(pos, dtype=np.float32)

        self.vel = direction * PARTICLE_SPEED

        self.life = PARTICLE_LIFE

    def update(self):

        self.pos += self.vel

        self.life -= 1

    def draw(self, screen):

        pygame.draw.circle(
            screen,
            BLUE,
            self.pos.astype(int),
            3
        )


# ============================================================
# LocalizationNet Wrapper
# ============================================================

class SensorModel:

    def __init__(self, ckpt_path):

        self.cfg = DEFAULT

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.net = build_net(
            self.cfg,
            with_range=True
        ).to(self.device)

        ckpt = torch.load(
            ckpt_path,
            map_location=self.device
        )

        self.net.load_state_dict(ckpt["model"])

        self.net.eval()

        print("模型載入成功")

    # --------------------------------------------------------

    def simulate_audio(self, src_pos):

        room = pra.ShoeBox(
            [10, 8],
            fs=48000,
            max_order=2,
        )

        mic_world = MIC_LAYOUT + np.array([5.0, 4.0])

        room.add_microphone_array(
            mic_world.T
        )

        duration = 0.02

        t = np.linspace(
            0,
            duration,
            int(48000 * duration)
        )

        signal = chirp(
            t,
            4000,
            t[-1],
            8000
        )

        room.add_source(
            src_pos,
            signal=signal
        )

        room.simulate()

        audio = room.mic_array.signals.T

        return audio

    # --------------------------------------------------------

    def extract_feature(self, audio):

        feat = []

        ref = audio[:, 0]

        for i in range(1, 6):

            mic = audio[:, i]

            phase = np.angle(
                np.sum(ref * np.conj(mic))
            )

            feat.append(np.sin(phase))
            feat.append(np.cos(phase))

            amp = np.log(
                np.std(mic) /
                (np.std(ref) + 1e-6)
            )

            feat.append(amp)

        return np.array(
            feat,
            dtype=np.float32
        )

    # --------------------------------------------------------

    @torch.no_grad()
    def predict_angle(self, target_pos):

        room_pos = np.array([
            5 + (target_pos[0] - WIDTH/2)/100,
            4 + (target_pos[1] - HEIGHT/2)/100,
        ])

        audio = self.simulate_audio(room_pos)

        feat = self.extract_feature(audio)

        x = torch.tensor(
            feat,
            device=self.device
        ).unsqueeze(0)

        out = self.net(x)

        az_logits = out[0] if isinstance(out, tuple) else out

        pred_bin = torch.argmax(
            az_logits,
            dim=-1
        ).item()

        pred_angle = pred_bin * (
            360 / self.cfg.task.n_azimuth_bins
        )

        return pred_angle


# ============================================================
# 主程式
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        type=str,
        required=True
    )

    args = ap.parse_args()

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "Acoustic Localization Sandbox"
    )

    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 28)

    sensor = SensorModel(args.model)

    particles = []

    waves = []

    obstacles = []

    prediction_ready = False
    prediction_timer = 0
    delay_frames = 0

    pred_angle = 0

    # ========================================================
    # 發射粒子
    # ========================================================

    def spawn_particles():

        pred_dir = angle_to_vec(pred_angle)

        for _ in range(150):

            noise = np.random.normal(
                0,
                0.08,
                size=2
            )

            direction = pred_dir + noise

            direction /= np.linalg.norm(direction)

            particles.append(
                Particle(
                    CENTER.copy(),
                    direction
                )
            )

    # ========================================================
    # 新目標
    # ========================================================

    def reset_target():

        nonlocal prediction_ready
        nonlocal prediction_timer
        nonlocal delay_frames
        nonlocal pred_angle

        particles.clear()
        waves.clear()

        prediction_ready = False
        prediction_timer = 0

        angle = random.uniform(0, 360)

        dist = random.uniform(180, 320)

        target_pos = (
            CENTER +
            angle_to_vec(angle) * dist
        )

        waves.append(
            SoundWave(target_pos.copy())
        )

        # ================================================
        # propagation delay
        # ================================================

        distance_m = dist / 100

        travel_time = distance_m / 343

        delay_frames = max(
            1,
            int(travel_time * 60)
        )

        pred_angle = sensor.predict_angle(
            target_pos
        )

        return angle, target_pos

    true_angle, target_pos = reset_target()

    # ========================================================
    # loop
    # ========================================================

    running = True

    while running:

        dt = clock.tick(60)

        # ----------------------------------------------------
        # event
        # ----------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:

                    true_angle, target_pos = (
                        reset_target()
                    )

                elif event.key == pygame.K_SPACE:

                    particles.clear()

                    spawn_particles()

            # =================================================
            # mouse
            # =================================================

            if event.type == pygame.MOUSEBUTTONDOWN:

                mx, my = pygame.mouse.get_pos()

                # 左鍵新增 obstacle
                if event.button == 1:

                    rect = pygame.Rect(
                        mx - 25,
                        my - 25,
                        50,
                        50
                    )

                    obstacles.append(
                        Obstacle(rect)
                    )

                # 右鍵刪除 obstacle
                elif event.button == 3:

                    for obs in obstacles[:]:

                        if obs.rect.collidepoint(mx, my):

                            obstacles.remove(obs)

        # ----------------------------------------------------
        # propagation delay
        # ----------------------------------------------------

        if not prediction_ready:

            prediction_timer += 1

            if prediction_timer >= delay_frames:

                spawn_particles()

                prediction_ready = True

        # ----------------------------------------------------
        # update
        # ----------------------------------------------------

        for p in particles:
            p.update()

        particles[:] = [
            p for p in particles
            if p.life > 0
        ]

        for w in waves:
            w.update()

        waves[:] = [
            w for w in waves
            if w.life > 0
        ]

        # ----------------------------------------------------
        # draw
        # ----------------------------------------------------

        screen.fill(BG)

        # obstacle
        for obs in obstacles:
            obs.draw(screen)

        # sound waves
        for w in waves:
            w.draw(screen)

        # ====================================================
        # mic pair lines
        # ====================================================

        for a, b in MIC_PAIRS:

            pa = mic_to_screen(MIC_LAYOUT[a])
            pb = mic_to_screen(MIC_LAYOUT[b])

            pygame.draw.line(
                screen,
                (70, 70, 70),
                pa.astype(int),
                pb.astype(int),
                1
            )

        # ====================================================
        # microphones
        # ====================================================

        for i, mic in enumerate(MIC_LAYOUT):

            pos = mic_to_screen(mic)

            pygame.draw.circle(
                screen,
                WHITE,
                pos.astype(int),
                7
            )

            txt = font.render(
                f"{i}",
                True,
                WHITE
            )

            screen.blit(
                txt,
                (pos[0] + 10, pos[1] - 10)
            )

        # ====================================================
        # target
        # ====================================================

        rect = pygame.Rect(
            int(target_pos[0] - TARGET_SIZE/2),
            int(target_pos[1] - TARGET_SIZE/2),
            TARGET_SIZE,
            TARGET_SIZE
        )

        pygame.draw.rect(
            screen,
            RED,
            rect
        )

        # ====================================================
        # real direction
        # ====================================================

        true_end = (
            CENTER +
            angle_to_vec(true_angle) * 170
        )

        pygame.draw.line(
            screen,
            GREEN,
            CENTER.astype(int),
            true_end.astype(int),
            3
        )

        # ====================================================
        # predicted direction
        # ====================================================

        pred_end = (
            CENTER +
            angle_to_vec(pred_angle) * 200
        )

        pygame.draw.line(
            screen,
            BLUE,
            CENTER.astype(int),
            pred_end.astype(int),
            3
        )

        # ====================================================
        # particles
        # ====================================================

        for p in particles:
            p.draw(screen)

        # ====================================================
        # UI
        # ====================================================

        err = abs(
            (pred_angle - true_angle + 180) % 360 - 180
        )

        texts = [
            f"True Angle: {true_angle:.1f}",
            f"Pred Angle: {pred_angle:.1f}",
            f"Error: {err:.1f}",
            f"Particles: {len(particles)}",
            f"Prediction Delay: {delay_frames} frames",
            "",
            "GREEN = Real Direction",
            "BLUE = AI Prediction",
            "",
            "LEFT CLICK = Add Obstacle",
            "RIGHT CLICK = Remove Obstacle",
            "R = New Target",
            "SPACE = Respawn Particles",
        ]

        for i, t in enumerate(texts):

            surf = font.render(
                t,
                True,
                YELLOW
            )

            screen.blit(
                surf,
                (20, 20 + i * 30)
            )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()