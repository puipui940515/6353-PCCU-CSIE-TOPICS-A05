"""
sensor_model_demo.py

改良版功能:
--------------------------------------------------
- 改善 mic 幾何比例
- 保持相對結構
- 降低 spatial aliasing
- C 切換:
    cw / chirp / pulse_train
- 更直覺 Hz UI
- 更細緻 Hz 調整
- 顯示 kHz
- 更安全的 mic spacing

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
import signal_processing as sp


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

PIXELS_PER_METER = 1000
SIDE_GROUND_Y = HEIGHT - 150
SIDE_ORIGIN_X = WIDTH // 2
MIN_TARGET_HEIGHT_M = 0.00
MAX_TARGET_HEIGHT_M = 0.18

VIEW_MODES = [
    "top",
    "side",
]


# ============================================================
# 訊號設定
# ============================================================

SIGNAL_TYPES = [
    "cw",
    "chirp",
    "pulse_train",
]

signal_mode_idx = 1

# 使用 Hz
START_FREQ = 38000
END_FREQ = 42000

# 調整步進
FREQ_STEP = 250


# ============================================================
# mic layout
# ============================================================
#
# 必須跟 config.py / 訓練資料一致。若 demo 自行縮小陣列，
# 畫面準確率會和 checkpoint 真實評估脫節。
#
# ============================================================

MIC_LAYOUT = np.asarray(DEFAULT.audio.mic_layout, dtype=np.float32)[:, :2]

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

def hz_to_text(hz):

    return f"{hz/1000:.2f} kHz"


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


def side_to_screen(distance_m, height_m):

    return np.array([
        SIDE_ORIGIN_X + distance_m * PIXELS_PER_METER,
        SIDE_GROUND_Y - height_m * PIXELS_PER_METER,
    ], dtype=np.float32)


def range_bin_to_text(cfg, range_bin):

    if range_bin is None:
        return "-"

    edges = list(cfg.range_head.bin_edges_m)

    if range_bin <= 0:
        return f"< {edges[0]:.2f} m"

    if range_bin >= len(edges):
        return f">= {edges[-1]:.2f} m"

    return f"{edges[range_bin - 1]:.2f}-{edges[range_bin]:.2f} m"


def range_bin_midpoint(cfg, range_bin):

    if range_bin is None:
        return None

    edges = list(cfg.range_head.bin_edges_m)

    if range_bin <= 0:
        return edges[0] * 0.5

    if range_bin >= len(edges):
        span = edges[-1] - edges[-2] if len(edges) > 1 else edges[-1]
        return edges[-1] + span * 0.5

    return (edges[range_bin - 1] + edges[range_bin]) * 0.5


def height_bin_to_text(cfg, height_bin):

    if height_bin is None:
        return "-"

    edges = list(cfg.height_head.bin_edges_m)

    if height_bin <= 0:
        return f"< {edges[0]:.2f} m"

    if height_bin >= len(edges):
        return f">= {edges[-1]:.2f} m"

    return f"{edges[height_bin - 1]:.2f}-{edges[height_bin]:.2f} m"


def height_bin_midpoint(cfg, height_bin):

    if height_bin is None:
        return None

    edges = list(cfg.height_head.bin_edges_m)

    if height_bin <= 0:
        return edges[0] * 0.5

    if height_bin >= len(edges):
        span = edges[-1] - edges[-2] if len(edges) > 1 else edges[-1]
        return edges[-1] + span * 0.5

    return (edges[height_bin - 1] + edges[height_bin]) * 0.5


# ============================================================
# 障礙物參數（全域，供 main 迴圈讀寫）
# ============================================================

next_obstacle_attenuation = 0.5   # 0.0(全擋) ~ 1.0(全透)，[ ] 鍵調整
next_obstacle_size        = 50    # 像素邊長，- = 鍵調整


# ============================================================
# 干擾方塊參數（全域）
# ============================================================

distractor_amplitude = 0.5    # 干擾源相對於主聲源的振幅，E/F 鍵調整


# ============================================================
# 障礙物
# ============================================================

class Obstacle:

    def __init__(self, rect, attenuation):

        self.rect = rect
        self.attenuation = attenuation

    def draw(self, screen):

        shade = int(70 + 120 * self.attenuation)
        pygame.draw.rect(
            screen,
            (shade, shade, shade),
            self.rect
        )
        # 顯示衰減值
        font_small = pygame.font.SysFont(None, 20)
        label = font_small.render(f"{self.attenuation:.2f}", True, (200, 200, 200))
        screen.blit(label, (self.rect.x + 2, self.rect.y + 2))


# ============================================================
# 干擾方塊（額外聲源，混入 pyroomacoustics 迷惑模型）
# ============================================================

ORANGE = (255, 160, 40)

class DistractorSource:
    """畫面上可手動放置的干擾聲源。
    中鍵點擊放置，右鍵點擊移除，E/F 鍵調整全域振幅。
    聲學上以 distractor_amplitude 倍振幅加入 pyroomacoustics 場景，
    讓模型在多聲源環境下嘗試定位主目標。
    """

    SIZE = 24   # 半徑像素

    def __init__(self, pos, amplitude):
        self.pos = np.array(pos, dtype=np.float32)
        self.amplitude = amplitude

    def draw(self, screen):
        cx, cy = int(self.pos[0]), int(self.pos[1])
        r = self.SIZE
        # 橘色菱形
        pts = [
            (cx,     cy - r),
            (cx + r, cy    ),
            (cx,     cy + r),
            (cx - r, cy    ),
        ]
        pygame.draw.polygon(screen, ORANGE, pts)
        pygame.draw.polygon(screen, (255, 220, 100), pts, 2)
        # 振幅標籤
        font_small = pygame.font.SysFont(None, 20)
        label = font_small.render(f"x{self.amplitude:.2f}", True, (255, 255, 200))
        screen.blit(label, (cx - 18, cy + r + 2))


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
            with_range=True,
            with_height=True,
        ).to(self.device)

        try:
            ckpt = torch.load(
                ckpt_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            ckpt = torch.load(
                ckpt_path,
                map_location=self.device,
            )

        state = ckpt["model"]
        has_height = any("height" in k for k in state.keys())
        if not has_height:
            self.net = build_net(
                self.cfg,
                with_range=True,
                with_height=False,
            ).to(self.device)
        self.net.load_state_dict(state)
        self.has_height = has_height

        self.net.eval()

        print("Model loaded")

    # --------------------------------------------------------

    def simulate_audio(
        self,
        src_pos,
        target_screen=None,
        obstacles=None,
        distractors=None,
    ):

        global signal_mode_idx
        global START_FREQ
        global END_FREQ

        room = pra.ShoeBox(
            [10, 8],
            fs=192000,
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
            int(192000 * duration)
        )

        signal_type = SIGNAL_TYPES[signal_mode_idx]

        # ====================================================
        # CW
        # ====================================================

        if signal_type == "cw":

            signal = np.sin(
                2 * np.pi * START_FREQ * t
            )

        # ====================================================
        # CHIRP
        # ====================================================

        elif signal_type == "chirp":

            signal = chirp(
                t,
                START_FREQ,
                t[-1],
                END_FREQ
            )

        # ====================================================
        # PULSE TRAIN
        # ====================================================

        elif signal_type == "pulse_train":

            base = np.sin(
                2 * np.pi * START_FREQ * t
            )

            period = max(
                1,
                len(t) // 10
            )

            gate = (
                (np.arange(len(t)) % period)
                < (period // 2)
            ).astype(np.float32)

            signal = base * gate

        else:

            signal = np.sin(
                2 * np.pi * START_FREQ * t
            )

        room.add_source(
            src_pos,
            signal=signal
        )

        # ---- 干擾方塊：以各自振幅加入同一 room ----
        if distractors:
            for d in distractors:
                d_rel_m = (d.pos - CENTER) / PIXELS_PER_METER
                d_room = np.array([
                    5.0 + d_rel_m[0],
                    4.0 + d_rel_m[1],
                ])
                # 限制在房間邊界內（避免 pyroomacoustics 報錯）
                d_room = np.clip(d_room, [0.1, 0.1], [9.9, 7.9])
                room.add_source(d_room, signal=signal * d.amplitude)

        room.simulate()

        audio = room.mic_array.signals.T

        if target_screen is not None and obstacles:
            for mic_idx, mic_xy in enumerate(MIC_LAYOUT):
                mic_screen = mic_to_screen(mic_xy)
                blocked = any(
                    obs.rect.clipline(
                        tuple(target_screen.astype(int)),
                        tuple(mic_screen.astype(int)),
                    )
                    for obs in obstacles
                )
                if blocked:
                    gain = min(
                        obs.attenuation
                        for obs in obstacles
                        if obs.rect.clipline(
                            tuple(target_screen.astype(int)),
                            tuple(mic_screen.astype(int)),
                        )
                    )
                    audio[:, mic_idx] *= gain

        return audio

    # --------------------------------------------------------

    def extract_feature(self, audio):

        signals = audio.T.astype(np.float32)
        filtered = sp.bandpass(self.cfg, signals)
        return sp.extract_features_v2(self.cfg, filtered)

    # --------------------------------------------------------

    @torch.no_grad()
    def predict_angle(self, target_pos, obstacles=None, distractors=None):

        rel_m = (target_pos - CENTER) / PIXELS_PER_METER

        room_pos = np.array([
            5 + rel_m[0],
            4 + rel_m[1],
        ])

        audio = self.simulate_audio(
            room_pos,
            target_screen=target_pos,
            obstacles=obstacles,
            distractors=distractors,
        )

        feat = self.extract_feature(audio)

        x = torch.tensor(
            feat,
            device=self.device
        ).unsqueeze(0)

        out = self.net(x)

        az_logits = out[0] if isinstance(out, tuple) else out
        az_prob = torch.softmax(az_logits, dim=-1)

        pred_bin = torch.argmax(
            az_logits,
            dim=-1
        ).item()

        pred_angle = pred_bin * (
            360 / self.cfg.task.n_azimuth_bins
        )

        confidence = az_prob.max(dim=-1).values.item()

        range_bin = None
        range_confidence = None
        height_bin = None
        height_confidence = None
        if isinstance(out, tuple):
            range_prob = torch.softmax(out[1], dim=-1)
            range_bin = torch.argmax(out[1], dim=-1).item()
            range_confidence = range_prob.max(dim=-1).values.item()
            if len(out) > 2:
                height_prob = torch.softmax(out[2], dim=-1)
                height_bin = torch.argmax(out[2], dim=-1).item()
                height_confidence = height_prob.max(dim=-1).values.item()

        return pred_angle, confidence, range_bin, range_confidence, height_bin, height_confidence


# ============================================================
# 主程式
# ============================================================

def main():

    global signal_mode_idx
    global START_FREQ
    global END_FREQ
    global next_obstacle_attenuation
    global next_obstacle_size
    global distractor_amplitude

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

    distractors = []

    prediction_ready = False
    prediction_timer = 0
    delay_frames = 0

    pred_angle = 0
    pred_confidence = 0
    pred_range_bin = None
    pred_range_confidence = None
    pred_height_bin = None
    pred_height_confidence = None
    true_distance_m = 0
    true_height_m = 0
    view_mode_idx = 0

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
        nonlocal pred_confidence
        nonlocal pred_range_bin
        nonlocal pred_range_confidence
        nonlocal pred_height_bin
        nonlocal pred_height_confidence
        nonlocal true_distance_m
        nonlocal true_height_m

        particles.clear()
        waves.clear()

        prediction_ready = False
        prediction_timer = 0

        angle = random.uniform(0, 360)

        dist = random.uniform(180, 320)
        true_height_m = random.uniform(
            MIN_TARGET_HEIGHT_M,
            MAX_TARGET_HEIGHT_M,
        )

        target_pos = (
            CENTER +
            angle_to_vec(angle) * dist
        )

        waves.append(
            SoundWave(target_pos.copy())
        )

        true_distance_m = dist / PIXELS_PER_METER

        travel_time = true_distance_m / 343

        delay_frames = max(
            1,
            int(travel_time * 60)
        )

        (pred_angle, pred_confidence, pred_range_bin, pred_range_confidence,
         pred_height_bin, pred_height_confidence) = sensor.predict_angle(
            target_pos,
            obstacles,
            distractors,
        )

        return angle, target_pos

    def refresh_prediction():

        nonlocal pred_angle
        nonlocal pred_confidence
        nonlocal pred_range_bin
        nonlocal pred_range_confidence
        nonlocal pred_height_bin
        nonlocal pred_height_confidence
        nonlocal prediction_ready
        nonlocal prediction_timer

        (pred_angle, pred_confidence, pred_range_bin, pred_range_confidence,
         pred_height_bin, pred_height_confidence) = sensor.predict_angle(
            target_pos,
            obstacles,
            distractors,
        )
        particles.clear()
        prediction_ready = False
        prediction_timer = 0

    true_angle, target_pos = reset_target()

    # ========================================================
    # loop
    # ========================================================

    running = True

    while running:

        clock.tick(60)

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

                elif event.key == pygame.K_t:

                    view_mode_idx = (
                        view_mode_idx + 1
                    ) % len(VIEW_MODES)

                # ============================================
                # 切換訊號
                # ============================================

                elif event.key == pygame.K_c:

                    signal_mode_idx = (
                        signal_mode_idx + 1
                    ) % len(SIGNAL_TYPES)

                    true_angle, target_pos = (
                        reset_target()
                    )

                # ============================================
                # 頻率調整
                # ============================================

                elif event.key == pygame.K_q:

                    START_FREQ += FREQ_STEP
                    refresh_prediction()

                elif event.key == pygame.K_a:

                    START_FREQ = max(
                        1000,
                        START_FREQ - FREQ_STEP
                    )
                    refresh_prediction()

                elif event.key == pygame.K_w:

                    END_FREQ += FREQ_STEP
                    refresh_prediction()

                elif event.key == pygame.K_s:

                    END_FREQ = max(
                        START_FREQ,
                        END_FREQ - FREQ_STEP
                    )
                    refresh_prediction()

                # ============================================
                # 障礙物衰減調整（[ 降 / ] 升）
                # ============================================

                elif event.key == pygame.K_LEFTBRACKET:
                    next_obstacle_attenuation = max(
                        0.05,
                        round(next_obstacle_attenuation - 0.05, 2)
                    )

                elif event.key == pygame.K_RIGHTBRACKET:
                    next_obstacle_attenuation = min(
                        1.0,
                        round(next_obstacle_attenuation + 0.05, 2)
                    )

                # ============================================
                # 障礙物大小調整（- 縮 / = 放）
                # ============================================

                elif event.key == pygame.K_MINUS:
                    next_obstacle_size = max(20, next_obstacle_size - 10)

                elif event.key == pygame.K_EQUALS:
                    next_obstacle_size = min(200, next_obstacle_size + 10)

                # ============================================
                # 干擾方塊振幅（E 降 / F 升）
                # ============================================

                elif event.key == pygame.K_e:
                    distractor_amplitude = max(
                        0.05,
                        round(distractor_amplitude - 0.05, 2)
                    )

                elif event.key == pygame.K_f:
                    distractor_amplitude = min(
                        2.0,
                        round(distractor_amplitude + 0.05, 2)
                    )

                # ============================================
                # 清除所有干擾方塊（D）
                # ============================================

                elif event.key == pygame.K_d:
                    distractors.clear()
                    refresh_prediction()

            if event.type == pygame.MOUSEBUTTONDOWN:

                mx, my = pygame.mouse.get_pos()

                if event.button == 1:

                    half = next_obstacle_size // 2
                    rect = pygame.Rect(
                        mx - half,
                        my - half,
                        next_obstacle_size,
                        next_obstacle_size,
                    )

                    obstacles.append(
                        Obstacle(
                            rect,
                            next_obstacle_attenuation,
                        )
                    )
                    refresh_prediction()

                # ---- 中鍵：放置干擾方塊 ----
                elif event.button == 2:
                    distractors.append(
                        DistractorSource(
                            (mx, my),
                            distractor_amplitude,
                        )
                    )
                    refresh_prediction()

                elif event.button == 3:

                    removed = False

                    # 先嘗試移除干擾方塊
                    for d in distractors[:]:
                        dx = abs(d.pos[0] - mx)
                        dy = abs(d.pos[1] - my)
                        if dx <= DistractorSource.SIZE and dy <= DistractorSource.SIZE:
                            distractors.remove(d)
                            removed = True

                    # 再嘗試移除障礙物
                    for obs in obstacles[:]:
                        if obs.rect.collidepoint(mx, my):
                            obstacles.remove(obs)
                            removed = True

                    if removed:
                        refresh_prediction()

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

        view_mode = VIEW_MODES[view_mode_idx]
        pred_range_m = range_bin_midpoint(sensor.cfg, pred_range_bin)

        if view_mode == "top":

            for obs in obstacles:
                obs.draw(screen)

            for d in distractors:
                d.draw(screen)

            for w in waves:
                w.draw(screen)

            # mic pair
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

            # mic
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

            # target
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

            # true direction
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

            # predicted direction
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

            if pred_range_m is not None:
                pygame.draw.circle(
                    screen,
                    BLUE,
                    CENTER.astype(int),
                    int(pred_range_m * PIXELS_PER_METER),
                    1
                )

            for p in particles:
                p.draw(screen)

        else:

            pygame.draw.line(
                screen,
                (95, 95, 95),
                (80, SIDE_GROUND_Y),
                (WIDTH - 80, SIDE_GROUND_Y),
                2
            )

            origin = side_to_screen(0.0, 0.0)
            pygame.draw.circle(
                screen,
                WHITE,
                origin.astype(int),
                8
            )

            target_side = side_to_screen(
                true_distance_m,
                true_height_m,
            )

            pygame.draw.line(
                screen,
                GREEN,
                origin.astype(int),
                target_side.astype(int),
                3
            )

            pygame.draw.line(
                screen,
                RED,
                (int(target_side[0]), SIDE_GROUND_Y),
                target_side.astype(int),
                2
            )

            pygame.draw.rect(
                screen,
                RED,
                pygame.Rect(
                    int(target_side[0] - TARGET_SIZE / 2),
                    int(target_side[1] - TARGET_SIZE / 2),
                    TARGET_SIZE,
                    TARGET_SIZE,
                )
            )

            if pred_range_m is not None:
                pred_height_m = height_bin_midpoint(sensor.cfg, pred_height_bin) or 0.0
                pred_side = side_to_screen(pred_range_m, pred_height_m)
                pygame.draw.line(
                    screen,
                    BLUE,
                    origin.astype(int),
                    pred_side.astype(int),
                    3
                )
                pygame.draw.circle(
                    screen,
                    BLUE,
                    pred_side.astype(int),
                    8,
                    2
                )

        err = abs(
            (pred_angle - true_angle + 180) % 360 - 180
        )

        texts = [

            f"Signal Type : {SIGNAL_TYPES[signal_mode_idx]}",
            f"Start Freq  : {hz_to_text(START_FREQ)}",
            f"End Freq    : {hz_to_text(END_FREQ)}",
            f"Freq Step   : {FREQ_STEP} Hz",

            "",

            "--- Controls ----------------",
            "C      = Change signal type",
            "Q/A    = Start Freq +/-",
            "W/S    = End Freq +/-",
            "T      = Toggle top/side view",
            "R      = New target",
            "SPACE  = Respawn particles",

            "",

            "--- Obstacles ---------------",
            f"Gain   : {next_obstacle_attenuation:.2f}  ([ ] adjust)",
            f"Size   : {next_obstacle_size} px  (-/= adjust)",
            f"Count  : {len(obstacles)}",
            "Left   = Add obstacle",
            "Right  = Remove obstacle/distractor",

            "",

            "--- Distractors -------------",
            f"Amp    : x{distractor_amplitude:.2f}  (E/F adjust)",
            f"Count  : {len(distractors)}",
            "Middle = Add distractor (orange)",
            "D      = Clear all distractors",

            "",

            "--- Prediction --------------",
            f"True Angle  : {true_angle:.1f} deg",
            f"Pred Angle  : {pred_angle:.1f} deg",
            f"Error       : {err:.1f} deg",
            f"Confidence  : {pred_confidence:.2f}",
            f"True Dist   : {true_distance_m:.2f} m",
            f"Height      : {true_height_m:.2f} m",
            f"Pred Height : {height_bin_to_text(sensor.cfg, pred_height_bin)}",
            (f"Height Conf : {pred_height_confidence:.2f}"
             if pred_height_confidence is not None else "Height Conf : -"),
            f"Pred Dist   : {range_bin_to_text(sensor.cfg, pred_range_bin)}",
            (f"Dist Conf   : {pred_range_confidence:.2f}"
             if pred_range_confidence is not None else "Dist Conf   : -"),

            "",

            "GREEN = true direction",
            "BLUE  = AI prediction",
            f"View  : {view_mode.upper()}",
        ]

        for i, t in enumerate(texts):

            surf = font.render(
                t,
                True,
                YELLOW
            )

            screen.blit(
                surf,
                (20, 20 + i * 28)
            )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
