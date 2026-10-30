"""Hardware-in-the-loop at 20 Hz.

    python hil.py
    python hil.py --port /dev/ttyACM0

This PC simulates the pendulum and draws it. Every tick it sends the raw
observation to the board and waits for the torque. The board runs the
float32 student. Pendulum.step clips the torque at ±MAX_TORQUE.

A seed is drawn at random and the pendulum keeps that initial condition.
Next seed draws another one. Unplugging the board keeps the physics
running with zero torque. Replugging reopens the port and the board
answers the current state.

Frame to the board: 0xA5, 12-byte observation, XOR of those 12 bytes.
Frame back: 0x5A, 4-byte torque, 4-byte cycle count, XOR of those 8 bytes.
Floats are little-endian. Flash firmware/ before starting.
"""

import argparse
import os
import random
import select
import subprocess
import sys
import time
from collections import deque

import numpy as np
import serial
import tkinter as tk

from policy_contract import CONTROL_HZ
from pendulum_env import Pendulum

OBS_MAGIC = 0xA5
TORQUE_MAGIC = 0x5A
TITLE = "Pendulum on STM32"
PERIOD_MS = int(1000 / CONTROL_HZ)
REPLY_TIMEOUT_S = 0.05
REPLY_SIZE = 10
BOOT_WAIT_S = 0.5
CPU_HZ = 216_000_000
TIMING_WINDOW = 200
GREEN = "#1f8a4c"
RED = "#c0392b"


def checksum(payload):
    total = 0
    for byte in payload:
        total ^= byte
    return total


def observation_frame(obs):
    payload = np.asarray(obs, dtype=np.float32).tobytes()
    return bytes([OBS_MAGIC]) + payload + bytes([checksum(payload)])


def torque_from_frame(frame):
    if len(frame) != REPLY_SIZE or frame[0] != TORQUE_MAGIC:
        raise ValueError("bad torque frame")
    payload = frame[1:9]
    if frame[9] != checksum(payload):
        raise ValueError("bad torque checksum")
    torque = float(np.frombuffer(payload[:4], dtype=np.float32)[0])
    cycles = int(np.frombuffer(payload[4:], dtype=np.uint32)[0])
    return torque, cycles


class Timing:
    """Recent link round-trips and on-chip network times."""

    def __init__(self):
        self.link_s = deque(maxlen=TIMING_WINDOW)
        self.board_s = deque(maxlen=TIMING_WINDOW)

    def add(self, link_s, board_s):
        self.link_s.append(link_s)
        self.board_s.append(board_s)

    @staticmethod
    def _mean_jitter(samples):
        values = np.asarray(samples, dtype=np.float64)
        return float(values.mean()), float(values.max() - values.min())

    def line(self):
        if not self.link_s:
            return "link —    board —"
        link_mean, link_jitter = self._mean_jitter(self.link_s)
        board_mean, board_jitter = self._mean_jitter(self.board_s)
        return (
            f"link {link_mean * 1e3:.2f} ms, jitter {link_jitter * 1e3:.2f} ms"
            f"    board {board_mean * 1e6:.0f} us, jitter {board_jitter * 1e6:.0f} us"
        )


class Board:
    """Serial link that can disappear and come back."""

    def __init__(self, port):
        self.port = port
        self.link = None
        self.ready_at = 0.0
        self.dropped = False
        self.link_s = None
        self.board_s = None

    def close(self):
        if self.link is None:
            return
        try:
            self.link.close()
        except (serial.SerialException, OSError):
            pass
        self.link = None
        self.dropped = True

    def _open(self):
        if self.link is not None or not os.path.exists(self.port):
            return
        try:
            # Non-blocking: unplugging the ST-Link must not stall the window.
            self.link = serial.Serial(self.port, 115200, timeout=0, write_timeout=0)
        except (serial.SerialException, OSError):
            self.link = None
            return
        # Opening the ST-Link port resets the MCU. Wait out the reboot.
        self.ready_at = time.monotonic() + BOOT_WAIT_S
        try:
            self.link.reset_input_buffer()
        except (serial.SerialException, OSError):
            self.close()

    def _read_exact(self, size):
        buf = bytearray()
        deadline = time.monotonic() + REPLY_TIMEOUT_S
        fd = self.link.fileno()
        while len(buf) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                ready, _, _ = select.select([fd], [], [], remaining)
            except (OSError, ValueError):
                self.close()
                return None
            if not ready:
                return None
            try:
                chunk = os.read(fd, size - len(buf))
            except OSError:
                self.close()
                return None
            if not chunk:
                self.close()
                return None
            buf.extend(chunk)
        return bytes(buf)

    def torque(self, obs):
        """Return the board torque, or None when the board is not answering."""
        self.dropped = False
        self.link_s = None
        self.board_s = None
        self._open()
        if self.link is None or time.monotonic() < self.ready_at:
            return None
        if not os.path.exists(self.port):
            self.close()
            return None
        try:
            self.link.reset_input_buffer()
            started = time.perf_counter()
            self.link.write(observation_frame(obs))
        except (serial.SerialException, OSError):
            self.close()
            return None
        frame = self._read_exact(REPLY_SIZE)
        if frame is None:
            return None
        try:
            torque, cycles = torque_from_frame(frame)
        except ValueError:
            return None
        self.link_s = time.perf_counter() - started
        self.board_s = cycles / CPU_HZ
        return torque


def main():
    if sys.byteorder != "little":
        raise SystemExit("The board expects little-endian floats.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    args = parser.parse_args()
    board = Board(args.port)

    # Light axle friction so an unplugged pendulum slows down instead of swinging forever.
    env = Pendulum(seed=0, damping=0.5)
    state = {
        "seed": random.randrange(2**31),
        "obs": None,
        "total": 0.0,
        "trace": [],
        "skip": False,
        "detected": False,
    }

    def restart(seed):
        env.rng = np.random.default_rng(seed)
        env.steps = 0
        state["seed"] = seed
        state["obs"] = env.reset()
        state["total"] = 0.0
        state["trace"] = []

    restart(state["seed"])
    timing = Timing()

    root = tk.Tk()
    root.title(TITLE)
    root.geometry("640x720+120+60")
    canvas = tk.Canvas(root, width=640, height=640, bg="white", highlightthickness=0)
    canvas.pack()
    info = tk.Label(root, text="", font=("sans-serif", 13))
    info.pack(pady=4)
    timing_label = tk.Label(root, text="link —    board —", font=("sans-serif", 12))
    timing_label.pack()
    bar = tk.Frame(root)
    bar.pack(fill="x", padx=16, pady=4)
    mark = tk.Frame(bar, width=18, height=18, bg=RED)
    mark.pack_propagate(False)
    mark.pack(side="left")
    tk.Label(bar, text="STM32", font=("sans-serif", 13)).pack(side="left", padx=6)

    def tip_upright():
        # Unplugging leaves a balanced pendulum on the unstable point, so it
        # looks frozen. A small speed makes the tilt grow and it falls.
        theta, theta_dot = env.state
        if np.cos(theta) <= 0.0 or abs(theta_dot) >= 0.5:
            return
        direction = 1.0 if theta >= 0.0 else -1.0
        env.state = np.array([theta, theta_dot + direction * 0.8])

    def skip_seed():
        state["skip"] = True

    tk.Button(bar, text="Next seed", command=skip_seed).pack(side="right")

    def to_screen(x, y):
        return 320 + x * 200, 320 - y * 200

    def draw():
        try:
            if state["skip"]:
                state["skip"] = False
                restart(random.randrange(2**31))

            was_up = state["detected"]
            action = board.torque(state["obs"])
            state["detected"] = action is not None
            if action is None:
                action = 0.0
                if was_up and board.dropped:
                    tip_upright()

            obs, reward, _done = env.step(action)
            state["obs"] = obs
            state["total"] += reward
            x, y = float(obs[1]), float(obs[0])
            state["trace"].append((x, y))
            state["trace"] = state["trace"][-50:]

            canvas.delete("all")
            canvas.create_oval(*to_screen(-1, 1), *to_screen(1, -1), outline="#bfbfbf", dash=(4, 4))
            if len(state["trace"]) >= 2:
                pts = [c for point in state["trace"] for c in to_screen(*point)]
                canvas.create_line(*pts, fill="#e7b08a", width=2, smooth=True)
            px, py = to_screen(0, 0)
            bx, by = to_screen(x, y)
            canvas.create_line(px, py, bx, by, fill="#1f4b99", width=5, capstyle="round")
            canvas.create_rectangle(px - 7, py - 7, px + 7, py + 7, fill="black", outline="black")
            canvas.create_oval(bx - 14, by - 14, bx + 14, by + 14, fill="#e07a3d", outline="#e07a3d")
            if board.link_s is not None and board.board_s is not None:
                timing.add(board.link_s, board.board_s)
            info.config(
                text=f"seed {state['seed']}    return {state['total']:7.1f}    torque {action:+.2f}"
            )
            timing_label.config(text=timing.line())
            mark.config(bg=GREEN if state["detected"] else RED)
        except Exception:
            state["detected"] = False
            mark.config(bg=RED)
        root.after(PERIOD_MS, draw)

    def bring_to_front():
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        try:
            subprocess.run(
                ["xdotool", "search", "--name", TITLE, "windowactivate"],
                check=False, capture_output=True,
            )
        except FileNotFoundError:
            pass
        root.after(2500, lambda: root.attributes("-topmost", False))

    def on_close():
        board.close()
        root.destroy()

    print(f"Window open: {TITLE} on {args.port}, seed {state['seed']}", flush=True)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(PERIOD_MS, draw)
    root.after(200, bring_to_front)
    root.mainloop()


if __name__ == "__main__":
    main()
