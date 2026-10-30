"""Tune a PID and compare it with the distilled student.

    python pid/tune.py

The live pendulum starts tilted, so the sliders show the balance.
Compare runs the course protocol: 50 episodes, seed 123, no friction.
A linear PID holds a small tilt. It does not swing up from the bottom,
so its mean return stays far below the student.
"""

import math
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embedded"))

from pendulum_env import Pendulum
from policy_contract import (
    CONTROL_HZ, DT, EVAL_EPISODES, EVAL_SEED, MAX_TORQUE,
    evaluate, load_student,
)

TITLE = "PID"
PERIOD_MS = int(1000 / CONTROL_HZ)
UPRIGHT_TILT = 0.4


def pid_torque(obs, gains, memory):
    """Angle and speed toward upright. Torque before the clip, then clipped."""
    theta = math.atan2(float(obs[1]), float(obs[0]))
    memory["i"] = max(-1.0, min(1.0, memory["i"] + theta * DT))
    raw = -gains["kp"] * theta - gains["ki"] * memory["i"] - gains["kd"] * float(obs[2])
    return float(np.clip(raw, -MAX_TORQUE, MAX_TORQUE))


def mean_return(controller, n=EVAL_EPISODES, seed=EVAL_SEED):
    env = Pendulum(seed=seed, damping=0.0)
    returns = []
    for _ in range(n):
        obs, total, done = env.reset(), 0.0, False
        memory = {"i": 0.0}
        while not done:
            obs, reward, done = env.step(controller(obs, memory))
            total += reward
        returns.append(total)
    return float(np.mean(returns))


def main():
    env = Pendulum(seed=EVAL_SEED, damping=0.0)
    memory = {"i": 0.0}
    state = {
        "obs": None,
        "trace": [],
        "student": None,
        "busy": False,
    }

    def stand_up():
        env.state = np.array([UPRIGHT_TILT, 0.0])
        env.steps = 0
        memory["i"] = 0.0
        state["obs"] = env._obs()
        state["trace"] = []

    stand_up()

    root = tk.Tk()
    root.title(TITLE)
    root.geometry("640x900+120+40")
    canvas = tk.Canvas(root, width=640, height=520, bg="white", highlightthickness=0)
    canvas.pack()
    info = tk.Label(root, text="", font=("sans-serif", 13))
    info.pack(pady=4)

    controls = tk.Frame(root)
    controls.pack(fill="x", padx=16)

    def slider(label, start, stop, value):
        scale = tk.Scale(
            controls, from_=start, to=stop, resolution=0.1,
            orient=tk.HORIZONTAL, label=label, length=360,
        )
        scale.set(value)
        scale.pack(fill="x")
        return scale

    kp = slider("Kp  angle", 0, 40, 15)
    ki = slider("Ki  intégrale", 0, 5, 0)
    kd = slider("Kd  vitesse", 0, 10, 2)

    def gains():
        return {"kp": kp.get(), "ki": ki.get(), "kd": kd.get()}

    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=16, pady=8)
    tk.Button(buttons, text="Remettre près de la verticale", command=stand_up).pack(side="left")
    compare_button = tk.Button(buttons, text="Comparer")
    compare_button.pack(side="right")

    result = tk.Label(root, text="50 épisodes, graine 123, sans frottement", font=("sans-serif", 12))
    result.pack(pady=4)

    def to_screen(x, y):
        return 320 + x * 160, 260 - y * 160

    def draw():
        action = pid_torque(state["obs"], gains(), memory)
        obs, _reward, _done = env.step(action)
        state["obs"] = obs
        x, y = float(obs[1]), float(obs[0])
        state["trace"].append((x, y))
        state["trace"] = state["trace"][-50:]
        theta = math.degrees(math.atan2(y, x))

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
        info.config(text=f"angle {theta:+5.1f}°    couple {action:+.2f}")
        root.after(PERIOD_MS, draw)

    def show_result(pid_mean, student_mean):
        state["busy"] = False
        compare_button.config(state=tk.NORMAL)
        result.config(text=f"PID  {pid_mean:7.1f}      élève  {student_mean:7.1f}")

    def compare():
        if state["busy"]:
            return
        state["busy"] = True
        compare_button.config(state=tk.DISABLED)
        result.config(text="calcul…")
        chosen = gains()

        def work():
            pid_mean = mean_return(lambda obs, mem: pid_torque(obs, chosen, mem))
            if state["student"] is None:
                state["student"] = evaluate(load_student())
            root.after(0, lambda: show_result(pid_mean, state["student"]))

        threading.Thread(target=work, daemon=True).start()

    compare_button.config(command=compare)

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

    print(f"Window open: {TITLE}", flush=True)
    root.after(PERIOD_MS, draw)
    root.after(200, bring_to_front)
    root.mainloop()


if __name__ == "__main__":
    main()
