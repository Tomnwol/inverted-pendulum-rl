"""Show the pendulum driven by a policy from the embedded folder.

    python simulation.py            PPO reference (weights/actor.pt)
    python simulation.py --student  float32 student (weights/student.pt)
    python simulation.py --int8     int8 student (weights/student_int8.pt)
"""

import subprocess
import sys

import tkinter as tk

from policy_contract import CONTROL_HZ, load_reference, load_student, policy as reference_action
from pendulum_env import Pendulum
from quantize import load_int8

if "--int8" in sys.argv:
    TITLE = "Pendulum int8"
elif "--student" in sys.argv:
    TITLE = "Distilled pendulum"
else:
    TITLE = "Reference pendulum"
PERIOD_MS = int(1000 / CONTROL_HZ)


def main():
    if "--int8" in sys.argv:
        network = load_int8()
    elif "--student" in sys.argv:
        network = load_student()
    else:
        network = load_reference()

    def torque(obs):
        return reference_action(network, obs)

    env = Pendulum(seed=123)
    state = {"obs": env.reset(), "total": 0.0, "episode": 1, "trace": []}

    root = tk.Tk()
    root.title(TITLE)
    root.geometry("640x700+120+60")
    canvas = tk.Canvas(root, width=640, height=640, bg="white", highlightthickness=0)
    canvas.pack()
    info = tk.Label(root, text="", font=("sans-serif", 13))
    info.pack(pady=8)

    def to_screen(x, y):
        return 320 + x * 200, 320 - y * 200

    def draw():
        obs = state["obs"]
        action = torque(obs)
        obs, reward, done = env.step(action)
        state["obs"] = obs
        state["total"] += reward
        x, y = float(obs[1]), float(obs[0])
        state["trace"].append((x, y))
        state["trace"] = state["trace"][-50:]

        canvas.delete("all")
        canvas.create_oval(*to_screen(-1, 1), *to_screen(1, -1), outline="#bfbfbf", dash=(4, 4))
        if len(state["trace"]) >= 2:
            pts = [c for p in state["trace"] for c in to_screen(*p)]
            canvas.create_line(*pts, fill="#e7b08a", width=2, smooth=True)
        px, py = to_screen(0, 0)
        bx, by = to_screen(x, y)
        canvas.create_line(px, py, bx, by, fill="#1f4b99", width=5, capstyle="round")
        canvas.create_rectangle(px - 7, py - 7, px + 7, py + 7, fill="black", outline="black")
        canvas.create_oval(bx - 14, by - 14, bx + 14, by + 14, fill="#e07a3d", outline="#e07a3d")
        info.config(text=f"episode {state['episode']}    return {state['total']:7.1f}    torque {action:+.2f}")

        if done:
            state.update(obs=env.reset(), total=0.0, episode=state["episode"] + 1, trace=[])
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

    print(f"Window open: {TITLE}", flush=True)
    root.after(PERIOD_MS, draw)
    root.after(200, bring_to_front)
    root.mainloop()


if __name__ == "__main__":
    main()
