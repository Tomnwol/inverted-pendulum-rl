"""Affiche le pendule conduit par une politique enregistrée.

    python simulation_pendule.py        acteur PPO (pendule_acteur.pt)
    python simulation_pendule.py --ch3  recherche aléatoire (ch3_poids.npy)
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import tkinter as tk
import torch
import torch.nn as nn

from pendule_env import Pendule
from ch3_recherche_aleatoire import politique_depuis

POIDS = Path(__file__).with_name("pendule_acteur.pt")
POIDS_CH3 = Path(__file__).with_name("ch3_poids.npy")
MODE_CH3 = "--ch3" in sys.argv
TITRE = "Pendule recherche aléatoire" if MODE_CH3 else "Pendule entraîné"


def mlp():
    return nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))


def prep(obs):
    o = torch.tensor(np.array(obs), dtype=torch.float32)
    o[..., 2] /= 8
    return o


def charger():
    if not POIDS.exists():
        raise SystemExit("Pas de poids enregistrés. Lance d'abord : python ppo.py")
    actor = mlp()
    actor.load_state_dict(torch.load(POIDS, weights_only=True))
    actor.eval()
    return actor


def charger_ch3():
    if not POIDS_CH3.exists():
        raise SystemExit("Pas de poids chapitre 3. Lance d'abord : python ch3_recherche_aleatoire.py")
    return politique_depuis(np.load(POIDS_CH3))


def main():
    if MODE_CH3:
        politique = charger_ch3()
        def couple(o):
            return float(politique(o))
    else:
        actor = charger()
        def couple(o):
            with torch.no_grad():
                return actor(prep(o)[None]).item()

    env = Pendule(seed=123)
    etat = {"obs": env.reset(), "total": 0.0, "episode": 1, "trace": []}

    root = tk.Tk()
    root.title(TITRE)
    root.geometry("640x700+120+60")
    toile = tk.Canvas(root, width=640, height=640, bg="white", highlightthickness=0)
    toile.pack()
    info = tk.Label(root, text="", font=("sans-serif", 13))
    info.pack(pady=8)

    def vers_ecran(x, y):
        return 320 + x * 200, 320 - y * 200

    def dessiner():
        o = etat["obs"]
        u = couple(o)
        o, r, fini = env.step(u)
        etat["obs"] = o
        etat["total"] += r
        x, y = float(o[1]), float(o[0])
        etat["trace"].append((x, y))
        etat["trace"] = etat["trace"][-50:]

        toile.delete("all")
        toile.create_oval(*vers_ecran(-1, 1), *vers_ecran(1, -1), outline="#bfbfbf", dash=(4, 4))
        if len(etat["trace"]) >= 2:
            pts = [c for p in etat["trace"] for c in vers_ecran(*p)]
            toile.create_line(*pts, fill="#e7b08a", width=2, smooth=True)
        px, py = vers_ecran(0, 0)
        bx, by = vers_ecran(x, y)
        toile.create_line(px, py, bx, by, fill="#1f4b99", width=5, capstyle="round")
        toile.create_rectangle(px - 7, py - 7, px + 7, py + 7, fill="black", outline="black")
        toile.create_oval(bx - 14, by - 14, bx + 14, by + 14, fill="#e07a3d", outline="#e07a3d")
        info.config(text=f"épisode {etat['episode']}    retour {etat['total']:7.1f}    couple {u:+.2f}")

        if fini:
            etat.update(obs=env.reset(), total=0.0, episode=etat["episode"] + 1, trace=[])
        root.after(50, dessiner)

    def devant():
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        try:
            subprocess.run(
                ["xdotool", "search", "--name", TITRE, "windowactivate"],
                check=False, capture_output=True,
            )
        except FileNotFoundError:
            pass
        root.after(2500, lambda: root.attributes("-topmost", False))

    print("Fenêtre ouverte : Pendule entraîné", flush=True)
    root.after(50, dessiner)
    root.after(200, devant)
    root.mainloop()


if __name__ == "__main__":
    main()
