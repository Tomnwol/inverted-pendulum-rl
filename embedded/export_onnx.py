"""Export the float32 student to ONNX.

    python export_onnx.py

The graph includes the division of angular velocity by MAX_SPEED, so the
ONNX input is the raw observation (cos θ, sin θ, θ̇). The output is the
torque before the ±MAX_TORQUE clip.
"""

import numpy as np
import torch
import torch.nn as nn

from policy_contract import MAX_SPEED, STUDENT_ONNX, load_student, policy

OPSET = 17


class StudentOnnx(nn.Module):
    """Raw observation (1, 3) → torque. Scaling is a multiply, so ONNX can record it."""

    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, obs):
        scale = obs.new_tensor([1.0, 1.0, 1.0 / MAX_SPEED])
        return self.student(obs * scale)


def export(model, path=STUDENT_ONNX):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, 3),),
        path,
        input_names=["observation"],
        output_names=["torque"],
        opset_version=OPSET,
        dynamo=False,
    )


def check(model, student):
    samples = np.array(
        [
            [0.3, -0.4, 4.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, -8.0],
            [-0.7, 0.7, 2.5],
        ],
        dtype=np.float32,
    )
    worst = 0.0
    with torch.no_grad():
        for obs in samples:
            reference = policy(student, obs)
            exported = model(torch.tensor(obs)[None]).item()
            worst = max(worst, abs(reference - exported))
            print(f"obs {obs.tolist()} | policy {reference:+.6f} | onnx {exported:+.6f}")
    print(f"max abs gap {worst:.3e}")
    if worst > 1e-5:
        raise SystemExit("ONNX wrapper does not match policy()")


def main():
    student = load_student()
    model = StudentOnnx(student).eval()
    export(model)
    check(model, student)
    print(f"Student ONNX saved: weights/{STUDENT_ONNX.name}")


if __name__ == "__main__":
    main()
