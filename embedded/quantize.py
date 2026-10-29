"""Quantize the student to int8 and measure the gap against float32.

    python quantize.py

Symmetric per-tensor quantization. Each tensor's scale comes from the
maximum absolute value seen on the distill trajectories: one tick is
max / 127. Weights and activations are integers from −127 to 127.
tanh is computed on the value converted back to float, then the result
is quantized again: that is the equivalent of an integer lookup table.
The prepared input and the returned torque stay float32.
"""

import torch
import torch.nn as nn

from distill import collect
from policy_contract import (
    STUDENT_INT8_WEIGHTS,
    evaluate,
    load_reference,
    load_student,
)

QUANT_MAX = 127


class StudentInt8(nn.Module):
    """Same signature as the float32 student: prepared float in, torque out."""

    def __init__(self, blob):
        super().__init__()
        self.n = len(blob["weights"])
        for i, (weight, bias) in enumerate(zip(blob["weights"], blob["biases"])):
            self.register_buffer(f"w{i}", weight.to(torch.int8))
            self.register_buffer(f"b{i}", bias.to(torch.int32))
            self.register_buffer(f"sw{i}", blob["weight_scales"][i].float().reshape(()))
            self.register_buffer(f"se{i}", blob["input_scales"][i].float().reshape(()))
            self.register_buffer(f"ss{i}", blob["output_scales"][i].float().reshape(()))

    def _to_int(self, value, scale):
        return torch.clamp(torch.round(value / scale), -QUANT_MAX, QUANT_MAX)

    def forward(self, x):
        q = self._to_int(x, self.se0)
        for i in range(self.n):
            weight = getattr(self, f"w{i}").float()
            input_scale = getattr(self, f"se{i}")
            weight_scale = getattr(self, f"sw{i}")
            output_scale = getattr(self, f"ss{i}")
            accumulator = nn.functional.linear(q, weight)
            output = (accumulator + getattr(self, f"b{i}").float()) * (input_scale * weight_scale)
            output = self._to_int(output, output_scale) * output_scale
            if i < self.n - 1:
                q = self._to_int(torch.tanh(output), getattr(self, f"se{i + 1}"))
            else:
                q = output
        return q


def _scale(tensor):
    maximum = tensor.detach().abs().amax().clamp(min=1e-8)
    return maximum / QUANT_MAX


def _quantize(tensor, scale):
    return torch.clamp(torch.round(tensor / scale), -QUANT_MAX, QUANT_MAX)


def _quantize_bias(bias, scale):
    """The bias is added in the int32 accumulator, not stored on 8 bits."""
    return torch.round(bias / scale).to(torch.int32)


def quantize(student, inputs):
    """Calibrate scales on `inputs` (already passed through to_network_input) and freeze the integers."""
    activations = [inputs]
    hidden = inputs
    with torch.no_grad():
        for layer in student:
            hidden = layer(hidden)
            activations.append(hidden)

    linears = [layer for layer in student if isinstance(layer, nn.Linear)]
    weights, biases = [], []
    weight_scales, input_scales, output_scales = [], [], []
    # activations: input, linear output, tanh, linear output, tanh, torque
    input_indices = (0, 2, 4)
    output_indices = (1, 3, 5)
    for i, layer in enumerate(linears):
        weight_scale = _scale(layer.weight)
        input_scale = _scale(activations[input_indices[i]])
        output_scale = _scale(activations[output_indices[i]])
        accumulator_scale = input_scale * weight_scale
        weights.append(_quantize(layer.weight, weight_scale).to(torch.int8))
        biases.append(_quantize_bias(layer.bias, accumulator_scale))
        weight_scales.append(weight_scale.reshape(()))
        input_scales.append(input_scale.reshape(()))
        output_scales.append(output_scale.reshape(()))
    return {
        "weights": weights,
        "biases": biases,
        "weight_scales": weight_scales,
        "input_scales": input_scales,
        "output_scales": output_scales,
    }


def load_int8(path=STUDENT_INT8_WEIGHTS):
    if not path.exists():
        raise SystemExit("No int8 student. Run first: python quantize.py")
    blob = torch.load(path, weights_only=True)
    model = StudentInt8(blob)
    model.eval()
    return model


def main():
    student = load_student()
    print("collecting calibration trajectories...")
    inputs, _ = collect(load_reference())
    blob = quantize(student, inputs)
    STUDENT_INT8_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, STUDENT_INT8_WEIGHTS)
    quantized = load_int8()
    with torch.no_grad():
        gap = (student(inputs) - quantized(inputs)).squeeze(-1).abs()
    print(f"teacher       | eval {evaluate(load_reference()):8.1f}")
    print(f"student float | eval {evaluate(student):8.1f}")
    print(f"student int8  | eval {evaluate(quantized):8.1f} "
          f"| mean gap {gap.mean():.4f} | max gap {gap.max():.4f}")
    print(f"Int8 student saved: weights/{STUDENT_INT8_WEIGHTS.name}")


if __name__ == "__main__":
    main()
