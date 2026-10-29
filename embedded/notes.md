# ONNX vers STM32CubeAI

`weights/student.onnx` (élève float32) est importé dans STM32CubeMX avec X-CUBE-AI 10.2.1, cible `stm32f7`, sans compression. L'analyse donne la place occupée en flash et en RAM. La carte est une Nucleo-F756ZG : 1 Mo de flash (1 048 576 o) et 320 Ko de RAM (327 680 o).

Mesure du 29 octobre 2026, modèle `student`, hash `0x4e5f3b75917d09021c4dc2411cfc5cab`.

| | Occupé | Disponible | Part |
|---|---:|---:|---:|
| Flash | 11 544 o (11,3 Ko) | 1 048 576 o | 1,1 % |
| RAM | 1 940 o (1,9 Ko) | 327 680 o | 0,6 % |

Détail :

| | Flash | RAM |
|---|---:|---:|
| Poids | 1 412 o | — |
| Bibliothèque X-CUBE-AI et `libm` | 10 132 o | 1 812 o |
| Activations | — | 128 o |

L'entrée `observation` (12 o) et la sortie `torque` (4 o) sont comprises dans les activations. Le réseau lui-même est petit ; l'essentiel du flash est le code qui calcule les couches.
