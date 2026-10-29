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

## Test sur entrées fixes

Le 30 octobre 2026, le code généré tourne sur la Nucleo. `firmware/Core/Src/main.c` envoie les quatre observations de `export_onnx.py` au réseau et compare le couple à `policy()` sur le PC. La carte écrit le résultat sur USART3, 115200 bauds.

| Observation (cos θ, sin θ, θ̇) | Couple carte | Couple PC | Écart affiché |
|---|---:|---:|---:|
| +0,3, −0,4, +4 | −0,687599 | −0,687599 | 0 |
| +1, 0, 0 | +0,031638 | +0,031638 | 0 |
| 0, +1, −8 | +1,002009 | +1,002009 | 0 |
| −0,7, +0,7, +2,5 | +2,492419 | +2,492420 | 0 |

Les quatre lignes finissent par `ok`. À six décimales, la carte et le PC donnent le même couple. Sur la dernière ligne, l'attendu affiché est arrondi à `+2,492420` et le couple carte à `+2,492419` ; l'écart réel reste sous un demi-millionième, donc l'affichage le montre à `0`. Le réseau float32 embarqué reproduit la politique du PC sur ces entrées.
