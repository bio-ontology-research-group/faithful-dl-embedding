# Objectives & Prisms: `pizza`

- Provenance: `experiments` `d2dcde6,fc6b75f` (dirty), `mowl` `8a3e95a,f74b622` (dirty).
- Margin semantics: `Mod strict` evaluates GCI losses at `margin = 0`; `Mod@γ` evaluates at the training margin `γ` (`relaxing` for Box²EL/MultiBoxEL, `tightening` for ELBE/MultiELBE/PrismEL).
- Suite membership: the `suite` column distinguishes `objectives` methods (30 objective-grid methods + `prism`/`prism-init`) from `baseline` 10-29 methods.

| method | suite | backbone | k | objective | lr | d | Mod strict | Mod@γ | point-Mod (pos/hardneg) | HVR | Deg | eff-k | seeds n/k |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `elbe` | baseline | ELBE | 1 | axiom | 0.001 | 50 | 0.7110 ± 0.0097 | 0.1255 ± 0.0093 | — | — | — | 0.36 ± 0.02 | 3/3 |
| `elbe-init` | baseline | ELBE | 1 | axiom | 0.001 | 50 | 0.4808 ± 0.0239 | 0.0507 ± 0.0056 | — | — | — | 0.38 ± 0.01 | 3/3 |
| `elbe-anchor` | baseline | ELBE | 1 | axiom | 0.001 | 50 | 0.7046 ± 0.0024 | 0.0887 ± 0.0152 | — | — | — | 0.36 ± 0.01 | 3/3 |
| `box2el` | baseline | BoxSquaredEL | 1 | axiom | 0.001 | 50 | 0.2655 ± 0.0779 | 0.4129 ± 0.0665 | — | — | — | 0.05 ± 0.01 | 3/3 |
| `box2el-init` | baseline | BoxSquaredEL | 1 | axiom | 0.001 | 50 | 0.0085 ± 0.0009 | 0.0897 ± 0.0016 | — | — | — | 0.97 ± 0.00 | 3/3 |
| `box2el-anchor` | baseline | BoxSquaredEL | 1 | axiom | 0.001 | 50 | 0.0085 ± 0.0009 | 0.0897 ± 0.0016 | — | — | — | 0.97 ± 0.00 | 3/3 |
| `box2el-milp` | baseline | BoxSquaredEL | 1 | axiom | 0.001 | 50 | 0.0085 ± 0.0009 | 0.0897 ± 0.0016 | — | — | — | 0.97 ± 0.00 | 3/3 |
| `multibox-k1` | baseline | MultiBoxEL | 1 | axiom | 0.001 | 50 | 0.2655 ± 0.0779 | 0.4129 ± 0.0665 | — | — | — | 0.05 ± 0.01 | 3/3 |
| `multibox-k2` | baseline | MultiBoxEL | 2 | axiom | 0.001 | 50 | 0.0358 ± 0.0103 | 0.2559 ± 0.0676 | — | — | — | 0.20 ± 0.01 | 3/3 |
| `multibox-k3` | baseline | MultiBoxEL | 3 | axiom | 0.001 | 50 | 0.0000 ± 0.0000 | 0.0267 ± 0.0158 | — | — | — | 0.34 ± 0.04 | 3/3 |
| `multibox-k4` | baseline | MultiBoxEL | 4 | axiom | 0.001 | 50 | 0.0000 ± 0.0000 | 0.0069 ± 0.0024 | — | — | — | 0.50 ± 0.02 | 3/3 |
| `multibox-k2-init` | baseline | MultiBoxEL | 2 | axiom | 0.001 | 50 | 0.1966 ± 0.3169 | 0.2447 ± 0.2850 | — | — | — | 0.90 ± 0.61 | 3/3 |
| `multibox-k2-soft` | objectives | multibox | 2 | soft | 0.001 | 50 | 0.1715 | 0.3349 | — | — | — | 0.10 | 1/1 |
| `multibox-k2-point` | objectives | multibox | 2 | point | 0.001 | 50 | 0.0369 | 0.0769 | 1.0000 / 0.9997 | — | — | 1.73 | 1/1 |
| `multibox-k2-mixed` | objectives | multibox | 2 | mixed | 0.001 | 50 | 0.0016 | 0.0016 | 0.9065 / 0.9994 | — | — | 0.91 | 1/1 |
| `multibox-k2-hybrid` | objectives | multibox | 2 | hybrid | 0.001 | 50 | 0.1026 | 0.4263 | 0.7477 / 1.0000 | — | — | 0.44 | 1/1 |
| `multielbe-k2-soft` | objectives | multielbe | 2 | soft | 0.001 | 50 | 0.5913 | 0.0721 | — | — | — | 0.67 | 1/1 |
| `multielbe-k2-point` | objectives | multielbe | 2 | point | 0.001 | 50 | 0.0016 | 0.0000 | 1.0000 / 0.9994 | — | — | 1.72 | 1/1 |
| `multielbe-k2-mixed` | objectives | multielbe | 2 | mixed | 0.001 | 50 | 0.0321 | 0.0016 | 0.9252 / 0.9982 | — | — | 1.07 | 1/1 |
| `multielbe-k2-hybrid` | objectives | multielbe | 2 | hybrid | 0.001 | 50 | 0.3574 | 0.1731 | 0.9346 / 0.9979 | — | — | 1.00 | 1/1 |
