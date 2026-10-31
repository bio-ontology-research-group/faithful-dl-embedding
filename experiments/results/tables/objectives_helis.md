# Objectives & Prisms: `helis`

- Provenance: `experiments` `385523a,fc6b75f` (dirty), `mowl` `e5d9a42,f61dc5a,f74b622` (dirty).
- Margin semantics: `Mod strict` evaluates GCI losses at `margin = 0`; `Mod@γ` evaluates at the training margin `γ` (`relaxing` for Box²EL/MultiBoxEL, `tightening` for ELBE/MultiELBE/PrismEL).
- Suite membership: the `suite` column distinguishes `objectives` methods (30 objective-grid methods + `prism`/`prism-init`) from `baseline` 10-29 methods.

| method | suite | backbone | k | objective | lr | d | Mod strict | Mod@γ | point-Mod (pos/hardneg) | HVR | Deg | eff-k | seeds n/k |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `elbe` | baseline | ELBE | 1 | axiom | 0.001 | 50 | 0.8633 ± 0.0026 | — | — | — | — | 0.68 ± 0.01 | 3/3 |
| `elbe-init` | baseline | ELBE | 1 | axiom | 0.001 | 50 | 0.9576 ± 0.0076 | — | — | — | — | 0.77 ± 0.00 | 3/3 |
| `box2el` | baseline | BoxSquaredEL | 1 | axiom | 0.001 | 50 | 0.0085 ± 0.0020 | — | — | — | — | 0.40 ± 0.02 | 3/3 |
| `box2el-init` | baseline | BoxSquaredEL | 1 | axiom | 0.001 | 50 | 0.1059 ± 0.0078 | — | — | — | — | 0.74 ± 0.02 | 3/3 |
| `canonical-mixture` | baseline | CanonicalMixture | 1 | axiom | 0.001 | 50 | 0.2694 ± 0.0000 | — | — | — | — | 1.00 ± 0.00 | 3/3 |
| `elem` | baseline | ELEm | 1 | axiom | 0.001 | 50 | 0.0600 ± 0.0027 | — | — | — | — | 1.00 ± 0.00 | 3/3 |
| `multibox-k2-axiom` | objectives | multibox | 2 | axiom | 0.001 | 50 | 0.0095 ± 0.0026 | 0.2766 ± 0.1025 | — | — | — | 0.92 ± 0.14 | 3/3 |
| `multibox-k2-point` | objectives | multibox | 2 | point | 0.001 | 50 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9600 ± 0.0009 / 1.0000 ± 0.0000 | — | — | 1.20 ± 0.02 | 3/3 |
| `multibox-k2-mixed` | objectives | multibox | 2 | mixed | 0.001 | 50 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9573 ± 0.0003 / 1.0000 ± 0.0000 | — | — | 1.13 ± 0.03 | 3/3 |
| `multibox-k2-hybrid` | objectives | multibox | 2 | hybrid | 0.001 | 50 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8769 ± 0.0653 / 1.0000 ± 0.0000 | — | — | 0.97 ± 0.02 | 3/3 |
| `multielbe-k2-axiom` | objectives | multielbe | 2 | axiom | 0.001 | 50 | 0.8818 ± 0.0036 | 0.6458 ± 0.0171 | — | — | — | 1.32 ± 0.04 | 3/3 |
| `multielbe-k2-point` | objectives | multielbe | 2 | point | 0.001 | 50 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9906 ± 0.0002 / 1.0000 ± 0.0000 | — | — | 1.56 ± 0.02 | 3/3 |
| `multielbe-k2-mixed` | objectives | multielbe | 2 | mixed | 0.001 | 50 | 0.0002 ± 0.0001 | 0.0000 ± 0.0000 | 0.9716 ± 0.0024 / 1.0000 ± 0.0000 | — | — | 1.38 ± 0.07 | 3/3 |
| `multielbe-k2-hybrid` | objectives | multielbe | 2 | hybrid | 0.001 | 50 | 0.6162 ± 0.0028 | 0.5091 ± 0.0049 | 0.9671 ± 0.0002 / 1.0000 ± 0.0000 | — | — | 1.31 ± 0.01 | 3/3 |
| `prism` | objectives | prism | 1 | axiom | 0.001 | 50 | 0.2436 ± 0.0009 | 0.2158 ± 0.0015 | — | — | — | 0.91 ± 0.02 | 3/3 |
| `prism-init` | objectives | prism | 1 | axiom | 0.001 | 50 | 0.2604 ± 0.0008 | 0.2571 ± 0.0010 | — | — | — | 0.97 ± 0.00 | 3/3 |
