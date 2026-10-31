# Objectives & Prisms: `yeast-function`

- Provenance: `experiments` `199295f,385523a,512b186,52d39ae,e7365eb,fc6b75f,fd972f8` (dirty), `mowl` `23c69aa,95cb121,ac6bd30,db4d483,e5d9a42,f61dc5a` (dirty).
- Margin semantics: `Mod strict` evaluates GCI losses at `margin = 0`; `Mod@γ` evaluates at the training margin `γ` (`relaxing` for Box²EL/MultiBoxEL, `tightening` for ELBE/MultiELBE/PrismEL).
- Suite membership: the `suite` column distinguishes `objectives` methods (30 objective-grid methods + `prism`/`prism-init`) from `baseline` 10-29 methods.

| method | suite | backbone | k | objective | lr | d | Mod strict | Mod@γ | point-Mod (pos/hardneg) | HVR | Deg | eff-k | seeds n/k |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `elbe` | baseline | ELBE | 1 | axiom | 0.001 | 50 | 0.1666 ± 0.0856 | — | — | — | — | 0.28 ± 0.02 | 3/3 |
| `elbe-init` | baseline | ELBE | 1 | axiom | 0.001 | 50 | 0.2914 ± 0.0002 | — | — | — | — | 0.37 ± 0.00 | 3/3 |
| `box2el` | baseline | BoxSquaredEL | 1 | axiom | 0.001 | 50 | 0.0035 ± 0.0004 | — | — | — | — | 0.06 ± 0.01 | 3/3 |
| `box2el-init` | baseline | BoxSquaredEL | 1 | axiom | 0.001 | 50 | 0.0089 ± 0.0002 | — | — | — | — | 0.08 ± 0.01 | 3/3 |
| `canonical-mixture` | baseline | CanonicalMixture | 1 | axiom | 0.001 | 50 | 0.6156 ± 0.0000 | — | — | — | — | 1.00 ± 0.00 | 3/3 |
| `elem` | baseline | ELEm | 1 | axiom | 0.001 | 50 | 0.0893 ± 0.0004 | — | — | — | — | 1.00 ± 0.00 | 3/3 |
| `multibox-k2-axiom` | objectives | multibox | 2 | axiom | 0.001 | 50 | 0.0045 ± 0.0003 | 0.0448 ± 0.0017 | — | — | — | 0.18 ± 0.02 | 3/3 |
| `multibox-k2-point` | objectives | multibox | 2 | point | 0.001 | 50 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6162 ± 0.0532 / 0.9970 ± 0.0015 | — | — | 0.66 ± 0.01 | 3/3 |
| `multibox-k2-mixed` | objectives | multibox | 2 | mixed | 0.001 | 50 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5111 ± 0.0609 / 0.9971 ± 0.0016 | — | — | 0.65 ± 0.01 | 3/3 |
| `multibox-k2-hybrid` | objectives | multibox | 2 | hybrid | 0.001 | 50 | 0.0117 ± 0.0020 | 0.0213 ± 0.0034 | 0.8278 ± 0.0026 / 0.9913 ± 0.0009 | — | — | 0.42 ± 0.03 | 3/3 |
| `multielbe-k2-axiom` | objectives | multielbe | 2 | axiom | 0.001 | 50 | 0.1340 ± 0.0653 | 0.0545 ± 0.0343 | — | — | — | 0.49 ± 0.02 | 3/3 |
| `multielbe-k2-point` | objectives | multielbe | 2 | point | 0.001 | 50 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0040 ± 0.0006 / 1.0000 ± 0.0000 | — | — | 0.69 ± 0.00 | 3/3 |
| `multielbe-k2-mixed` | objectives | multielbe | 2 | mixed | 0.001 | 50 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0025 ± 0.0002 / 1.0000 ± 0.0000 | — | — | 0.69 ± 0.00 | 3/3 |
| `multielbe-k2-hybrid` | objectives | multielbe | 2 | hybrid | 0.001 | 50 | 0.0038 ± 0.0000 | 0.0010 ± 0.0000 | 0.0054 ± 0.0004 / 1.0000 ± 0.0000 | — | — | 0.57 ± 0.00 | 3/3 |
| `prism` | objectives | prism | 1 | axiom | 0.001 | 50 | 0.1610 ± 0.0010 | 0.0904 ± 0.0006 | — | — | — | 0.33 ± 0.00 | 3/3 |
| `prism-init` | objectives | prism | 1 | axiom | 0.001 | 50 | 0.3092 ± 0.0006 | 0.2039 ± 0.0016 | — | — | — | 0.39 ± 0.00 | 3/3 |
