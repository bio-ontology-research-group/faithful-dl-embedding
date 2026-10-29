# Experiment Dataset Statistics (`Task 6`)

Generated from the training ontologies of the 7 datasets (`DATASETS = ["tk3", "tk4", "tk5", "tk6", "pizza", "go", "foodon"]`) using `mowl.geometric.normalise`, `CanonicalModel`, `certificate_triangles`, and `helly_check`.

## 1. Main Dataset Statistics Table

| Dataset | Task | Dataset Classes (`len(ds.classes)`) | Normalised Concept Names (incl. `⊤`, `⊥`, aux) | Roles (`len(no.role_names)`) | Total Normalised Train Axioms | NF1 (`A ⊑ B` / `⊥`) | NF2 (`A1 ⊓ A2 ⊑ B` / `⊥`) | NF3 (`A ⊑ ∃r.B`) | NF4 (`∃r.A ⊑ B` / `⊥`) | NF6 (`r ⊑ s`) | NF7 (`r1 ∘ r2 ⊑ s`) | `⊥`-Axioms (NF1 / NF2 / NF4) | Dropped Axioms (Train) | CM Satisfiable Names (`len(cm.names)`) | CM Unsat Names | `n_certificate_triangles` (Train) | `helly_check` Nondeg / Strong Certs (`k_max=k`) | Train / Valid / Test Split Sizes | Deductive Closure Axioms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tk3` | `satisfaction` | 9 | 9 | 0 | 12 | 6 | 6 | 0 | 0 | 0 | 0 | 3 (0 / 3 / 0) | 0 | 8 | 0 | 1 | 1 / 1 | 12 / 0 / 0 | — |
| `tk4` | `satisfaction` | 14 | 14 | 0 | 27 | 14 | 13 | 0 | 0 | 0 | 0 | 6 (0 / 6 / 0) | 0 | 13 | 0 | 0 | 4 / 7 | 27 / 0 / 0 | — |
| `tk5` | `satisfaction` | 20 | 20 | 0 | 46 | 24 | 22 | 0 | 0 | 0 | 0 | 10 (0 / 10 / 0) | 0 | 19 | 0 | 0 | 10 / 23 | 46 / 0 / 0 | — |
| `tk6` | `satisfaction` | 27 | 27 | 0 | 69 | 36 | 33 | 0 | 0 | 0 | 0 | 15 (0 / 15 / 0) | 0 | 26 | 0 | 0 | 20 / 56 | 69 / 0 / 0 | — |
| `pizza` | `subsumption` | 109 | 109 | 8 | 628 | 56 | 404 | 153 | 9 | 4 | 2 | 398 (0 / 398 / 0) | 0 (52 dropped when extracting full EL fragment) | 106 | 2 | 0 | 0 / 0 (`k_max=4`) | 628 / 17 / 17 | 435 |
| `go` | `subsumption` | 44,246 | 56,191 | 9 | 105,296 | 61,503 | 11,910 | 19,996 | 11,878 | 3 | 6 | 30 (0 / 30 / 0) | 1 | 56,190 | 0 | 0 | — | 105,296 (81,844 OWL TBox) / 7,260 / 14,521 | 447,026 |
| `foodon` | `subsumption` | 28,184 | 29,136 | 83 | 35,253 | 21,795 | 1,762 | 10,719 | 897 | 60 | 20 | 495 (0 / 495 / 0) | 2,272 | 29,135 | 0 | 0 | — | 35,253 (34,224 OWL TBox) / 2,977 / 5,957 | 375,820 |

## 2. File Sizes and Timing Breakdown

| Dataset | Download Time (s) | Tarball Size (bytes / MB) | `ontology.owl` Size (bytes / MB) | `valid.owl` Size (bytes / MB) | `test.owl` Size (bytes / MB) | `ontology_deductive_closure.owl` Size (bytes / MB) | Load Time (s) | Normalise Time (s) | CM Saturation Time (s) | `certificate_triangles` Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tk3` | — | — | 3,836 (0.004 MB) | — | — | — | 0.37 | 0.10 | 0.0004 | 0.0007 |
| `tk4` | — | — | 6,887 (0.007 MB) | — | — | — | 0.04 | 0.02 | 0.0003 | 0.0005 |
| `tk5` | — | — | 10,788 (0.010 MB) | — | — | — | 0.03 | 0.15 | 0.0006 | 0.0017 |
| `tk6` | — | — | 15,551 (0.015 MB) | — | — | — | 0.03 | 0.03 | 0.0008 | 0.0018 |
| `pizza` | — | — | 249,002 (0.237 MB) | 7,324 (0.007 MB) | 7,200 (0.007 MB) | 62,687 (0.060 MB) | 0.30 | 0.33 | 0.0028 | 0.0001 |
| `go` | 8.74 | 8,055,490 (7.68 MB) | 151,742,068 (144.71 MB) | 2,085,356 (1.99 MB) | 3,735,624 (3.56 MB) | 43,831,468 (41.80 MB) | 16.79 | 13.33 | 29.45 | 0.1334 |
| `foodon` | 6.20 | 2,933,449 (2.80 MB) | 32,926,859 (31.40 MB) | 959,359 (0.91 MB) | 1,758,741 (1.68 MB) | 37,062,961 (35.35 MB) | 5.84 | 5.61 | 1.81 | 0.0210 |

## 3. Protocol Notes & Recorded Rulings

1. **Synthetic `T_k` (`tk4`–`tk6`) Certificate Structure**:
   - In `T_k` for `k >= 4`, the ontology defines the `(k - 1)`-wise conjunctions `A_i ≡ ⊓_{j != i} X_j` (binary-normalised with shared prefix auxiliary equivalences `Aux_1_2 ≡ X_1 ⊓ X_2`, etc.) rather than naming all three pairwise edges of an unsatisfiable 3-clique, and every 3-subset of `{X_1, ..., X_k}` is satisfiable.
   - Consequently, `certificate_triangles(cm)` returns `[]` (`n_certificate_triangles = 0`) on `tk4`, `tk5`, and `tk6`, while `helly_check(cm, k_max=k)` finds the minimal unsatisfiable set of size `k` as a non-degenerate `HellyCertificate` with `c1=(X1,)`, `c2=(X2,)`, `c3=(X3, ..., Xk)` (a tuple of `k - 2` names) and `pairwise_names=(Aux_1_2, A2, A1)`.
   - The Helly metrics in Task 7 therefore evaluate `helly_check` certificates with tuple `c3` on `tk4`–`tk6`.
2. **Pizza-EL Split Fraction Ruling (Spec §5.1, §7)**:
   - Normalising `~/tmp/gfe-data/pizza/pizza.owl` produces `662` normalised EL++ axioms (`90` `NF1`, `404` `NF2`, `153` `NF3`, `9` `NF4`, `4` `NF6`, `2` `NF7`; `52` non-EL++ axioms dropped).
   - Of the `90` `NF1` axioms, `1` involves an auxiliary normalisation entity (`AuxiliaryEntity111 ⊑ PizzaTopping`) and `3` have an unsatisfiable subclass (`CheeseyVegetableTopping ⊑ CheeseTopping`, `CheeseyVegetableTopping ⊑ VegetableTopping`, `IceCream ⊑ Food`), leaving `86` asserted atomic subsumptions between distinct satisfiable original named classes.
   - With `seed = 0`, an initial `10 % / 10 %` split yields `round(0.10 * 86) = 9 < 20` test axioms. Per Spec §7, the split fractions were raised to `20 % / 20 %`, yielding `17` validation axioms (`valid.owl`), `17` test axioms (`test.owl`), and `628` training axioms (`ontology.owl`, containing `56` `NF1` axioms).
   - Because the four generated Pizza-EL OWL files total `326,213` bytes (`< 1 MB`), they are stored in `~/tmp/gfe-data/pizza-el/` and mirrored in `experiments/data/pizza-el/`.
