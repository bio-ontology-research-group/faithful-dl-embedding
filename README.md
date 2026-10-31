# faithful-dl-embedding

Code for the project on *faithful geometric embeddings of description logics*
(Robert Hoehndorf, KAUST / Bio-Ontology Research Group). Companion to the write-up
"Helly-type limits of region-based models of EL and ALC theories" (paper repository
`geometric-faithful-embeddings`, Overleaf).

## Contents

* `lean/` — Lean 4 + Mathlib formalisation, machine-checked on the group workstation
  (`lean/BUILD_LOG.txt` has the build and `#print axioms` audit). The theory
  `T₃ = {A ≡ X₁⊓X₂, B ≡ X₂⊓X₃, C ≡ X₁⊓X₃, A⊓B ⊑ ⊥, B⊓C ⊑ ⊥, A⊓C ⊑ ⊥}` is stated in the
  EL++ normal-form semantics of the [kobayashi-marust](https://github.com/bio-ontology-research-group/kobayashi-marust)
  certification library. Checked results: the generic Helly-2 obstruction; no box model in
  any dimension; no convex model on the line; no ball model in `ℝᵈ`, `d ≥ 2`; a convex
  planar model of `T₃`; Helly's dimension lower bound for convex regions.
  See `lean/README.md` for build instructions.
* `scripts/check_models.py` — numeric sanity checks of the constructions (discs, half-space
  models of `T_k`, box Helly property, al-cones, unions of intervals).
* `experiments/` — learnable geometric DL embedding experiments across 9 datasets,
  6 sweep suites, and 52 method configurations:
  * **Datasets (9):**
    * Synthetic Helly stress suite (`satisfaction`): `tk3`, `tk4`, `tk5`, `tk6` (`T_k` for `k ∈ {3, 4, 5, 6}`).
    * TBox subsumption benchmarks (`subsumption`): `pizza` (Pizza-EL), `go` (Gene Ontology), `foodon` (FoodOn).
    * ABox assertion prediction benchmarks (`abox`): `yeast-function` (B1 yeast protein function prediction over the GO EL++ TBox and PPI network) and `helis` (B3 HeLiS food/nutrient class membership, OWL2Vec\* protocol).
  * **Suites (6):** `synthetic` (`tk3`–`tk6`), `pizza` (`pizza`), `real` (`go`, `foodon`), `objectives` (`tk3`–`tk6` across `lr ∈ {0.001, 0.01}`), `abox` (`yeast-function`, `helis`), and `all` (all 9 datasets).
  * **Method families (52 method IDs in `experiments/config.py`):**
    * Single-box and multi-box baselines (`ELBE`, `BoxSquaredEL`, `MultiBoxEL`): `elbe`, `elbe-init`, `elbe-anchor`, `box2el`, `box2el-init`, `box2el-anchor`, `box2el-milp` (`pizza` only), `multibox-k1`..`multibox-k4`, `multibox-k2-init`.
    * Ball baseline (`ELEm`): `elem` (subsumption and ABox benchmarks).
    * Canonical-model mixture (`CanonicalMixture`): `canonical-mixture` (ABox benchmarks `yeast-function`, `helis`).
    * Multi-component objective grid (`MultiBoxEL` and `MultiELBE`): 36 method IDs `{multibox,multielbe}-k{1,2,3}-{axiom,soft,softdiv,point,mixed,hybrid}`.
    * Triangular-prism embeddings (`PrismEL`): `prism`, `prism-init`.
  * **Main result reports and tables:**
    * [`experiments/results/OBJECTIVES.md`](experiments/results/OBJECTIVES.md) — objective sweep verdict on `tk3`–`tk6`, Prism results, and Pizza sanity check.
    * [`experiments/results/ABOX.md`](experiments/results/ABOX.md) — ABox benchmark results and diagnostics on `yeast-function` (B1) and `helis` (B3).
    * [`experiments/results/tables/`](experiments/results/tables/README.md) — per-dataset summary tables (`{dataset}.md`, `objectives_{dataset}.md`, `multibox_curve.md`) and [`experiments/results/dataset_stats.md`](experiments/results/dataset_stats.md).
  * See [`experiments/README.md`](experiments/README.md) and [`experiments/data/README.md`](experiments/data/README.md) for layout, data provenance, and CLI usage.
