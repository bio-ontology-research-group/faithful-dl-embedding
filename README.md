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

The embedding method itself will be added here.
