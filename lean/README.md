# Lean formalisation

Machine-checked versions of the results in `../impossibility.tex`.

* `GeometricFaithful/Theory.lean` — the theory `T₃` in the EL++ normal form of the
  kobayashi-marust certification library (`ContextCalculus.ELCompletion`), its
  three-element model, and the generic Helly-type obstruction
  (`no_nondegenerate_model_of_helly2`).
* `GeometricFaithful/Boxes.lean` — one-dimensional Helly for intervals, Helly number 2
  of boxes, `no_box_model` (any dimension), `no_convex_model_dim1`.
* `GeometricFaithful/Balls.lean` — two balls whose intersection is a ball are nested;
  `no_ball_model`, `no_ball_model_euclidean` (`ℝᵈ`, `d ≥ 2`).
* `GeometricFaithful/Convex.lean` — a convex planar model of `T₃` (`planeModel`), and
  the Helly dimension bound for convex regions (`helly_dimension_bound`).

## Building (on ws only)

The project depends on Mathlib (`v4.30.0-rc2`, the version used by kobayashi-marust)
and on `/home/leechuck/Public/software/kobayashi-marust/lean` (path dependency).

```
rsync -a --exclude .lake lean/ ws:lean-work/gfe/
ssh ws 'cd lean-work/gfe && export PATH=$HOME/.elan/bin:$PATH && lake exe cache get && lake build'
```
