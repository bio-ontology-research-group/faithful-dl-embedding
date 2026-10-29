# Helly-Satisfiability Diagnostics on Real Ontologies

This directory contains `helly_diagnostic.py`, which evaluates the EL++ normalisation, canonical model completion, full Helly-satisfiability check (`mowl.geometric.helly_check`), and targeted NF2 certificate-triangle search (`mowl.geometric.certificate_triangles`) on OWL ontologies.

## Usage

Run with the `mowl-gpu` virtual environment (`~/.venvs/mowl-gpu/bin/python`):

```bash
~/.venvs/mowl-gpu/bin/python experiments/diagnostics/helly_diagnostic.py \
  --ontology /path/to/ontology.owl \
  --name pizza \
  --out experiments/diagnostics/results/ \
  --k-max 4 \
  --max-certificates 100000 \
  --timeout-seconds 900 \
  --jvm-memory 4g
```

### CLI Options

- `--ontology PATH`: Path to the input OWL ontology file (loaded via `mowl.datasets.PathDataset`).
- `--name NAME`: Short identifier for the ontology; writes `<out>/<NAME>.json` and updates the corresponding row in `<out>/summary.md`.
- `--out DIR`: Output directory for `<NAME>.json` and `summary.md`.
- `--k-max INT` (default: `4`): Maximum minimal unsatisfiable concept-name set size inspected by `helly_check`.
- `--max-certificates INT` (default: `100000`): Cutoff on the number of strong certificates collected by `helly_check`.
- `--timeout-seconds INT` (default: `1800`): Hard wall-clock timeout in seconds for the spawned `helly_check` child process.
- `--jvm-memory STR` (default: `24g`): Maximum JVM heap size passed to `mowl.init_jvm`.

## Columns in `results/summary.md`

| Column | Meaning |
|---|---|
| `Ontology` | Short identifier (`--name`) of the ontology. |
| `Concepts` | Total number of concept names in `NormalisedOntology.concept_names` (including `owl:Thing`, `owl:Nothing`, and auxiliary concepts introduced by `ELNormalizer`). |
| `Roles` | Total number of object property names in `NormalisedOntology.role_names` (including any auxiliary chain roles). |
| `NF1` | Count of `NF1` (`A ⊑ B`, including `A ⊑ ⊥`) axioms after normalisation. |
| `NF2` | Count of `NF2` (`A1 ⊓ A2 ⊑ B`, including `A1 ⊓ A2 ⊑ ⊥`) axioms after normalisation. |
| `NF3` | Count of `NF3` (`A ⊑ ∃r.B`) axioms after normalisation. |
| `NF4` | Count of `NF4` (`∃r.A ⊑ B`, including `∃r.A ⊑ ⊥`) axioms after normalisation. |
| `NF6` | Count of `NF6` (`r ⊑ s`) role-inclusion axioms after normalisation. |
| `NF7` | Count of `NF7` (`r1 ∘ r2 ⊑ s`) role-composition axioms after normalisation. |
| `⊥-axioms` | Total count of normal-form axioms (`NF1`, `NF2`, `NF4`) with `sup == owl:Nothing`. |
| `Dropped` | Count of non-EL++ TBox and RBox axioms dropped during `normalise(ds)`. |
| `CM names` | Number of satisfiable concept names (`len(cm.names)`, including `owl:Thing`), equal to the domain size of the canonical model. |
| `Unsat` | Number of unsatisfiable concept names in the signature (excluding `owl:Nothing` itself). |
| `CM time (s)` | Total wall-clock time (seconds) for `normalise(ds)` plus `CanonicalModel` saturation. |
| `⊥-rel sat` | Number of `⊥`-relevant satisfiable concept names (`(_bottom_relevant_names(cm.axioms) & set(cm.names)) - {TOP}`). |
| `Helly status` | Completion status of `helly_check`: `complete`, `incomplete(roles)`, `incomplete(cutoff)`, or `TIMEOUT (Ns)`. |
| `Strong certs` | Number of strong Helly certificates returned by `helly_check` (`—` on timeout). |
| `Nondeg certs` | Number of non-degenerate Helly certificates returned by `helly_check` (`—` on timeout). |
| `Helly time (s)` | Wall-clock time (seconds) spent inside `helly_check` (or elapsed time until timeout). |
| `Named conj` | Number of unordered concept-name pairs `{A, B}` with a direct `NF2(A, B, N)` definition satisfying `O ⊨ N ≡ A ⊓ B` (`A, B, N` satisfiable and distinct from `⊤, ⊥`). |
| `Graph △` | Number of triangles `{X1, X2, X3}` in the named-conjunction graph before joint-satisfiability filtering. |
| `Cert △` | Number of non-degenerate singleton Helly certificate triangles returned by `mowl.geometric.certificate_triangles(cm, max_triangles=None)`. |
| `△ time (s)` | Wall-clock time (seconds) taken by `certificate_triangles(cm, max_triangles=None)`. |
| `Discrepancies` | Number of singleton non-degenerate certificates where `helly_check` and `certificate_triangles` disagree (`0` when in full agreement or when `helly_check` timed out). |
