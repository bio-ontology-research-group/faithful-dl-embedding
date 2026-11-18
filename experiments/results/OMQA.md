# E-OMQA (Andresel et al. 2023) ontology-mediated query answering on LUBM and NELL: GroundedEL

Pipeline: `experiments/omqa/build_eomqa.py` (KB tables from the released integer-id query/answer pickles: individuals, class
entities, mined DL-Lite TBox — class hierarchy, role inclusions, inverses, range axioms; `onto` tables = `train_plain` facts plus
the certain answers of `train_onto`), `experiments/omqa/train.py --benchmark eomqa` (GroundedEL with translation-box roles trained
on the KB tables only; corrupted-fact negatives; HITS@3 checkpoint selection on the first test case's validation queries),
Query2Box protocol (`experiments/omqa/protocol.py`: filtered ranking of every hard answer against all entities, macro over queries,
HITS@3 per shape, macro over the nine shapes). Test cases: I (inductive: missing facts), D (deductive: answers that need the
ontology), I+D. **Scorer of record: `--scorer beam --beam-k 16`** (max–min over the top-k witnesses per hop; crisp region depth
for the anchored hops). The region scorer (`compile_query` + `score_entities`) remains the crisp membership test (soundness); as
a *ranking* it is unreliable on type queries because it ranks class answers among all individuals inside the class boxes (note
§4.6, §4.8). Raw `results.json` under `runs/qa/<cell>/` on orix and `~/tmp/gfe/omqa/results/runs/` locally; details in
`notes/experiments-2026-11-08-qa.md` §4 (paper repo). The previous, region-scored version of this file is kept at
`~/tmp/gfe/omqa/OMQA.md.region-scorer.bak`.

Recipe: d 128, K 2, K_r 4, `onto` tables, corrupted-fact negatives (weight 1, k 4), τ floor 0.1 after 20 τ-epochs, beam-16
validation every 25 epochs with HITS@3 selection. LUBM: fixed constraint weights, 200 epochs. NELL: Lagrangian dual weights
(`--constraint-mode lagrangian --dual-rho 1.0`), 100 epochs. Everything else identical.

## Headline: macro HITS@3 over the nine shapes (beam-16; mean ± population sd over seeds 0–2)

| KB | model | D | I+D | I |
|---|---|---|---|---|
| LUBM | **GroundedEL, 200 epochs** | **0.689 ± 0.006** | **0.489 ± 0.002** | – (not evaluated; 100-epoch cells: 0.205) |
| LUBM | GroundedEL, 100 epochs | 0.638 ± 0.011 | 0.461 ± 0.008 | 0.205 |
| LUBM | GroundedEL, 400 epochs (seed 0, best of a 500-epoch run; seeds 1–2 running) | 0.720 | 0.503 | |
| LUBM | GroundedEL, 300 epochs (seed 0) | 0.710 | 0.500 | |
| LUBM | Query2Box plain / rew (paper) | 0.189 / 0.248 | 0.218 / – | |
| LUBM | Query2Box gen / O2B gen (paper) | – | 0.458 / 0.490 | |
| LUBM | CQD plain / rew / ASR (paper) | 0.225 / 0.228 / – | 0.179 / – / 0.560 | |
| LUBM | CQD-Beam, own run (pipeline check, see below) | 0.072 | 0.087 | 0.453 |
| LUBM | QTO, own run (pipeline check) | 0.076 | 0.095 | 0.657 |
| NELL | **GroundedEL, Lagrangian ρ = 1** | **0.632 ± 0.005** | **0.505 ± 0.003** | 0.115 |
| NELL | GroundedEL, Lagrangian ρ = 1, 200 epochs (seed 0; seeds 1–2 and 400 epochs running) | 0.707 | 0.559 | 0.129 |
| NELL | GroundedEL, fixed weights | 0.615 ± 0.003 | 0.495 ± 0.001 | 0.115 |
| NELL | Query2Box plain / rew (paper) | 0.617 / 0.683 | 0.458 / – | |
| NELL | Query2Box gen / O2B gen (paper) | – | 0.636 / 0.655 | |
| NELL | CQD plain / rew / ASR (paper) | 0.656 / 0.743 / – | 0.555 / – / 0.770 | |
| NELL | CQD-Beam, own run (pipeline check) | 0.520 | 0.479 | 0.279 |
| NELL | QTO, own run (pipeline check) | 0.527 | 0.494 | 0.346 |

## Per shape, HITS@3, test D (headline cells, seed 0; three-seed sd ≤ 0.03 per shape)

| KB | cell | 1p | 2p | 3p | 2i | 3i | ip | pi | 2u | up |
|---|---|---|---|---|---|---|---|---|---|---|
| LUBM | 200 epochs, s0 | .845 | .646 | .558 | .832 | .817 | .639 | .464 | .783 | .630 |
| LUBM | 200 epochs, s1 | .842 | .652 | .552 | .854 | .832 | .638 | .465 | .813 | .624 |
| LUBM | 200 epochs, s2 | .831 | .633 | .518 | .841 | .826 | .618 | .463 | .810 | .598 |
| LUBM | 400 epochs, s0 | .825 | .711 | .598 | .838 | .821 | .674 | .494 | .807 | .713 |
| LUBM | 300 epochs, s0 | .820 | .688 | .613 | .834 | .822 | .665 | .477 | .799 | .675 |
| LUBM | 100 epochs, s0 | .845 | .558 | .439 | .852 | .831 | .569 | .442 | .811 | .528 |
| NELL | ρ = 1, s0 | .848 | .488 | .317 | .936 | .938 | .500 | .595 | .753 | .291 |
| NELL | ρ = 1, s1 | .839 | .488 | .322 | .937 | .936 | .494 | .595 | .747 | .284 |
| NELL | ρ = 1, s2 | .851 | .509 | .322 | .938 | .941 | .517 | .616 | .762 | .296 |
| NELL | ρ = 1, 200 epochs, s0 | .867 | .631 | .369 | .957 | .955 | .660 | .738 | .837 | .350 |
| NELL | fixed weights, s0 | .846 | .518 | .298 | .878 | .858 | .508 | .594 | .765 | .307 |

Test I+D, same cells (seed 0): LUBM 200 ep 1p .617, 2p .483, 3p .413, 2i .583, 3i .536, ip .412, pi .316, 2u .573, up .461;
NELL ρ = 1 1p .724, 2p .398, 3p .276, 2i .754, 3i .732, ip .394, pi .419, 2u .597, up .245.

## Ablations (beam-16 unless stated; macro HITS@3 D / I+D, seed 0)

| ablation | LUBM | NELL | reading |
|---|---|---|---|
| plain tables (no `train_onto` answers), neg k 4 / k 1 | .102 / .076 | .134 / .121 | the ontology tables are essential |
| negatives k 16 instead of 4 (LUBM, ep 90) | .532 / .385 | – | more negatives hurt |
| d 512, K 4, K_r 8 (last epoch) | .484 / .348 | – / .428 | more capacity hurts under the same schedule |
| Lagrangian ρ = 1 | .344 / .271 | **.629 / .504** | KB-specific: collapses on LUBM, best on NELL |
| Lagrangian ρ = 3 | – | .583 / .467 | projections pay; ρ = 10 collapsed (region-scored) |
| τ floor 0.3 instead of 0.1 | .538 / .405 | .622 / .504 | KB-specific; ρ = 1 + τ 0.3 on NELL is worse than either (.554 / .442) |
| 100 → 200 → 300 → 400 epochs | LUBM .653 → .687 → .710 → .720 | NELL ρ = 1: .629 → .707 at 200 | projection shapes carry the gain; LUBM anchored shapes oscillate late (selection at 400 of 500) |
| beam width k = 1 / 4 / 16 / 64 / 256 (NELL D) | – | .535 / .619 / .619 / .620 / .621 | greedy loses 2p/pi/up; k ≥ 4 saturates |
| region scorer instead of beam (same checkpoints) | .408 (s1) vs .618; .278 (s0) vs .653 | .541 vs .620 | ranking artefact on type queries; see note §4.6/§4.8 |

## Own-run baselines (pipeline check; `~/tmp/gfe/omqa/baselines/RESULTS.md`)

CQD-Beam (KGReasoning fork, ComplEx-N3 d 1000, step 50 k; t-norm prod, k 4 chosen on the I+D validation subset) and QTO (ComplEx
rank 1000 via ssl-relation-prediction, fraction 10, threshold 2e-4) were run on our converted data (`train.txt` = the E-OMQA
`train_plain` 1p facts, which are not materialised: one asserted class per individual). Both land far below the published plain
CQD (LUBM D .07 vs .225, NELL .52 vs .656) with the same shape profile (2i/3i strong, 2u/up weak), and far above GroundedEL on I.
The link predictor is the limit: CQD's 1p validation MRR on LUBM is ≈ .115 for N3 ∈ {0.1, 0.01, 0.001} from step 2 k on, QTO's
ComplEx has valid MRR .015. The paper therefore compares against the published numbers; the own runs document that the data
conversion and metric code path agree with the baselines' expectations (easy answers rank at ≈ 1.0 in all runs).

## Reading

GroundedEL is a reasoner, not a link predictor: on D it is 2.8× the best published LUBM number and within .05–.11 of the best
published NELL numbers (which use a strong link predictor plus query rewriting); on I it is far below any KGE (LUBM .20 vs QTO
.66), and I+D inherits that. The remaining NELL gap sits in the role-composition shapes (3p ≈ .32, up ≈ .29), not in search (beam
width settled), seeds (sd ≤ .005) or checkpoint selection. Schedule is the main lever on both KBs: LUBM saturates near 400 epochs (D .720), NELL ρ = 1 reaches D .707 at 200 epochs and
is still rising; seeds for both and a 400-epoch NELL run are in progress (2026-11-10) and will replace the headline rows.

## Query2Box, our runs of the released code (orix jobs 390577/390578, 2026-11-15)

Authors' `code_q2b_onto` with the `example.sh` settings (d 400, γ 24, batch 512, 128 negatives, lr 1e-4, 450,000 steps,
one seed); macro HITS@3 over the nine shapes at step 450,000. `plain` = plain training queries, `gen` = the benchmark's
ontology-generalised training queries.

| KB | training queries | D | I+D |
|---|---|---|---|
| LUBM | plain | 0.166 | 0.167 |
| LUBM | gen | 0.435 | 0.412 |
| NELL | plain | 0.523 | 0.458 |
| NELL | gen | 0.734 | 0.641 |

O2B (`--tbox --mean_gen`) did not finish: about 5,000–15,000 of 450,000 steps after 7 hours in the same jobs
(NELL `gen` cell: BrokenPipeError in the data loader). The published O2B numbers are used instead.

## Verified TBox and "deduce from predicted facts" (2026-11-18; code in `experiments/omqa/hybrid/`)

The TBox mined from the generalisation files contains 33 LUBM axioms (24 range axioms, 9 class inclusions) that do not
hold in the benchmark's own enriched training answers (`hybrid/tbox.py`, list in `~/tmp/gfe/hybrid/tbox_lubm.verify.log`;
NELL: all axioms hold). Verified TBox: `~/tmp/gfe/hybrid/tbox_{lubm,nell}.json`; tables `*_vt` (`build_eomqa.py --tbox-json`).
All LUBM numbers above this section use the unfiltered TBox.

Macro HITS@3 (D / I+D), one seed / one run each:

| KB | input | exact reasoning (`hybrid/exact_ceiling.py`) | GroundedEL (K 2, d 128, 400 ep, beam 16) | QTO | QTO + rule closure (`hybrid/closure_adj.py`) |
|---|---|---|---|---|---|
| LUBM | plain | .740 / .491 (unfiltered TBox .696 / .479) | .488 / .352 (unfiltered .461 / .332) | .076 / .095 | .847 / .702 |
| LUBM | enriched | .845 / .558 (unfiltered .779 / .531) | .758 / .547 (unfiltered .707 / .490) | .876 / .720 | .905 / .747 |
| NELL | plain | .730 / .497 | .516 / .437 | .527 / .494 | .968 / .837 |
| NELL | enriched | .924 / .595 | .743 / .574 | .952 / .813 | .997 / .848 |

GroundedEL with predicted facts added to the enriched LUBM input (QTO adjacency score ≥ t, at most 3 per (h, r);
`hybrid/predicted_facts.py`): verified TBox t 0.5 .752 / .568, t 0.3 .740 / .575; unfiltered TBox t 0.5 .721 / .533,
t 0.3 .723 / .547. Rule closure: decay 1, no score floor, nothing tuned; orix jobs 399149, 399150, 399310, 399372.

Other diagnostics (LUBM plain, unfiltered TBox): K 1 vs K 2 boxes D .43–.47 vs .46–.48; of the hard D answers of the
K 2 seed-0 model 48 % are in the top 3, 6 % are true in the model but outranked, 47 % are not true in the model
(`ensemble.py --diagnose`, job 398953).
