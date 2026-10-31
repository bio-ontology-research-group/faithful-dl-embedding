# ABox benchmark results (B1 `yeast-function`, B3 `helis-membership`)

Generated from `results/raw/{yeast-function,helis}__*.json` by a script (no hand-copied numbers). d = 50, 3 seeds each, mean ± sd over seeds. 2000 epochs for all EL models (best checkpoint by validation MRR, checked every 50 epochs); `canonical-mixture` trains with its own mini-batch loop for 200 epochs (time budget; its initialisation already reproduces the training ABox).

Columns. MRR / H@1 / H@10: filtered ranking of held-out class assertions among candidates (KB-entailed candidates, the individual's other held-out positives and, for HeLiS, the OWL2Vec* skip list are masked; ties pessimistic). MRR(all): positives without a candidate class count as rank n+1 (B1: 7 % of held-out positives). Fmax: protein-centric CAFA Fmax over propagated held-out annotations, KB-entailed cells excluded (B1 only). HierViol: fraction of sampled (a, G ⊑ G') with a scored more plausibly in G than in G' (a model of the TBox has 0). ABoxMod: training assertions satisfied strictly. AUC(+/?)kb: KB-entailed vs unknown candidates. AUC(+/−)gold, FVR-gold: gold-conditional labels (HeLiS only; B1 has no contradictions by construction). point-Mod: canonical-element positives inside their regions (point objectives only).

## yeast-function

| method | n | MRR | H@1 | H@10 | MRR(all) | Fmax | HierViol ↓ | ABoxMod | AUC(+/?)kb | point-Mod |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| elem | 3 | 0.035 ± 0.004 | 0.013 ± 0.003 | 0.074 ± 0.006 | 0.033 ± 0.004 | 0.036 ± 0.003 | 0.421 ± 0.007 | 0.052 ± 0.000 | 0.772 ± 0.001 | — |
| elbe | 3 | 0.022 ± 0.003 | 0.005 ± 0.003 | 0.045 ± 0.005 | 0.020 ± 0.003 | 0.026 ± 0.003 | 0.422 ± 0.004 | 0.051 ± 0.040 | 0.752 ± 0.012 | — |
| elbe-init | 3 | 0.015 ± 0.000 | 0.005 ± 0.001 | 0.031 ± 0.002 | 0.014 ± 0.000 | 0.023 ± 0.002 | 0.035 ± 0.000 | 0.083 ± 0.000 | 0.898 ± 0.001 | — |
| box2el | 3 | 0.123 ± 0.007 | 0.043 ± 0.005 | 0.298 ± 0.011 | 0.114 ± 0.007 | 0.128 ± 0.013 | 0.389 ± 0.026 | 0.004 ± 0.000 | 0.822 ± 0.017 | — |
| box2el-init | 3 | 0.093 ± 0.002 | 0.019 ± 0.002 | 0.272 ± 0.003 | 0.087 ± 0.002 | 0.128 ± 0.004 | 0.342 ± 0.025 | 0.012 ± 0.000 | 0.845 ± 0.012 | — |
| prism | 3 | 0.030 ± 0.002 | 0.008 ± 0.003 | 0.057 ± 0.005 | 0.028 ± 0.002 | 0.038 ± 0.003 | 0.442 ± 0.009 | 0.063 ± 0.001 | 0.753 ± 0.003 | — |
| prism-init | 3 | 0.012 ± 0.000 | 0.003 ± 0.000 | 0.024 ± 0.002 | 0.011 ± 0.000 | 0.024 ± 0.001 | 0.032 ± 0.002 | 0.086 ± 0.000 | 0.915 ± 0.001 | — |
| multibox-k2-axiom | 3 | 0.116 ± 0.003 | 0.035 ± 0.002 | 0.291 ± 0.008 | 0.108 ± 0.003 | 0.133 ± 0.009 | 0.420 ± 0.006 | 0.005 ± 0.000 | 0.812 ± 0.008 | — |
| multielbe-k2-axiom | 3 | 0.016 ± 0.001 | 0.003 ± 0.002 | 0.028 ± 0.003 | 0.015 ± 0.001 | 0.021 ± 0.002 | 0.459 ± 0.020 | 0.051 ± 0.034 | 0.744 ± 0.009 | — |
| multibox-k2-point | 3 | 0.060 ± 0.009 | 0.024 ± 0.005 | 0.125 ± 0.019 | 0.056 ± 0.008 | 0.038 ± 0.001 | 0.102 ± 0.003 | 0.000 ± 0.000 | 0.940 ± 0.001 | 0.616 ± 0.053 |
| multibox-k2-mixed | 3 | 0.061 ± 0.009 | 0.025 ± 0.005 | 0.126 ± 0.019 | 0.057 ± 0.008 | 0.037 ± 0.001 | 0.101 ± 0.003 | 0.000 ± 0.000 | 0.940 ± 0.001 | 0.511 ± 0.061 |
| multibox-k2-hybrid | 3 | 0.127 ± 0.002 | 0.062 ± 0.005 | 0.246 ± 0.001 | 0.118 ± 0.002 | 0.051 ± 0.001 | 0.085 ± 0.006 | 0.000 ± 0.000 | 0.936 ± 0.006 | 0.828 ± 0.003 |
| multielbe-k2-point | 3 | 0.038 ± 0.008 | 0.014 ± 0.003 | 0.072 ± 0.015 | 0.035 ± 0.007 | 0.032 ± 0.001 | 0.108 ± 0.003 | 0.000 ± 0.000 | 0.939 ± 0.001 | 0.004 ± 0.001 |
| multielbe-k2-mixed | 3 | 0.038 ± 0.008 | 0.014 ± 0.003 | 0.072 ± 0.015 | 0.035 ± 0.007 | 0.032 ± 0.001 | 0.108 ± 0.003 | 0.000 ± 0.000 | 0.939 ± 0.001 | 0.003 ± 0.000 |
| multielbe-k2-hybrid | 3 | 0.044 ± 0.003 | 0.016 ± 0.001 | 0.091 ± 0.011 | 0.041 ± 0.003 | 0.029 ± 0.000 | 0.110 ± 0.001 | 0.000 ± 0.000 | 0.931 ± 0.001 | 0.005 ± 0.000 |
| canonical-mixture | 3 | 0.040 ± 0.000 | 0.014 ± 0.000 | 0.102 ± 0.000 | 0.037 ± 0.000 | 0.176 ± 0.002 | 0.000 ± 0.000 | 0.616 ± 0.000 | 1.000 ± 0.000 | — |

## helis

| method | n | MRR | H@1 | H@10 | HierViol ↓ | ABoxMod | AUC(+/?)kb | AUC(+/−)gold | FVR-gold ↓ | point-Mod |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| elem | 3 | 0.101 ± 0.007 | 0.058 ± 0.006 | 0.161 ± 0.013 | 0.193 ± 0.003 | 0.059 ± 0.003 | 0.918 ± 0.003 | 0.614 ± 0.005 | 0.810 ± 0.003 | — |
| elbe | 3 | 0.132 ± 0.001 | 0.079 ± 0.002 | 0.191 ± 0.000 | 0.119 ± 0.001 | 0.864 ± 0.003 | 0.986 ± 0.000 | 0.923 ± 0.002 | 0.683 ± 0.005 | — |
| elbe-init | 3 | 0.041 ± 0.002 | 0.006 ± 0.001 | 0.063 ± 0.007 | 0.116 ± 0.001 | 0.958 ± 0.008 | 0.956 ± 0.001 | 0.970 ± 0.001 | 0.611 ± 0.003 | — |
| box2el | 3 | 0.120 ± 0.002 | 0.075 ± 0.005 | 0.170 ± 0.005 | 0.169 ± 0.014 | 0.008 ± 0.002 | 0.991 ± 0.000 | 0.774 ± 0.010 | 0.770 ± 0.010 | — |
| box2el-init | 3 | 0.097 ± 0.029 | 0.032 ± 0.039 | 0.210 ± 0.004 | 0.243 ± 0.001 | 0.106 ± 0.008 | 0.993 ± 0.001 | 0.977 ± 0.002 | 0.588 ± 0.015 | — |
| prism | 3 | 0.133 ± 0.001 | 0.096 ± 0.002 | 0.143 ± 0.001 | 0.005 ± 0.002 | 0.242 ± 0.001 | 0.945 ± 0.002 | 0.832 ± 0.018 | 0.740 ± 0.014 | — |
| prism-init | 3 | 0.091 ± 0.011 | 0.046 ± 0.014 | 0.162 ± 0.001 | 0.026 ± 0.005 | 0.258 ± 0.001 | 0.794 ± 0.008 | 0.915 ± 0.003 | 0.835 ± 0.018 | — |
| multibox-k2-axiom | 3 | 0.123 ± 0.003 | 0.075 ± 0.003 | 0.186 ± 0.007 | 0.132 ± 0.021 | 0.010 ± 0.003 | 0.985 ± 0.005 | 0.804 ± 0.107 | 0.739 ± 0.158 | — |
| multielbe-k2-axiom | 3 | 0.136 ± 0.006 | 0.090 ± 0.008 | 0.190 ± 0.014 | 0.115 ± 0.009 | 0.882 ± 0.004 | 0.987 ± 0.002 | 0.915 ± 0.003 | 0.682 ± 0.012 | — |
| multibox-k2-point | 3 | 0.171 ± 0.002 | 0.130 ± 0.002 | 0.224 ± 0.002 | 0.065 ± 0.004 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.988 ± 0.000 | 0.439 ± 0.003 | 0.960 ± 0.001 |
| multibox-k2-mixed | 3 | 0.171 ± 0.001 | 0.130 ± 0.001 | 0.226 ± 0.003 | 0.125 ± 0.009 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.987 ± 0.000 | 0.450 ± 0.008 | 0.957 ± 0.000 |
| multibox-k2-hybrid | 3 | 0.162 ± 0.002 | 0.126 ± 0.002 | 0.186 ± 0.002 | 0.066 ± 0.003 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.968 ± 0.001 | 0.444 ± 0.013 | 0.877 ± 0.065 |
| multielbe-k2-point | 3 | 0.160 ± 0.002 | 0.124 ± 0.000 | 0.193 ± 0.008 | 0.028 ± 0.004 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.975 ± 0.001 | 0.579 ± 0.002 | 0.991 ± 0.000 |
| multielbe-k2-mixed | 3 | 0.161 ± 0.004 | 0.125 ± 0.003 | 0.191 ± 0.005 | 0.062 ± 0.002 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.982 ± 0.004 | 0.547 ± 0.029 | 0.972 ± 0.002 |
| multielbe-k2-hybrid | 3 | 0.160 ± 0.001 | 0.124 ± 0.001 | 0.193 ± 0.004 | 0.085 ± 0.009 | 0.616 ± 0.003 | 1.000 ± 0.000 | 0.946 ± 0.001 | 0.482 ± 0.000 | 0.967 ± 0.000 |
| canonical-mixture | 3 | 0.167 ± 0.000 | 0.121 ± 0.000 | 0.244 ± 0.001 | 0.000 ± 0.000 | 0.269 ± 0.000 | 1.000 ± 0.000 | 0.972 ± 0.001 | 0.758 ± 0.001 | — |

### HeLiS reference rows (OWL2Vec* paper, Chen et al., Machine Learning 2021, Table 4; not re-run)

MRR / H@1 / H@10, same candidate and skip protocol, but a Random-Forest classifier on concatenated embeddings and the authors' own training negatives: OWL2Vec* 0.953 / 0.932 / 0.987 (uses labels/text); pre-trained Word2Vec 0.899 / 0.877 / 0.933 (text); RDF2Vec 0.345 / 0.219 / 0.655; OWL2Vec (no text) 0.335 / 0.215 / 0.601; TransR 0.298 / 0.184 / 0.559; DistMult 0.253 / 0.166 / 0.437; OPA2Vec 0.237 / 0.146 / 0.408; Onto2Vec 0.211 / 0.108 / 0.397; TransE 0.181 / 0.090 / 0.355; Quantum embedding 0.159 / 0.132 / 0.190.
