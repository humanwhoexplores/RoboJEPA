
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.601 | 0.738 | 0.989 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.613 | 0.745 | 0.994 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.571 | 0.720 | 0.974 |
| position_instr | 1232 | 0.88 | 0.101 | 0.585 | 0.722 | 0.983 |
| color_instr | 1232 | 0.88 | 0.094 | 0.617 | 0.754 | 0.994 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.854 | 0.826 | 0.995 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.438 | 0.681 | 0.990 |

Candidate-level AUC: direct 0.941, jepa 0.944, oracle 0.999
World-model feature MSE: 1push: pred 0.318 vs copy 1.320, 2push: pred 0.493 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.461 | 0.413 | 0.943 |
| seen_templates | 910 | 0.76 | 0.089 | 0.470 | 0.427 | 0.947 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.437 | 0.376 | 0.931 |
| position_instr | 637 | 0.76 | 0.080 | 0.427 | 0.421 | 0.940 |
| color_instr | 637 | 0.76 | 0.102 | 0.495 | 0.405 | 0.945 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.697 | 0.442 | 0.938 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.444 | 0.392 | 0.981 |

Candidate-level AUC: direct 0.926, jepa 0.835, oracle 0.998
World-model feature MSE: 1push: pred 0.542 vs copy 1.915, 2push: pred 0.768 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms