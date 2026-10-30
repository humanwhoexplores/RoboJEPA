
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.538 | 0.623 | 0.960 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.539 | 0.645 | 0.994 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.536 | 0.567 | 0.875 |
| position_instr | 1232 | 0.88 | 0.101 | 0.525 | 0.599 | 0.926 |
| color_instr | 1232 | 0.88 | 0.094 | 0.551 | 0.647 | 0.994 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.797 | 0.779 | 0.956 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.462 | 0.595 | 0.958 |

Candidate-level AUC: direct 0.911, jepa 0.916, oracle 0.998
World-model feature MSE: 1push: pred 0.352 vs copy 1.320, 2push: pred 0.599 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.495 | 0.347 | 0.892 |
| seen_templates | 910 | 0.76 | 0.089 | 0.497 | 0.357 | 0.923 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.489 | 0.321 | 0.813 |
| position_instr | 637 | 0.76 | 0.080 | 0.542 | 0.327 | 0.849 |
| color_instr | 637 | 0.76 | 0.102 | 0.447 | 0.367 | 0.934 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.674 | 0.452 | 0.878 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.423 | 0.363 | 0.929 |

Candidate-level AUC: direct 0.902, jepa 0.825, oracle 0.996
World-model feature MSE: 1push: pred 0.617 vs copy 1.915, 2push: pred 0.929 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms