
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.616 | 0.761 | 0.989 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.633 | 0.770 | 0.994 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.572 | 0.739 | 0.974 |
| position_instr | 1232 | 0.88 | 0.101 | 0.607 | 0.749 | 0.983 |
| color_instr | 1232 | 0.88 | 0.094 | 0.624 | 0.773 | 0.994 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.846 | 0.880 | 0.995 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.398 | 0.668 | 0.990 |

Candidate-level AUC: direct 0.935, jepa 0.950, oracle 0.999
World-model feature MSE: 1push: pred 0.302 vs copy 1.320, 2push: pred 0.477 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.405 | 0.368 | 0.943 |
| seen_templates | 910 | 0.76 | 0.089 | 0.414 | 0.377 | 0.947 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.382 | 0.346 | 0.931 |
| position_instr | 637 | 0.76 | 0.080 | 0.407 | 0.366 | 0.940 |
| color_instr | 637 | 0.76 | 0.102 | 0.403 | 0.370 | 0.945 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.690 | 0.430 | 0.938 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.363 | 0.370 | 0.981 |

Candidate-level AUC: direct 0.920, jepa 0.838, oracle 0.998
World-model feature MSE: 1push: pred 0.539 vs copy 1.915, 2push: pred 0.772 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms