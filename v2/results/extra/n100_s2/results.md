
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.380 | 0.562 | 0.940 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.404 | 0.585 | 0.970 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.321 | 0.506 | 0.862 |
| position_instr | 1232 | 0.88 | 0.101 | 0.468 | 0.558 | 0.911 |
| color_instr | 1232 | 0.88 | 0.094 | 0.292 | 0.567 | 0.968 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.600 | 0.682 | 0.954 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.272 | 0.512 | 0.943 |

Candidate-level AUC: direct 0.877, jepa 0.858, oracle 0.995
World-model feature MSE: 1push: pred 0.369 vs copy 1.320, 2push: pred 0.603 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.372 | 0.405 | 0.881 |
| seen_templates | 910 | 0.76 | 0.089 | 0.395 | 0.430 | 0.913 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.316 | 0.343 | 0.802 |
| position_instr | 637 | 0.76 | 0.080 | 0.419 | 0.396 | 0.848 |
| color_instr | 637 | 0.76 | 0.102 | 0.325 | 0.414 | 0.915 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.562 | 0.572 | 0.850 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.355 | 0.365 | 0.930 |

Candidate-level AUC: direct 0.878, jepa 0.808, oracle 0.987
World-model feature MSE: 1push: pred 0.607 vs copy 1.915, 2push: pred 0.908 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms