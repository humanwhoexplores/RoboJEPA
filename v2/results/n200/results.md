
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.446 | 0.638 | 0.989 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.447 | 0.655 | 1.000 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.443 | 0.594 | 0.960 |
| position_instr | 1232 | 0.88 | 0.101 | 0.567 | 0.620 | 0.977 |
| color_instr | 1232 | 0.88 | 0.094 | 0.325 | 0.655 | 1.000 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.660 | 0.775 | 0.988 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.350 | 0.542 | 0.982 |

Candidate-level AUC: direct 0.895, jepa 0.871, oracle 0.998
World-model feature MSE: 1push: pred 0.340 vs copy 1.320, 2push: pred 0.543 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.364 | 0.349 | 0.937 |
| seen_templates | 910 | 0.76 | 0.089 | 0.367 | 0.353 | 0.954 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.357 | 0.338 | 0.896 |
| position_instr | 637 | 0.76 | 0.080 | 0.429 | 0.353 | 0.922 |
| color_instr | 637 | 0.76 | 0.102 | 0.300 | 0.344 | 0.953 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.551 | 0.439 | 0.906 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.406 | 0.290 | 0.958 |

Candidate-level AUC: direct 0.897, jepa 0.706, oracle 0.995
World-model feature MSE: 1push: pred 0.571 vs copy 1.915, 2push: pred 0.830 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms