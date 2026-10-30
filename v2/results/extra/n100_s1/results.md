
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.454 | 0.565 | 0.941 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.461 | 0.581 | 0.974 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.435 | 0.523 | 0.859 |
| position_instr | 1232 | 0.88 | 0.101 | 0.503 | 0.554 | 0.913 |
| color_instr | 1232 | 0.88 | 0.094 | 0.404 | 0.575 | 0.969 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.752 | 0.741 | 0.949 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.385 | 0.526 | 0.958 |

Candidate-level AUC: direct 0.896, jepa 0.907, oracle 0.997
World-model feature MSE: 1push: pred 0.372 vs copy 1.320, 2push: pred 0.604 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.361 | 0.396 | 0.905 |
| seen_templates | 910 | 0.76 | 0.089 | 0.364 | 0.409 | 0.938 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.354 | 0.365 | 0.821 |
| position_instr | 637 | 0.76 | 0.080 | 0.396 | 0.381 | 0.879 |
| color_instr | 637 | 0.76 | 0.102 | 0.327 | 0.411 | 0.931 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.555 | 0.539 | 0.910 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.353 | 0.439 | 0.920 |

Candidate-level AUC: direct 0.892, jepa 0.854, oracle 0.995
World-model feature MSE: 1push: pred 0.605 vs copy 1.915, 2push: pred 0.907 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms