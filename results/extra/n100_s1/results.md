
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.429 | 0.585 | 0.941 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.448 | 0.605 | 0.974 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.382 | 0.536 | 0.859 |
| position_instr | 1232 | 0.88 | 0.101 | 0.426 | 0.579 | 0.913 |
| color_instr | 1232 | 0.88 | 0.094 | 0.433 | 0.591 | 0.969 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.732 | 0.739 | 0.949 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.353 | 0.518 | 0.958 |

Candidate-level AUC: direct 0.904, jepa 0.907, oracle 0.997
World-model feature MSE: 1push: pred 0.359 vs copy 1.320, 2push: pred 0.598 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.348 | 0.472 | 0.905 |
| seen_templates | 910 | 0.76 | 0.089 | 0.362 | 0.486 | 0.938 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.313 | 0.437 | 0.821 |
| position_instr | 637 | 0.76 | 0.080 | 0.334 | 0.429 | 0.879 |
| color_instr | 637 | 0.76 | 0.102 | 0.361 | 0.515 | 0.931 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.458 | 0.617 | 0.910 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.336 | 0.455 | 0.920 |

Candidate-level AUC: direct 0.899, jepa 0.864, oracle 0.995
World-model feature MSE: 1push: pred 0.601 vs copy 1.915, 2push: pred 0.913 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.7 ms