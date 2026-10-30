
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.434 | 0.648 | 0.989 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.461 | 0.662 | 1.000 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.366 | 0.612 | 0.960 |
| position_instr | 1232 | 0.88 | 0.101 | 0.442 | 0.644 | 0.977 |
| color_instr | 1232 | 0.88 | 0.094 | 0.426 | 0.651 | 1.000 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.738 | 0.810 | 0.988 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.394 | 0.553 | 0.982 |

Candidate-level AUC: direct 0.917, jepa 0.892, oracle 0.998
World-model feature MSE: 1push: pred 0.322 vs copy 1.320, 2push: pred 0.522 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.349 | 0.309 | 0.937 |
| seen_templates | 910 | 0.76 | 0.089 | 0.368 | 0.313 | 0.954 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.299 | 0.299 | 0.896 |
| position_instr | 637 | 0.76 | 0.080 | 0.363 | 0.308 | 0.922 |
| color_instr | 637 | 0.76 | 0.102 | 0.334 | 0.311 | 0.953 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.546 | 0.399 | 0.906 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.353 | 0.292 | 0.958 |

Candidate-level AUC: direct 0.915, jepa 0.709, oracle 0.995
World-model feature MSE: 1push: pred 0.589 vs copy 1.915, 2push: pred 0.841 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms