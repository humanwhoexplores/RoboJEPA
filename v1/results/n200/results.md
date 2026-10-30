
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.437 | 0.575 | 0.951 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.444 | 0.598 | 0.980 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.419 | 0.517 | 0.879 |
| position_instr | 1232 | 0.88 | 0.101 | 0.584 | 0.553 | 0.913 |
| color_instr | 1232 | 0.88 | 0.094 | 0.289 | 0.597 | 0.989 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.584 | 0.714 | 0.933 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.347 | 0.504 | 0.960 |

Candidate-level AUC: direct 0.886, jepa 0.878, oracle 0.996
World-model feature MSE: 1push: pred 0.366 vs copy 1.320, 2push: pred 0.572 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.270 | 0.322 | 0.908 |
| seen_templates | 910 | 0.76 | 0.089 | 0.277 | 0.330 | 0.935 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.253 | 0.302 | 0.841 |
| position_instr | 637 | 0.76 | 0.080 | 0.396 | 0.283 | 0.873 |
| color_instr | 637 | 0.76 | 0.102 | 0.144 | 0.361 | 0.943 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.450 | 0.469 | 0.921 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.270 | 0.321 | 0.938 |

Candidate-level AUC: direct 0.894, jepa 0.803, oracle 0.995
World-model feature MSE: 1push: pred 0.589 vs copy 1.915, 2push: pred 0.845 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.7 ms