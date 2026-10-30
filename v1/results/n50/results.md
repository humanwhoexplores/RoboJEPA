
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.246 | 0.247 | 0.546 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.244 | 0.246 | 0.547 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.251 | 0.250 | 0.545 |
| position_instr | 1232 | 0.88 | 0.101 | 0.244 | 0.248 | 0.547 |
| color_instr | 1232 | 0.88 | 0.094 | 0.249 | 0.247 | 0.545 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.308 | 0.389 | 0.593 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.211 | 0.276 | 0.610 |

Candidate-level AUC: direct 0.754, jepa 0.743, oracle 0.940
World-model feature MSE: 1push: pred 0.535 vs copy 1.320, 2push: pred 0.791 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.308 | 0.275 | 0.606 |
| seen_templates | 910 | 0.76 | 0.089 | 0.311 | 0.275 | 0.605 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.299 | 0.275 | 0.607 |
| position_instr | 637 | 0.76 | 0.080 | 0.314 | 0.275 | 0.606 |
| color_instr | 637 | 0.76 | 0.102 | 0.301 | 0.275 | 0.606 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.333 | 0.322 | 0.582 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.310 | 0.300 | 0.691 |

Candidate-level AUC: direct 0.722, jepa 0.802, oracle 0.952
World-model feature MSE: 1push: pred 0.814 vs copy 1.915, 2push: pred 1.079 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.7 ms