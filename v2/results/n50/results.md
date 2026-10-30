
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.356 | 0.500 | 0.917 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.367 | 0.515 | 0.945 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.328 | 0.464 | 0.847 |
| position_instr | 1232 | 0.88 | 0.101 | 0.453 | 0.493 | 0.885 |
| color_instr | 1232 | 0.88 | 0.094 | 0.259 | 0.508 | 0.950 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.548 | 0.580 | 0.925 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.277 | 0.430 | 0.902 |

Candidate-level AUC: direct 0.834, jepa 0.782, oracle 0.980
World-model feature MSE: 1push: pred 0.401 vs copy 1.320, 2push: pred 0.687 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.286 | 0.272 | 0.857 |
| seen_templates | 910 | 0.76 | 0.089 | 0.280 | 0.279 | 0.878 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.299 | 0.253 | 0.805 |
| position_instr | 637 | 0.76 | 0.080 | 0.353 | 0.262 | 0.801 |
| color_instr | 637 | 0.76 | 0.102 | 0.218 | 0.281 | 0.914 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.415 | 0.409 | 0.813 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.385 | 0.267 | 0.885 |

Candidate-level AUC: direct 0.821, jepa 0.738, oracle 0.978
World-model feature MSE: 1push: pred 0.660 vs copy 1.915, 2push: pred 1.019 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms