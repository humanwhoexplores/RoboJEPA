
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.420 | 0.322 | 0.549 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.431 | 0.322 | 0.548 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.392 | 0.322 | 0.551 |
| position_instr | 1232 | 0.88 | 0.101 | 0.543 | 0.321 | 0.550 |
| color_instr | 1232 | 0.88 | 0.094 | 0.296 | 0.323 | 0.548 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.568 | 0.428 | 0.601 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.325 | 0.295 | 0.615 |

Candidate-level AUC: direct 0.873, jepa 0.776, oracle 0.951
World-model feature MSE: 1push: pred 0.431 vs copy 1.320, 2push: pred 0.671 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.242 | 0.253 | 0.635 |
| seen_templates | 910 | 0.76 | 0.089 | 0.253 | 0.252 | 0.635 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.214 | 0.255 | 0.635 |
| position_instr | 637 | 0.76 | 0.080 | 0.300 | 0.253 | 0.639 |
| color_instr | 637 | 0.76 | 0.102 | 0.184 | 0.253 | 0.631 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.385 | 0.340 | 0.639 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.280 | 0.250 | 0.701 |

Candidate-level AUC: direct 0.867, jepa 0.766, oracle 0.961
World-model feature MSE: 1push: pred 0.668 vs copy 1.915, 2push: pred 0.948 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms