
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.401 | 0.523 | 0.917 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.415 | 0.534 | 0.945 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.368 | 0.494 | 0.847 |
| position_instr | 1232 | 0.88 | 0.101 | 0.451 | 0.512 | 0.885 |
| color_instr | 1232 | 0.88 | 0.094 | 0.351 | 0.533 | 0.950 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.570 | 0.643 | 0.925 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.295 | 0.432 | 0.902 |

Candidate-level AUC: direct 0.834, jepa 0.796, oracle 0.980
World-model feature MSE: 1push: pred 0.393 vs copy 1.320, 2push: pred 0.683 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.297 | 0.275 | 0.857 |
| seen_templates | 910 | 0.76 | 0.089 | 0.315 | 0.282 | 0.878 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.253 | 0.255 | 0.805 |
| position_instr | 637 | 0.76 | 0.080 | 0.342 | 0.251 | 0.801 |
| color_instr | 637 | 0.76 | 0.102 | 0.253 | 0.298 | 0.914 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.377 | 0.343 | 0.813 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.338 | 0.271 | 0.885 |

Candidate-level AUC: direct 0.793, jepa 0.740, oracle 0.978
World-model feature MSE: 1push: pred 0.659 vs copy 1.915, 2push: pred 1.025 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms