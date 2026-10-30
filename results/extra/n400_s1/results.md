
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.556 | 0.717 | 0.962 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.570 | 0.741 | 0.994 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.521 | 0.655 | 0.881 |
| position_instr | 1232 | 0.88 | 0.101 | 0.558 | 0.692 | 0.929 |
| color_instr | 1232 | 0.88 | 0.094 | 0.555 | 0.741 | 0.994 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.876 | 0.875 | 0.961 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.433 | 0.649 | 0.950 |

Candidate-level AUC: direct 0.940, jepa 0.954, oracle 0.998
World-model feature MSE: 1push: pred 0.301 vs copy 1.320, 2push: pred 0.472 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.481 | 0.423 | 0.929 |
| seen_templates | 910 | 0.76 | 0.089 | 0.495 | 0.442 | 0.962 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.448 | 0.376 | 0.846 |
| position_instr | 637 | 0.76 | 0.080 | 0.469 | 0.413 | 0.901 |
| color_instr | 637 | 0.76 | 0.102 | 0.493 | 0.433 | 0.956 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.679 | 0.489 | 0.887 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.433 | 0.402 | 0.954 |

Candidate-level AUC: direct 0.933, jepa 0.870, oracle 0.997
World-model feature MSE: 1push: pred 0.543 vs copy 1.915, 2push: pred 0.771 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms