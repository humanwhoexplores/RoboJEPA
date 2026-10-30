
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.489 | 0.596 | 0.960 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.491 | 0.615 | 0.994 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.482 | 0.547 | 0.875 |
| position_instr | 1232 | 0.88 | 0.101 | 0.494 | 0.588 | 0.926 |
| color_instr | 1232 | 0.88 | 0.094 | 0.483 | 0.604 | 0.994 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.781 | 0.742 | 0.956 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.380 | 0.517 | 0.958 |

Candidate-level AUC: direct 0.911, jepa 0.894, oracle 0.998
World-model feature MSE: 1push: pred 0.374 vs copy 1.320, 2push: pred 0.600 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.450 | 0.423 | 0.892 |
| seen_templates | 910 | 0.76 | 0.089 | 0.454 | 0.444 | 0.923 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.440 | 0.371 | 0.813 |
| position_instr | 637 | 0.76 | 0.080 | 0.446 | 0.405 | 0.849 |
| color_instr | 637 | 0.76 | 0.102 | 0.454 | 0.441 | 0.934 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.620 | 0.579 | 0.878 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.402 | 0.323 | 0.929 |

Candidate-level AUC: direct 0.890, jepa 0.819, oracle 0.996
World-model feature MSE: 1push: pred 0.605 vs copy 1.915, 2push: pred 0.895 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms