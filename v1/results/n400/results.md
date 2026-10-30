
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.372 | 0.656 | 0.957 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.368 | 0.695 | 0.994 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.382 | 0.558 | 0.866 |
| position_instr | 1232 | 0.88 | 0.101 | 0.487 | 0.670 | 0.929 |
| color_instr | 1232 | 0.88 | 0.094 | 0.256 | 0.641 | 0.986 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.564 | 0.761 | 0.958 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.305 | 0.624 | 0.964 |

Candidate-level AUC: direct 0.894, jepa 0.912, oracle 0.997
World-model feature MSE: 1push: pred 0.317 vs copy 1.320, 2push: pred 0.496 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.291 | 0.341 | 0.906 |
| seen_templates | 910 | 0.76 | 0.089 | 0.296 | 0.367 | 0.936 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.280 | 0.277 | 0.830 |
| position_instr | 637 | 0.76 | 0.080 | 0.386 | 0.356 | 0.878 |
| color_instr | 637 | 0.76 | 0.102 | 0.196 | 0.327 | 0.934 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.419 | 0.421 | 0.912 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.305 | 0.353 | 0.929 |

Candidate-level AUC: direct 0.879, jepa 0.806, oracle 0.996
World-model feature MSE: 1push: pred 0.537 vs copy 1.915, 2push: pred 0.772 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms