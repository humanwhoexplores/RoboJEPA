
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.494 | 0.590 | 0.940 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.513 | 0.614 | 0.970 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.446 | 0.530 | 0.862 |
| position_instr | 1232 | 0.88 | 0.101 | 0.501 | 0.568 | 0.911 |
| color_instr | 1232 | 0.88 | 0.094 | 0.487 | 0.612 | 0.968 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.734 | 0.710 | 0.954 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.399 | 0.456 | 0.943 |

Candidate-level AUC: direct 0.895, jepa 0.845, oracle 0.995
World-model feature MSE: 1push: pred 0.358 vs copy 1.320, 2push: pred 0.597 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.349 | 0.334 | 0.881 |
| seen_templates | 910 | 0.76 | 0.089 | 0.363 | 0.347 | 0.913 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.316 | 0.299 | 0.802 |
| position_instr | 637 | 0.76 | 0.080 | 0.344 | 0.323 | 0.848 |
| color_instr | 637 | 0.76 | 0.102 | 0.355 | 0.344 | 0.915 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.567 | 0.499 | 0.850 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.359 | 0.317 | 0.930 |

Candidate-level AUC: direct 0.874, jepa 0.776, oracle 0.987
World-model feature MSE: 1push: pred 0.595 vs copy 1.915, 2push: pred 0.915 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.7 ms