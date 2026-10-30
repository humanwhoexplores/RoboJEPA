
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.414 | 0.737 | 0.962 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.428 | 0.760 | 0.994 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.381 | 0.680 | 0.881 |
| position_instr | 1232 | 0.88 | 0.101 | 0.524 | 0.710 | 0.929 |
| color_instr | 1232 | 0.88 | 0.094 | 0.305 | 0.764 | 0.994 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.637 | 0.867 | 0.961 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.361 | 0.670 | 0.950 |

Candidate-level AUC: direct 0.902, jepa 0.955, oracle 0.998
World-model feature MSE: 1push: pred 0.316 vs copy 1.320, 2push: pred 0.495 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.373 | 0.381 | 0.929 |
| seen_templates | 910 | 0.76 | 0.089 | 0.389 | 0.395 | 0.962 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.332 | 0.349 | 0.846 |
| position_instr | 637 | 0.76 | 0.080 | 0.430 | 0.370 | 0.901 |
| color_instr | 637 | 0.76 | 0.102 | 0.316 | 0.392 | 0.956 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.543 | 0.461 | 0.887 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.378 | 0.379 | 0.954 |

Candidate-level AUC: direct 0.894, jepa 0.860, oracle 0.997
World-model feature MSE: 1push: pred 0.545 vs copy 1.915, 2push: pred 0.772 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms