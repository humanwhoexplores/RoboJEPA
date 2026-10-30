
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.429 | 0.724 | 0.962 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.431 | 0.756 | 0.998 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.423 | 0.642 | 0.872 |
| position_instr | 1232 | 0.88 | 0.101 | 0.538 | 0.696 | 0.930 |
| color_instr | 1232 | 0.88 | 0.094 | 0.319 | 0.751 | 0.994 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.664 | 0.836 | 0.972 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.296 | 0.673 | 0.961 |

Candidate-level AUC: direct 0.894, jepa 0.937, oracle 0.999
World-model feature MSE: 1push: pred 0.318 vs copy 1.320, 2push: pred 0.497 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.341 | 0.375 | 0.922 |
| seen_templates | 910 | 0.76 | 0.089 | 0.345 | 0.390 | 0.956 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.330 | 0.338 | 0.838 |
| position_instr | 637 | 0.76 | 0.080 | 0.425 | 0.350 | 0.890 |
| color_instr | 637 | 0.76 | 0.102 | 0.256 | 0.400 | 0.954 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.482 | 0.483 | 0.900 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.388 | 0.392 | 0.935 |

Candidate-level AUC: direct 0.891, jepa 0.834, oracle 0.998
World-model feature MSE: 1push: pred 0.536 vs copy 1.915, 2push: pred 0.771 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms