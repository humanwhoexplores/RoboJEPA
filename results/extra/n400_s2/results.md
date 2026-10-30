
## test_id

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 2464 | 0.88 | 0.098 | 0.494 | 0.741 | 0.962 |
| seen_templates | 1760 | 0.88 | 0.096 | 0.516 | 0.773 | 0.998 |
| heldout_paraphrases | 704 | 0.88 | 0.102 | 0.439 | 0.659 | 0.872 |
| position_instr | 1232 | 0.88 | 0.101 | 0.506 | 0.718 | 0.930 |
| color_instr | 1232 | 0.88 | 0.094 | 0.482 | 0.764 | 0.994 |
| 1push_candidates | 1932 | 0.69 | 0.097 | 0.856 | 0.834 | 0.972 |
| 2push_candidates | 2226 | 0.80 | 0.128 | 0.383 | 0.693 | 0.961 |

Candidate-level AUC: direct 0.937, jepa 0.942, oracle 0.999
World-model feature MSE: 1push: pred 0.298 vs copy 1.320, 2push: pred 0.471 vs copy 1.760
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms

## test_ood

Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):

| group | n | solvable | random | direct | jepa | oracle |
|---|---|---|---|---|---|---|
| all | 1274 | 0.76 | 0.091 | 0.469 | 0.346 | 0.922 |
| seen_templates | 910 | 0.76 | 0.089 | 0.489 | 0.356 | 0.956 |
| heldout_paraphrases | 364 | 0.76 | 0.096 | 0.420 | 0.321 | 0.838 |
| position_instr | 637 | 0.76 | 0.080 | 0.444 | 0.333 | 0.890 |
| color_instr | 637 | 0.76 | 0.102 | 0.495 | 0.359 | 0.954 |
| 1push_candidates | 938 | 0.56 | 0.091 | 0.683 | 0.443 | 0.900 |
| 2push_candidates | 1078 | 0.64 | 0.118 | 0.441 | 0.376 | 0.935 |

Candidate-level AUC: direct 0.933, jepa 0.821, oracle 0.998
World-model feature MSE: 1push: pred 0.528 vs copy 1.915, 2push: pred 0.766 vs copy 2.187
Decision latency (48 candidates): direct 6.3 ms, jepa 14.6 ms