# JEPA vs direct: action selection (3 seeds, mean ± std)

Encoder: dinov2-small, text: all-MiniLM-L6-v2

## val (48 decisions, 8 candidates each)

| method | selection success | AUROC | position | color |
|---|---|---|---|---|
| direct | 0.771 ± 0.045 | 0.921 | 0.806 | 0.736 |
| jepa | 0.868 ± 0.026 | 0.892 | 0.889 | 0.847 |
| jepa_oracle | 0.958 ± 0.000 | 1.000 | 0.958 | 0.958 |
| state_ref | 0.875 ± 0.000 | 0.932 | 0.875 | 0.875 |
| random | 0.203 ± 0.000 |  | 0.203 | 0.203 |
| best_possible | 0.958 ± 0.000 |  | 0.958 | 0.958 |

Predictor: MSE 0.1021 vs copy-initial 0.2783; cosine 0.9442 vs copy-initial 0.8569
Latency per decision (ms): {'direct': 6.84, 'jepa': 11.12}

## test (52 decisions, 8 candidates each)

| method | selection success | AUROC | position | color |
|---|---|---|---|---|
| direct | 0.417 ± 0.059 | 0.741 | 0.436 | 0.397 |
| jepa | 0.596 ± 0.000 | 0.859 | 0.513 | 0.679 |
| jepa_oracle | 0.679 ± 0.055 | 0.963 | 0.551 | 0.808 |
| state_ref | 0.731 ± 0.000 | 0.933 | 0.731 | 0.731 |
| random | 0.178 ± 0.000 |  | 0.178 | 0.178 |
| best_possible | 0.885 ± 0.000 |  | 0.885 | 0.885 |

Predictor: MSE 0.1177 vs copy-initial 0.2774; cosine 0.9344 vs copy-initial 0.857
Latency per decision (ms): {'direct': 7.12, 'jepa': 11.33}
