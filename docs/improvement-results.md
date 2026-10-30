# Actual v2 results

| Method | Top5 seconds MAP@5 | Top5 watches MAP@5 | All watched MAP@5 | Fit s | Predict s |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.177472 | 0.310072 | 0.777052 | 1.245 | 5.883 |
| Content TF-IDF | 0.024564 | 0.027860 | 0.274944 | 1.410 | 8.046 |
| Original PPR (alpha 0.50) | 0.190441 | 0.323826 | 0.791118 | 1.970 | 9.338 |
| Selected PPR (alpha 0.25, sqrt seconds) | 0.304660 | 0.391758 | 0.774628 | 2.598 | 8.567 |

The protocol and limits are recorded in `config/improvement-v2.json`; full private evidence is in `experiments/improvement-v2-01/`. A genuine user-provided completed `submission_v2.csv` crop displays public 0.32589/private 0.28919, compared with first-crop 0.29103/0.28587. The displayed private gain is 0.00332 (1.16% relative), far smaller than the local primary gain. Context/filename match v2; exact submitted bytes, account, ID and description remain unverified. No new model or validation search was run after receiving the v2 score.
