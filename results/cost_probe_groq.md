# Cost probe

Provider groq, model openai/gpt-oss-120b, D1 dev, run 1.

**Incomplete: ['B1', 'B2', 'B3', 'B4', 'OURS'] did not finish all 20 problems.**

| system | problems | api_errors | mean_calls | mean_input_tokens | mean_output_tokens | mean_seconds | parse_failures | truncation_rate | rate_limit_retries |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 6 | 4 | 1.00 | 710 | 7030 | 15.3 | 0/6 | 0.000 (0/6) | 2.0 |
| B2 | 0 | 3 |  |  |  |  |  |  |  |
| B3 | 0 | 3 |  |  |  |  |  |  |  |
| B4 | 0 | 3 |  |  |  |  |  |  |  |
| OURS | 0 | 3 |  |  |  |  |  |  |  |

## Projection for the full run

Blank: it needs all 5 systems on all 20 problems, and this probe did not finish. Projecting from what did finish would not be a measurement.
