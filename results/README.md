# Result tables

Tables of the two experiments of the paper. The names follow the paper: ACLO, ACLO-NC (the base search without curvature learning) and ACLO-NC63 (the base search with the 63% line-probe share).

| File | Content |
| --- | --- |
| `bbob_preregistered_criteria.json` | Thresholds and observed values of the five pre-registered criteria (ACLO against ACLO-NC) |
| `bbob_pairwise_vs_aclo.csv` | Median log-ratio, bootstrap interval, wins, losses and tests of ACLO against every other method, by budget and function group |
| `bbob_mean_ranks.csv` | Mean ranks by budget, function group and dimension |
| `bbob_ecdf_final.csv` | Share of (problem, target) pairs reached, by budget, method, group and dimension |
| `bbob_aRT.csv` | Average running time to the targets |
| `bbob_timing.csv` | CPU and wall-clock time |
| `bbob_gate_by_group.csv` | Skill-gate opening rate and first opening of ACLO |
| `bbob_friedman_tests.json`, `bbob_completeness.json` | Friedman tests; run counts |
| `app_statistics.csv` | Median, mean, spread and success rate per problem, budget and method |
| `app_ranksum_tests.csv` | The 30 primary and 15 secondary Mann-Whitney tests with Holm correction and A12 |
| `app_ranks.csv`, `app_summary.json` | Ranks and Friedman test |
| `app_reports_median.csv`, `app_reports_per_run.csv` | Calibration and validation KGE and NSE; residuals for alpha-pinene |
| `app_best_parameters.csv` | Best parameter sets per problem, budget and method |
| `app_aclo_gate.csv` | Skill-gate opening rate in the applications |
