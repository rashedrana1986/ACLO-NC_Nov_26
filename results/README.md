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

## Columns used in several tables

| Column | Meaning |
| --- | --- |
| `budget_xD`, `budget_mult`, `run_budget_xD` | Budget in evaluations per dimension (per parameter for the applications): 100, 300 or 1000 |
| `label` | Campaign of a run. ACLO, ACLO-NC, ACLO-NC63 and L-SHADE were run once per budget (for example `ACLO_100xD`); BIPOP-CMA-ES, BOBYQA and SPSO-2011 were run once at the largest budget and read at each budget from their anytime records (`*_anytime`) |
| `group` | BBOB function group: separable (f1-f5), moderate (f6-f9), ill_conditioned (f10-f14), multimodal_adequate (f15-f19), multimodal_weak (f20-f24), or all |
| `median_R`, `lower95`, `upper95` | Median of the log-ratio R = log10((gap of B + 1e-9) / (gap of A + 1e-9)) over paired runs, with a 95% stratified bootstrap interval; positive values favour A (ACLO) |
| `wins`, `losses`, `ties` | Paired runs in which A had the smaller, larger or equal final gap |
| `hits_a`, `hits_b` | Runs that reached the final COCO target (precision 1e-8) |
| `median_cpu_ratio` | Median ratio of CPU time, A over B |
| `wilcoxon_p`, `wilcoxon_holm_p` | Wilcoxon signed-rank test on the non-zero values of R, before and after Holm correction |
| `A12_aclo_better` | Vargha-Delaney effect size: probability that a random ACLO run is better than a random run of the competitor |
| `p_ranksum`, `p_holm`, `family` | Mann-Whitney test, Holm-adjusted within its family (the 30 primary tests or the 15 secondary tests at 1000 x D) |
| `kge_cal`, `nse_cal`, `kge_val`, `nse_val` | Kling-Gupta and Nash-Sutcliffe efficiency in the calibration and validation periods (hydrological problems) |
| `sse`, `rmse_y1` ... `rmse_y5` | Sum of squared errors and the error of each of the five species (alpha-pinene) |
| `k1_x1e5` ... `k5_x1e5` | Rate constants of the alpha-pinene model, multiplied by 1e5 |
