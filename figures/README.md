# Figures

Figures of the paper, each as a PNG file at 400 dpi and as a PDF file. Figures 1 and 4 of the paper are drawn in LaTeX (TikZ and pgfplots), so they have no file here.

| Files | Paper | Content |
| --- | --- | --- |
| `fig2_aclo_vs_nc_by_function` | Figure 2 | Median log-ratio R of ACLO against ACLO-NC for each of the 24 BBOB functions at 100, 300 and 1000 x D (90 problem instances per bar, symmetric logarithmic axis). Positive values mean a smaller final gap for ACLO. An asterisk marks a Holm-significant loss in the one-sided sign test; an asterisk without a visible bar marks a significant loss whose median log-ratio is zero or too small to show. |
| `fig3_ecdf` | Figure 3 | ECDF of runtimes over 51 targets and six dimensions for all seven methods: all 24 functions in the top row and the ill-conditioned functions f10-f14 in the bottom row, at 100, 300 and 1000 x D from left to right. |
| `fig5_pinene_fit` | Figure 5 | Measured compositions of the five species in the alpha-pinene isomerisation and the curves simulated with the parameters of the median ACLO run at 100 x D. |
| `fig6_convergence` | Figure 6 | Best objective value above the best known value against evaluations per parameter for the three application problems: median (line) and interquartile range (band) of 25 runs with a budget of 1000 x D. Dotted lines mark 100 x D and 300 x D. |
| `fig7_hydrograph_gr4j`, `fig7_hydrograph_hbv` | Figure 7 | Observed and simulated daily streamflow of CAMELS basin 01031500 in the validation water year 2000 (1 October 1999 to 30 September 2000), for GR4J with snow and for HBV, with the parameter sets of the median ACLO runs at 300 x D. |

The labels follow the paper: ACLO is the full method, ACLO-NC is the base search without curvature learning and ACLO-NC63 is the base search with the same 63% line-probe share as ACLO.
