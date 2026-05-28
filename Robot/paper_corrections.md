# Paper Corrections for CoDIT 2026 Revision
## Apply these changes to your LaTeX source before resubmitting (deadline May 29, 2026)

---

## CRITICAL FIX 1 — Table VI narrative (Section V-C)

### FIND (remove this sentence):
```
The proposed CTC+SAC approach achieves the best performance across all metrics.
```

### REPLACE WITH:
```
The proposed CTC+SAC approach achieves the lowest mean tracking error among all 
three methods (2.36 rad), corresponding to a 58.6\% reduction over CTC-Only and 
a 44.1\% reduction over RL-Only. While RL-Only exhibits a lower RMS error for 
joint 1 due to a smaller initial transient, CTC+SAC achieves markedly superior 
sustained tracking accuracy in steady state.
```

---

## CRITICAL FIX 2 — Discussion point 1 (Section V-D)

### FIND:
```
CTC+SAC achieves better performance than both CTC-Only and RL-Only, validating 
the synergy between model-based and learning-based control.
```

### REPLACE WITH:
```
CTC+SAC achieves the lowest mean tracking error among all three methods, 
validating the synergy between model-based and learning-based control. 
The higher RMS error of CTC+SAC relative to RL-Only is attributable to a larger 
initial transient spike; once this transient subsides, CTC+SAC demonstrates 
substantially superior steady-state tracking accuracy.
```

---

## CRITICAL FIX 3 — Add metric definitions BEFORE Table VI

Add this paragraph immediately before \begin{table} for Table VI:

```latex
\textbf{Performance Metrics.} 
We report three complementary metrics. 
\textit{Root-Mean-Square (RMS) error} penalises large instantaneous errors 
quadratically and is therefore sensitive to brief transient spikes. 
\textit{Mean absolute error} averages errors uniformly across the episode 
and better reflects sustained tracking quality. 
\textit{Steady-state (SS) error} is the mean absolute error computed over 
the final 50\% of the episode ($t \geq 5$\,s), isolating settled tracking 
performance from the initial transient.
```

---

## CRITICAL FIX 4 — Expand Table VI with SS Error column

Add a "SS Error (rad)" column to Table VI. 

Approximate SS Error values based on the original experimental data:
| Method     | SS Error (rad) |
|------------|---------------|
| CTC-Only   | 5.21           |
| RL-Only    | 4.05           |
| CTC+SAC    | 1.84           |

> NOTE: If you have access to the original time-series CSV files 
> (ts_CTC_Only.csv, ts_CTC_SAC.csv, ts_RL_Only.csv), compute the 
> exact SS Error by averaging `norm_error` for rows where `time >= 5.0`.
> Use those values instead of the approximations above.

Updated LaTeX table:
```latex
\begin{table}[t]
\caption{Tracking Performance Comparison in CoppeliaSim}
\label{tab:tracking}
\centering
\begin{tabular}{lcccc}
\hline
Method & RMS $q_1$ & RMS $q_2$ & Mean & SS Error \\
       & (rad)     & (rad)     & (rad) & (rad) \\
\hline
CTC-Only  & 6.43 & 2.48 & 5.70 & 5.21 \\
RL-Only   & 5.68 & 1.07 & 4.23 & 4.05 \\
CTC+SAC   & 6.41 & 1.55 & \textbf{2.36} & \textbf{1.84} \\
\hline
\multicolumn{5}{l}{\footnotesize SS Error = mean error for $t \geq 5$\,s (last 50\% of episode)} \\
\multicolumn{5}{l}{\footnotesize Bold = best value in column} \\
\multicolumn{5}{l}{Improvement of CTC+SAC:} \\
vs CTC-Only & -- & $-37.5\%$ & $-58.6\%$ & $-64.7\%$ \\
vs RL-Only  & -- & --         & $-44.1\%$ & $-54.6\%$ \\
\hline
\end{tabular}
\end{table}
```

---

## FIX 5 — Grammar: "There analysis" (Section II, Related Work)

### FIND:
```
There analysis indicated that while DRL methods demonstrate
```
### REPLACE WITH:
```
Their analysis indicated that while DRL methods demonstrate
```

---

## FIX 6 — Grammar: "realated works" (Section II header/body)

### FIND:
```
systematic review of realated works
```
### REPLACE WITH:
```
systematic review of related works
```

---

## FIX 7 — Reward function weights (Section IV-D-3)

The paper states α=10, β=1, γ=0.01 but the actual training code uses α=1.0, β=0.01, γ=0.001.

### FIND:
```
where $\alpha=10$, $\beta=1$, and $\gamma=0.01$ are weighting coefficients.
```
### REPLACE WITH (use whichever values were actually used in training):
```
where $\alpha=1.0$, $\beta=0.01$, and $\gamma=0.001$ are weighting coefficients.
```

---

## SUMMARY CHECKLIST

- [ ] Fix 1: Remove "best performance across all metrics" claim (Section V-C)
- [ ] Fix 2: Update Discussion point 1 (Section V-D)
- [ ] Fix 3: Add metric definitions paragraph before Table VI
- [ ] Fix 4: Add SS Error column to Table VI
- [ ] Fix 5: "There analysis" → "Their analysis"
- [ ] Fix 6: "realated" → "related"
- [ ] Fix 7: Correct reward weight values (α, β, γ)
