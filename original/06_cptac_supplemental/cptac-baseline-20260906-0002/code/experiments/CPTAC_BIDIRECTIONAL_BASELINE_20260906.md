# BRCA/LUAD bidirectional baseline — fixed first-pass development design

Fixed before running: use only the verified BRCA (122) and LUAD (110) blocks.
RNA log2 values and protein assay values are not transformed a second time.
This is a direct-prediction comparator, not CREED FM/agent efficacy evidence.

- RNA: exact unique source symbols. Protein: BRCA uses SYMBOL|RefSeq accession
  (NP/XP), LUAD uses SYMBOL|SYMBOL. Project only recognized structures to gene
  labels; remove every ambiguous or duplicate gene-label row before joining.
  This is a gene-level feature bridge, not equivalence of protein isoforms.
  Attempt 0001 found zero exact composite-ID overlap and produced no predictions;
  this identity correction defines attempt 0002. Model settings are unchanged.
- Predict RNA→protein and protein→RNA separately.
- Five cohort-stratified patient-disjoint folds plus BRCA→LUAD and LUAD→BRCA
  reciprocal transfers. All data previously inspected; no new blind-test claim.
- Predictor eligibility >=80% observed and nonconstant on training rows only;
  choose at most 1,000 largest training-SD predictors. Training-median imputation
  plus missingness indicators, train-only scaling/PCA, rank 16.
- Target eligibility >=80% observed on training rows only, SD>1e-8. Training
  mean/SD define evaluation scale. Missing/censored labels never enter the loss.
- Multi-output masked ridge minimizes per-target observed mean squared loss
  plus 0.1 L2 regularization; intercept unpenalized. No hyperparameter search.
- Controls: training mean; train-only cohort means (global mean for unseen
  cohort); same-gene univariate ridge; row-shuffled target matrix including mask.
- Evaluate all arms on identical target panels and finite truth masks, reporting
  mean/median per-target r, defined-correlation counts and standardized RMSE.
  Constant-control correlations are undefined, not silently reported as zero.
- Preserve predictions, target scale, IDs and split indices on A6000. Independent
  metric replay and patient overlap checks precede interpretation.

This first pass measures available predictive signal and transfer difficulty.
It is not a tuned best-in-class baseline, a quantitative cross-assay calibration,
or evidence that sparse-input completion/agent orchestration already succeeds.
