# Known issues and environment notes

## shap cannot read XGBoost ≥ 3.0 multiclass models

**Symptom.** `shap.TreeExplainer(xgb_classifier)` raises:

```
ValueError: could not convert string to float: '[3.3333167E-1,3.3333436E-1,3.3333388E-1]'
```

**Cause.** XGBoost 3.x writes a *vector* `base_score` for multi-class models (one
init score per class). shap's `XGBTreeModelLoader` still parses that field with
`float()`, which fails on the JSON array. Confirmed with shap 0.49.1 and
xgboost 3.1.2.

**What this project does.** `explain._xgboost_shap_values` asks XGBoost for the
values directly:

```python
booster.predict(xgb.DMatrix(X), pred_contribs=True)
```

`pred_contribs` is the same exact TreeSHAP computation, implemented inside
XGBoost, so the fix costs no fidelity. Returned shape is
`(n, classes, features + 1)` with the bias in the trailing column; the helper
drops it and transposes to the `(n, features, classes)` convention used
throughout `explain.py`.

Without this, the code would silently fall back to permutation importance and a
figure captioned "SHAP" would not be showing SHAP.
`tests/test_explain.py::test_xgboost_gets_exact_treeshap` asserts the exact path
is the one taken.

## `LinearExplainer` returns all zeros without a background distribution

**Symptom.** Every SHAP contribution for a single case is exactly `0.0`. No
error, no warning. Downstream, `LocalExplanation.drivers()` returns an empty list
and the risk message silently drops its "what stood out" clause — so the tool
still produces a plausible-looking output, just one that explains nothing.

**Cause.** A linear SHAP value is the coefficient times the feature's deviation
*from its background mean*. Construct the explainer with the row you are
explaining as its own background —

```python
Xt = transform(row)
shap.LinearExplainer(est, Xt).shap_values(Xt)   # every deviation is zero
```

— and every attribution is zero by construction.

**What this project does.** `_linear_shap_values` takes an explicit `background`
argument and `explain_instance` passes the training distribution through to it.
Two tests guard this: one asserts local attributions are never all zero, and one
deliberately reproduces the degenerate call to prove the failure mode is real
rather than hypothetical.

This is worth flagging because it is the most dangerous class of bug in an
explainability layer — it degrades the explanation to nothing while every other
part of the output continues to look correct.

## `cross_val_predict(..., params=...)` needs a recent scikit-learn

Per-sample weights are routed to XGBoost during inner cross-validation via the
`params=` argument, which requires scikit-learn ≥ 1.4 (verified on 1.7.2). On
older versions the argument is named `fit_params` and behaves differently. The
pinned floor in `requirements.txt` is ≥ 1.3, which is enough for everything
*except* this path — if you are on 1.3, drop XGBoost from the zoo or upgrade.

## Runtime

Full run (5 folds × 4 repeats, 6 models, inner 3-fold threshold tuning) is
roughly **6–8 minutes** on a laptop CPU. Roughly 480 model fits, dominated by the
random forest and the MLP.

```bash
python scripts/train.py --quick     # 1 repeat, no MLP: about 90 seconds
```

## Threshold instability at small folds

With `--folds 3` on ~1000 rows, each fold's inner tuning sees ~440 rows, and the
tuned `t_high` can vary noticeably between folds. This is a genuine property of
the data volume, not a bug — `fig7_threshold_stability.png` exists to show it. A
threshold that swings across folds is not one to deploy, and seeing the spread is
more useful than reading a single averaged number.

## Windows console encoding

Urdu output through a redirected pipe can raise `UnicodeEncodeError` on Windows
because the default console encoding is not UTF-8:

```bash
PYTHONIOENCODING=utf-8 python scripts/predict.py --age 34 ...
```

Files written by the pipeline are always explicitly `encoding="utf-8"`, so
artifacts are unaffected — this only affects terminal output.

## `HeartRate` and severe diastolic hypertension can never fire

Not a bug. The dataset's `HeartRate` never exceeds 90 (tachycardia needs > 100)
and `DiastolicBP` never reaches 110. Both criteria are kept in the rule set and
reported as *silent* by `calibration_report`, because a criterion the data cannot
reach tells you something about the data. Dropping them would hide that.

## The dataset is not vendored

`data/raw/` is gitignored. Run `python scripts/download_data.py`, or follow the
manual instructions it prints if the network is unavailable. The synthetic
stand-in in `data/bundled/` *is* committed, so a fresh clone runs offline — with
every output stamped `SYNTHETIC`.
