---
title: "Safety-First Maternal Risk Screening: Documented Clinical Thresholds as a Baseline, Label-Transfer Calibration, and Urdu Risk Communication as a Deliverable"
author: "Laiba Mazhar"
date: "August 2026"
keywords: "maternal health, clinical prediction models, cost-sensitive learning, external validity, explainable AI, low-resource languages, Pakistan"
abstract: |
  Maternal risk classification on the UCI Maternal Health Risk dataset is a
  well-worn exercise. The existing literature on it converges on a single shape
  — preprocess, train a tree ensemble, report accuracy in the low-to-mid
  eighties, conclude that machine learning is promising for maternal healthcare
  — and that shape leaves three gaps unaddressed. It optimises a metric that
  cannot distinguish the two error types that matter clinically; it treats risk
  labels produced by an unpublished protocol in one country as ground truth for
  deployment in another; and it stops at a number, when the artefact a Lady
  Health Worker needs is a sentence she can say to a family.
  This paper restructures the task around those gaps. We contribute (i) a
  safety-first evaluation protocol built on a separately reported *critical
  miss* rate, an asymmetric cost matrix, and referral thresholds tuned under an
  operational budget inside the cross-validation loop; (ii) a label-transfer
  calibration study that encodes eleven sourced obstetric criteria as a
  transparent rule baseline and quantifies how far the dataset's labels sit from
  published guidance; and (iii) hand-authored Urdu risk communication whose
  safety properties are enforced by continuous-integration lint rules rather
  than reviewer diligence.
  Our headline result is negative and, we argue, the most useful finding here:
  **no learned model beat the eleven documented thresholds on high-risk recall,
  even after equalising referral load** (0.918 versus 0.847 for the best learned
  model). The learned models earn their place differently — they cut the
  catastrophic high-to-low error from 3.5% to 1.2% or below, reduce expected
  cost, and lift balanced accuracy from 0.642 to 0.700 — which supports a hybrid
  rules-plus-model design that the accuracy-reporting framing never puts on the
  table. We further find that the sampling condition of the dataset's
  *undocumented* blood-glucose column moves the clinical baseline more than any
  modelling choice available to us: reading it as fasting rather than post-load
  glucose shifts guideline–label agreement from 64.0% to 31.4%.
---

> **Provenance notice.** Every number in this manuscript was computed on a
> deterministic **synthetic** stand-in for the UCI dataset, not the real file.
> They demonstrate that the pipeline, the metrics, and the analyses behave as
> specified; they are **not** findings about maternal health. Section 4.2
> explains why and how, and `paper/README.md` gives the command sequence for
> regenerating every table on the real data. Claims whose direction could
> plausibly reverse are flagged in place.

---

# 1. Introduction

## 1.1 The decision this is really about

Pakistan's maternal mortality ratio remains among the higher figures in South
Asia [@pmms2019], and the majority of maternal deaths worldwide are caused by
conditions that are detectable in advance — hypertensive disorders, haemorrhage,
sepsis, and complications of pre-existing disease [@say2014global]. In much of
rural Pakistan the first clinical contact is not an obstetrician but a Lady
Health Worker (LHW) [@lhw-evaluation; @bhutta2011community], who can take a blood
pressure, a temperature, a pulse, and a glucose strip, and who must then make one
decision: refer, or not.

That decision is the intervention. Everything a screening tool does that does not
improve it — including its accuracy figure — is decoration. This framing drives
every choice in this paper, and it is worth stating plainly at the outset because
it is where we depart from the existing literature.

## 1.2 Three properties of a useful screening tool, and how current work scores

**It must weight errors the way the clinic does.** In this cohort roughly 37% of
mothers are labelled low risk, so a classifier that predicts "low risk" for
everyone attains 37% accuracy while missing 100% of high-risk cases. That is not
a straw man: it is the direction any unweighted loss pushes a model on imbalanced
data. Worse, accuracy treats *predicting mid risk for a high-risk mother* — which
still escalates her — as identical to *predicting low risk*, which sends her
home. Those errors differ in kind, not degree.

**It must not assume its labels transfer.** The dataset was collected through a
community health-monitoring initiative in Bangladesh [@ahmed-iot-maternal], and
its labelling protocol was never published: we do not know which guideline, if
any, the labellers applied, nor how disagreements were resolved. Treating those
labels as ground truth for a Pakistani deployment embeds an untested transfer
assumption at the root of the model — the failure mode systematic reviews of
clinical prediction models find repeatedly [@wynants2020covid;
@vancalster2019calibration].

**Its output must be deliverable at the point of care.** A probability, or a risk
band in clinical English, is not something an LHW can say to a family in rural
Punjab. The last mile — a sentence that is understood, believed, and acted on —
is where a screening programme succeeds or fails [@kelly2019challenges], and it is
almost entirely absent from the machine-learning literature on this dataset.

## 1.3 Contributions

1. **A safety-first evaluation protocol** (§5) whose primary quantities are
   high-risk recall, a separately tracked critical-miss rate, expected cost under
   an asymmetric matrix, and referral load; in which the decision rule is
   separated from the classifier and tuned under an explicit referral budget
   *inside* the cross-validation loop; and in which model selection is on recall
   rather than accuracy.
2. **A label-transfer calibration study** (§5.2, §6.2) encoding eleven sourced
   obstetric criteria as a transparent rule baseline, and measuring agreement,
   directionality of disagreement, and per-criterion discriminative contribution
   against the dataset's own labels.
3. **A matched-referral comparison** (§5.5, §6.3) that equalises operational cost
   before comparing a threshold-tuned model against a fixed clinical rule — the
   only form of that comparison we consider honest.
4. **Urdu risk communication as a first-class, testable deliverable** (§5.7),
   hand-authored rather than machine-translated, with three safety commitments
   enforced as executable lint rules.
5. **A reproducible artefact** (§8) in which every reported number is regenerated
   from a single command and stamped with its data provenance.
6. **A negative headline result** (§6.3) and an analysis of what the learned
   models do buy, which points to a hybrid design.

---

# 2. Existing work, and what each strand leaves undone

We organise prior work into six strands. For each we state what exists, the
limitation that matters for our problem, and the resulting gap. Section 3
consolidates the gaps and maps them to the sections that address them.

## 2.1 Machine learning on the UCI maternal health dataset

**What exists.** Since the dataset's release [@uci-maternal-health;
@ahmed-iot-maternal] a substantial body of student projects, notebooks, and
conference papers has applied the standard tabular toolkit to it: decision trees,
random forests, gradient boosting, occasionally a small neural network, usually
with SMOTE [@chawla2002smote] or class weighting for the imbalance, and typically
reporting accuracy and macro F1 in the low-to-mid eighties, sometimes with a
feature-importance plot.

**Limitations.**

* *Metric choice.* Accuracy and macro F1 are symmetric across error types. Neither
  can express that high→low is categorically worse than high→mid, so neither can
  be optimised toward the behaviour a clinic wants.
* *No clinical floor.* We are not aware of prior work on this dataset that
  compares against a guideline-derived rule baseline. Without one, "84% accuracy"
  is unanchored: the reader cannot tell whether the model has learned anything a
  threshold sheet does not already encode.
* *Implicit thresholds.* Bands are assigned by `argmax`, which fixes the decision
  threshold at an arbitrary point and makes the referral load an accident of the
  fitted probabilities rather than a design parameter.
* *Provenance treated as settled.* The dataset's collection context and its
  undocumented units are generally passed over.

**Gap.** *(G1, G2, G3)* An evaluation of this dataset in which errors are priced
asymmetrically, the decision threshold is an explicit tuned object under an
operational constraint, and a documented clinical rule set is present as the
comparator.

## 2.2 Maternal risk prediction and obstetric risk scoring more broadly

**What exists.** Obstetric practice already uses threshold-based risk
identification: blood-pressure cut-offs for gestational and severe hypertension
[@isshp2018], glycaemic cut-offs for hyperglycaemia first detected in pregnancy
[@who-hip-2013], fever and danger-sign checklists in WHO's antenatal and
complication-management guidance [@who-anc-2016; @who-imppc]. These are designed
for exactly our setting: few measurements, non-specialist users, referral as the
output.

**Limitations.** Guideline rules are deliberately conservative and have no graded
confidence. When no criterion fires they return "low risk" with no hedge
available, and they cannot express *how close* a mother is to a threshold. They
also cannot be tuned to a referral budget: a rule set refers whoever it refers.

**Gap.** *(G3)* A quantitative account of what a learned model adds *over* these
rules, measured at equal operational cost rather than at each system's own
convenient operating point — and, symmetrically, of what the rules still do
better, so a hybrid design can be argued for rather than assumed.

## 2.3 Evaluation methodology for clinical prediction models

**What exists.** The methodological literature has been explicit for over a
decade that discrimination alone is insufficient. TRIPOD [@tripod2015] specifies
reporting requirements including data provenance and validation; Steyerberg and
Vergouwe [@steyerberg2014abcd] set out development and validation steps; Van
Calster et al. [@vancalster2019calibration] argue calibration is where deployed
models fail; Vickers and Elkin [@vickers2006dca] formalise the decision-analytic
question of net benefit across thresholds; cost-sensitive learning has a clean
theoretical basis [@elkan2001cost]; and precision–recall analysis is known to be
more informative than ROC under imbalance [@saito2015prc].

**Limitations.** This is prescription, not instantiation. Systematic review finds
the prescriptions are widely ignored in practice — Wynants et al.
[@wynants2020covid] judged nearly every model in a large COVID-19 review at high
risk of bias. Decision curve analysis in particular integrates net benefit across
a threshold range, which is the right idea but a continuous, somewhat abstract
summary; a programme manager's actual question is discrete and concrete: *at the
referral load I can staff, how many high-risk mothers do I catch?*

**Limitation specific to threshold tuning.** Where thresholds are tuned at all,
they are frequently tuned on the same data used to report performance. A tuned
threshold is a fitted parameter; tuning it outside the resampling loop leaks.

**Gap.** *(G1, G4)* A concrete, discrete instantiation of net-benefit reasoning
for this problem — a referral budget as an optimisation constraint — combined
with a resampling protocol in which threshold selection is nested inside the
outer folds so it cannot leak.

## 2.4 Explainability in clinical machine learning

**What exists.** SHAP [@lundberg2017shap] and its exact tree variant
[@lundberg2020trees] are the default attribution tools in clinical tabular ML,
and "we applied SHAP" is now a near-standard sentence in this literature.

**Limitations.**

* *Method substitution goes unreported.* Exact SHAP exists only for particular
  model classes. In practice a pipeline that cannot run an exact explainer falls
  back to an approximate method, and the figure caption still says "SHAP". We hit
  precisely this: shap 0.49 cannot parse XGBoost ≥ 3.0 multiclass models, and the
  default behaviour of a naively written pipeline is a silent downgrade.
* *Silent degenerate output.* Attribution code can return all-zero values without
  raising — for instance a linear explainer constructed with the explained row as
  its own background — producing an explanation that says nothing while every
  other part of the interface looks correct.
* *Post-hoc explanation of an unnecessary black box.* Rudin [@rudin2019stop] argues
  that for high-stakes decisions on structured data an interpretable model should
  be used rather than a black box plus a post-hoc story. On six tabular vitals
  this critique lands hard.

**Gap.** *(G5)* An attribution layer that names its own method in every output,
uses the exact explainer wherever one exists, is tested against the degenerate
all-zero failure, and reports honestly when a simple interpretable model is
competitive.

## 2.5 Health communication, localisation, and low-resource languages

**What exists.** mHealth and community health-worker programmes in Pakistan have
a real evidence base [@bhutta2011community; @lhw-evaluation], and WHO's antenatal
guidance stresses communication quality as part of care [@who-anc-2016]. In NLP,
Joshi et al. [@joshi2020linguistic] document how systematically under-served
languages like Urdu are relative to their speaker populations.

**Limitations.** The clinical-ML literature on this dataset ends at the metric;
where an interface is presented, it is English and clinical. Where localisation
appears in health-AI work it is typically machine translation of clinical text —
which for medical instructions risks inverting urgency, asserting a diagnosis the
system cannot support, or landing in a register that reads as either frightening
or dismissive. Crucially, message quality is treated as a review activity, not an
engineering property: nothing in the artefact prevents a later edit from making
the wording unsafe.

**Gap.** *(G6)* Hand-authored bilingual risk communication whose safety
properties — no catastrophising, no diagnosis claims, an actionable step with a
timeframe, family-inclusive framing where decisions are collective — are encoded
as automated tests that fail the build, plus honest surfacing of review status.

## 2.6 Data documentation, dataset shift, and proxy labels

**What exists.** Datasheets for Datasets [@gebru2021datasheets], Model Cards
[@mitchell2019modelcards], and Data Statements [@bender2018datastatements]
established that provenance, collection context, and intended use should ship with
the artefact. Obermeyer et al. [@obermeyer2019bias] showed how a proxy label can
encode a systematic inequity that accuracy metrics cannot detect.

**Limitations.** These frameworks tell you to *document* provenance; they do not
tell you what to do when the documentation is already missing and cannot be
recovered. The UCI file's blood-glucose column has no recorded sampling
condition, and the two plausible readings imply thresholds that differ by more
than a factor of two.

**Gap.** *(G2, G7)* A method for handling irrecoverably undocumented units:
state the assumption explicitly, and *quantify the sensitivity of every
downstream result to it*, so the reader can see how much of the analysis rests on
a guess.

---

# 3. Gap analysis and positioning

| # | Gap left by existing work | How this work fills it | Where |
|---|---|---|---|
| **G1** | Symmetric metrics cannot express that high→low is categorically worse than high→mid | Critical-miss rate reported separately from recall; asymmetric cost matrix pricing a critical miss at 25× an unnecessary referral; selection on recall, not accuracy | §5.3, §6.1 |
| **G2** | Dataset labels assumed to transfer; provenance and units treated as settled | Eleven sourced criteria compared against the labels (agreement, κ, directionality, per-rule activity, silent criteria); glucose-interpretation sensitivity quantified | §5.2, §6.2 |
| **G3** | No clinical floor; no account of what learning adds over documented rules | Guideline rule set evaluated as a first-class model in the same loop, plus a matched-referral comparison at equal operational cost | §5.4, §6.3 |
| **G4** | Thresholds implicit (`argmax`) or tuned with leakage; referral load an accident | Ordered-threshold decision rule separated from the model; tuning under a referral budget nested inside the outer folds; threshold stability reported | §5.3, §5.6, §6.5 |
| **G5** | "We applied SHAP" conceals method substitution and silent degenerate output | Exact TreeSHAP and exact LinearSHAP routed per model family, method named in every output, degenerate-attribution regression tests, interpretable model reported as competitive | §5.6, §6.4 |
| **G6** | Localisation absent or machine-translated; message quality a review activity | Hand-authored Urdu with three commitments enforced as CI lint; review status surfaced in every artefact and on screen | §5.7, §6.6 |
| **G7** | Documentation frameworks presume recoverable provenance | Assumption stated explicitly and its downstream sensitivity measured as a reported result | §4.1, §6.2 |

Our aim is explicitly **not** to improve on published accuracy figures — §6.1
shows our models land in the same band. It is to argue that the figure was never
the interesting quantity.

---

# 4. Data

## 4.1 The real dataset, and one load-bearing ambiguity

The UCI Maternal Health Risk Data Set contains approximately 1,014 records with
six features — age, systolic and diastolic blood pressure, blood glucose, body
temperature, resting heart rate — and an ordinal label in {low risk, mid risk,
high risk}, distributed roughly 406/336/272. The imbalance runs in the worst
direction: the class we least want to miss is the smallest.

Three data properties shape everything downstream.

**Units are undocumented, and one is load-bearing.** Body temperature is
Fahrenheit (reading it as Celsius would be catastrophic) and glucose is mmol/L,
but the *sampling condition* of the glucose column is unrecorded. WHO thresholds
for hyperglycaemia in pregnancy differ sharply by sampling condition
[@who-hip-2013]:

| Reading | Gestational diabetes | Diabetes in pregnancy |
|---|---|---|
| Fasting plasma glucose | 5.1–6.9 mmol/L | ≥ 7.0 mmol/L |
| 2-hour 75 g OGTT | 8.5–11.0 mmol/L | ≥ 11.1 mmol/L |

The observed range is roughly 6–19 mmol/L with a median near 8. Read as fasting
values, over 80% of a community *screening* cohort would be diabetic, which is
not credible; read as post-load values the distribution is clinically ordinary.
We therefore assume the OGTT reading, say so explicitly, and report the
sensitivity of every guideline result to that assumption (§6.2) — following the
spirit of @gebru2021datasheets where the documentation itself is unrecoverable.

**Some values are physiologically impossible.** Heart rates of 7 bpm are dropped
digits, not bradycardic patients; ages up to 70 are not maternal ages. We repair
these by class-conditional median imputation rather than dropping the row, which
preserves the five other usable vitals on an affected record.

**Two clinical criteria cannot fire.** Diastolic pressure never reaches the
severe-hypertension threshold of 110 mmHg, and heart rate never exceeds 90 bpm,
so the tachycardia criterion (>100) is unreachable. We retain both and report
them as *silent*: a criterion whose cut-off the data cannot reach is a fact about
the data, and dropping it would conceal that fact.

## 4.2 The synthetic stand-in, and the status of every number here

All numbers in this manuscript were computed on a deterministic synthetic dataset
generated to resemble published summaries of the real file. This was a project
constraint rather than a methodological choice, and we state it plainly:
**nothing in §6 is a finding about maternal health.** The results demonstrate
that the pipeline runs, that the metrics behave as designed, and that the
analyses produce the shape of answer they claim to.

The generator samples a class from the published class proportions, then samples
vitals from class-conditional normal distributions clipped to the published
per-column ranges and discretised as a clinic form would record them. It
deliberately injects the real file's awkward properties — implausible heart
rates, exact duplicate rows, and 7% label noise concentrated on the mid/high
boundary — because a stand-in cleaner than the real data would let the pipeline
pass tests it would fail in practice. Class-conditional centres were chosen so
that guideline thresholds *partially* agree with the labels: perfect agreement
would render the calibration study vacuous, and total disagreement would mean
measuring the generator rather than anything real.

Synthetic provenance propagates into every artefact — the metadata record, every
generated table, the interface banner, and the figures, where the stamp is burned
into the image rather than left in the caption, because a figure pasted into a
slide deck loses its caption long before it loses its axes. An automated test
asserts this propagation.

---

# 5. Method

## 5.1 Overview

```
load → clean → guideline calibration → repeated CV with nested threshold tuning
     → model selection on recall → matched-referral comparison
     → refit + freeze operating point → attribution → bilingual message
     → artefacts (tables, figures, metadata, model bundle)
```

One command produces all of it, so no reported number is transcribed by hand.

## 5.2 Guideline criteria as a first-class baseline

We encode eleven obstetric criteria drawn from published guidance, each stored as
*data* with its threshold, severity, and source, so a clinician can review the
rule set without reading code:

* **Hypertension** [@isshp2018]: systolic 140–159 or diastolic 90–109 mmHg
  (moderate); systolic ≥ 160 or diastolic ≥ 110 (severe).
* **Hyperglycaemia in pregnancy** [@who-hip-2013]: 8.5–11.0 mmol/L (moderate);
  ≥ 11.1 (severe), under the OGTT reading of §4.1.
* **Fever** [@who-imppc]: 100.4–101.9 °F, i.e. ≥ 38 °C (moderate); ≥ 102 °F
  (severe).
* **Tachycardia**: resting heart rate > 100 bpm (moderate).
* **Maternal age**: < 18 or ≥ 35 years (moderate).

Escalation is deliberately simple enough to defend in a viva: any severe feature
→ high risk; two or more moderate features → high risk; exactly one → mid risk;
none → low risk.

This serves two purposes. As a *comparator* it sets the floor a learned model must
clear to justify its complexity. As an *instrument* it lets us ask whether the
dataset's labels agree with published guidance at all — the calibration study
addressing G2.

## 5.3 Safety-first metrics (G1)

Reported in this order:

* **High-risk recall** — the fraction of high-risk mothers escalated at all.
* **Critical miss rate** — the fraction of high-risk mothers predicted *low*
  risk. Tracked separately because "mid risk" still escalates her; the
  distinction is between a delayed referral and no referral.
* **Expected cost** — mean cost under an asymmetric matrix [@elkan2001cost]
  pricing high→low at 25, high→mid at 8, mid→low at 4, and unnecessary
  escalation at 1–2. These encode a clinical judgement, not a measurement; they
  live in one configuration object so they can be argued with and swept.
* **Referral rate** — the share escalated. A first-class metric because the quiet
  way a screening tool fails is by referring so many mothers that the programme
  stops believing it.
* Conventional metrics — accuracy, balanced accuracy, macro F1, and
  quadratic-weighted κ [@cohen1968weighted] — reported for comparability with
  published work, never for selection.

## 5.4 A decision rule separate from the model (G4)

Fitting a classifier and choosing where to put the referral threshold are
different acts; conflating them is how safety-critical thresholds end up wherever
`argmax` happened to land. We assign bands by ordered thresholds on the predicted
distribution: high when P(high) ≥ t_high; at least mid when P(mid) + P(high) ≥
t_escalate; low otherwise. A 40% chance of high risk therefore escalates even
when low risk holds a 45% plurality — a decision `argmax` cannot express.

Thresholds maximise high-risk recall **subject to a referral budget** (55% here),
ties broken on expected cost. The constraint is part of the objective, not a
footnote: without it the optimum is always "refer everybody", which has perfect
recall and is switched off within a month. This is a discrete, operational
instantiation of the net-benefit reasoning of @vickers2006dca.

## 5.5 Protocol, leakage control, and the matched comparison (G3, G4)

Repeated stratified cross-validation, 5 folds × 4 repeats. Because the referral
thresholds are fitted parameters, each outer fold tunes its operating point on
**inner** 3-fold cross-validated probabilities from that fold's training portion
only, then applies the frozen point to the held-out fold.

The model zoo is a majority-class baseline, the guideline baseline, logistic
regression, random forest, XGBoost [@chen2016xgboost], and a small MLP, all built
on scikit-learn [@scikit-learn], with class balancing throughout — `class_weight`
where available, inverse-frequency sample weights for XGBoost. Selection is on
mean high-risk recall, tie-broken on expected cost, baselines excluded from
selection but reported alongside.

**The matched-referral comparison.** The guideline rule has no threshold to tune,
so comparing it against a threshold-tuned model at each system's own operating
point is not a comparison — whichever refers more people wins on recall by
construction. We therefore also push every learned model to the guideline
baseline's exact referral load and compare there. This is the honest form of
"does learning add anything over the guidelines?"

## 5.6 Attribution that names itself (G5)

Exact explainers wherever one exists: TreeSHAP for the ensembles
[@lundberg2020trees], the closed-form linear explainer for logistic regression.
XGBoost ≥ 3.0 multiclass models are routed through XGBoost's own `pred_contribs`,
because shap 0.49 cannot parse their vector-valued `base_score` — the exact same
values, obtained from the library that can compute them. Where no exact explainer
exists (the MLP) the fallback is permutation-based and **the method name is
carried into every table, caption, and screen**. An attribution silently swapped
underneath a plot labelled "SHAP" is worse than no plot.

Two regression tests guard the degenerate failure identified in §2.4: one asserts
local attributions are never all-zero, and one deliberately reproduces the
all-zero call to prove the failure mode is real rather than hypothetical.

## 5.7 Communication as part of the system (G6)

Model output becomes a bilingual message through hand-authored templates. Urdu is
written by hand, not machine-translated, because a mistranslated clinical
instruction is a safety incident and because register matters as much as
vocabulary. Three commitments are enforced by lint rules that run in CI, so a
well-meaning later edit fails the build rather than shipping:

1. **No catastrophising.** Urgency lives in the recommended action and its
   timeframe ("today", "within a few days"), never in frightening adjectives. A
   family that panics may go nowhere, or to the wrong place.
2. **No diagnosis.** The tool reports observations and names who to see. It never
   names a condition, because it cannot confirm one.
3. **Family-inclusive framing at the high band.** Where care decisions are made
   collectively, a message addressed only to the patient can stall against a
   household that was never consulted.

A fourth rule emerged from reviewing rendered output rather than from design:
**the low band mentions no drivers at all.** Telling a mother her result is
normal and then listing which of her vitals "stood out" plants worry the result
does not justify, and worry is what makes a family ignore the next message.

Review status is explicit. The templates ship as `UNREVIEWED`; the interface
renders that status on every result and the metadata record carries it, because
"plausible Urdu written by the project author" is not "signed off by a
native-speaker reviewer and a clinician".

---

# 6. Results

All numbers are from the synthetic stand-in (§4.2). Full tables are in
`paper/generated_tables.md`, regenerated from artefacts by script.

## 6.1 Cross-validated performance

| Model | High-risk recall | Critical miss | Expected cost | Referral rate | Balanced acc. | Accuracy |
|---|---|---|---|---|---|---|
| guideline_baseline | **0.900** ± 0.041 | 0.036 ± 0.027 | **1.010** ± 0.221 | 0.633 ± 0.019 | 0.656 ± 0.031 | 0.640 ± 0.033 |
| logistic_regression | 0.730 ± 0.059 | 0.044 ± 0.029 | 1.290 ± 0.201 | 0.542 ± 0.019 | **0.759** ± 0.020 | **0.764** ± 0.018 |
| mlp | 0.716 ± 0.054 | 0.036 ± 0.022 | 1.269 ± 0.161 | 0.548 ± 0.025 | 0.758 ± 0.020 | 0.764 ± 0.019 |
| xgboost | 0.663 ± 0.065 | 0.034 ± 0.024 | 1.395 ± 0.227 | 0.535 ± 0.023 | 0.740 ± 0.027 | 0.750 ± 0.025 |
| random_forest | 0.648 ± 0.047 | **0.029** ± 0.019 | 1.383 ± 0.186 | 0.542 ± 0.024 | 0.744 ± 0.022 | 0.755 ± 0.021 |
| majority_baseline | 0.000 | 1.000 | 8.417 ± 0.034 | 0.000 | 0.333 | 0.370 ± 0.001 |

**The majority baseline earns 37% accuracy and misses every high-risk mother**,
at an expected cost roughly six times the worst real model's. This is the concrete
form of the G1 argument: a metric on which the degenerate model looks merely
mediocre rather than disqualified is not a safety metric.

**Our models land in the accuracy band the literature reports** (0.750–0.764), so
this is not a story about a weak implementation; they are being asked a different
question.

**Selection on recall changes the answer, and exposes a trade-off two numbers
apart.** Logistic regression is selected — the simplest and most inspectable
learned model, consistent with @rudin2019stop. Note that random forest achieves
the *lowest* critical-miss rate (0.029) while having the *worst* high-risk recall
(0.648): it rarely commits the catastrophic error but under-escalates into the
mid band more often. Reporting either number alone misrepresents it.

## 6.2 Do the dataset's labels agree with published guidance? (G2, G7)

Exact agreement between guideline bands and dataset labels is **64.0%**, linear
κ 0.584, quadratic-weighted κ **0.696** — substantial ordinal agreement, well
short of the interchangeability that treating these labels as clinical ground
truth assumes.

| | guideline: low | guideline: mid | guideline: high |
|---|---|---|---|
| **dataset: low** | 267 | 88 | 15 |
| **dataset: mid** | 90 | 121 | 139 |
| **dataset: high** | 10 | 18 | 253 |

The disagreement has structure, not just noise.

* **The guidelines are stricter in the middle.** Of 350 mothers the dataset calls
  mid risk, 139 (40%) reach the guideline high band. Whatever protocol the
  labellers used was more permissive than published thresholds for precisely the
  ambiguous cases.
* **28 of 281 dataset high-risk mothers (10%) fall below the guideline high
  band** — published thresholds alone would not escalate them, which is a direct
  argument against replacing a model with a rule sheet.
* **15 dataset low-risk mothers reach the guideline high band** — the rules
  over-escalate relative to the labellers, at a low but non-zero rate.

**Rule activity shows a third of the rule set doing nearly all the work.** The
diabetes-in-pregnancy criterion fires for 69.4% of high-risk mothers and 0% of
low-risk ones; advanced maternal age for 51.2% versus 4.1%. Two criteria never
fire at all (§4.1).

**One undocumented column dominates every modelling choice.** Switching the
glucose interpretation from post-load to fasting moves agreement from 64.0% to
31.4%, quadratic κ from 0.696 to 0.136, and the share of mothers the guidelines
would escalate from 40.7% to 82.0%:

| Interpretation | Moderate cut-off | Severe cut-off | Agreement | Quadratic κ | Flagged high risk |
|---|---|---|---|---|---|
| 2-hour OGTT | 8.5 | 11.1 | 0.640 | 0.696 | 40.7% |
| Fasting | 5.1 | 7.0 | 0.314 | 0.136 | 82.0% |

This is our clearest methodological finding and the substance of G7. The sampling
condition of one column the dataset never documented changes the clinical
baseline more than any architecture, hyperparameter, or threshold in the study.
Work that trains on this dataset without stating its glucose interpretation has
left the most consequential assumption in the analysis unstated.

## 6.3 Does learning beat the guidelines at equal cost? (G3)

Forcing every learned model to the guideline baseline's referral load (≈ 0.648):

| Model | Referral rate | High-risk recall | Critical miss | Expected cost | Balanced acc. |
|---|---|---|---|---|---|
| guideline_baseline | 0.648 | **0.918** | 0.035 | 0.993 | 0.642 |
| logistic_regression | 0.648 | 0.847 | 0.012 | **0.907** | 0.700 |
| mlp | 0.638 | 0.835 | 0.012 | 0.887 | **0.731** |
| xgboost | 0.648 | 0.788 | **0.000** | 0.930 | 0.715 |
| random_forest | 0.651 | 0.776 | **0.000** | 0.987 | 0.699 |
| majority_baseline | 0.000 | 0.000 | 1.000 | 8.455 | 0.333 |

**No learned model beats eleven documented thresholds on high-risk recall, even
at matched referral load** — 0.918 against 0.847 for the best. We report this as
the headline because it is the result a reader most needs and the one this
literature's framing tends to obscure. On the project's own primary metric, the
machine learning did not win.

What the learned models *do* buy is a different error profile:

* **The catastrophic error nearly disappears.** The rule baseline sends 3.5% of
  high-risk mothers home as low risk; logistic regression 1.2%; both tree
  ensembles none. The rules catch more high-risk mothers overall yet are *more*
  likely to place one in the worst band — because a fixed rule has no graded
  confidence: when no criterion fires it says "low risk" with no hedge available
  (§2.2).
* **Expected cost is lower** for logistic regression (0.907) and the MLP (0.887)
  than for the rules (0.993), following directly from the point above given a
  cost matrix that prices a critical miss at 25.
* **Balanced accuracy is substantially better** — 0.700–0.731 versus 0.642 —
  because the rule baseline buys recall partly by escalating indiscriminately.

The defensible claim is narrower and more interesting than "our model is better":
at equal operational cost the learned models trade a little high-risk recall for
near-elimination of the worst individual error and a better overall cost profile.
A deployment should therefore use **both** — the rules as a non-overridable
safety net, the model to grade everything the rules leave silent. **This is the
claim most at risk of reversing on real data** and should be re-derived before
being repeated.

## 6.4 Attribution (G5)

Exact LinearSHAP on the selected model ranks blood glucose first by a wide margin
(mean |SHAP| 1.09), then systolic pressure (0.53), diastolic pressure (0.45), age
(0.25), heart rate (0.19), body temperature (0.12). Exact TreeSHAP on random
forest and XGBoost yields the same ordering.

Agreement across three model families and two independent exact SHAP
implementations is a validity signal: glucose and blood pressure dominating a
maternal-risk model is clinically expected, and a model leaning on body
temperature would have been suspect regardless of accuracy. The ordering also
matches the rule-activity analysis of §6.2 — two methods, one built from published
thresholds and one from fitted coefficients, agreeing on which vitals carry
signal.

## 6.5 Threshold stability (G4)

Tuned `t_high` varies across folds (per-model spreads in Table 7 of
`generated_tables.md`), as expected at ~1,000 rows with ~440 rows per inner
tuning set. We report the spread rather than an averaged point estimate: a
threshold that swings across folds is not one to deploy, and a single mean hides
that.

## 6.6 Message safety (G6)

All three band templates pass the lint gate in both languages. The suite includes
adversarial tests that inject a catastrophising phrase, a diagnosis claim, and an
action-free message, and assert the linter catches each — the linter is verified
to have teeth rather than assumed to. Review status is `UNREVIEWED` for all three
bands and a test asserts the artefact does not claim otherwise.

---

# 7. Discussion

**The negative result is the useful one.** A study designed to show a model beats
a rule sheet, which found the opposite, is more informative than one reporting
84% accuracy and stopping. It also has a direct design consequence: run the
documented criteria as a non-overridable safety net and use the model to grade the
cases where no criterion fires. That hybrid is invisible to the accuracy-reporting
framing, because that framing never puts the rule baseline on the table.

**Undocumented units are a bigger risk than model choice.** §6.2 found the glucose
sampling condition moves the clinical baseline far more than anything we varied in
modelling. The lesson is unglamorous: for small clinical tabular datasets, effort
spent resolving provenance with whoever collected the data will usually beat effort
spent on architecture. This extends the datasheet literature
[@gebru2021datasheets] with a concrete recipe for the case where documentation is
already unrecoverable — state the assumption, then measure what it costs you.

**Label transfer is measurable, and cheap to measure.** Quadratic κ of 0.696
between dataset labels and published thresholds is substantial agreement with a
structured residue: the guidelines are systematically stricter on ambiguous mid
cases, so a model trained on these labels inherits the labellers'
permissiveness. This is an afternoon's work and it is what TRIPOD asks for
[@tripod2015].

**Interpretability was not a sacrifice here.** Logistic regression was selected
on the project's own safety criterion, not chosen for simplicity. On six tabular
vitals that is the outcome @rudin2019stop predicts, and it means the deployed
model has exactly-computable attributions and reviewable coefficients rather than
a post-hoc story about a black box.

**Communication is not downstream polish.** The one design rule we did not
anticipate — suppressing driver phrases at the low band — came from reading
rendered Urdu, not from the model. It is invisible to every metric in §6 and
would plausibly matter more to real-world outcomes than the difference between
our best and worst classifier. Encoding message constraints as CI lint rather
than review guidelines is the component of this project we would most readily
reuse elsewhere.

---

# 8. Threats to validity and limitations

**Construct validity.** The cost matrix is a judgement, not a measurement:
pricing a critical miss at 25× an unnecessary referral is defensible but
arbitrary, and expected-cost comparisons move with it. It is isolated in
configuration so it can be challenged; a sensitivity sweep over it is the obvious
next step.

**Internal validity.** Threshold tuning is nested to prevent leakage, but with
~1,000 rows every difference we report carries wide uncertainty and several gaps
in §6.1 lie within one standard deviation. The matched-referral comparison uses a
single held-out split at a fixed operating point, deliberately outside the tuning
loop; it is therefore noisier than the cross-validated table.

**External validity.** All numbers come from synthetic data (§4.2). Beyond that,
the real dataset is Bangladeshi and nothing here establishes transfer to
Pakistan — the calibration study *measures* the gap between labels and published
thresholds; it does not close it. Six features are available: no gestational age,
parity, obstetric history, proteinuria, or haemoglobin, several of which are more
clinically informative than anything in this feature set. The ceiling is set by
the data, not the models.

**Conclusion validity.** The §6.3 finding is the most likely to reverse on real
data, and it is the paper's headline. We flag it as conditional rather than
burying the caveat.

**Deployment validity.** The Urdu templates are `UNREVIEWED` — not signed off by a
native-speaker reviewer or a clinician, and never tested with a health worker or
a patient. This is not a medical device and nothing here should inform a clinical
decision.

---

# 9. Future work

1. **Re-run on the real dataset** and re-derive §6, particularly §6.3.
2. **Resolve the glucose sampling condition** with the original collectors; this
   is the single highest-value action available (§6.2).
3. **Sweep the cost matrix** to establish which conclusions are robust to the
   clinical judgement encoded in it.
4. **Validate on Pakistani data**, which is the only way to close rather than
   measure the transfer gap.
5. **Formalise the hybrid** suggested by §6.3 — rules as a safety net, model as a
   grader over the silent cases — and evaluate it as a system rather than
   inferring it from two tables.
6. **Get the Urdu reviewed** by a native speaker and a clinician, and test the
   messages with LHWs; promote template review status accordingly.
7. **Establish accountability** for a missed case before the tool reaches anyone's
   hands. A screening tool without a named owner for its failures is not
   deployable regardless of its metrics.

---

# 10. Conclusion

Recast as a decision problem rather than a classification benchmark, the UCI
maternal health dataset yields a different and more useful set of answers. Eleven
documented obstetric thresholds beat every model we trained on high-risk recall
even at matched referral load; the learned models earn their place by nearly
eliminating the catastrophic high-to-low error rather than by being more
accurate; the dataset's labels agree with published guidance at quadratic κ 0.696
with a systematic permissiveness on ambiguous cases; and the undocumented
sampling condition of a single glucose column moves the clinical baseline more
than any modelling choice available to us.

None of these findings is reachable from an accuracy number, and none required a
larger dataset or a better model — only asking what the tool is for, and measuring
that instead.

---

# Reproducibility

Single seed (20260822) through data generation, splits, and every model. Each run
writes a metadata record with the data SHA-256, row counts, cross-validation
configuration, glucose interpretation, selected model, frozen operating point,
attribution method, template review status, and library versions. Every table in
§6 is regenerated from those artefacts by `scripts/make_paper_tables.py`. The
test suite (155 tests) covers the metric definitions against hand-worked cases,
every guideline threshold at its boundary, the message-safety lint gate including
adversarial cases, the degenerate-attribution regression, and provenance
propagation into artefacts.

```bash
pip install -r requirements.txt
python scripts/download_data.py          # fetch the real dataset
python scripts/train.py --source real
python scripts/make_figures.py
python scripts/make_paper_tables.py
python -m pytest
```

# References
