# Data card

Two datasets live under `data/`. They are not interchangeable and the code never
treats them as such.

---

## 1. `data/raw/` — the real dataset (not vendored)

**Name.** UCI Maternal Health Risk Data Set (UCI ML Repository, dataset 863).

**Size.** ~1,014 records, 6 features, 1 ordinal label.

**Collection.** Gathered through a community health-monitoring initiative in a
South Asian setting — rural hospitals, community clinics, and maternal
health-care facilities — with risk labels assigned by health professionals.
**Not collected in Pakistan.** That distinction is the premise of the calibration
study in `guidelines.py`, not a detail.

**Licence and access.** Public, free, no registration. Fetch with:

```bash
python scripts/download_data.py
```

**Label definition.** `RiskLevel` ∈ {low risk, mid risk, high risk}. The
labelling protocol is not published — we do not know which guideline, if any,
the labellers applied, or how disagreements were resolved. This is why the
project measures agreement against published thresholds rather than assuming the
labels encode them.

**Class balance.** Roughly 406 / 336 / 272 (low / mid / high). Imbalanced in the
worst direction: the class we least want to miss is the smallest.

**Known quality problems.** `HeartRate` values of 7 bpm; `Age` values up to 70;
exact duplicate rows; `BodyTemp` quantised to manually-recorded Fahrenheit
values; `BS` with no documented sampling condition. See
[../docs/UNITS.md](../docs/UNITS.md).

**Not vendored** because it is freely downloadable and because a copy in the repo
drifts from the source. `data/raw/` is gitignored.

---

## 2. `data/bundled/maternal_health_risk_SYNTHETIC.csv` — the stand-in (committed)

**What it is.** Deterministic synthetic data from
`mhrisk.data.generate_synthetic`, seeded with `config.RANDOM_SEED`. Committed
deliberately: a fresh clone runs the entire pipeline offline, and demonstration
results are reproducible.

**What it is not.** Not real. Not derived from any real record. Contains no
person's data, real or re-identifiable. Every artifact generated from it is
stamped `SYNTHETIC`, including inside the figure images.

**How it is built.** Sample a class from the published class proportions, then
sample vitals from class-conditional normal distributions whose centres and
spreads are chosen to land near published descriptive statistics of the real
file. Clip to the published per-column ranges. Discretise the way a clinic form
would (integer BP and heart rate, one decimal for glucose, Fahrenheit
temperatures on a weighted grid dominated by 98.0 °F).

**Deliberate imperfections.** A stand-in that is cleaner than the real thing
would let the pipeline pass tests it would fail in reality, so the generator
injects:

| Artifact | Why |
|---|---|
| `HeartRate = 7` on two rows | The real file contains them; the cleaning path must handle implausible vitals. |
| 12 exact duplicate rows | The real file contains duplicates; deduplication must be exercised. |
| 7% label noise on the mid/high boundary | Keeps the classes non-separable so accuracy lands in the published band (~76–84%) rather than at a suspicious 99%, and makes ordinal-aware metrics meaningful. |

**Calibrated to be interesting, not to be flattering.** The class-conditional
centres are chosen so guideline thresholds *partially* agree with the labels. If
they agreed perfectly, the calibration study would be vacuous; if they disagreed
completely, it would be measuring the generator rather than anything real.

**Fidelity check** (synthetic vs. published summaries of the real file):

| Column | Real (published) | Synthetic |
|---|---|---|
| Age, mean | ~30 | ~28.6 |
| SystolicBP, mean | ~113 | ~119.8 |
| DiastolicBP, mean | ~76 | ~78.2 |
| BS, mean | ~8.7 | ~9.3 |
| BodyTemp, mean | ~98.7 | ~98.6 |
| HeartRate, mean | ~74 | ~74.7 |

Close in the aggregate, and not claimed to be closer than that. The blood
pressure means in particular sit a few mmHg high. **The synthetic numbers are
not a substitute for the real ones and no result computed on them belongs in a
write-up without the SYNTHETIC qualifier.**

**Regenerate:**

```python
from mhrisk import data
data.write_bundled()
```

---

## Reference sources (not datasets)

Clinical thresholds in `guidelines.py` come from published guidance, each rule
carrying its own `source` string:

* ISSHP 2018 / WHO — hypertensive disorders of pregnancy (140/90 and 160/110);
* WHO 2013 — hyperglycaemia in pregnancy (fasting and 2-hour OGTT cut-offs);
* WHO integrated management of pregnancy and childbirth — fever (≥ 38 °C);
* WHO / obstetric consensus — adolescent (< 18) and advanced (≥ 35) maternal age.

`tests/test_guidelines.py::test_every_rule_cites_a_source` fails the build if a
threshold is added without one.
