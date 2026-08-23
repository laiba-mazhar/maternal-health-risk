# Units, and the one column nobody documented

The UCI Maternal Health Risk Data Set ships six feature columns and no units.
Five of them are unambiguous. The sixth is the most important feature in every
model trained on this data, and its interpretation is a guess.

| Column | Unit | Confidence | Notes |
|---|---|---|---|
| `Age` | years | certain | Range 10–70. The upper tail is not obstetrically plausible; see below. |
| `SystolicBP` | mmHg | certain | Range 70–160. |
| `DiastolicBP` | mmHg | certain | Range 49–100. Never reaches the severe cut-off of 110. |
| `BS` | mmol/L | **unit certain, sampling condition unknown** | Range ~6–19. See below. |
| `BodyTemp` | degrees **Fahrenheit** | high | Range 98–103. Values cluster on whole/half degrees, consistent with manual recording. Reading these as Celsius would be catastrophic. |
| `HeartRate` | bpm, resting | certain | Range 7–90. The 7 bpm entries are recording errors. |

## The blood-sugar problem

`BS` is in mmol/L. But a glucose measurement means nothing without knowing
*when it was taken*, and WHO thresholds for hyperglycaemia in pregnancy differ
sharply by sampling condition:

| Reading | GDM range | Diabetes in pregnancy |
|---|---|---|
| Fasting plasma glucose | 5.1–6.9 mmol/L | ≥ 7.0 mmol/L |
| 2-hour 75 g OGTT | 8.5–11.0 mmol/L | ≥ 11.1 mmol/L |

The observed range is roughly 6–19 mmol/L, with a median around 8.

* **Read as fasting values**, essentially the entire cohort is diabetic — over
  80% of rows clear the ≥ 7.0 threshold. For a community *screening* population
  that is not credible.
* **Read as 2-hour post-load values**, the distribution is clinically ordinary:
  most mothers normal, a meaningful minority in the GDM band, a smaller group in
  the diabetes range.

This project therefore assumes the **OGTT reading**, sets
`guidelines.BS_INTERPRETATION = "ogtt_2h"`, and — because an assumption that
changes the answer should never be invisible — reports both readings side by
side in every run:

```bash
python scripts/train.py          # writes artifacts/bs_interpretation_sensitivity.csv
```

The assumption is not free. Switching to the fasting reading moves
guideline–dataset agreement from roughly 64% to 31%, and the share of mothers
the guidelines would escalate from about 41% to 82%. A single undocumented
column drives the clinical baseline more than any modelling choice in the
project. That is a finding about the dataset, and it belongs in the write-up
rather than in a footnote.

## Two ranges that do not survive contact with obstetrics

**`Age` up to 70.** Pregnancy at 70 does not occur. Either the column is not
maternal age, or those rows are data-entry errors. `data.PLAUSIBLE` caps age at
60 and repairs anything beyond it, which affects a handful of rows.

**`HeartRate` down to 7.** A resting heart rate of 7 bpm is not a bradycardic
patient, it is a typo — most likely a dropped digit. These are repaired rather
than dropped, so the row's other five usable vitals survive.

**`DiastolicBP` never reaches 110** and `HeartRate` never exceeds 90. Both
matter: the severe-diastolic-hypertension and tachycardia criteria therefore
*cannot fire* on this dataset. They are carried in the rule set anyway, and
`calibration_report` lists them as silent, because a criterion that cannot fire
is not evidence of anything — and quietly dropping it would hide a limitation of
the data rather than the rule.

## Why hand-authored Urdu, not translation

Numbers cross languages; clinical instructions do not. "Get checked today" and
"آج ہی معائنہ کروائیں" carry the same content, but a machine translation of a
clinical sentence can invert urgency, name a condition the tool has no business
naming, or land in a register that reads as either alarming or dismissive.

All Urdu strings in `localization.py` are written by hand and checked by
automated lint rules (`tests/test_localization.py`) against three commitments:
no catastrophising vocabulary, no diagnosis claims, and a named action with a
timeframe in every message. They are nonetheless marked `UNREVIEWED` — see
[ETHICS.md](ETHICS.md).
