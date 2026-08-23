"""Urdu/English risk communication.

A risk score that a Lady Health Worker cannot deliver to a family is not a
finished tool. This module turns a model output into something sayable, and it
is deliberately *not* machine translation: the Urdu here is hand-authored,
because a mistranslated clinical instruction is a safety incident, and because
the register matters as much as the vocabulary.

Three design commitments, each enforced by ``lint_templates`` rather than left
to good intentions:

1. **No catastrophising.** Urgency is carried by the recommended action and its
   timeframe ("today", "within a few days"), never by frightening adjectives. A
   family that panics may go to the wrong place, or nowhere.
2. **No diagnosis.** The tool reports what the vitals suggest and who to see. It
   does not name a condition; it is not licensed to, and a wrong disease name
   spreads faster than a correct referral.
3. **Family-inclusive framing at the high band.** Where care decisions are made
   collectively, a message addressed only to the patient can stall. High-risk
   messages therefore name the family as part of the plan.

Every template carries a ``review`` status. They ship as ``UNREVIEWED``: the
strings are plain, checked Urdu, but "plausible Urdu written by the project
author" is not "signed off by a native-speaker reviewer and a clinician". The
distinction is recorded in the artifact rather than glossed over, and
``review_summary`` reports it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from . import config as C

Language = Literal["ur", "en"]
ReviewStatus = Literal["UNREVIEWED", "LANGUAGE_REVIEWED", "CLINICALLY_REVIEWED"]


# ---------------------------------------------------------------------------
# Shared strings
# ---------------------------------------------------------------------------
DISCLAIMER = {
    "ur": "یہ تشخیص نہیں ہے۔ یہ صرف ایک احتیاطی مشورہ ہے تاکہ آپ درست جگہ پر مشورہ لے سکیں۔",
    "en": "This is not a diagnosis. It is a precautionary suggestion so you can seek advice in the right place.",
}

# Named so the front-end cannot accidentally show a risk band without it.
NOT_A_DEVICE = {
    "ur": "یہ ایک تحقیقی نمونہ ہے، طبی آلہ نہیں۔",
    "en": "This is a research prototype, not a medical device.",
}


@dataclass(frozen=True)
class Template:
    """One risk band's message, in both languages."""

    band: str
    opening: dict[str, str]        # neutral framing of the result
    action: dict[str, str]         # what to do
    timeframe: dict[str, str]      # how soon
    reassurance: dict[str, str]    # closing, non-alarming
    family: dict[str, str] | None  # collective-decision framing (high band)
    review: ReviewStatus = "UNREVIEWED"
    reviewer_notes: str = ""


# ---------------------------------------------------------------------------
# Band templates
# ---------------------------------------------------------------------------
TEMPLATES: dict[str, Template] = {
    "low risk": Template(
        band="low risk",
        opening={
            "ur": "اس وقت آپ کی جو علامات درج کی گئی ہیں، وہ معمول کے قریب لگ رہی ہیں۔",
            "en": "The measurements recorded for you right now appear close to normal.",
        },
        action={
            "ur": "معمول کے مطابق حمل کا اپنا اگلا چیک اپ وقت پر کروا لیں۔",
            "en": "Continue with your next routine antenatal check-up as scheduled.",
        },
        timeframe={
            "ur": "معمول کے وقت پر۔",
            "en": "At the usual scheduled time.",
        },
        reassurance={
            "ur": "اگر کوئی نئی تکلیف محسوس ہو — مثلاً سر درد، نظر میں دھندلاہٹ، یا پیٹ میں درد — تو انتظار نہ کریں، رابطہ کریں۔",
            "en": "If anything new comes up - headache, blurred vision, or stomach pain - do not wait; get in touch.",
        },
        family=None,
    ),
    "mid risk": Template(
        band="mid risk",
        opening={
            "ur": "آپ کی چند علامات ایسی ہیں جن پر نظر رکھنے کی ضرورت ہے۔",
            "en": "A few of your measurements need to be kept an eye on.",
        },
        action={
            "ur": "قریبی مرکزِ صحت یا لیڈی ہیلتھ ورکر سے دوبارہ معائنہ کروا لیں تاکہ یہ علامات دیکھی جا سکیں۔",
            "en": "Arrange a follow-up check at your nearest health centre or with your Lady Health Worker so these can be reviewed.",
        },
        timeframe={
            "ur": "اگلے چند دن کے اندر۔",
            "en": "Within the next few days.",
        },
        reassurance={
            "ur": "اس کا مطلب یہ نہیں کہ کوئی بیماری ہے۔ وقت پر دیکھ لینے سے اکثر معاملات آسانی سے سنبھل جاتے ہیں۔",
            "en": "This does not mean something is wrong. Checked in time, most such things are managed easily.",
        },
        family=None,
    ),
    "high risk": Template(
        band="high risk",
        opening={
            "ur": "آپ کی کچھ علامات ایسی ہیں جن پر جلد توجہ دینا بہتر ہوگا۔",
            "en": "Some of your measurements would be better looked at soon.",
        },
        action={
            "ur": "براہِ کرم قریبی ہسپتال یا مرکزِ صحت پر کسی ڈاکٹر یا دائی سے معائنہ کروائیں۔",
            "en": "Please get checked by a doctor or midwife at your nearest hospital or health centre.",
        },
        timeframe={
            "ur": "آج ہی، یا آج ممکن نہ ہو تو کل صبح۔",
            "en": "Today, or tomorrow morning if today is not possible.",
        },
        reassurance={
            "ur": "جلد معائنہ کروانے کا مقصد صرف احتیاط ہے تاکہ ضرورت پڑنے پر بروقت مدد مل سکے۔",
            "en": "Going early is simply a precaution, so that help is available in good time if it is needed.",
        },
        family={
            "ur": "اگر ممکن ہو تو گھر کے کسی بڑے یا شوہر کو ساتھ لے جائیں، تاکہ فیصلہ مل کر کیا جا سکے اور سفر میں آسانی ہو۔",
            "en": "If possible, take a senior family member or your husband along, so the decision can be made together and travel is easier.",
        },
    ),
}


# ---------------------------------------------------------------------------
# Per-feature driver phrases
# ---------------------------------------------------------------------------
# Keyed by feature, then by direction. Phrased as observations ("higher than
# usual"), not verdicts ("you have hypertension") -- see commitment 2 above.
DRIVER_PHRASES: dict[str, dict[str, dict[str, str]]] = {
    "SystolicBP": {
        "high": {"ur": "خون کا دباؤ معمول سے کچھ زیادہ ہے",
                 "en": "blood pressure is somewhat higher than usual"},
        "low": {"ur": "خون کا دباؤ معمول سے کم ہے",
                "en": "blood pressure is lower than usual"},
    },
    "DiastolicBP": {
        "high": {"ur": "خون کے دباؤ کا نچلا درجہ بڑھا ہوا ہے",
                 "en": "the lower blood-pressure reading is raised"},
        "low": {"ur": "خون کے دباؤ کا نچلا درجہ کم ہے",
                "en": "the lower blood-pressure reading is low"},
    },
    "BS": {
        "high": {"ur": "خون میں شوگر کی مقدار زیادہ ہے",
                 "en": "blood sugar is high"},
        "low": {"ur": "خون میں شوگر کی مقدار کم ہے",
                "en": "blood sugar is low"},
    },
    "BodyTemp": {
        "high": {"ur": "جسم کا درجۂ حرارت بڑھا ہوا ہے (بخار)",
                 "en": "body temperature is raised (fever)"},
        "low": {"ur": "جسم کا درجۂ حرارت معمول سے کم ہے",
                "en": "body temperature is below normal"},
    },
    "HeartRate": {
        "high": {"ur": "دل کی دھڑکن معمول سے تیز ہے",
                 "en": "the heartbeat is faster than usual"},
        "low": {"ur": "دل کی دھڑکن معمول سے سست ہے",
                "en": "the heartbeat is slower than usual"},
    },
    "Age": {
        "high": {"ur": "عمر کے باعث حمل میں کچھ زیادہ احتیاط کی ضرورت ہوتی ہے",
                 "en": "age means a little extra care is advisable in pregnancy"},
        "low": {"ur": "کم عمری کے باعث حمل میں زیادہ احتیاط کی ضرورت ہوتی ہے",
                "en": "younger age means extra care is advisable in pregnancy"},
    },
}

FEATURE_NAMES = {
    "Age": {"ur": "عمر", "en": "Age"},
    "SystolicBP": {"ur": "خون کا دباؤ (اوپر)", "en": "Blood pressure (systolic)"},
    "DiastolicBP": {"ur": "خون کا دباؤ (نیچے)", "en": "Blood pressure (diastolic)"},
    "BS": {"ur": "خون میں شوگر", "en": "Blood sugar"},
    "BodyTemp": {"ur": "درجۂ حرارت", "en": "Body temperature"},
    "HeartRate": {"ur": "دل کی دھڑکن", "en": "Heart rate"},
}

BAND_NAMES = {
    "low risk": {"ur": "کم خطرہ", "en": "Low risk"},
    "mid risk": {"ur": "درمیانہ خطرہ", "en": "Medium risk"},
    "high risk": {"ur": "زیادہ توجہ درکار", "en": "Needs closer attention"},
}

_CONNECTORS = {
    "ur": {"noticed": "جو بات سامنے آئی:", "and": " اور "},
    "en": {"noticed": "What stood out:", "and": " and "},
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
@dataclass
class RiskMessage:
    band: str
    language: Language
    band_name: str
    text: str
    lines: list[str] = field(default_factory=list)
    drivers_used: list[str] = field(default_factory=list)
    review: ReviewStatus = "UNREVIEWED"


def render(
    band: str,
    drivers: list[str] | None = None,
    values: dict[str, float] | None = None,
    language: Language = "ur",
    max_drivers: int = 2,
) -> RiskMessage:
    """Compose the message for one result.

    ``drivers`` are feature names from the model explanation, strongest first;
    at most ``max_drivers`` are mentioned. Two is a deliberate cap -- a spoken
    message listing six vitals is one nobody remembers, and the point is that
    the mother acts on it.
    """
    if band not in TEMPLATES:
        raise KeyError(f"Unknown risk band {band!r}; expected one of {list(TEMPLATES)}")
    if language not in ("ur", "en"):
        raise ValueError(f"language must be 'ur' or 'en', got {language!r}")

    tpl = TEMPLATES[band]
    conn = _CONNECTORS[language]
    lines = [tpl.opening[language]]
    used: list[str] = []

    # Driver phrases are withheld at the low band. Telling a mother her result is
    # normal and then reciting which vitals "stood out" plants worry the result
    # does not justify, and worry is what makes a family ignore the next message.
    mentionable = [] if band == "low risk" else (drivers or [])

    phrases = []
    for feature in mentionable[:max_drivers]:
        if feature not in DRIVER_PHRASES:
            continue
        direction = _direction_for(feature, (values or {}).get(feature))
        phrases.append(DRIVER_PHRASES[feature][direction][language])
        used.append(feature)
    if phrases:
        lines.append(f"{conn['noticed']} " + conn["and"].join(phrases) + "۔"
                     if language == "ur" else
                     f"{conn['noticed']} " + conn["and"].join(phrases) + ".")

    lines.append(f"{tpl.action[language]} ({tpl.timeframe[language]})")
    if tpl.family:
        lines.append(tpl.family[language])
    lines.append(tpl.reassurance[language])
    lines.append(DISCLAIMER[language])

    return RiskMessage(
        band=band,
        language=language,
        band_name=BAND_NAMES[band][language],
        text=" ".join(lines),
        lines=lines,
        drivers_used=used,
        review=tpl.review,
    )


def _direction_for(feature: str, value: float | None) -> str:
    """Whether a value sits above or below the usual range for that vital.

    Uses simple reference midpoints rather than the model, so the wording stays
    interpretable even when the attribution method changes underneath it.
    """
    midpoints = {"Age": 27, "SystolicBP": 120, "DiastolicBP": 80,
                 "BS": 8.0, "BodyTemp": 98.6, "HeartRate": 80}
    if value is None:
        return "high"
    return "high" if float(value) >= midpoints.get(feature, 0) else "low"


def render_both(
    band: str, drivers: list[str] | None = None, values: dict[str, float] | None = None
) -> dict[str, RiskMessage]:
    """Urdu and English side by side, for the bilingual interface."""
    return {lang: render(band, drivers, values, language=lang) for lang in ("ur", "en")}


# ---------------------------------------------------------------------------
# Template safety lint
# ---------------------------------------------------------------------------
# Catastrophising vocabulary. Banned because urgency belongs in the action and
# its timeframe, not in adjectives that frighten a family into paralysis.
BANNED_TERMS = {
    "ur": ["خطرناک", "جان لیوا", "موت", "مہلک", "ہلاکت", "تشویشناک"],
    "en": ["dangerous", "fatal", "deadly", "life-threatening", "death", "will die",
           "critical condition", "severe risk"],
}

# Condition names the tool must never assert, since it cannot diagnose.
BANNED_DIAGNOSES = {
    "ur": ["پری ایکلیمپسیا", "ایکلیمپسیا", "ذیابیطس", "شوگر کی بیماری"],
    "en": ["pre-eclampsia", "eclampsia", "diabetes", "gestational diabetes",
           "hypertension", "infection", "sepsis"],
}

# Every message must tell the reader what to do. Checked by looking for an
# action template that is non-empty and mentions a place or a person to see.
REQUIRED_ACTION_HINTS = {
    "ur": ["مرکزِ صحت", "ہسپتال", "ڈاکٹر", "دائی", "چیک اپ", "لیڈی ہیلتھ ورکر", "معائنہ"],
    "en": ["health centre", "hospital", "doctor", "midwife", "check-up",
           "Lady Health Worker", "checked"],
}


@dataclass
class LintFinding:
    band: str
    language: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.band}/{self.language}] {self.rule}: {self.detail}"


def lint_templates(max_chars: int = 700) -> list[LintFinding]:
    """Check every template against the three design commitments.

    Run in CI (``tests/test_localization.py``). The point of encoding these as
    tests is that a well-meaning later edit -- adding "this is dangerous" to make
    the high band feel more urgent -- fails the build instead of shipping.
    """
    findings: list[LintFinding] = []

    for band, tpl in TEMPLATES.items():
        for lang in ("ur", "en"):
            msg = render(band, drivers=["SystolicBP", "BS"],
                         values={"SystolicBP": 150, "BS": 12.0}, language=lang)
            text = msg.text
            low = text.lower()

            for term in BANNED_TERMS[lang]:
                if term.lower() in low:
                    findings.append(LintFinding(band, lang, "catastrophising", f"contains {term!r}"))

            for term in BANNED_DIAGNOSES[lang]:
                if term.lower() in low:
                    findings.append(LintFinding(band, lang, "diagnosis-claim", f"names condition {term!r}"))

            if not any(h.lower() in low for h in REQUIRED_ACTION_HINTS[lang]):
                findings.append(LintFinding(band, lang, "no-actionable-step",
                                            "no place or person to consult is named"))

            if DISCLAIMER[lang] not in text:
                findings.append(LintFinding(band, lang, "missing-disclaimer",
                                            "the not-a-diagnosis line is absent"))

            if len(text) > max_chars:
                findings.append(LintFinding(band, lang, "too-long",
                                            f"{len(text)} chars > {max_chars}"))

            if not tpl.timeframe[lang].strip():
                findings.append(LintFinding(band, lang, "missing-timeframe",
                                            "urgency has nowhere to live but adjectives"))

        if band == "high risk" and not tpl.family:
            findings.append(LintFinding(band, "-", "missing-family-framing",
                                        "high band must address the family"))

    # Driver phrases must exist in both languages and both directions.
    for feature, directions in DRIVER_PHRASES.items():
        for direction, langs in directions.items():
            for lang in ("ur", "en"):
                if not langs.get(lang, "").strip():
                    findings.append(LintFinding(feature, lang, "missing-driver-phrase",
                                                f"{direction} direction is empty"))

    return findings


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------
# Decimal places appropriate to how each vital is actually measured. Blood
# pressure and pulse are read as whole numbers; glucose and temperature to one
# decimal. Anything more is false precision.
VITAL_PRECISION = {
    "Age": 0, "SystolicBP": 0, "DiastolicBP": 0, "HeartRate": 0,
    "BS": 1, "BodyTemp": 1,
}


def format_vital(feature: str, value: float) -> str:
    """Render one measurement the way a clinic form would.

    Exists because float steppers drift: a glucose input nudged through a UI
    arrives as 14.5216, and showing that back to a health worker next to a
    clinical threshold undermines confidence in everything around it.
    """
    places = VITAL_PRECISION.get(feature, 1)
    return f"{round(float(value), places):.{places}f}"


def round_vitals(values: dict[str, float]) -> dict[str, float]:
    """Snap collected measurements to their measurable precision before scoring.

    Applied at the point of collection so the model never sees a precision the
    instrument cannot produce, and so the displayed value and the scored value
    are the same number.
    """
    return {
        k: round(float(v), VITAL_PRECISION.get(k, 1))
        for k, v in values.items()
    }


def review_summary() -> dict[str, str]:
    """Review status per band -- surfaced in the UI and in generated artifacts."""
    return {band: tpl.review for band, tpl in TEMPLATES.items()}


def unreviewed_bands() -> list[str]:
    return [b for b, s in review_summary().items() if s != "CLINICALLY_REVIEWED"]
