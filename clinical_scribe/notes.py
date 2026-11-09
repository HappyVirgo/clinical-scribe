"""SOAP note generation from a (redacted) visit transcript.

Two providers share the :class:`NoteProvider` interface:

* :class:`RuleBasedNoteProvider` - offline and deterministic. Each sentence is assigned to
  Subjective / Objective / Assessment / Plan (or dropped as small talk) by weighted lexical
  cues plus the speaker's role.
* :class:`ClaudeNoteProvider` - asks Claude for a structured :class:`SOAPNote` using
  structured outputs. It must only ever be given *redacted* text.

Output is a draft for research and demonstration only. NOT FOR CLINICAL USE.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from clinical_scribe.text import Role, Sentence, parse_transcript

if TYPE_CHECKING:
    import anthropic

CLAUDE_MODEL = "claude-opus-5-5"


class Section(StrEnum):
    """Gold/predicted label for one transcript sentence."""

    SUBJECTIVE = "S"
    OBJECTIVE = "O"
    ASSESSMENT = "A"
    PLAN = "P"
    OTHER = "X"


class Medication(BaseModel):
    """A medication mentioned in the plan."""

    name: str = Field(description="Generic or brand drug name, lower case.")
    dose: str | None = Field(default=None, description="Strength, e.g. '10 mg'.")
    frequency: str | None = Field(default=None, description="e.g. 'daily', 'twice a day'.")


class SOAPNote(BaseModel):
    """A structured SOAP note. Each section is a list of short statements."""

    subjective: list[str] = Field(description="Symptoms, history, patient-reported information.")
    objective: list[str] = Field(description="Vitals, examination findings, labs and imaging.")
    assessment: list[str] = Field(description="Clinician impressions and diagnoses.")
    plan: list[str] = Field(description="Orders, medications, counselling, referrals.")
    medications: list[Medication] = Field(description="Medications started or continued.")
    follow_up: str | None = Field(description="When and why the patient should return.")


class NoteGenerationError(RuntimeError):
    """Raised when a provider cannot produce a note."""


class NoteProvider(Protocol):
    """Anything that turns a redacted transcript into a :class:`SOAPNote`."""

    name: str

    def generate(self, redacted_transcript: str) -> SOAPNote:
        """Build a SOAP note from an already de-identified transcript."""
        ...


def _cues(*patterns: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(patterns) + r")", re.I)


# Each section has strong (weight 2) and weak (weight 1) cue sets.
_CUES: dict[Section, tuple[re.Pattern[str], re.Pattern[str]]] = {
    Section.ASSESSMENT: (
        _cues(
            r"impression",
            r"assessment",
            r"diagnos[ie]s",
            r"most likely",
            r"consistent with",
            r"i think (?:this|it|you|that|we're|what)",
            r"i suspect",
            r"i believe",
            r"(?:well|poorly|not well|under) controlled",
            r"uncontrolled",
            r"differential",
            r"(?:this|it|that) (?:is|looks like|sounds like) (?:a |an )?(?:likely|probably|classic|typical)",
            r"looks like (?:you have|a |an )",
            r"sounds like (?:you have|a |an )",
            r"suggestive of",
            r"meets criteria",
            r"you have (?:a |an )?(?:mild|moderate|severe|acute|chronic)",
        ),
        _cues(
            r"likely",
            r"probably",
            r"secondary to",
            r"due to",
            r"suggests?",
            r"(?:it|this)'s a",
            # Copular statements opening a clinician sentence ("This is shingles.") are
            # usually impressions; objective/plan cues outweigh this when present.
            r"^(?:this|these|that|it) (?:is|are|looks like|sounds like)\b",
            r"^(?:you|she|he|they) (?:have|has)\b",
            r"looks like",
        ),
    ),
    Section.PLAN: (
        _cues(
            r"i(?:'ll| will| am going to|'m going to|'d like to) (?:order|prescribe|start|send|refer|"
            r"put|check|give|schedule|increase|decrease|recommend|switch|add|get|have|draw|write)",
            r"we(?:'ll| will| should| can| are going to|'re going to) ",
            r"let's",
            r"prescri",
            r"referr?al",
            r"refer you",
            r"follow[- ]?up",
            r"come back",
            r"return (?:in|if|to)",
            r"see you (?:in|back)",
            r"recheck",
            r"\d+(?:\.\d+)? ?(?:mg|mcg|units|ml)\b",
            r"once (?:a|per) day",
            r"twice (?:a|per) day",
            r"(?:three|four) times a day",
            r"as needed",
            r"every \d+ hours",
            r"at bedtime",
            r"please ",
            # Imperative mood at the start of a clinician sentence is advice or an order.
            r"^(?:take|use|apply|drink|raise|elevate|avoid|keep|stop|start|call|try|rest|ice|"
            r"limit|aim|strain|stay|go|return|continue|switch|cut|eat|walk|wear|get)\b",
        ),
        _cues(
            r"start",
            r"continue",
            r"increase",
            r"decrease",
            r"stop\b",
            r"take\b",
            r"daily",
            r"recommend",
            r"avoid",
            r"try\b",
            r"make sure",
            r"call (?:us|me|the office)",
            r"emergency",
            r"vaccine",
            r"shots?\b",
            r"physical therapy",
            r"rest\b",
            r"ice\b",
            r"hydrat",
            r"order",
            r"schedule",
            r"plan\b",
            r"should",
            r"keep",
            r"would help",
            r"weight loss",
            r"losing (?:some )?weight",
            r"fluids",
        ),
    ),
    Section.OBJECTIVE: (
        _cues(
            r"blood pressure",
            r"bp\b",
            r"heart rate",
            r"pulse",
            r"temperature",
            r"afebrile",
            r"oxygen",
            r"o2",
            r"sat(?:uration|s)\b",
            r"spo2",
            r"respiratory rate",
            r"bmi",
            r"weigh(?:t|s)",
            r"on exam",
            r"exam(?:ination)? (?:shows|reveals|is)",
            r"auscultat",
            r"palpat",
            r"lungs?",
            r"heart sounds",
            r"murmur",
            r"tender(?:ness)?",
            r"erythema",
            r"range of motion",
            r"reflexes",
            r"a1c",
            r"hemoglobin",
            r"cholesterol",
            r"ldl",
            r"tsh",
            r"creatinine",
            r"glucose",
            r"x-?ray",
            r"ultrasound",
            r"mri",
            r"ct scan",
            r"ecg",
            r"ekg",
            r"urinalysis",
            r"culture",
            r"came back",
            r"results?",
            r"labs?",
            r"\d{2,3}/\d{2,3}",
            r"\d+(?:\.\d+)? ?(?:%|bpm|mg/dl|degrees)",
            r"rapid (?:strep|flu|covid)",
            r"potassium",
            r"sodium",
            r"egfr",
            r"uric acid",
            r"vitamin d level",
            r"ferritin",
            r"psa\b",
            r"white (?:blood )?count",
            r"platelets",
            r"(?:bone density|ct|pet) scan",
            r"score (?:today )?(?:is|was)",
            r"gait",
        ),
        _cues(
            r"i (?:see|hear|feel|notice)",
            r"appears?",
            r"swollen",
            r"swelling",
            r"clear",
            r"edema",
            r"pupils",
            r"tympanic",
            r"abdomen",
            r"soft",
            r"positive",
            r"negative",
            r"normal",
            r"strength",
            r"intact",
            r"test",
            r"level",
            r"looks? (?:red|inflamed|normal)",
            r"^there (?:is|are)\b",
            r"no (?:sign|evidence) of",
            r"scan",
            r"(?:is|are|was) \d",
        ),
    ),
    Section.SUBJECTIVE: (
        _cues(
            r"you (?:said|mentioned|told me|reported|describe)",
            r"history of",
            r"family history",
            r"started",
            r"for the (?:past|last)",
            r"since",
            r"denies",
            r"allerg",
        ),
        _cues(
            r"pain",
            r"hurts?",
            r"days?\b",
            r"weeks?\b",
            r"months?\b",
            r"feel(?:ing|s)?",
            r"symptoms?",
            r"cough",
            r"fever",
            r"nause",
            r"tired",
            r"worse",
            r"better",
            r"smok",
            r"drink",
        ),
    ),
}

_ACK = re.compile(
    r"^(?:ok(?:ay)?|yes|yeah|no|nope|sure|thanks?|thank you|sounds good|great|alright|all right|"
    r"got it|will do|perfect|i understand|that makes sense|(?:good )?(?:morning|afternoon)|hi|hello|bye)"
    r"\b[\s\w,'!.-]{0,30}$",
    re.I,
)
_TIE_ORDER = (Section.ASSESSMENT, Section.PLAN, Section.OBJECTIVE, Section.SUBJECTIVE)


def _scores(text: str) -> dict[Section, int]:
    return {
        section: 2 * len(strong.findall(text)) + len(weak.findall(text)) for section, (strong, weak) in _CUES.items()
    }


def classify_sentence(sentence: Sentence) -> Section:
    """Assign one sentence to a SOAP section, or ``Section.OTHER`` for small talk.

    Patients' statements default to Subjective. Clinician questions are treated as
    conversational (the answer carries the content); other clinician statements go to the
    highest-scoring section, ties broken A > P > O > S.
    """
    text = sentence.text.strip()
    if not text or _ACK.match(text):
        return Section.OTHER
    if sentence.role is Role.PATIENT:
        return Section.OTHER if text.endswith("?") else Section.SUBJECTIVE
    if text.endswith("?"):
        return Section.OTHER
    scores = _scores(text)
    best = max(_TIE_ORDER, key=lambda s: scores[s])
    if scores[best] == 0:
        return Section.OTHER
    return best


_DRUGS = frozenset(
    """
    acetaminophen albuterol allopurinol amlodipine amoxicillin amoxicillin-clavulanate atorvastatin
    azithromycin benzonatate budesonide bupropion cephalexin cetirizine ciprofloxacin citalopram
    clopidogrel colchicine cyclobenzaprine diclofenac doxycycline duloxetine escitalopram
    famotidine fluconazole fluoxetine fluticasone furosemide gabapentin glipizide
    hydrochlorothiazide ibuprofen insulin ipratropium levothyroxine lisinopril loratadine losartan
    meloxicam metformin methylprednisolone metoprolol metronidazole montelukast naproxen
    nitrofurantoin omeprazole ondansetron pantoprazole prednisone propranolol rosuvastatin
    sertraline simvastatin spironolactone sumatriptan tamsulosin trazodone triamcinolone
    valacyclovir vitamin warfarin apixaban empagliflozin semaglutide hydrocortisone mupirocin
    alendronate tiotropium oseltamivir prednisolone nitroglycerin
    """.split()
)
_DOSE = re.compile(r"^\s*(\d+(?:\.\d+)?\s?(?:mg|mcg|g|units|ml|%|puffs?))", re.I)
_FREQUENCY = re.compile(
    r"\b(once (?:a|per) day|twice (?:a|per) day|(?:three|four) times (?:a|per) day|daily|"
    r"nightly|at bedtime|every \d+(?: to \d+)? hours|as needed|weekly|bid|tid|qid|prn)\b",
    re.I,
)
_FOLLOW_UP = re.compile(
    r"\b(?:follow[- ]?up|come back|return|see you(?: back)?|recheck)\b.*?"
    r"\b(?:in|within|after)\s+(?:\w+\s+)?(?:days?|weeks?|months?|years?)\b",
    re.I,
)


def extract_medications(plan: list[str]) -> list[Medication]:
    """Pull drug names with optional dose and frequency out of plan sentences."""
    meds: dict[str, Medication] = {}
    for sentence in plan:
        for word in re.finditer(r"[A-Za-z][A-Za-z-]+", sentence):
            name = word.group().lower()
            if name not in _DRUGS or name in meds:
                continue
            tail = sentence[word.end() :]
            dose = _DOSE.match(tail)
            freq = _FREQUENCY.search(tail)
            meds[name] = Medication(
                name=name,
                dose=dose.group(1) if dose else None,
                frequency=freq.group(1).lower() if freq else None,
            )
    return list(meds.values())


def extract_follow_up(plan: list[str]) -> str | None:
    """Return the first plan sentence that schedules a return visit."""
    return next((s for s in plan if _FOLLOW_UP.search(s)), None)


class RuleBasedNoteProvider:
    """Offline, deterministic SOAP drafting by cue-based sentence classification."""

    name = "rules"

    def generate(self, redacted_transcript: str) -> SOAPNote:
        """Classify each sentence and assemble the note sections in transcript order."""
        buckets: dict[Section, list[str]] = {s: [] for s in Section}
        for sentence in parse_transcript(redacted_transcript):
            buckets[classify_sentence(sentence)].append(sentence.text)
        plan = buckets[Section.PLAN]
        return SOAPNote(
            subjective=buckets[Section.SUBJECTIVE],
            objective=buckets[Section.OBJECTIVE],
            assessment=buckets[Section.ASSESSMENT],
            plan=plan,
            medications=extract_medications(plan),
            follow_up=extract_follow_up(plan),
        )


_SYSTEM_PROMPT = """You are a clinical documentation assistant drafting a SOAP note from a \
doctor-patient visit transcript. The transcript has been de-identified: placeholders such as \
[NAME], [DATE:2091] or [MRN] replace identifiers. Keep placeholders as they are and never try to \
infer the original values. Only document what was said in the visit; do not invent findings, \
doses or diagnoses. Write concise clinical statements. Subjective: symptoms and history reported \
by the patient. Objective: vitals, exam, labs, imaging. Assessment: the clinician's impressions. \
Plan: orders, medications, counselling, referrals. List every medication started or continued, \
and the follow-up interval if one was given. This draft is for review by a clinician."""


class ClaudeNoteProvider:
    """SOAP drafting with Claude structured outputs. Only pass redacted transcripts."""

    name = "claude"

    def __init__(self, client: anthropic.Anthropic | None = None, model: str = CLAUDE_MODEL) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def generate(self, redacted_transcript: str) -> SOAPNote:
        """Request a :class:`SOAPNote` from Claude; raise NoteGenerationError on failure."""
        import anthropic

        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=16000,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"<transcript>\n{redacted_transcript}\n</transcript>"}],
                output_format=SOAPNote,
                output_config={"effort": "medium"},
                extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
                extra_body={"fallbacks": "default"},
            )
        except anthropic.RateLimitError as exc:
            raise NoteGenerationError("Claude rate limit reached; retry later.") from exc
        except anthropic.APIStatusError as exc:
            raise NoteGenerationError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise NoteGenerationError("Could not reach the Claude API.") from exc

        if response.stop_reason == "refusal":
            raise NoteGenerationError("Claude declined to draft a note for this transcript.")
        if response.stop_reason == "max_tokens":
            raise NoteGenerationError("Claude's response was truncated before the note was complete.")
        if response.parsed_output is None:
            raise NoteGenerationError("Claude returned no structured note.")
        return response.parsed_output
