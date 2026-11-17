"""Labelled synthetic datasets for classifiers (intent routing, moderation...) from a grammar.

A *spec* lists, per label, templates with ``{slots}`` and the context each one is said in::

    {
      "slots": {"project": ["Atlas", "Nimbus"], "time": ["a las ocho", "mañana"]},
      "labels": {
        "project": [{"t": "arregla el test que falla en {project}"},
                    {"t": "sigue", "context": ["active_project"]}],
        "general": [{"t": "pon una alarma {time}"},
                    {"t": "sigue", "context": ["recent_general"]}]
      }
    }

Generation fills the slots at random, applies light *noise* the way people type in a chat
(lower-casing, dropped accents and punctuation, a filler prefix such as "oye", a typo), and drops
near-duplicates (character-trigram Jaccard above ``dedup_threshold``). Every sample records the
template it came from, so :func:`split` can hold out **whole templates** for evaluation: the
evaluation set then measures whether a model generalizes to phrasings it never saw, not whether
it memorized the templates.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

_SLOT = re.compile(r"\{(\w+)\}")
FILLERS = ["oye ", "porfa ", "a ver, ", "venga, ", "eh ", "una cosa: "]


@dataclass(frozen=True)
class LabeledSample:
    text: str
    label: str
    context: List[str] = field(default_factory=list)
    template: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _trigrams(text: str) -> Set[str]:
    clean = re.sub(r"\s+", " ", text.lower())
    return {clean[i : i + 3] for i in range(max(1, len(clean) - 2))}


def _typo(text: str, rng: random.Random) -> str:
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) > 4]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(1, len(w) - 1)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]  # swap two letters
    return " ".join(words)


def noisy(text: str, rng: random.Random, rate: float) -> str:
    """Chat-style variations of ``text``; each applied with probability ``rate``."""
    if rng.random() < rate:
        text = text.lower()
    if rng.random() < rate:
        text = _strip_accents(text)
    if rng.random() < rate:
        text = re.sub(r"[¿¡?!.,]", "", text).strip()
    if rng.random() < rate / 2:
        text = rng.choice(FILLERS) + text[0].lower() + text[1:] if text else text
    if rng.random() < rate / 2:
        text = _typo(text, rng)
    return text


def fill(template: str, slots: Dict[str, Sequence[str]], rng: random.Random) -> str:
    def pick(m: "re.Match[str]") -> str:
        name = m.group(1)
        if name not in slots:
            raise KeyError(f"template uses unknown slot {{{name}}}: {template!r}")
        return rng.choice(list(slots[name]))

    return _SLOT.sub(pick, template)


def generate_labeled(
    spec: Dict[str, Any],
    n_per_label: int,
    seed: int = 0,
    noise: float = 0.3,
    dedup_threshold: float = 0.9,
    max_tries_factor: int = 20,
) -> List[LabeledSample]:
    """``n_per_label`` samples per label (fewer if the grammar cannot produce that many distinct
    ones), balanced across each label's templates."""
    rng = random.Random(seed)
    slots: Dict[str, Sequence[str]] = spec.get("slots", {})
    out: List[LabeledSample] = []
    seen: List[Set[str]] = []
    for label, templates in spec["labels"].items():
        produced = 0
        tries = 0
        while produced < n_per_label and tries < n_per_label * max_tries_factor:
            tries += 1
            entry = templates[(produced + tries) % len(templates)]
            text = noisy(fill(entry["t"], slots, rng), rng, noise)
            grams = _trigrams(text)
            if any(len(grams & g) / max(1, len(grams | g)) > dedup_threshold for g in seen):
                continue
            seen.append(grams)
            out.append(LabeledSample(text, label, list(entry.get("context", [])), entry["t"]))
            produced += 1
    rng.shuffle(out)
    return out


def split(
    samples: Sequence[LabeledSample], eval_fraction: float = 0.25, seed: int = 0, by_template: bool = True
) -> Tuple[List[LabeledSample], List[LabeledSample]]:
    """Train/eval split. With ``by_template`` whole templates go to one side (per label, at
    least one template stays in training)."""
    rng = random.Random(seed)
    if not by_template:
        items = list(samples)
        rng.shuffle(items)
        cut = int(len(items) * (1 - eval_fraction))
        return items[:cut], items[cut:]
    held: Set[Tuple[str, str]] = set()
    by_label: Dict[str, List[str]] = {}
    for s in samples:
        key = s.template + "|" + ",".join(s.context)
        if key not in by_label.setdefault(s.label, []):
            by_label[s.label].append(key)
    for label, keys in by_label.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        k = min(len(keys) - 1, max(1, round(len(keys) * eval_fraction))) if len(keys) > 1 else 0
        held.update((label, key) for key in keys[:k])
    train = [s for s in samples if (s.label, s.template + "|" + ",".join(s.context)) not in held]
    evaluation = [s for s in samples if (s.label, s.template + "|" + ",".join(s.context)) in held]
    return train, evaluation


def save_jsonl(samples: Sequence[LabeledSample], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(s.to_json() + "\n")


def load_jsonl(path: str) -> List[LabeledSample]:
    with open(path, encoding="utf-8") as fh:
        return [LabeledSample(**json.loads(line)) for line in fh if line.strip()]


def load_spec(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
