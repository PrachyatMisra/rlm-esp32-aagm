"""Corpus builder for the ESP32 edge track.

Two sources:
  * ``hf``        -> HuggingFace ``imdb`` (matches the main repo protocol).
  * ``synthetic`` -> Deterministic lexicon-bootstrapped movie-review corpus.

The synthetic fallback exists because edge/embedded labs are often offline.
It encodes exactly the phenomena the RLM recursion is designed for:
long-range contrast ("slow start but brilliant ending" -> positive),
negation ("not a bad film" -> positive), and mixed-aspect reviews.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

Label = int
Text = str


@dataclass
class Corpus:
    train_texts: List[Text]
    train_labels: List[Label]
    val_texts: List[Text]
    val_labels: List[Label]
    test_texts: List[Text]
    test_labels: List[Label]


# --------------------------------------------------------------------------
# HF route (primary repo protocol, used when the machine is online)
# --------------------------------------------------------------------------

def load_hf_imdb(seed: int, train: int, val: int, test: int) -> Corpus:
    from datasets import load_dataset

    ds = load_dataset("imdb")
    tr = ds["train"].shuffle(seed=seed).select(range(train + val))
    te = ds["test"].shuffle(seed=seed).select(range(test))
    return Corpus(
        list(tr["text"][:train]), list(tr["label"][:train]),
        list(tr["text"][train:]), list(tr["label"][train:]),
        list(te["text"]), list(te["label"]),
    )


# --------------------------------------------------------------------------
# Synthetic fallback (offline lab-safe)
# --------------------------------------------------------------------------

POS_STRONG = [
    "masterpiece", "brilliant", "outstanding", "superb", "phenomenal", "excellent",
    "captivating", "gripping", "remarkable", "wonderful", "exceptional", "stunning",
    "powerful", "moving", "flawless", "unforgettable", "dazzling", "gripping",
    "heartfelt", "ingenious", "mesmerizing", "magnificent", "sublime", "terrific",
    "fantastic", "amazing", "awesome", "incredible", "touching", "riveting",
    "compelling", "masterful", "glorious", "sensational", "electrifying",
]
POS_MILD = [
    "good", "enjoyable", "solid", "pleasant", "likable", "charming", "fun",
    "engaging", "entertaining", "impressive", "refreshing", "witty", "satisfying",
    "well-crafted", "watchable", "delightful", "smart", "worthwhile", "genuine", "neat",
    "decent", "amusing", "tender", "graceful", "spirited", "commendable",
]
NEG_STRONG = [
    "terrible", "atrocious", "dreadful", "abysmal", "unwatchable", "painful",
    "disaster", "horrendous", "appalling", "insufferable", "lifeless", "tedious",
    "excruciating", "hopeless", "wretched", "miserable", "catastrophic", "laughable",
    "soulless", "broken", "unbearable", "pathetic", "disgraceful", "vapid",
    "awful", "horrible", "worst", "cringeworthy", "incoherent", "lame",
]
NEG_MILD = [
    "boring", "dull", "predictable", "slow", "weak", "shallow", "clumsy",
    "forgettable", "mediocre", "generic", "flawed", "uneven", "disappointing",
    "confusing", "hollow", "tiresome", "bland", "overlong", "stale", "flat",
    "plodding", "sluggish", "messy", "monotonous", "grating", "bad",
]
FILLER_SUBJ = [
    "the film", "the movie", "the story", "the plot", "the screenplay", "the script",
    "the first act", "the second act", "the opening scene", "the climax", "the ending",
    "the dialogue", "the pacing", "the soundtrack", "the cinematography", "the visuals",
    "the editing", "the action scenes", "the romance", "the comedy", "the horror scenes",
    "the acting", "the direction", "the writing", "the performances", "the premise",
]
FILLER_ACTOR = [
    "the lead actor", "the supporting cast", "the director", "the villain",
    "the protagonist", "the ensemble", "the narrator", "the child actor",
]
POS_VERBS = ["loved", "adored", "enjoyed", "recommend", "applaud"]
NEG_VERBS = ["hated", "regret", "disliked", "endured", "dropped"]
INTENS = ["truly", "absolutely", "remarkably", "so", "very", "utterly", "genuinely", "really", ""]
OPENERS = [
    "i watched this last night", "saw it in theatres", "picked it on a whim",
    "a friend recommended it", "i had low expectations", "i had high hopes",
    "caught this on a flight", "finally got around to it", "watched it twice",
    "short review:", "honestly", "i usually love this genre",
]
FILM_WORDS = ["film", "movie", "picture", "feature", "flick", "production", "drama"]

LEXICON_PAIRS = [(w, 1) for w in POS_STRONG + POS_MILD] + \
                [(w, 0) for w in NEG_STRONG + NEG_MILD]

# Phrases whose sentiment flips under negation.
def _neg(w: str) -> str: return f"not {w}"
def _hardly(w: str) -> str: return f"hardly {w}"


def _one(rng: random.Random) -> Tuple[Text, Label]:
    """Render one review. Returns (text, label)."""
    kind = rng.random()
    intens = rng.choice(INTENS)
    opener = rng.choice(OPENERS) if rng.random() < 0.35 else ""
    film = rng.choice(FILM_WORDS)
    use_tail = rng.random() < 0.6          # tail clause is NOT always present

    if kind < 0.30:  # plain polar review
        sent = 1 if rng.random() < 0.5 else 0
        words = (POS_STRONG + POS_MILD) if sent else (NEG_STRONG + NEG_MILD)
        w1, w2 = rng.sample(words, 2)
        verb = rng.choice(POS_VERBS if sent else NEG_VERBS)
        s = rng.choice([
            f"{rng.choice(FILLER_SUBJ)} is {intens} {w1}".replace("  ", " "),
            f"this {film} is {intens} {w1} and {w2}".replace("  ", " "),
            f"{rng.choice(FILLER_ACTOR)} delivers {intens} {w1} work".replace("  ", " "),
            f"this {film} is {w1}",
            f"the {film} was {w1}",
            f"i {verb} this {film}" if verb != "recommend" else f"i recommend this {film}",
            f"a {w1} {film}",
        ])
        label = sent
        tail = f"overall a {w1} {film} i would {'recommend' if sent else 'avoid'}"
    elif kind < 0.52:  # contrast: BUT clause is decisive (recursion test)
        decisive_pos = rng.random() < 0.5
        if decisive_pos:
            w_weak = rng.choice(NEG_MILD + NEG_STRONG)
            w_win = rng.choice(POS_STRONG + POS_MILD)
            tmpl = rng.choice([
                f"{rng.choice(FILLER_SUBJ)} is {w_weak} but {rng.choice(FILLER_SUBJ)} is {intens} {w_win}",
                f"although the film feels {w_weak} at first, the {rng.choice(['ending', 'finale', 'payoff'])} is {w_win}",
                f"it starts {w_weak} yet ends {w_win}",
            ])
            label = 1
            tail = f"despite the {w_weak} parts, this {film} is {w_win}"
        else:
            w_weak = rng.choice(POS_MILD + POS_STRONG)
            w_win = rng.choice(NEG_STRONG + NEG_MILD)
            tmpl = rng.choice([
                f"{rng.choice(FILLER_SUBJ)} is {w_weak} but {rng.choice(FILLER_SUBJ)} is {intens} {w_win}",
                f"great {rng.choice(['premise', 'idea', 'cast'])} but the {rng.choice(['execution', 'script', 'result'])} is {w_win}",
                f"starts {w_weak} yet turns {w_win}",
            ])
            label = 0
            tail = f"for all the {w_weak} hype, the {film} is {w_win}"
        s = tmpl.replace("  ", " ")
    elif kind < 0.72:  # negation (flips polarity)
        if rng.random() < 0.5:
            w = rng.choice(POS_STRONG + POS_MILD)
            s = rng.choice([
                f"this {film} is {_neg(w)} at all".replace("  ", " "),
                f"i can {_hardly('call it')} {w}" if rng.random() < 0.5 else f"the {film} is {_hardly(w)}",
                f"{rng.choice(FILLER_SUBJ)} is {_neg(w)}".replace("  ", " "),
            ])
            label = 0
            tail = f"honestly not {w}"
        else:
            w = rng.choice(NEG_STRONG + NEG_MILD)
            s = rng.choice([
                f"this {film} is {_neg(w)} one".replace("  ", " "),
                f"it is {_neg('half as')} {w} as critics say",
                f"call it anything but {w}",
            ])
            label = 1
            tail = f"definitely not {w}"
    elif kind < 0.88:  # mixed-aspect, longer review (depth stress test)
        pw, nw = rng.choice(POS_MILD + POS_STRONG), rng.choice(NEG_MILD + NEG_STRONG)
        end_pos = rng.random() < 0.5
        label = int(end_pos)          # the FINAL clause is decisive
        first, last = (nw, pw) if end_pos else (pw, nw)
        parts = [
            f"{rng.choice(FILLER_SUBJ)} is {first}",
            f"meanwhile {rng.choice(FILLER_ACTOR)} {'shines' if end_pos else 'struggles'}",
            f"some scenes run {rng.choice(NEG_MILD)}",
            f"yet the {rng.choice(['third act', 'resolution', 'final scene'])} feels {last}",
        ]
        s = ", ".join(parts)
        tail = f"in the end this {film} is {last}"
    else:  # mild/mixed near-boundary
        sent = 1 if rng.random() < 0.5 else 0
        w = rng.choice(POS_MILD) if sent else rng.choice(NEG_MILD)
        w2 = rng.choice(NEG_MILD) if sent else rng.choice(POS_MILD)
        s = (f"{rng.choice(FILLER_SUBJ)} is {w}, though {rng.choice(FILLER_SUBJ)} is {w2}")
        label = sent
        tail = f"a {w} {film} overall"

    if opener and use_tail:
        text = f"{opener}, {s}. {tail}."
    elif opener:
        text = f"{opener}, {s}."
    elif use_tail:
        text = f"{s}. {tail}."
    else:
        text = f"{s}."
    return text.strip(), label


def build_synthetic(seed: int, n_train: int, n_val: int, n_test: int) -> Corpus:
    rng = random.Random(seed)
    xs: List[Tuple[Text, Label]] = []
    seen = set()
    while len(xs) < n_train + n_val + n_test:
        t, y = _one(rng)
        if t in seen:        # keep the split deduplicated
            continue
        seen.add(t)
        xs.append((t, y))
    texts = [t for t, _ in xs]
    labels = [y for _, y in xs]
    return Corpus(
        texts[:n_train], labels[:n_train],
        texts[n_train:n_train + n_val], labels[n_train:n_train + n_val],
        texts[n_train + n_val:], labels[n_train + n_val:],
    )


def build_corpus(source: str, seed: int, n_train: int, n_val: int, n_test: int) -> Corpus:
    if source == "hf":
        try:
            return load_hf_imdb(seed, n_train, n_val, n_test)
        except Exception as exc:  # offline lab -> deterministic fallback
            print(f"[corpus] HF imdb unavailable ({exc.__class__.__name__}); "
                  f"falling back to synthetic corpus")
            source = "synthetic"
    print(f"[corpus] using '{source}' corpus "
          f"({n_train}/{n_val}/{n_test} train/val/test)")
    return build_synthetic(seed, n_train, n_val, n_test)
