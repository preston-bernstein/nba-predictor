import itertools
import string
import pytest

from src.service.normalizer import (
    normalize_team,
    TeamNormalizeError,
    CANON,
    ALIASES,
)

# ---- Helpers ---------------------------------------------------------------

def variants(word: str):
    """Generate simple case variants to ensure case-insensitivity."""
    yield word.lower()
    yield word.upper()
    # Mixed case: alternate lower/upper
    yield "".join(ch.lower() if i % 2 else ch.upper() for i, ch in enumerate(word))


def with_noise(s: str):
    """Sprinkle benign punctuation and whitespace; normalizer should handle it."""
    punct = " -_.,!:/\\|—–"  # includes em/en dashes
    pieces = [p for p in s.split()]
    if not pieces:
        return [s]
    # Insert punctuation between tokens and pad ends
    combos = []
    for sep in [" ", "  ", "   ", punct[0], " " + punct[3] + " "]:
        combos.append(sep.join(pieces))
    # Pad with punctuation at ends
    results = []
    for base in combos:
        results.append(base)
        results.append(punct[:2] + base + punct[-2:])
        results.append("   " + base + "   ")
        results.append(punct[-1] + "  " + base + "  " + punct[0])
    # Dedup
    seen = set()
    out = []
    for r in results:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ---- Tests: Canonical codes ------------------------------------------------

@pytest.mark.parametrize("code", sorted(CANON.keys()))
def test_canon_codes_pass_through(code):
    assert normalize_team(code) == code
    # case-insensitive
    for v in variants(code):
        assert normalize_team(v) == code


# ---- Tests: Aliases map to existing canonical codes -----------------------

def test_all_aliases_target_valid_canon():
    # Every alias value must point to a known canonical code
    for alias, target in ALIASES.items():
        assert target in CANON, f"Alias '{alias}' points to unknown code '{target}'"

@pytest.mark.parametrize("alias,target", sorted(ALIASES.items()))
def test_aliases_normalize(alias, target):
    # base alias
    assert normalize_team(alias) == target
    # case tolerance
    for v in variants(alias):
        assert normalize_team(v) == target
    # punctuation/whitespace tolerance
    for noisy in with_noise(alias):
        assert normalize_team(noisy) == target


# ---- Tests: Representative real-world names/nicknames ---------------------

EXAMPLES = [
    ("New York Knicks", "NYK"),
    ("Knicks", "NYK"),
    ("BOSTON CELTICS", "BOS"),
    ("Celtics", "BOS"),
    ("Los Angeles Lakers", "LAL"),
    ("Lakers", "LAL"),
    ("Los   Angeles     Clippers", "LAC"),
    ("Clippers", "LAC"),
    ("Golden State", "GSW"),
    ("Warriors", "GSW"),
    ("Sixers", "PHI"),
    ("76ers", "PHI"),
    ("Blazers", "POR"),
    ("Trail   Blazers", "POR"),
    ("Wolves", "MIN"),
    ("Cavs", "CLE"),
    ("NOLA", "NOP"),
    ("OKLAHOMA CITY", "OKC"),
    ("San Antonio", "SAS"),
]

@pytest.mark.parametrize("raw,code", EXAMPLES)
def test_examples_normalize(raw, code):
    assert normalize_team(raw) == code
    for noisy in with_noise(raw):
        assert normalize_team(noisy) == code


# ---- Tests: Idempotence & closure -----------------------------------------

@pytest.mark.parametrize("input_str,expected", [
    *[(c, c) for c in CANON.keys()],
    *list(EXAMPLES),
    *[(k, v) for k, v in ALIASES.items()],
])
def test_idempotent_normalization(input_str, expected):
    # First normalization
    first = normalize_team(input_str)
    assert first == expected
    # Re-normalizing stays stable (idempotent)
    assert normalize_team(first) == expected


# ---- Tests: Bad inputs -----------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "???", "Gotham Rogues", "Seattle Supersonics"])
def test_unknown_team_raises(bad):
    with pytest.raises(TeamNormalizeError) as e:
        normalize_team(bad)
    assert "Unknown team" in str(e.value)

@pytest.mark.parametrize("bad", [None, 123, 3.14, object(), [], {}])
def test_non_string_inputs_raise(bad):
    with pytest.raises(TeamNormalizeError):
        normalize_team(bad)  # type: ignore[arg-type]


# ---- Optional: Property-based fuzz (skipped if Hypothesis not installed) ---

hypothesis = pytest.importorskip("hypothesis", reason="Install hypothesis for property tests")
st = pytest.importorskip("hypothesis.strategies", reason="Install hypothesis").__dict__

from hypothesis import given, settings
from hypothesis.strategies import sampled_from, characters, lists, text

# Build a strategy that takes a known alias and injects random benign punctuation/spaces
BENIGN_PUNCT = list(" -_.,!:/\\|—–")
@settings(max_examples=150)
@given(
    base_alias=sampled_from(list(ALIASES.keys()) + [n for n, _ in EXAMPLES]),
    left=text(alphabet=sampled_from(BENIGN_PUNCT + [" "]), min_size=0, max_size=3),
    right=text(alphabet=sampled_from(BENIGN_PUNCT + [" "]), min_size=0, max_size=3),
    mid=lists(sampled_from(BENIGN_PUNCT + [" "]), min_size=0, max_size=2),
)
def test_property_noise_tolerance(base_alias, left, right, mid):
    target = ALIASES.get(base_alias.upper(), dict(EXAMPLES).get(base_alias, None))
    if target is None:
        # If example not in ALIASES, look it up via normalizer once to derive canonical;
        # if that fails, skip (shouldn't happen for EXAMPLES).
        try:
            target = normalize_team(base_alias)
        except TeamNormalizeError:
            pytest.skip(f"Unresolvable base_alias: {base_alias}")
    # Build a noisy string: optional left pad, random mid punct, optional right pad
    mid_str = "".join(mid)
    noisy = f"{left}{base_alias}{mid_str}{right}"
    assert normalize_team(noisy) == target
