import pytest

from src.service.normalizer import (
    ALIASES,
    BR_FULL,
    CODES,
    TeamNormalizeError,
    normalize_team,
)

try:
    from hypothesis import given, settings
    from hypothesis.strategies import lists, sampled_from, text

    HAVE_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    HAVE_HYPOTHESIS = False

# ---- Helpers ---------------------------------------------------------------


def variants(word: str):
    """Generate simple case variants to ensure case-insensitivity."""
    yield word.lower()
    yield word.upper()
    yield "".join(ch.lower() if i % 2 else ch.upper() for i, ch in enumerate(word))


def with_noise(s: str):
    """Sprinkle benign punctuation and whitespace; normalizer should handle it."""
    punct = " -_.,!:/\\|—–"
    parts = s.split()
    if not parts:
        return [s]
    combos = []
    for sep in [" ", "  ", "   ", punct[0], " " + punct[3] + " "]:
        combos.append(sep.join(parts))
    results = []
    for base in combos:
        results.append(base)
        results.append(punct[:2] + base + punct[-2:])
        results.append("   " + base + "   ")
        results.append(punct[-1] + "  " + base + "  " + punct[0])
    # de-dup
    seen, out = set(), []
    for r in results:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ---- Tests: canonical codes pass-through -----------------------------------


@pytest.mark.parametrize("code", sorted(CODES))
def test_codes_pass_through(code):
    assert normalize_team(code) == code
    for v in variants(code):
        assert normalize_team(v) == code


# ---- Tests: full names round-trip -----------------------------------------


@pytest.mark.parametrize("full,code", sorted(BR_FULL.items()))
def test_full_names_round_trip(full, code):
    assert normalize_team(full) == code
    for noisy in with_noise(full):
        assert normalize_team(noisy) == code


# ---- Tests: aliases map to valid codes ------------------------------------


def test_all_aliases_point_to_valid_codes():
    for alias, target in ALIASES.items():
        assert target in CODES, f"Alias '{alias}' -> '{target}' is not a valid code"


@pytest.mark.parametrize("alias,target", sorted(ALIASES.items()))
def test_aliases_normalize(alias, target):
    assert normalize_team(alias) == target
    for v in variants(alias):
        assert normalize_team(v) == target
    for noisy in with_noise(alias):
        assert normalize_team(noisy) == target


# ---- Representative real-world examples -----------------------------------

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


# ---- Idempotence -----------------------------------------------------------


@pytest.mark.parametrize(
    "input_str,expected",
    [
        *[(c, c) for c in CODES],
        *list(EXAMPLES),
        *[(k, v) for k, v in ALIASES.items()],
        *[(full, code) for full, code in BR_FULL.items()],
    ],
)
def test_idempotent_normalization(input_str, expected):
    first = normalize_team(input_str)
    assert first == expected
    assert normalize_team(first) == expected


# ---- Bad inputs ------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "???", "Gotham Rogues", "Seattle Supersonics"])
def test_unknown_team_raises(bad):
    with pytest.raises(TeamNormalizeError) as e:
        normalize_team(bad)
    assert "Unknown team" in str(e.value)


@pytest.mark.parametrize("bad", [None, 123, 3.14, object(), [], {}])
def test_non_string_inputs_raise(bad):
    with pytest.raises(TeamNormalizeError):
        normalize_team(bad)  # type: ignore[arg-type]


BENIGN_PUNCT = list(" -_.,!:/\\|—–")


@settings(max_examples=120)
@given(
    base_alias=sampled_from(list(ALIASES.keys()) + [n for n, _ in EXAMPLES]),
    left=text(alphabet=sampled_from(BENIGN_PUNCT + [" "]), min_size=0, max_size=3),
    right=text(alphabet=sampled_from(BENIGN_PUNCT + [" "]), min_size=0, max_size=3),
    mid=lists(sampled_from(BENIGN_PUNCT + [" "]), min_size=0, max_size=2),
)
def test_property_noise_tolerance(base_alias, left, right, mid):
    target = ALIASES.get(base_alias.upper(), dict(EXAMPLES).get(base_alias))
    if target is None:
        try:
            target = normalize_team(base_alias)
        except TeamNormalizeError:
            pytest.skip(f"Unresolvable base_alias: {base_alias}")
    noisy = f"{left}{base_alias}{''.join(mid)}{right}"
    assert normalize_team(noisy) == target
