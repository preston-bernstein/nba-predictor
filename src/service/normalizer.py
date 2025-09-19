from __future__ import annotations

import re

__all__ = [
    "normalize_team",
    "TeamNormalizeError",
    "CODES",
    "BR_FULL",
    "ALIASES",
    "canonical_name",
    "_TEAMS_MAP_VERSION",
    "codes",
    "aliases",
    "br_full",
]


class TeamNormalizeError(ValueError):
    """Raised when a team string can't be normalized to a canonical Basketball-Reference code."""


# Bump when alias map changes (handy to spot stale installs)
_TEAMS_MAP_VERSION = "2025-09-08"

# Canonical Basketball-Reference 3-letter team codes (current franchises)
CODES = {
    "ATL",
    "BOS",
    "BRK",
    "CHO",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHO",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
}

# Exact BR full names -> code (UPPERCASE, single-spaced)
BR_FULL = {
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BRK",
    "CHARLOTTE HORNETS": "CHO",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHO",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",
}

# Friendly aliases & legacy codes -> code (UPPERCASE, single-spaced)
ALIASES = {
    # Alt/legacy codes
    "PHX": "PHO",
    "BKN": "BRK",
    "CHA": "CHO",
    "WSH": "WAS",
    "NJN": "BRK",
    "NOH": "NOP",
    "NOK": "NOP",
    # City-only (unambiguous)
    "ATLANTA": "ATL",
    "BOSTON": "BOS",
    "BROOKLYN": "BRK",
    "CHARLOTTE": "CHO",
    "CHICAGO": "CHI",
    "CLEVELAND": "CLE",
    "DALLAS": "DAL",
    "DENVER": "DEN",
    "DETROIT": "DET",
    "GOLDEN STATE": "GSW",
    "HOUSTON": "HOU",
    "INDIANA": "IND",
    "MEMPHIS": "MEM",
    "MIAMI": "MIA",
    "MILWAUKEE": "MIL",
    "MINNESOTA": "MIN",
    "NEW ORLEANS": "NOP",
    "NEW YORK": "NYK",
    "OKLAHOMA CITY": "OKC",
    "ORLANDO": "ORL",
    "PHILADELPHIA": "PHI",
    "PHOENIX": "PHO",
    "PORTLAND": "POR",
    "SACRAMENTO": "SAC",
    "SAN ANTONIO": "SAS",
    "TORONTO": "TOR",
    "UTAH": "UTA",
    "WASHINGTON": "WAS",
    # LA disambiguations
    "LA CLIPPERS": "LAC",
    "L A CLIPPERS": "LAC",
    "LOS ANGELES CLIPPERS": "LAC",
    "LA LAKERS": "LAL",
    "L A LAKERS": "LAL",
    "LOS ANGELES LAKERS": "LAL",
    # Nickname-only
    "HAWKS": "ATL",
    "CELTICS": "BOS",
    "NETS": "BRK",
    "HORNETS": "CHO",
    "BULLS": "CHI",
    "CAVALIERS": "CLE",
    "MAVERICKS": "DAL",
    "NUGGETS": "DEN",
    "PISTONS": "DET",
    "WARRIORS": "GSW",
    "ROCKETS": "HOU",
    "PACERS": "IND",
    "CLIPPERS": "LAC",
    "LAKERS": "LAL",
    "GRIZZLIES": "MEM",
    "HEAT": "MIA",
    "BUCKS": "MIL",
    "TIMBERWOLVES": "MIN",
    "PELICANS": "NOP",
    "KNICKS": "NYK",
    "THUNDER": "OKC",
    "MAGIC": "ORL",
    "76ERS": "PHI",
    "SUNS": "PHO",
    "TRAIL BLAZERS": "POR",
    "BLAZERS": "POR",
    "KINGS": "SAC",
    "SPURS": "SAS",
    "RAPTORS": "TOR",
    "JAZZ": "UTA",
    "WIZARDS": "WAS",
    # Fan/press nicknames
    "MAVS": "DAL",
    "CAVS": "CLE",
    "DUBS": "GSW",
    "CLIPS": "LAC",
    "GRIZZ": "MEM",
    "WOLVES": "MIN",
    "PELIS": "NOP",
    "SIXERS": "PHI",
    "TWOLVES": "MIN",
    "T WOLVES": "MIN",
    "RAPS": "TOR",
    "WIZ": "WAS",
    "NUGS": "DEN",
    "CELTS": "BOS",
    "ROX": "HOU",
    "SPURS SA": "SAS",
    "NOLA": "NOP",
    "NOLA PELICANS": "NOP",
    "N O PELICANS": "NOP",
    "PHILA": "PHI",
    "PHILA 76ERS": "PHI",
}

CODE_TO_FULL = {code: full for full, code in BR_FULL.items()}


def canonical_name(code: str) -> str:
    """Return the Basketball-Reference display name for a canonical code."""
    if not isinstance(code, str) or not code.strip():
        raise TeamNormalizeError("Code must be a non-empty string.")
    c = code.strip().upper()
    if c not in CODES:
        raise TeamNormalizeError(f"Unknown team code '{code}'.")
    return CODE_TO_FULL.get(c, c)


# Back-compat lowercase exports for older tests/code
codes = CODES
aliases = ALIASES
br_full = BR_FULL

# Punctuation/whitespace cleanup (keep word chars and spaces only)
_PUNC_RE = re.compile(r"[^A-Z0-9\s]+")


def _clean(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = s.upper()
    # convert any non A–Z / 0–9 to spaces (includes _, dashes, punctuation, unicode dashes)
    s = _PUNC_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_team(raw: str) -> str:
    """
    Normalize a team string into a Basketball-Reference canonical 3-letter code.
    Accepts already-canonical codes, full names, and a variety of aliases.
    """
    if not isinstance(raw, str) or not raw.strip():
        # Keep legacy wording so older tests asserting substring still pass
        raise TeamNormalizeError("Unknown team: empty or whitespace.")
    s = _clean(raw)

    # Already a canonical code?
    if s in CODES:
        return s
    # Exact BR full names
    if s in BR_FULL:
        return BR_FULL[s]
    # Friendly aliases
    if s in ALIASES:
        return ALIASES[s]

    raise TeamNormalizeError(f"Unknown team '{raw}'. Try a code like NYK or a full team name.")
