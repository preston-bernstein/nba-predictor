from __future__ import annotations
from typing import Dict

CANON: Dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BRK": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

ALIASES: Dict[str, str] = {
    "ATLANTA": "ATL", "HAWKS": "ATL", "ATLANTA HAWKS": "ATL",
    "BOSTON": "BOS", "CELTICS": "BOS", "BOSTON CELTICS": "BOS",
    "BROOKLYN": "BRK", "NETS": "BRK", "BROOKLYN NETS": "BRK",
    "CHARLOTTE": "CHA", "HORNETS": "CHA",
    "CHICAGO": "CHI", "BULLS": "CHI",
    "CLEVELAND": "CLE", "CAVALIERS": "CLE", "CAVS": "CLE",
    "DALLAS": "DAL", "MAVERICKS": "DAL", "MAVS": "DAL",
    "DENVER": "DEN", "NUGGETS": "DEN",
    "DETROIT": "DET", "PISTONS": "DET",
    "GOLDEN STATE": "GSW", "WARRIORS": "GSW",
    "HOUSTON": "HOU", "ROCKETS": "HOU",
    "INDIANA": "IND", "PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC", "CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL", "LAKERS": "LAL",
    "MEMPHIS": "MEM", "GRIZZLIES": "MEM",
    "MIAMI": "MIA", "HEAT": "MIA",
    "MILWAUKEE": "MIL", "BUCKS": "MIL",
    "MINNESOTA": "MIN", "TIMBERWOLVES": "MIN", "WOLVES": "MIN",
    "NEW ORLEANS": "NOP", "PELICANS": "NOP", "NOLA": "NOP",
    "NEW YORK": "NYK", "KNICKS": "NYK", "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY": "OKC", "THUNDER": "OKC",
    "ORLANDO": "ORL", "MAGIC": "ORL",
    "PHILADELPHIA": "PHI", "SIXERS": "PHI", "76ERS": "PHI",
    "PHOENIX": "PHO", "SUNS": "PHO",
    "PORTLAND": "POR", "TRAIL BLAZERS": "POR", "BLAZERS": "POR",
    "SACRAMENTO": "SAC", "KINGS": "SAC",
    "SAN ANTONIO": "SAS", "SPURS": "SAS",
    "TORONTO": "TOR", "RAPTORS": "TOR",
    "UTAH": "UTA", "JAZZ": "UTA",
    "WASHINGTON": "WAS", "WIZARDS": "WAS",
}

class TeamNormalizeError(ValueError):
    pass

def _canon_text(s: str) -> str:
    """
    Uppercase, remove punctuation to spaces, collapse all whitespace to single spaces.
    Handles doubled/mixed spaces and weird dashes.
    """
    # Normalize to uppercase
    s = s.upper()
    # Replace any non-alnum with space
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    # Collapse runs of whitespace to a single space and strip ends
    return " ".join(s.split())

# Pre-normalize alias keys so spacing variants resolve
_NORMALIZED_ALIASES: Dict[str, str] = {_canon_text(k): v for k, v in ALIASES.items()}

def normalize_team(s: str) -> str:
    # Non-string? Treat as unknown team for a consistent error message the tests expect.
    if not isinstance(s, str):
        raise TeamNormalizeError(f"Unknown team '{s}'. Try a code like NYK or a full team name.")
    raw = s
    # Fast path: explicit 3-letter code
    code = raw.strip().upper()
    if code in CANON:
        return code
    # Canonicalize text (handles punctuation + double spaces)
    key = _canon_text(raw)
    if not key:
        # Tests expect "Unknown team" phrasing for empty-ish input
        raise TeamNormalizeError(f"Unknown team '{raw}'. Try a code like NYK or a full team name.")
    # Direct alias match on normalized key
    if key in _NORMALIZED_ALIASES:
        return _NORMALIZED_ALIASES[key]
    # Still nothing — unknown
    raise TeamNormalizeError(f"Unknown team '{raw}'. Try a code like NYK or a full team name.")
