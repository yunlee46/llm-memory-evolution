"""Canonical MLB team abbreviations and mappings from each data source.

Canonical codes follow MLB StatsAPI / Kalshi 2026 usage (AZ, ATH, CWS, KC, LAD, SD, SF, TB, WSH ...).
"""
from __future__ import annotations

CANONICAL = {
    "AZ", "ATH", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET", "HOU", "KC", "LAA", "LAD",
    "MIA", "MIL", "MIN", "NYM", "NYY", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
}

# Retrosheet game-log team ids (2005+)
RETROSHEET = {
    "ANA": "LAA", "ARI": "AZ", "ATH": "ATH", "OAK": "ATH", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CWS", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "FLO": "MIA",
    "HOU": "HOU", "KCA": "KC", "LAN": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYA": "NYY",
    "NYN": "NYM", "PHI": "PHI", "PIT": "PIT", "SDN": "SD", "SEA": "SEA", "SFN": "SF", "SLN": "STL",
    "TBA": "TB", "TEX": "TEX", "TOR": "TOR", "WAS": "WSH", "MON": "WSH",
}

# Kalshi ticker / StatsAPI abbreviations (mostly identical to canonical; a few aliases)
KALSHI = {c: c for c in CANONICAL}
KALSHI.update({"ARI": "AZ", "OAK": "ATH", "WAS": "WSH", "CHW": "CWS", "SFG": "SF", "SDP": "SD", "TBR": "TB",
               "KCR": "KC", "WSN": "WSH"})

# sportsbookreviewsonline xlsx team strings
SBRO = {
    "ANA": "LAA", "LAA": "LAA", "ARI": "AZ", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "BRS": "BOS", "CHC": "CHC",
    "CUB": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "FLA": "MIA",
    "MIA": "MIA", "HOU": "HOU", "KAN": "KC", "KC": "KC", "LAD": "LAD", "LOS": "LAD", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "ATH", "ATH": "ATH", "PHI": "PHI", "PIT": "PIT",
    "SDG": "SD", "SD": "SD", "SEA": "SEA", "SFO": "SF", "SFG": "SF", "SF": "SF", "STL": "STL", "TAM": "TB", "TB": "TB",
    "TEX": "TEX", "TOR": "TOR", "WAS": "WSH", "WSH": "WSH",
}


def map_codes(codes, table: dict[str, str], source: str) -> list[str]:
    """Map a sequence of source codes to canonical; fail loudly on anything unknown."""
    out = []
    unknown = set()
    for c in codes:
        c = str(c).strip().upper()
        if c in table:
            out.append(table[c])
        else:
            unknown.add(c)
            out.append(None)
    if unknown:
        raise ValueError(f"{source}: unmapped team codes {sorted(unknown)} - extend team_map.py")
    return out
