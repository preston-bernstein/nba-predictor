import logging
import time

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}


def fetch_season_html(end_year: int, retries: int = 3, timeout: int = 60) -> str:
    """Fetch the Basketball-Reference season index HTML (e.g. 2024 -> 2023–24 season)."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_games.html"
    err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            err = e
            logging.warning("fetch %s failed (attempt %d/%d): %s", end_year, attempt, retries, e)
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"could not fetch season {end_year}") from err
