"""
Builds local gw2.csv / gw3.csv / gw4.csv files, matching the format this
project's other scripts expect, by pulling data directly from the official
Fantasy Premier League API (fantasy.premierleague.com). This is a
different, always-up-to-date source from the community GitHub repo
(vaastav/Fantasy-Premier-League) used elsewhere in this project, which
sometimes lags a few gameweeks behind real life for a brand-new season.

Caveat: two fields aren't available per-gameweek from this API the way the
GitHub source provides them, so they're approximated rather than exact:
  - `value` uses each player's *current* price, not their price at the time
    of that gameweek (prices only drift slightly week to week, so this is a
    minor approximation).
  - `transfers_in` / `transfers_out` are set to 0, since this API doesn't
    expose historical per-gameweek transfer counts. This only affects the
    "transfer_activity" feature, which is a minor signal in the model.

Once the GitHub source publishes real files for these gameweeks, you can
delete the files this script creates (in data/cache/) so the project goes
back to using the real data automatically.

Run with:
    python3 fpl_api.py
(from the project's root folder.)
"""
from pathlib import Path

import pandas as pd
import requests

from src.data_collection import _cache_filename, get_gameweek_url

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Maps the official FPL API's full team names to the short names used
# elsewhere in this project (matching fixturedownload.com / vaastav's repo).
TEAM_NAME_MAP = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton & Hove Albion": "Brighton", "Chelsea": "Chelsea",
    "Coventry City": "Coventry", "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds", "Liverpool": "Liverpool", "Manchester City": "Man City",
    "Manchester United": "Man Utd", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Spurs",
    "Sunderland": "Sunderland",
}


def fetch_json(url: str) -> dict:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def build_gameweek_csv(gw: int) -> None:
    print(f"Building gw{gw}.csv from the official FPL API...")

    bootstrap = fetch_json("https://fantasy.premierleague.com/api/bootstrap-static/")
    elements = {e["id"]: e for e in bootstrap["elements"]}
    teams = {t["id"]: TEAM_NAME_MAP.get(t["name"], t["name"]) for t in bootstrap["teams"]}

    fixtures = fetch_json(f"https://fantasy.premierleague.com/api/fixtures/?event={gw}")
    team_fixture_info = {}
    for f in fixtures:
        team_fixture_info[f["team_h"]] = {"opponent_team": f["team_a"], "was_home": 1}
        team_fixture_info[f["team_a"]] = {"opponent_team": f["team_h"], "was_home": 0}

    live = fetch_json(f"https://fantasy.premierleague.com/api/event/{gw}/live/")

    rows = []
    for entry in live["elements"]:
        stats = entry["stats"]
        if stats["minutes"] == 0:
            continue  # didn't play this gameweek
        el = elements.get(entry["id"])
        if el is None:
            continue
        team_id = el["team"]
        fixture_info = team_fixture_info.get(team_id, {})
        rows.append({
            "name": f'{el["first_name"]} {el["second_name"]}',
            "position": POSITION_MAP.get(el["element_type"], "UNK"),
            "team": teams.get(team_id, "Unknown"),
            "minutes": stats["minutes"],
            "goals_scored": stats["goals_scored"],
            "assists": stats["assists"],
            "clean_sheets": stats["clean_sheets"],
            "goals_conceded": stats["goals_conceded"],
            "own_goals": stats["own_goals"],
            "bonus": stats["bonus"],
            "expected_goals": stats.get("expected_goals", 0),
            "expected_assists": stats.get("expected_assists", 0),
            "expected_goal_involvements": stats.get("expected_goal_involvements", 0),
            "expected_goals_conceded": stats.get("expected_goals_conceded", 0),
            "total_points": stats["total_points"],
            "starts": stats.get("starts", 1),
            "was_home": fixture_info.get("was_home", 0),
            "opponent_team": fixture_info.get("opponent_team", 0),
            "value": el["now_cost"],       # approximation — see caveat above
            "transfers_in": 0,             # approximation — see caveat above
            "transfers_out": 0,            # approximation — see caveat above
        })

    df = pd.DataFrame(rows)
    url_for_naming = get_gameweek_url("2026-27", gw)
    out_path = CACHE_DIR / _cache_filename(url_for_naming)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} player rows to {out_path}")


if __name__ == "__main__":
    for gameweek in [2, 3, 4]:
        build_gameweek_csv(gameweek)
    print("\nDone. You can now run the normal pipeline: "
          "python -m src.run_pipeline --gameweek 5 --fixtures-year 2026 --season 2026-27")
