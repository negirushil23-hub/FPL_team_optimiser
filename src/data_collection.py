"""
Data collection layer for the FPL points predictor.

Refactored from the original Kaggle notebook's PlayerData / TrainData / PredData
classes. Key changes vs. the original:
  - explicit season strings ("2023-24") instead of reconstructing from ints
  - on-disk caching of downloaded gameweek CSVs (data/cache/)
  - fixtures pulled from a configurable URL (fixturedownload.com) instead of
    a hardcoded local Kaggle input path
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Sending a normal browser User-Agent header to fixturedownload.com avoids a 
# 403 Forbidden response when fetching
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Resolves conflicting names between vaastav's GitHub source and the official 
# FPL API / fixture CSVs.
TEAM_NAME_NORMALIZE = {
    "Hull City": "Hull",
    "Coventry City": "Coventry",
    "Ipswich Town": "Ipswich",
}

# Columns present in a raw per-gameweek CSV that aren't useful for computing
# a player's rolling "form" average (either identifiers, leak-prone actual
# outcomes, or fixture-specific fields handled separately).
COLUMNS_NOT_NEEDED_FOR_FORM = [
    "xP", "element", "fixture", "kickoff_time", "round", "team_a_score",
    "team_h_score", "penalties_saved", "penalties_missed", "red_cards",
    "yellow_cards", "expected_goals_conceded", "was_home", "starts",
]

RENAME_MAP = {
    "expected_goals": "xG",
    "expected_assists": "xA",
    "expected_goal_involvements": "xGI",
}

FIXTURES_URL_TEMPLATE = "https://fixturedownload.com/download/csv/epl-{year}"


def _cache_filename(url: str) -> str:
    """Builds a cache filename from the FULL url path (not just the last
    segment), so e.g. gw1.csv from the 2023-24 season and gw1.csv from the
    2026-27 season get different cache files instead of colliding and
    silently serving the wrong season's data."""
    return url.split("//")[-1].replace("/", "_").replace("?", "_") + ".csv"


def _cached_read_csv(url: str) -> pd.DataFrame:
    """Read a CSV from a URL, caching it locally so repeated runs don't
    re-hit GitHub / fixturedownload for data that never changes (past
    gameweeks) or rarely changes (fixtures)."""
    cache_path = CACHE_DIR / _cache_filename(url)
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.to_csv(cache_path, index=False)
    if "team" in df.columns:
        df["team"] = df["team"].replace(TEAM_NAME_NORMALIZE)
    return df


def get_gameweek_url(season: str, gw: int) -> str:
    """season like '2023-24', gw an int gameweek number."""
    return (
        "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
        f"master/data/{season}/gws/gw{gw}.csv"
    )


def get_fixtures_url(start_year: int) -> str:
    """start_year e.g. 2026 for the 2026/27 season."""
    return FIXTURES_URL_TEMPLATE.format(year=start_year)


class PlayerData:

    def __init__(self, gw: int, season: str, form_range: int = 4):
        self.gw = gw
        self.season = season
        self.range = form_range
        self.form_data = self.get_form_data()

    def get_url(self, lag: int) -> str:
        return get_gameweek_url(self.season, self.gw - lag)

    def get_form_data(self) -> pd.DataFrame:
        frames = []
        for i in range(self.range):
            temp = _cached_read_csv(self.get_url(i + 1))
            temp = temp.drop(columns=COLUMNS_NOT_NEEDED_FOR_FORM, errors="ignore")
            temp = temp.groupby(["name", "position", "team"]).mean(numeric_only=True)
            frames.append(temp)
        data = pd.concat(frames)
        data = data.groupby(["name", "position", "team"]).mean(numeric_only=True)
        data = data.reset_index("position")
        data = data.rename(columns=RENAME_MAP)
        return data

    def calculate_team_defence(self) -> pd.Series:
        frames = []
        for n in range(self.range):
            temp = _cached_read_csv(self.get_url(n + 1))
            temp = temp.loc[temp.minutes == 90][["team", "expected_goals_conceded"]]
            frames.append(temp.groupby("team").mean(numeric_only=True))
        return pd.concat(frames).groupby("team").mean(numeric_only=True)["expected_goals_conceded"]

    def calculate_team_attack(self) -> pd.Series:
        frames = []
        for n in range(self.range):
            temp = _cached_read_csv(self.get_url(n + 1))
            temp = (
                temp[["name", "team", "expected_goals"]]
                .groupby(["name", "team"]).mean(numeric_only=True)
                .groupby("team").sum(numeric_only=True)
            )
            frames.append(temp)
        return pd.concat(frames).groupby("team").mean(numeric_only=True)["expected_goals"]


class TrainData(PlayerData):
    """Builds a training dataframe: form features + realized fixture/outcome."""

    def __init__(self, gw: int, season: str, form_range: int = 4):
        super().__init__(gw, season, form_range)
        self.fixture_data = self.training_fixture_data()
        self.df = self.create_final_df()

    def training_fixture_data(self) -> pd.DataFrame:
        data = _cached_read_csv(self.get_url(0))
        data = data[["name", "team", "was_home", "starts", "total_points", "opponent_team"]]
        data = data.rename(columns={"total_points": "points_scored"})
        data = self.add_team_stats(data)
        return data

    def add_team_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        team_defence = self.calculate_team_defence()
        team_attack = self.calculate_team_attack()
        data["opponent_xGC"] = data.opponent_team.apply(lambda x: team_defence.iloc[x - 1])
        data["opponent_xG"] = data.opponent_team.apply(lambda x: team_attack.iloc[x - 1])
        data["team_xGC"] = data.team.apply(lambda x: team_defence[x])
        data["team_xG"] = data.team.apply(lambda x: team_attack[x])
        data = data.drop(columns=["opponent_team"]).groupby(["name", "team"]).mean(numeric_only=True)
        data = data[data["was_home"] % 1 == 0]  # keep only players ever-present in one venue
        return data

    def create_final_df(self) -> pd.DataFrame:
        df = self.form_data.join(self.fixture_data, on=["name", "team"])
        df_refined = df[(df.starts == 1) & (df.minutes > 45)]
        return df_refined.drop(columns=["starts"])


class PredData(PlayerData):
    """Builds a prediction dataframe for an upcoming (not-yet-played) gameweek."""

    def __init__(self, gw: int, fixtures_start_year: int, look_ahead: int = 0,
                 form_range: int = 4, season: str = "2024-25"):
        super().__init__(gw, season, form_range)
        self.look_ahead = look_ahead
        self.fixtures_start_year = fixtures_start_year
        self.df = self.get_prediction_data()

    def get_prediction_data(self) -> pd.DataFrame:
        fixtures = self.load_fixtures_data()
        form = self.form_data.reset_index("team")
        mapping = self._map_fixtures_to_players(form, fixtures)
        data = self._join_form_and_fixture(form, mapping)
        data = self.add_team_stats(data)
        return data[data.minutes > 45]

    def load_fixtures_data(self) -> pd.DataFrame:
        # Prefer a locally saved fixtures CSV (e.g. data/fixtures_epl_2026.csv)
        # over a live download: fixturedownload.com's download URL serves an
        # HTML page rather than a raw CSV when fetched programmatically, so
        # you may need to open it in a browser and save the file yourself
        # if a gameweek isn't covered by the bundled sample file yet.
        local_path = (
            Path(__file__).resolve().parent.parent
            / "data" / f"fixtures_epl_{self.fixtures_start_year}.csv"
        )
        if local_path.exists():
            fixtures = pd.read_csv(local_path)
        else:
            url = get_fixtures_url(self.fixtures_start_year)
            fixtures = _cached_read_csv(url)
        return fixtures[fixtures["Round Number"] == self.gw + self.look_ahead]

    def _map_fixtures_to_players(self, form: pd.DataFrame, fixtures: pd.DataFrame) -> dict:
        mapping = {}
        for i, team in enumerate(form.team):
            home_match = fixtures["Home Team"] == team
            if home_match.any():
                oppo = fixtures.loc[home_match, "Away Team"].item()
                mapping[i] = [1, oppo]
            else:
                oppo = fixtures.loc[fixtures["Away Team"] == team, "Home Team"].item()
                mapping[i] = [0, oppo]
        return mapping

    def _join_form_and_fixture(self, form: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        df = pd.DataFrame(mapping).transpose().set_index(form.index)
        df.columns = ["was_home", "oppo"]
        return form.join(df, on=["name"])

    def add_team_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        team_defence = self.calculate_team_defence()
        team_attack = self.calculate_team_attack()
        data["opponent_xGC"] = data.oppo.apply(lambda x: team_defence[x])
        data["opponent_xG"] = data.oppo.apply(lambda x: team_attack[x])
        data["team_xGC"] = data.team.apply(lambda x: team_defence[x])
        data["team_xG"] = data.team.apply(lambda x: team_attack[x])
        return data
