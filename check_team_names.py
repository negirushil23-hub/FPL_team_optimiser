"""One-off diagnostic script to check that team names are consistent across the various data sources."""
import pandas as pd

gw1 = pd.read_csv(
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2026-27/gws/gw1.csv"
)
gw2 = pd.read_csv("data/cache/gw2.csv.csv")
gw3 = pd.read_csv("data/cache/gw3.csv.csv")
fixtures = pd.read_csv("data/fixtures_epl_2026.csv")

gw1_teams = set(gw1["team"].unique())
gw2_teams = set(gw2["team"].unique())
gw3_teams = set(gw3["team"].unique())
fixture_teams = set(fixtures["Home Team"].unique()) | set(fixtures["Away Team"].unique())

print("GW1 (vaastav) teams:   ", sorted(gw1_teams))
print()
print("GW2 (fetched) teams:   ", sorted(gw2_teams))
print()
print("GW3 (fetched) teams:   ", sorted(gw3_teams))
print()
print("Fixtures file teams:   ", sorted(fixture_teams))
print()
print("In GW1 but not fixtures:", gw1_teams - fixture_teams)
print("In GW2 but not fixtures:", gw2_teams - fixture_teams)
print("In GW3 but not fixtures:", gw3_teams - fixture_teams)
print("In fixtures but not GW1:", fixture_teams - gw1_teams)
