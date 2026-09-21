from src.data_collection import PredData


class PredDataDebug(PredData):
    def _map_fixtures_to_players(self, form, fixtures):
        form_teams = set(form.team.unique())
        fixture_teams = set(fixtures["Home Team"].unique()) | set(fixtures["Away Team"].unique())

        print("Teams in this week's form data:", sorted(form_teams))
        print()
        print("Teams in round 4 fixtures:     ", sorted(fixture_teams))
        print()
        print("In form data but NOT in fixtures:", form_teams - fixture_teams)
        print("In fixtures but NOT in form data:", fixture_teams - form_teams)
        print()
        print("Round 4 fixtures table:")
        print(fixtures.to_string())

        # Stop here instead of crashing further down.
        raise SystemExit("Diagnostic complete — see output above.")


PredDataDebug(gw=4, fixtures_start_year=2026, form_range=3, season="2026-27")
