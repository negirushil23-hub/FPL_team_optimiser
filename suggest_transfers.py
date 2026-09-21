"""
Suggests the best transfer(s) for any gameweek, given your current squad.
Uses the normal 4-week rolling form window (same as src/run_pipeline.py) —
so this needs at least 4 completed gameweeks of history to exist for the
season you're predicting. For a brand-new season's opening weeks (before
gameweek 5), you'd need a shortened-form-window version instead — ask if
you hit that situation again.

EDIT the CURRENT_SQUAD and BANK_BALANCE constants below to your own squad,
then run with:
    python3 suggest_transfers.py --gameweek 5

Optional flags:
    --fixtures-year 2026   (season start year for fixtures, default 2026)
    --season 2026-27       (season string for pulling player form data, default 2026-27)
"""
from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# EDIT THESE TWO THINGS:

# Your current 15-man squad, using the exact names as they appear in FPL
# (e.g. "Mohamed Salah", not "Salah" or "M.Salah" — if a name doesn't match,
# this script will tell you which ones it couldn't find).
CURRENT_SQUAD = [
    "David Raya Martín",
    "Luka Vušković",
    "Ronald Araujo",
    "Maxim De Cuyper",
    "Issa Diop",
    "Alex Scott",
    "Mamadou Sangare",
    "Bruno Borges Fernandes",
    "João Pedro Junqueira de Jesus",
    "Erling Haaland",
    "Jarred Braithwaite",
    "Brian Brobbey",
    "Ryan Gravenberch",
    "Patrick Dorgu"
    # ... add all 15 names here
]

# How much money you have sitting in the bank (in £m), e.g. 1.5 for £1.5m.
BANK_BALANCE = 0.3

# ---------------------------------------------------------------------------

from src.optimize_team import get_predictions, suggest_best_transfer
from src.run_pipeline import build_training_data, train_all_positions


def main():
    parser = argparse.ArgumentParser(description="Suggest a transfer for a given gameweek.")
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--fixtures-year", type=int, default=2026)
    parser.add_argument("--season", type=str, default="2026-27")
    args = parser.parse_args()

    print("Building training data (this pulls gameweek CSVs and may take a minute)...")
    data = build_training_data()

    print("Selecting and fitting the best model per position...")
    fitted_models = train_all_positions(data)

    print(f"Predicting gameweek {args.gameweek}...")
    predictions = get_predictions(fitted_models, args.gameweek, args.fixtures_year, args.season)

    # Sanity check: warn about any squad names that didn't match anyone in the predictions
    known_names = set(predictions.name)
    unmatched = [n for n in CURRENT_SQUAD if n not in known_names]
    if unmatched:
        print(f"\n⚠️  Could not find these squad names in the data — check "
              f"spelling (should match FPL's official name exactly): {unmatched}")

    print(f"\nFinding the best transfer(s) with £{BANK_BALANCE}m in the bank...\n")
    suggestions = suggest_best_transfer(CURRENT_SQUAD, predictions, BANK_BALANCE)

    if suggestions is None or suggestions.empty:
        print("No beneficial transfer found — your current squad already "
              "looks optimal for this gameweek within your budget.")
    else:
        print("Top suggested transfers (best first):")
        print(suggestions.to_string(index=False))


if __name__ == "__main__":
    main()
