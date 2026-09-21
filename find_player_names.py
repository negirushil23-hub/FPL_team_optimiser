"""
Helper — searches the official FPL player list for exact name spellings,
so you can fix names that didn't match in suggest_transfers.py.

Run with:
    python3 find_player_names.py Raya Araujo Sangare Fernandes "Joao Pedro" Braithwaite Dorgu Vuskovic
(pass as many partial names as you like, space-separated; wrap multi-word
searches in quotes)
"""
import sys

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def main():
    search_terms = sys.argv[1:]
    if not search_terms:
        print("Usage: python3 find_player_names.py <partial name> [<partial name> ...]")
        return

    response = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/", headers=HEADERS, timeout=30
    )
    response.raise_for_status()
    elements = response.json()["elements"]

    for term in search_terms:
        print(f"\nSearching for '{term}':")
        matches = [
            f'{e["first_name"]} {e["second_name"]}'
            for e in elements
            if term.lower() in f'{e["first_name"]} {e["second_name"]}'.lower()
            or term.lower() in e["web_name"].lower()
        ]
        if matches:
            for m in matches:
                print(f"  -> {m}")
        else:
            print("  (no matches found — try a shorter or different part of the name)")


if __name__ == "__main__":
    main()
