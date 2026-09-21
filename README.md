# FPL Points Predictor

A model that predicts Fantasy Premier League points for the upcoming gameweek based on each player's recent form and their fixture, then uses linear programming to pick the best squad within budget. I built this as a portfolio project, loosely following the approach in [Oscar Fraley's Kaggle notebook](https://www.kaggle.com/code/oscarfraley/predicting-fpl-points), but changed enough of it along the way that it's really its own thing now.

I also play FPL myself, so this isn't purely academic. I've been using it to help pick my own transfers this season, which is partly why half of this README is about things that broke.

See `PROJECT_GUIDE.md` for the longer version: why I made certain modelling choices, and a fairly honest account of everything that went wrong getting this running against a live, brand-new season.

## The idea

Two kinds of features drive the predictions.

- **Form**: rolling averages of a player's own stats (xG, xA, minutes, ICT index, etc.) over their last few gameweeks.
- **Fixture**: how leaky the upcoming opponent's defence has been recently, and how sharp the player's own team's attack has been. An attacker's rating should go up against a bad defence; a defender's or keeper's rating should go up against a blunt attack.

Each position (GK, DEF, MID, FWD) gets its own model, trained separately, because the things that predict a goalkeeper's points and a forward's points aren't really the same problem. For each position I grid-search a couple of regressors (Ridge, LightGBM) against a couple of preprocessing options and keep whichever combination does best.

Once I have predicted points for everyone, a small PuLP linear program picks the highest-scoring legal XI within a budget: right formation, no more than three players from any one real club.

## Running it

```bash
pip install -r requirements.txt
python -m src.run_pipeline --gameweek 6 --fixtures-year 2026 --season 2026-27
```

That pulls player data from [vaastav's Fantasy-Premier-League repo](https://github.com/vaastav/Fantasy-Premier-League) on GitHub and fixtures from fixturedownload.com. A small sample fixtures file for the first few gameweeks of 2026/27 is already in `data/fixtures_epl_2026.csv`. Grab the full season yourself once you're further along.

Two things worth knowing before you run this on a season that's only just started.

**If it's early in the season and the GitHub data hasn't caught up yet** (this happened to me, see below), use `fpl_api.py` to pull the missing gameweeks straight from the official FPL API instead, and `run_gw4_temp.py` as a stopgap predictor that doesn't need a full four weeks of history.

**If you want a transfer suggestion for your own squad**, fill in your 15 players and bank balance at the top of `suggest_transfers.py` and run it with `--gameweek <n>`. If any names don't match, `find_player_names.py` will look them up against the official player list. Accents and shortened names trip this up more often than you'd expect [Ronald Araújo vs Araujo, João Pedro when there happen to be two João Pedros in the league(one going by the name 'Costinha"), that kind of thing].

Tests: `pytest tests/`

## Layout

```
src/
├── data_collection.py     downloads and caches player + fixture data
├── feature_engineering.py
├── models.py               per-position model selection, fitting, prediction
├── evaluate.py             walk-forward validation, feature importance
├── optimize_team.py       squad optimizer + transfer suggestions
└── run_pipeline.py         normal CLI entry point

tests/
└── test_features.py        unit tests for feature engineering

notebooks/
└── exploration.ipynb       EDA and narrative walkthrough, for portfolio write-ups

run_gw4_temp.py              early-season stopgap (shortened form window)
fpl_api.py                   backfills gameweeks the GitHub source hasn't published  
suggest_transfers.py         transfer suggestions for any gameweek
find_player_names.py         looks up correct player name spellings
```

## Things that broke, and what I did about them

I'm leaving this in rather than cleaning it up, because most of what I actually learned building this was in the debugging, not the modelling. Full writeups are in `PROJECT_GUIDE.md`. Short version:

The GitHub data source (a community-maintained repo, not official) was several gameweeks behind by the time the season actually started, so predicting anything current meant pulling gameweek data straight from the official FPL API instead and stitching it in.

Team names don't match across the three sources I'm using. The GitHub data calls a newly-promoted club "Hull City," the fixtures list calls it "Hull," and matching a player to their opponent falls over instantly if those don't agree. Same story for Coventry, Ipswich and a couple of others. There's a small mapping in `data_collection.py` now that normalises everything to one spelling.

fixturedownload.com just returns a 403 if your request doesn't look like it came from a browser, which took a minute to figure out since the error gives you nothing useful. Sending a normal User-Agent header fixed it.

Probably the nastiest one: I was caching downloaded gameweek files by filename only, `gw1.csv`, `gw2.csv`, and so on, with no season attached. Since the training data pulls gameweek 1 from two different historical seasons and the live prediction also wants gameweek 1 for the current season, they were all landing in the same cache file. Whichever one got downloaded first just silently won, and everything downstream trained or predicted on the wrong season's players without throwing any error at all. Fixed by keying the cache off the full URL instead of just the last part of it.

There was also a subtler bug in the model selection step. I was reusing the same `Ridge()` and `LGBMRegressor()` objects across all four positions instead of creating fresh ones each time. If two positions both happened to pick Ridge as their best model, they were quietly sharing the same fitted object in memory, so training the second one overwrote the first one's coefficients. It surfaced as a bizarre "expected 22 features, got 6" error that took a while to trace back to its actual cause. `sklearn.base.clone()` fixed it.

The rest were more ordinary: a divide-by-zero in one of the derived features producing NaNs that Ridge doesn't like, a column getting dropped before I needed it later, that sort of thing.

## What I'd add next

Right now this doesn't account for injuries or rotation risk at all. A player can look great on paper and still not start. It also only optimises a starting XI rather than a full 15-man squad with a bench, and fixture difficulty is based purely on recent expected goals rather than anything like an Elo rating that would react faster to a managerial change. All reasonable next steps if I keep working on this past the portfolio stage.
