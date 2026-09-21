# Building this: notes on decisions, tradeoffs, and what went wrong

This is the longer companion to the README. Partly a build log, partly an explanation of why things are structured the way they are. I started from [Oscar Fraley's Kaggle notebook](https://www.kaggle.com/code/oscarfraley/predicting-fpl-points), which has a genuinely good core idea (form and fixture features, one model per position, an LP-based squad optimizer), but a single notebook that runs cleanly against historical data once is a different thing from a project that has to keep working against a live season. Most of what follows is about that gap.

## Where the data comes from

Player gameweek stats come from [vaastav's Fantasy-Premier-League repo](https://github.com/vaastav/Fantasy-Premier-League), a community-maintained GitHub project, not an official FPL data source. Each season has a folder of per-gameweek CSVs with things like minutes, xG, xA, ICT index, and price for every player who featured that week.

Fixtures come from fixturedownload.com, mostly because it's free and doesn't require an account. A small cached copy of the early 2026/27 fixtures sits in `data/fixtures_epl_2026.csv` so the project runs without needing that site to be reachable every time.

Both of these are third-party sources maintained by other people, which turned out to matter a lot more than I expected. See the second half of this document.

## Why per-position models instead of one model

A goalkeeper's points come almost entirely from saves and clean sheets. A forward's come almost entirely from goals. Pooling everyone into one regression means the model has to somehow learn both of those patterns at once from a shared set of coefficients, which just throws away signal. Splitting by position lets each model specialise, and it also means I can grid-search a slightly different set of transformers per position and see, for instance, that Ridge on power-transformed features tends to win for goalkeepers (low-variance target, mostly saves) while LightGBM wins more often for forwards, where the target is more skewed.

## Walk-forward validation instead of random K-fold

The original notebook validates with a random K-fold split. The problem is that a player's stats in nearby gameweeks are correlated (their underlying ability doesn't change week to week even if their score does), so a random split can put gameweek 8 in the training fold and gameweek 7 in the validation fold for the same player. That's not the situation this model will actually face in practice, where it only ever has access to the past. I added a walk-forward version instead: train on everything before gameweek k, validate on gameweek k, slide forward. The gap between the two numbers is worth reporting honestly rather than picking whichever one looks better.

## The squad optimizer

This is a small linear program using PuLP: maximise total predicted points subject to budget, a legal formation, and no more than three players from the same real club. It's a nice, self-contained piece of the original notebook and I mostly kept it as-is, though I added a transfer suggestion mode on top. Given a list of players you currently own and how much money is in the bank, it looks at every legal same-position swap and ranks them by predicted points gained.

## Everything that broke against a live season

This is the part I think is actually more useful to write down than the modelling choices above, because it's the part that doesn't show up if you only ever test against clean, finished historical data.

**The data source lagged real life.** By the time the season had actually reached its fourth gameweek, the GitHub repo had only published the first one. There's no code fix for someone else's repo being behind. The actual fix was pulling the missing gameweeks directly from the official Fantasy Premier League API (`fantasy.premierleague.com/api/...`) instead, which is always current since it's the real data behind the game. That API doesn't expose a couple of fields the same way, historical per-gameweek transfer counts in particular, so those get approximated with zeros rather than left out entirely. It's a small loss of signal, not a big one.

**Team names don't agree across sources.** This one cost the most time relative to how simple the eventual fix was. The GitHub data called a promoted club "Hull City." My fixtures file, and the official API, both call it "Hull." Same pattern for Coventry and Ipswich. Every time I tried to match a player to their upcoming opponent, one of these three would silently fail to line up, and the error you get back, `ValueError: can only convert an array of size 1 to a Python scalar`, tells you almost nothing about which team or why. I ended up writing a small throwaway diagnostic script each time this happened, subclassing the class that was crashing and overriding just the method that failed so I could print out both sets of team names before the crash actually landed. Once I could see the two lists side by side the mismatch was always obvious. There's now a single normalization mapping in `data_collection.py` that every downloaded file passes through, so this shouldn't come up again unless a new club gets promoted with yet another naming quirk.

**fixturedownload.com just returns a 403 if your request doesn't look like it came from a browser.** No useful error message, just a flat rejection. Fixed by sending a normal browser User-Agent header on every request.

**A caching bug that took the longest to actually understand.** The download cache was keyed on filename only, `gw1.csv`, `gw2.csv`, and so on, with the season stripped out. That's fine as long as you only ever care about one season. But this project needs gameweek 1 from two different historical seasons for training, and also gameweek 1 from the current season for live prediction, and they were all landing in the same cache file. Whichever one got downloaded first just stayed there. The failure mode wasn't a crash, it was quietly-wrong predictions, populated with players from clubs that weren't even in the league this season, which is a much worse kind of bug because nothing tells you it happened. I only found it because a diagnostic print showed players from Sheffield United and Burnley showing up in a 2026/27 prediction, and neither club has been in the Premier League for a couple of seasons now. The fix was to build the cache key from the entire URL path, not just the last segment.

**Fitted models silently overwriting each other.** In the model selection step, I was creating one `Ridge()` and one `LGBMRegressor()` at the top of the script and reusing those same two objects across all four position loops, rather than creating a fresh instance each time. If, say, both the goalkeeper model and the forward model ended up picking Ridge as their best performer, they were both wrapping the exact same object in memory. Not two separate Ridge instances that happen to have the same settings, the literal same one. Fitting the forward model second overwrote the coefficients the goalkeeper model had just been fit with. This surfaced as a genuinely confusing error, "X has 6 features, but Ridge is expecting 22 features as input," that only makes sense once you realise the "Ridge" object in question isn't the one you think it is. `sklearn.base.clone()` before each fit sorts it out, and it's a good general reminder that mutable objects reused across loop iterations in Python will bite you eventually, not just in this project.

**Divide-by-zero producing NaNs.** One of the derived features is a player's xG as a share of their team's total xG. Early in a season, or for a team with very little data yet, that denominator can be exactly zero, which turns into `inf` or `NaN` depending on the numerator. Ridge regression has no tolerance for either. Now anything that comes out non-finite gets treated as no signal yet and set to zero, with a second defensive `fillna(0)` right before fitting or predicting as a backstop.

**A missing column that only showed up after everything else was already fixed.** Once the model bug above was sorted, the pipeline started actually reaching the point of building the final output table, and immediately failed because it expected a `value` (price) column that had already been dropped by that point, since it wasn't one of the features the model itself needed. Simple fix: pull the columns you'll need later out before you narrow the dataframe down to just the model's inputs, the same way the team name was already being handled.

## If I were starting over

Most of the above would have been caught faster with better logging from the start rather than print-statement diagnostics written after each crash. I'd also validate the join between any two data sources, team names in particular, the moment I introduce a new source, rather than assuming a match and finding out three steps downstream that it silently didn't happen.
