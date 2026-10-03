This is a football projection model.



Current folder structure



LICENSE

README.md

AGENTS.md



WHAT THIS DOES/WILL DO:
take in stats, predict player stats over the next month to a year.

It does NOT need to predict over time periods LESS than 1 month and has a maximum prediction time of the end of the season.



This will predict component pieces of players. (over the time span)

This will predict the number of wins a team is projected to get over a time span

It should use separate technics (if needed) to try and predict who will win a postseason series.







DATA:

nba\_api

balldontlie

You can scrape somewhere too ig









PREDICTION BASELINES TO COMPARE AGAINST:

FOR TEAMS:


Home Team Advantage / Naïve Winner: Predict the home team wins every time (historically wins \~58–60% of NBA games). Any model failing to beat 60% accuracy on straight-up game outcomes is worse than guessing home-court advantage.

Simple ELO / Net Rating: Calculate team strength using rolling Net Rating (Offensive Rating minus Defensive Rating) adjusted for rest days and venue.



Closing Betting Lines (The Gold Standard):



&#x20;   Converting the Vegas closing point spread or moneyline into an implied win probability yields the most competitive baseline.



&#x20;   Target: Beating the Vegas spread against the vigorish (juice) requires a 52.4% win rate (at standard -110 odds) just to break even. Achieving 54%+ over a large sample is considered top-tier.



Public Analytics Models:



&#x20;   FiveThirtyEight (historical) / DRatings / DunksAndThreeds (EPM): Check model log-loss and Brier score against established public rating 
systems.



FOR PLAYERS:


Per-36 / Per-Minute Efficiency Model:Formula: $\\text{Predicted Output} = \\text{Weighted Stat per Minute} \\times \\text{Projected Minutes}$Why it matters: Playing time (minutes) accounts for \~70–80% of player production variance. A model predicting points directly without first predicting minutes will fail to beat this baseline.



Basketball-Reference Simple Projection System (SPS / Marcel):



&#x20;   Incorporates age curves, regresses to league average, and weights multi-year historical output.



The ultimate benchmark for player-level prediction is the Sportsbook Consensus Closing Line (e.g., DraftKings, FanDuel, Pinnacle).   Over/Under Win Rate: Converting model point predictions into Over/Under picks against bookmaker lines.Break-Even Target: Beat 52.4% accuracy against standard -110 juice.Profitable Target: 54%–56% over a sample size of >500 props.



Check median projections against established Daily Fantasy Sports (DFS) platforms and consensus aggregators:



&#x20;   RotoGrinders / DailyFantasyFuel / LineStar



&#x20;   NumberFire / SaberSim







