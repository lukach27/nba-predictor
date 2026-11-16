
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time
 
SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
 
all_games = []
for season in SEASONS:
    print(f"Dohvaćam sezonu {season}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    games = gamefinder.get_data_frames()[0]
    all_games.append(games)
    time.sleep(1)  # ne preopterećuj NBA server
 
full_data = pd.concat(all_games, ignore_index=True)
full_data.to_csv('data/raw/games_raw.csv', index=False)
print(f"Spremljeno {len(full_data)} redova u data/raw/games_raw.csv")