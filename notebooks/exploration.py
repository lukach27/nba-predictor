from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time

df = pd.read_csv('data/processed/games_merged.csv')
print(df.columns.tolist())