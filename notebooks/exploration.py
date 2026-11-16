"""Quick look at the processed matchup table.

Run after `python -m nba_predictor train` (or `features`), from the project root:
    python notebooks/exploration.py
"""

from nba_predictor import config
from nba_predictor.features import load_matchups, training_mask

df = load_matchups()
print(f"{len(df):,} rows, {df.shape[1]} columns ({int((~df['is_played']).sum())} upcoming fixtures)")
print("Model inputs:", config.all_feature_columns())
print(df.loc[training_mask(df), config.all_feature_columns()].describe().T[["mean", "std", "min", "max"]].round(3))
