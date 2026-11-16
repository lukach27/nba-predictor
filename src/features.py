import pandas as pd

# ============================================================
# 1. UČITAVANJE I FILTRIRANJE
# ============================================================
full_data = pd.read_csv('data/raw/games_raw.csv')

regular_season = full_data[
    full_data['GAME_ID'].astype(str).str.zfill(10).str.startswith('002')
].copy()

regular_season['GAME_DATE'] = pd.to_datetime(regular_season['GAME_DATE'])
regular_season = regular_season.sort_values(['TEAM_ID', 'GAME_DATE'])
regular_season['is_home'] = regular_season['MATCHUP'].str.contains('vs.', regex=False)

cols_to_roll = ['PTS', 'FG_PCT', 'REB', 'TOV', 'AST']


# ============================================================
# 2. ROLLING FORMA (opća i venue-specifična)
# ============================================================
for col in cols_to_roll:
    regular_season[f'rolling_5_{col}'] = (
        regular_season.groupby('TEAM_ID')[col]
        .transform(lambda x: x.shift(1).rolling(5).mean())
    )
    regular_season[f'rolling_5_{col}_by_venue'] = (
        regular_season.groupby(['TEAM_ID', 'is_home'])[col]
        .transform(lambda x: x.shift(1).rolling(5).mean())
    )


# ============================================================
# 3. ODMOR (days_rest, back-to-back)
# ============================================================
regular_season['days_rest'] = (
    regular_season.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
)
regular_season['days_rest'] = regular_season['days_rest'].clip(upper=14)
regular_season['is_back_to_back'] = (regular_season['days_rest'] <= 1).astype(int)


# ============================================================
# 4. ELO RATING (kronološki, red po red, kroz SVE utakmice)
# ============================================================
elo_ratings = {}
K = 20

games_chrono = regular_season[regular_season['is_home']][
    ['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'WL']
].merge(
    regular_season[~regular_season['is_home']][['GAME_ID', 'TEAM_ID']]
    .rename(columns={'TEAM_ID': 'TEAM_ID_away'}),
    on='GAME_ID'
).sort_values('GAME_DATE')

home_elo_before, away_elo_before = [], []

for _, row in games_chrono.iterrows():
    home_id, away_id = row['TEAM_ID'], row['TEAM_ID_away']
    home_elo = elo_ratings.get(home_id, 1500)
    away_elo = elo_ratings.get(away_id, 1500)

    home_elo_before.append(home_elo)
    away_elo_before.append(away_elo)

    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    actual_home = 1 if row['WL'] == 'W' else 0

    elo_ratings[home_id] = home_elo + K * (actual_home - expected_home)
    elo_ratings[away_id] = away_elo + K * ((1 - actual_home) - (1 - expected_home))

games_chrono['elo_home'] = home_elo_before
games_chrono['elo_away'] = away_elo_before

# --- Elo trend: potreban je Elo po TIMU kroz SVE njegove utakmice
# (doma i vani zajedno), zato pretvaramo u "dugi" format prije trenda.
home_elo_long = games_chrono[['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'elo_home']] \
    .rename(columns={'TEAM_ID': 'team_id', 'elo_home': 'elo_at_time'})
away_elo_long = games_chrono[['GAME_ID', 'TEAM_ID_away', 'GAME_DATE', 'elo_away']] \
    .rename(columns={'TEAM_ID_away': 'team_id', 'elo_away': 'elo_at_time'})

elo_long = pd.concat([home_elo_long, away_elo_long], ignore_index=True)
elo_long = elo_long.sort_values(['team_id', 'GAME_DATE'])
elo_long['elo_trend'] = (
    elo_long.groupby('team_id')['elo_at_time']
    .transform(lambda x: x - x.shift(10))
)

games_chrono = games_chrono.merge(
    elo_long[['GAME_ID', 'team_id', 'elo_trend']].rename(
        columns={'team_id': 'TEAM_ID', 'elo_trend': 'elo_trend_home'}),
    on=['GAME_ID', 'TEAM_ID']
)
games_chrono = games_chrono.merge(
    elo_long[['GAME_ID', 'team_id', 'elo_trend']].rename(
        columns={'team_id': 'TEAM_ID_away', 'elo_trend': 'elo_trend_away'}),
    on=['GAME_ID', 'TEAM_ID_away']
)


# ============================================================
# 5. SPAJANJE HOME/AWAY U JEDAN RED PO UTAKMICI
# ============================================================
home_df = regular_season[regular_season['is_home']].copy()
away_df = regular_season[~regular_season['is_home']].copy()

cols_to_drop = ['SEASON_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN']
away_df = away_df.drop(columns=cols_to_drop)

merged = home_df.merge(away_df, on='GAME_ID', suffixes=('_home', '_away'))
merged['home_win'] = (merged['WL'] == 'W').astype(int)

merged = merged.merge(
    games_chrono[['GAME_ID', 'elo_home', 'elo_away', 'elo_trend_home', 'elo_trend_away']],
    on='GAME_ID'
)


# ============================================================
# 6. DIFF KOLONE (home - away)
# ============================================================
for col in cols_to_roll:
    merged[f'{col}_diff'] = merged[f'rolling_5_{col}_home'] - merged[f'rolling_5_{col}_away']

merged['rest_diff'] = merged['days_rest_home'] - merged['days_rest_away']
merged['elo_diff'] = merged['elo_home'] - merged['elo_away']


# ============================================================
# 7. HEAD-TO-HEAD (H2H)
# ============================================================
merged = merged.sort_values('GAME_DATE')
merged['matchup_key'] = merged.apply(
    lambda row: '_'.join(sorted([str(row['TEAM_ID_home']), str(row['TEAM_ID_away'])])),
    axis=1
)

merged['point_margin'] = merged['PTS_home'] - merged['PTS_away']
h2h_history = {}  

h2h_avg_margin_list = []
h2h_home_win_rate_list = []

for _, row in merged.iterrows():
    key = row['matchup_key']
    past = h2h_history.get(key, [])[-3:]  

    if len(past) == 0:
        h2h_avg_margin_list.append(0)
        h2h_home_win_rate_list.append(0.5)
    else:
        margins, wins = [], []
        for past_home_id, past_margin, past_home_win in past:
            if past_home_id == row['TEAM_ID_home']:
                margins.append(past_margin)
                wins.append(past_home_win)
            else:
                margins.append(-past_margin)
                wins.append(1 - past_home_win)
        h2h_avg_margin_list.append(sum(margins) / len(margins))
        h2h_home_win_rate_list.append(sum(wins) / len(wins))

    h2h_history.setdefault(key, []).append(
        (row['TEAM_ID_home'], row['point_margin'], row['home_win'])
    )

merged['h2h_avg_margin'] = h2h_avg_margin_list
merged['h2h_home_win_rate'] = h2h_home_win_rate_list

# ============================================================
# 8. ČIŠĆENJE I SPREMANJE
# ============================================================
required_cols = (
    [f'rolling_5_{c}_home' for c in cols_to_roll] +
    [f'rolling_5_{c}_away' for c in cols_to_roll] +
    [f'rolling_5_{c}_by_venue_home' for c in cols_to_roll] +
    [f'rolling_5_{c}_by_venue_away' for c in cols_to_roll] +
    [f'{c}_diff' for c in cols_to_roll] +
    ['rest_diff', 'elo_home', 'elo_away', 'elo_diff', 'elo_trend_home', 'elo_trend_away',
     'h2h_avg_margin', 'h2h_home_win_rate']
)

model_ready = merged.dropna(subset=required_cols)
model_ready.to_csv('data/processed/games_merged.csv', index=False)

print(f"Spremljeno {len(model_ready)} utakmica spremnih za modeliranje.")
print(f"Home win rate: {model_ready['home_win'].mean():.1%}")
print(model_ready.columns.tolist())