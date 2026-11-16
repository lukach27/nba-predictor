import pandas as pd
import joblib

xgb_model = joblib.load('models/xgb_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

df = pd.read_csv('data/processed/games_merged.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE_home'] if 'GAME_DATE_home' in df.columns else df['GAME_DATE'])

feature_base_cols = ['rolling_5_PTS', 'rolling_5_FG_PCT', 'rolling_5_REB', 
                      'rolling_5_TOV', 'rolling_5_AST', 'elo']
venue_cols = ['rolling_5_PTS_by_venue', 'rolling_5_FG_PCT_by_venue', 'rolling_5_REB_by_venue', 
              'rolling_5_TOV_by_venue', 'rolling_5_AST_by_venue']

all_base_cols = feature_base_cols + venue_cols

home_stats_df = df[['GAME_DATE', 'TEAM_ID_home'] + [f'{c}_home' for c in all_base_cols]].copy()
home_stats_df.columns = ['GAME_DATE', 'team_id'] + all_base_cols

away_stats_df = df[['GAME_DATE', 'TEAM_ID_away'] + [f'{c}_away' for c in all_base_cols]].copy()
away_stats_df.columns = ['GAME_DATE', 'team_id'] + all_base_cols

all_team_stats = pd.concat([home_stats_df, away_stats_df], ignore_index=True)
all_team_stats = all_team_stats.sort_values(['team_id', 'GAME_DATE'])

def get_latest_team_stats(team_id):
    team_games = all_team_stats[all_team_stats['team_id'] == team_id]
    latest = team_games.sort_values('GAME_DATE').iloc[-1]
    return latest

def predict_matchup(home_team_id, away_team_id, game_date, model, feature_columns):
    home_stats = get_latest_team_stats(home_team_id)
    away_stats = get_latest_team_stats(away_team_id)

    matchup_features = {}
    for col in feature_base_cols:
        matchup_features[f'{col}_home'] = home_stats[col]
        matchup_features[f'{col}_away'] = away_stats[col]
        matchup_features[f'{col}_diff'] = home_stats[col] - away_stats[col]

    # days_rest -- razlika između planiranog datuma i zadnje utakmice svakog tima
    home_last_date = home_stats['GAME_DATE']
    away_last_date = away_stats['GAME_DATE']
    game_date = pd.to_datetime(game_date)

    days_rest_home = min((game_date - home_last_date).days, 14)
    days_rest_away = min((game_date - away_last_date).days, 14)

    matchup_features['days_rest_home'] = days_rest_home
    matchup_features['days_rest_away'] = days_rest_away
    matchup_features['rest_diff'] = days_rest_home - days_rest_away

    matchup_key = '_'.join(sorted([str(home_team_id), str(away_team_id)]))
    h2h_games = df[df['matchup_key'] == matchup_key].sort_values('GAME_DATE')

    if len(h2h_games) > 0:
        last_3 = h2h_games.tail(3)
        margins = []
        home_wins = []
        for _, g in last_3.iterrows():
            if g['TEAM_ID_home'] == home_team_id:
                margins.append(g['point_margin'])
                home_wins.append(g['home_win'])
            else:
                margins.append(-g['point_margin'])
                home_wins.append(1 - g['home_win'])
        matchup_features['h2h_avg_margin'] = sum(margins) / len(margins)
        matchup_features['h2h_home_win_rate'] = sum(home_wins) / len(home_wins)
    else:
        matchup_features['h2h_avg_margin'] = 0
        matchup_features['h2h_home_win_rate'] = 0.5

    input_df = pd.DataFrame([matchup_features])
    win_probability = model.predict_proba(input_df[feature_columns])[0, 1]
    return win_probability

prob = predict_matchup(
    home_team_id=1610612738,
    away_team_id=1610612754,
    game_date='2025-04-20',
    model=xgb_model,
    feature_columns=feature_columns
)

print(f"Vjerojatnost da domaćin pobijedi: {prob:.1%}")