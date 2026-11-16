import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib


df = pd.read_csv('data/processed/games_merged.csv')

df_sorted = df.sort_values('GAME_DATE').reset_index(drop=True)
X = df_sorted[[col for col in df_sorted.columns if col.startswith('rolling_5_')
                or col.startswith('days_rest')
                or col.endswith('_diff')
                or col.startswith('h2h_') and not col.endswith('_display')
                or col.startswith('elo_') and 'trend' not in col
                #or col.startswith('is_back_to_back')
                ]]
y = df_sorted['home_win']


train_split_index = int(len(df_sorted) * 0.7)
validation_split_index = int(len(df_sorted) * 0.8)

X_train, y_train = X.iloc[:train_split_index], y.iloc[:train_split_index]
X_validation, y_validation = X.iloc[train_split_index:validation_split_index], y.iloc[train_split_index:validation_split_index]
X_test, y_test = X.iloc[validation_split_index:], y.iloc[validation_split_index:]

param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.005, 0.01, 0.05]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid,
    scoring='neg_brier_score',
    cv=TimeSeriesSplit(n_splits=5),
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
xgb_model = grid_search.best_estimator_
# xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
# xgb_model.fit(X_train, y_train)

predictions = xgb_model.predict(X_test)
probabilities = xgb_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.3f}")

naive_baseline = y_test.mean()
print(f"Naivni baseline (uvijek predvidi home win): {naive_baseline:.3f}")

brier = brier_score_loss(y_test, probabilities)
print(f"Brier score: {brier:.3f}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)

log_predictions = log_model.predict(X_test_scaled)
log_probabilities = log_model.predict_proba(X_test_scaled)[:, 1]

log_accuracy = accuracy_score(y_test, log_predictions)
print(f"\nLogistic Regression Accuracy: {log_accuracy:.3f}")

log_brier = brier_score_loss(y_test, log_probabilities)
print(f"Logistic Regression Brier score: {log_brier:.3f}")

importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)


print(grid_search.best_params_)
print(importances)

joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(X_train.columns.tolist(), 'models/feature_columns.pkl')