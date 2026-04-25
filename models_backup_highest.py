import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import xgboost as xgb


class HybridForecaster:
    """
    Ridge (Trend) + XGBoost (Residual)
    Stable cho long-term forecasting (1.5 năm)
    """

    def __init__(self, ridge_alpha=10.0, xgb_params=None):
        self.ridge_alpha = ridge_alpha

        self.ridge_model = Ridge(alpha=self.ridge_alpha)

        self.xgb_params = xgb_params or {
            'n_estimators': 800,
            'learning_rate': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            # [ADDED] Light regularization — enough to reduce noise fitting
            # without suppressing the spike signal that made this model strong
            'reg_lambda'      : 2.0,    # gentle L2, much less than complex model's 10.0
            'min_child_weight': 10,     # in original revenue space — prevents splits on noise
                                        # 10 = need meaningful revenue gradient to split
            # [ADDED] MSE explicitly — same as default but now intentional
            # MSE squares spike residuals → XGB naturally prioritizes large revenue days
            'objective'       : 'reg:squarederror',
        }

        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)

        self.features_used = None

        self.t0 = None  # mốc thời gian

    # =========================
    # INTERNAL: create time features
    # =========================
    def _create_time_index(self, df, date_col):
        df = df.copy()
        df = df.sort_values(date_col)

        if self.t0 is None:
            self.t0 = df[date_col].min()

        df['t'] = (df[date_col] - self.t0).dt.days
        df['t2'] = df['t'] ** 2
        df['t3'] = df['t'] ** 3

        return df

    # =========================
    # FIT
    # =========================
    def fit(self, df, target_col='revenue', date_col='date'):
        df = df.copy()

        # --- TIME FEATURES ---
        df = self._create_time_index(df, date_col)

        # --- TREND (RIDGE) ---
        X_trend = df[['t', 't2', 't3']]
        y = df[target_col]

        self.ridge_model.fit(X_trend, y)

        trend_pred = self.ridge_model.predict(X_trend)

        # --- RESIDUAL ---
        residuals = y - trend_pred

        # --- XGB FEATURES ---
        X = df.drop(columns=[target_col, date_col], errors='ignore')

        # chống leak
        if 'cogs' in X.columns:
            X = X.drop(columns=['cogs'])

        self.features_used = X.columns.tolist()

        self.xgb_model.fit(X, residuals)

        print("✅ Hybrid (Ridge + XGB) trained")

        return self

    # =========================
    # PREDICT
    # =========================
    def predict(self, df, date_col='date'):
        df = df.copy()

        df = self._create_time_index(df, date_col)

        # --- TREND ---
        X_trend = df[['t', 't2', 't3']]
        trend_pred = self.ridge_model.predict(X_trend)

        # --- RESIDUAL ---
        X = df[self.features_used]
        residual_pred = self.xgb_model.predict(X)

        final_pred = trend_pred + residual_pred

        return np.maximum(final_pred, 0)