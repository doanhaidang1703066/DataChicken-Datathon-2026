import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Feature Engineering chuẩn long-term forecasting:
    - Không lag
    - Không rolling
    - Chỉ dùng feature future-safe
    """

    def __init__(self, sales_df, promos_df=None):
        self.sales_df = sales_df.copy()
        self.promos_df = promos_df

    def run_pipeline(self):
        print("⚙️ Running feature engineering...")

        df = self._build_base()
        df = self._time_features(df)
        df = self._fourier_features(df)
        df = self._promo_features(df)

        df = df.fillna(0)

        print(f"✅ Feature shape: {df.shape}")
        return df

    def _build_base(self):
        df = self.sales_df.copy()
        df = df.sort_values('date')

        full_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')

        df = (
            df.set_index('date')
              .reindex(full_range)
              .rename_axis('date')
              .reset_index()
        )

        if 'revenue' in df.columns:
            df['revenue'] = df['revenue'].fillna(0)

        return df

    def _time_features(self, df):
        df['t'] = np.arange(len(df))
        df['t2'] = df['t'] ** 2
        df['t3'] = df['t'] ** 3

        df['dayofweek'] = df['date'].dt.dayofweek
        df['month']     = df['date'].dt.month
        df['day']       = df['date'].dt.day

        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_payday']  = df['day'].isin([1, 15]).astype(int)

        # [ADDED] Quarter — captures Q4 peak and Q1 trough robustly across all years
        # This is one of the most stable seasonal patterns in retail/ecommerce
        df['quarter'] = df['date'].dt.quarter

        # [ADDED] Month-end spending window — structurally tied to salary cycles
        # Pattern holds across years regardless of business changes
        df['is_month_end']   = (df['day'] >= 28).astype(int)
        df['is_month_start'] = (df['day'] <= 3).astype(int)

        # [ADDED] Payday proximity decay — smoother signal than binary is_payday
        # Consumer spend builds toward payday, doesn't switch on exactly on day 1/15
        df['days_to_payday']   = df['day'].apply(
            lambda d: min(abs(d - 1), abs(d - 15), abs(d - 30))
        )
        df['payday_proximity'] = np.exp(-df['days_to_payday'] / 3)

        return df

    def _fourier_features(self, df):
        doy = df['date'].dt.dayofyear

        for k in range(1, 6):
            df[f'sin_{k}'] = np.sin(2 * np.pi * k * doy / 365)
            df[f'cos_{k}'] = np.cos(2 * np.pi * k * doy / 365)

        return df
    
    def _build_promo_calendar(self):
        """
        [ADDED] Build a typical promo calendar from historical data.
        Groups past promos by (month, day) and computes median values.
        Used to impute future dates where no promo schedule is known.
        Future-safe: derived entirely from past observed promo patterns.
        """
        if self.promos_df is None:
            return None

        # Expand all historical promos to daily rows (same as _promo_features)
        promo_expanded = []
        for _, row in self.promos_df.iterrows():
            dates = pd.date_range(row['start_date'], row['end_date'], freq='D')
            promo_len = len(dates)
            for i, d in enumerate(dates):
                promo_expanded.append({
                    'date'           : d,
                    'discount_value' : row['discount_value'],
                    'stackable_flag' : row.get('stackable_flag', 0),
                    'promo_duration' : promo_len,
                })

        promo_daily = pd.DataFrame(promo_expanded)
        promo_daily['month'] = promo_daily['date'].dt.month
        promo_daily['day']   = promo_daily['date'].dt.day

        # For each (month, day) combination: what does a typical promo day look like?
        # Use median to be robust against unusually large one-off events
        calendar = (
            promo_daily
            .groupby(['month', 'day'])
            .agg(
                active_promos          =('discount_value',  'count'),
                max_discount           =('discount_value',  'max'),
                stackable              =('stackable_flag',  'max'),
                promo_duration         =('promo_duration',  'median'),
                total_discount_exposure=('discount_value',  'sum'),
            )
            .reset_index()
        )

        # Normalize active_promos: divide by number of years in history
        # so count reflects "typical number of promos on this day" not total ever
        n_years = (
            self.promos_df['end_date'].max() - self.promos_df['start_date'].min()
        ).days / 365.25
        calendar['active_promos'] = (calendar['active_promos'] / n_years).round(2)
        calendar['total_discount_exposure'] = (
            calendar['total_discount_exposure'] / n_years
        ).round(2)
        
        # Remove Feb 29 to avoid issues on non-leap years
        calendar = calendar[~((calendar['month'] == 2) & (calendar['day'] == 29))]

        return calendar
    
    def _promo_features(self, df):
        if self.promos_df is None:
            df['active_promos'] = 0
            df['max_discount']  = 0
            df['stackable']     = 0
            return df

        promo_expanded = []
        for _, row in self.promos_df.iterrows():
            dates = pd.date_range(row['start_date'], row['end_date'], freq='D')
            for d in dates:
                promo_expanded.append({
                    'date':           d,
                    'discount_value': row['discount_value'],
                    'stackable_flag': row.get('stackable_flag', 0)
                })

        promo_daily = pd.DataFrame(promo_expanded)
        promo_agg   = promo_daily.groupby('date').agg(
            active_promos=('discount_value', 'count'),
            max_discount =('discount_value', 'max'),
            stackable    =('stackable_flag', 'max')
        ).reset_index()

        df = df.merge(promo_agg, on='date', how='left')

        # [ADDED] Impute future dates using historical promo calendar
        missing_mask = df['active_promos'].isna()
        if missing_mask.any():
            promo_calendar = self._build_promo_calendar()

            df['month_temp'] = df['date'].dt.month
            df['day_temp']   = df['date'].dt.day

            # [FIXED] Rename calendar's 'month'/'day' join keys to match df's temp keys
            # AND rename value columns to cal_ prefix to avoid collision
            # AND drop unused columns (promo_duration, total_discount_exposure)
            # that _build_promo_calendar returns but simple model doesn't need
            promo_calendar_slim = (
                promo_calendar[['month', 'day', 'active_promos', 'max_discount', 'stackable']]
                .rename(columns={
                    'month'        : 'month_temp',      # ← fix: align join keys
                    'day'          : 'day_temp',         # ← fix: align join keys
                    'active_promos': 'cal_active_promos',
                    'max_discount' : 'cal_max_discount',
                    'stackable'    : 'cal_stackable',
                })
            )

            df = df.merge(promo_calendar_slim, on=['month_temp', 'day_temp'], how='left')

            df['active_promos'] = df['active_promos'].fillna(df['cal_active_promos'])
            df['max_discount']  = df['max_discount'].fillna(df['cal_max_discount'])
            df['stackable']     = df['stackable'].fillna(df['cal_stackable'])

            # Drop all temporary columns cleanly
            df = df.drop(
                columns=[c for c in df.columns if c.startswith('cal_')] +
                        ['month_temp', 'day_temp'],
                errors='ignore'
            )

        df[['active_promos', 'max_discount', 'stackable']] = (
            df[['active_promos', 'max_discount', 'stackable']].fillna(0)
        )
        return df