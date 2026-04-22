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
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_payday'] = df['day'].isin([1, 15]).astype(int)

        return df

    def _fourier_features(self, df):
        doy = df['date'].dt.dayofyear

        for k in range(1, 6):
            df[f'sin_{k}'] = np.sin(2 * np.pi * k * doy / 365)
            df[f'cos_{k}'] = np.cos(2 * np.pi * k * doy / 365)

        return df

    def _promo_features(self, df):
        if self.promos_df is None:
            df['active_promos'] = 0
            df['max_discount'] = 0
            df['stackable'] = 0
            return df

        promo_features = []

        for _, row in df.iterrows():
            d = row['date']

            active = self.promos_df[
                (self.promos_df['start_date'] <= d) &
                (self.promos_df['end_date'] >= d)
            ]

            promo_features.append({
                'active_promos': len(active),
                'max_discount': active['discount_value'].max() if not active.empty else 0,
                'stackable': int(active['stackable_flag'].any()) if 'stackable_flag' in active else 0
            })

        promo_df = pd.DataFrame(promo_features)

        return pd.concat([df.reset_index(drop=True), promo_df], axis=1)