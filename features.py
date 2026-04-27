import numpy as np
import pandas as pd


# =========================
# CONSTANTS
# =========================

TET_DATES = {
    2013: '2013-02-10', 2014: '2014-01-31', 2015: '2015-02-19',
    2016: '2016-02-08', 2017: '2017-01-28', 2018: '2018-02-16',
    2019: '2019-02-05', 2020: '2020-01-25', 2021: '2021-02-12',
    2022: '2022-02-01', 2023: '2023-01-22', 2024: '2024-02-10',
}

# Promo schedule derived from historical promotions table
# Format: (name, start_month, start_day, duration_days, discount_pct, recurrence)
# recurrence: True = every year, 'odd' = odd years only
PROMO_SCHEDULE = [
    ('spring_sale',    3,  18, 30, 12,   True),
    ('mid_year',       6,  23, 29, 18,   True),
    ('fall_launch',    8,  30, 32, 10,   True),
    ('year_end',       11, 18, 45, 20,   True),
    ('urban_blowout',  7,  30, 33, None, 'odd'),  # odd years only
    ('rural_special',  1,  30, 30, 15,   'odd'),  # odd years only
]

# Vietnamese fixed-date holidays + commercial events
VN_FIXED_HOLIDAYS = [
    (1,  1,  'new_year'),
    (3,  8,  'womens_day'),
    (4,  30, 'reunification'),
    (5,  1,  'labor_day'),
    (9,  2,  'national_day'),
    (10, 20, 'vn_womens_day'),
    (11, 11, 'dd_1111'),
    (12, 12, 'dd_1212'),
    (12, 24, 'christmas_eve'),
    (12, 25, 'christmas'),
]


class FeatureEngineer:
    """
    Feature Engineering chuẩn long-term forecasting.
    Pure calendar features — fully future-safe, no lag, no rolling.
    All features computable from date alone → test matrix buildable upfront.

    Replaces old run_pipeline() with build_features(dates) which accepts
    any DatetimeIndex and returns the full feature matrix for those dates.
    """

    def __init__(self, sales_df):
        self.sales_df = sales_df.copy()

        # Pre-build Tet lookup: year → pd.Timestamp
        self._tet_lut = {
            yr: pd.Timestamp(dt) for yr, dt in TET_DATES.items()
        }

    # =========================
    # PUBLIC: BUILD FEATURES
    # =========================
    def build_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Main entry point. Accepts any DatetimeIndex (train or test).
        Returns a DataFrame with all calendar features for those dates.
        Revenue/COGS columns are NOT included — attach separately.
        """
        print("Building features...")

        df = pd.DataFrame({'date': dates})
        d  = df['date']

        df = self._calendar_features(df, d)
        df = self._regime_features(df, d)
        df = self._fourier_features(df, d)
        df = self._eom_features(df, d)
        df = self._tet_features(df, d)
        df = self._holiday_features(df, d)
        df = self._promo_features(df, d)

        df = df.fillna(0)

        print(f"Feature shape: {df.shape}")
        return df

    def get_train_df(self) -> pd.DataFrame:
        """
        Convenience: build features for the full training date range
        and attach revenue + cogs from sales_df.
        """
        sales = self.sales_df.copy().sort_values('date')

        full_range = pd.date_range(sales['date'].min(), sales['date'].max(), freq='D')
        sales = (
            sales.set_index('date')
                 .reindex(full_range)
                 .rename_axis('date')
                 .reset_index()
        )

        if 'revenue' in sales.columns:
            sales['revenue'] = sales['revenue'].fillna(0)
        if 'cogs' in sales.columns:
            sales['cogs'] = sales['cogs'].fillna(0)

        feat_df = self.build_features(pd.DatetimeIndex(sales['date']))

        # Attach targets — features carry no target information
        for col in ['revenue', 'cogs']:
            if col in sales.columns:
                feat_df[col] = sales[col].values

        return feat_df

    # =========================
    # INTERNAL: CALENDAR
    # =========================
    def _calendar_features(self, df, d):
        df['year']       = d.dt.year
        df['month']      = d.dt.month
        df['day']        = d.dt.day
        df['dayofweek']  = d.dt.dayofweek
        df['dayofyear']  = d.dt.dayofyear
        df['quarter']    = d.dt.quarter
        df['is_weekend'] = (d.dt.dayofweek >= 5).astype(int)
        
        # Two promotional campaigns run only in odd years (urban_blowout, rural_special),
        # producing a consistent August revenue gap of ~1.6x between even and odd years.
        # Test period spans 2023 (odd) and 2024 (even), making this feature essential.
        df['is_odd_year'] = (d.dt.year % 2).astype(int)

        # Days in the current month — used for proportional Fourier and EOM
        df['dim'] = d.dt.days_in_month

        return df

    # =========================
    # INTERNAL: REGIME
    # =========================
    def _regime_features(self, df, d):
        # Three regimes observed in EDA — critical for LGB, XGB to distinguish distributions
        # 2012-2018: peak era (high revenue, clear seasonality)
        # 2019: transition / sudden structural drop
        # 2020+: new lower regime (closer to test 2023-2024)
        df['regime_pre2019']  = (d.dt.year <= 2018).astype(int)
        df['regime_2019']     = (d.dt.year == 2019).astype(int)
        df['regime_post2019'] = (d.dt.year >= 2020).astype(int)

        # Continuous time anchor centered at 2020-01-01 rather than the training
        # start date. Centering reduces collinearity between t_days and t_years
        # and aligns the zero point with the post-2019 regime used by Ridge.
        t0 = pd.Timestamp('2020-01-01')
        df['t_days']  = (d - t0).dt.days
        df['t_years'] = df['t_days'] / 365.25

        return df

    # =========================
    # INTERNAL: FOURIER
    # =========================
    def _fourier_features(self, df, d):
        # All Fourier terms are pure date-math — zero dependency on predicted values,
        # fully future-safe for any horizon. No error stacking possible.
        TAU = 2 * np.pi

        # Annual cycle (k=1..5) — captures yearly seasonality at multiple harmonics
        for k in range(1, 6):
            df[f'sin_y{k}'] = np.sin(TAU * k * df['dayofyear'] / 365.25)
            df[f'cos_y{k}'] = np.cos(TAU * k * df['dayofyear'] / 365.25)

        # Weekly cycle (k=1..2) — captures day-of-week spending rhythm
        for k in range(1, 3):
            df[f'sin_w{k}'] = np.sin(TAU * k * df['dayofweek'] / 7.0)
            df[f'cos_w{k}'] = np.cos(TAU * k * df['dayofweek'] / 7.0)

        # Monthly cycle (k=1..2) — captures within-month payday/EOM rhythm
        # Uses proportional day position within month (not fixed 30/31)
        for k in range(1, 3):
            df[f'sin_m{k}'] = np.sin(TAU * k * (df['day'] - 1) / df['dim'])
            df[f'cos_m{k}'] = np.cos(TAU * k * (df['day'] - 1) / df['dim'])

        return df

    # =========================
    # INTERNAL: EDGE-OF-MONTH
    # =========================
    def _eom_features(self, df, d):
        # Graduated end-of-month flags: days 28-31 average ~7.5M vs mid-month ~3-4M
        # A binary is_month_end flag loses the gradual changes.
        # Continuous distance + graduated flags capture both the magnitude and the approach toward the boundary.

        df['days_to_eom']   = df['dim'] - df['day']      # 0 = last day of month
        df['days_from_som'] = df['day'] - 1               # 0 = first day of month

        # is_last1 = last day, is_last2 = last 2 days, is_last3 = last 3 days
        for k in [1, 2, 3]:
            df[f'is_last{k}']  = (df['days_to_eom']   <= k - 1).astype(int)
            df[f'is_first{k}'] = (df['days_from_som'] <= k - 1).astype(int)

        return df

    # =========================
    # INTERNAL: TET
    # =========================
    def _tet_features(self, df, d):
        # Tet falls on different Gregorian dates each year (Jan 21 – Feb 19)
        # Fixed month/Fourier cannot capture it — tet_days_diff is the correct encoding
        # Negative = before Tet, positive = after Tet, 0 = Tet day itself

        def nearest_tet_diff(dd):
            candidates = [
                self._tet_lut.get(dd.year - 1),
                self._tet_lut.get(dd.year),
                self._tet_lut.get(dd.year + 1),
            ]
            candidates = [c for c in candidates if c is not None]
            # Only consider Tet dates within ±45 days
            diffs = [(dd - c).days for c in candidates if abs((dd - c).days) <= 45]
            return min(diffs, key=abs) if diffs else 999

        diffs = np.array([nearest_tet_diff(dd) for dd in d])

        df['tet_days_diff'] = diffs

        # Window flags for XGB to learn Tet proximity effects on both pre and post phase:
        # Post-Tet ~20 days is the real revenue surge, not the day itself
        df['tet_in_7']     = (np.abs(diffs) <= 7).astype(int)
        df['tet_in_14']    = (np.abs(diffs) <= 14).astype(int)
        df['tet_before_7'] = ((diffs >= -7) & (diffs < 0)).astype(int)
        df['tet_after_7']  = ((diffs > 0)  & (diffs <= 7)).astype(int)
        df['tet_on']       = (diffs == 0).astype(int)

        return df

    # ====================================
    # INTERNAL: HOLIDAYS & SHOPPING EVENTS
    # ====================================
    def _holiday_features(self, df, d):
        # Fixed-date Vietnamese public holidays — known in advance, fully future-safe
        for (m, day, name) in VN_FIXED_HOLIDAYS:
            df[f'hol_{name}'] = (
                (df['month'] == m) & (df['day'] == day)
            ).astype(int)

        # Black Friday — last Friday of November each year
        def is_black_friday(dd):
            if dd.month != 11:
                return 0
            last_nov    = pd.Timestamp(year=dd.year, month=11, day=30)
            last_friday = last_nov - pd.Timedelta(days=(last_nov.dayofweek - 4) % 7)
            return int(dd == last_friday)

        df['hol_black_friday'] = [is_black_friday(dd) for dd in d]

        return df

    # =========================
    # INTERNAL: PROMO WINDOWS
    # =========================
    def _promo_features(self, df, d):
        # Promo schedule derived from historical promotions — fully future-safe
        # since patterns repeat annually (or biennially for odd-year campaigns)
        # Each promo generates 4 features per campaign:
        #   in_window, days_since_start, days_until_end, discount_pct

        # The since/until features allow the model to learn that launch-day spikes
        # and end-of-campaign urgency effects differ from mid-promo behavior.

        years = sorted(set(df['year'].tolist()))

        for (name, start_month, start_day, duration, discount, recurrence) in PROMO_SCHEDULE:
            in_window = np.zeros(len(df), dtype=int)
            since     = np.full(len(df), -1.0)
            until     = np.full(len(df), -1.0)
            disc_arr  = np.zeros(len(df))

            # Expand one year before and after to catch boundary promos (e.g. Dec 15 – Jan 15)
            for y in range(min(years) - 1, max(years) + 2):
                # Skip even years for odd-year-only campaigns
                if recurrence == 'odd' and y % 2 == 0:
                    continue

                start = pd.Timestamp(year=y, month=start_month, day=start_day)
                end   = start + pd.Timedelta(days=duration)
                mask  = (d >= start) & (d <= end)

                in_window[mask] = 1
                since[mask]     = (d[mask] - start).dt.days.values
                until[mask]     = (end - d[mask]).dt.days.values
                disc_arr[mask]  = discount or 0

            df[f'promo_{name}']         = in_window
            df[f'promo_{name}_since']   = since
            df[f'promo_{name}_until']   = until
            df[f'promo_{name}_disc']    = disc_arr

        return df