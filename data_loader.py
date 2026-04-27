import pandas as pd
import os


class DataLoader:
    """
    Loads and standardizes all raw data tables from the dataset.

    Handles column renaming, date parsing, and basic type coercion so that
    all downstream classes receive clean, consistently named DataFrames.
    """

    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    def load_all_data(self) -> dict:
        data = {}
        data['sales']       = self._load_sales()
        data['promotions']  = self._load_promotions()
        data['web_traffic'] = self._load_web_traffic()
        return data

    # --------------------------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------------------------

    def _load_sales(self) -> pd.DataFrame:
        path = os.path.join(self.raw_data_path, 'sales.csv')
        df   = pd.read_csv(path)

        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns={
            'date'   : 'date',
            'revenue': 'revenue',
            'cogs'   : 'cogs',
        })

        df['date']    = pd.to_datetime(df['date'])
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
        df['cogs']    = pd.to_numeric(df['cogs'],    errors='coerce').fillna(0)

        df = df.sort_values('date').reset_index(drop=True)

        print(f"Sales loaded       : {len(df):,} rows | "
              f"{df['date'].min().date()} -> {df['date'].max().date()}")
        return df

    def _load_promotions(self) -> pd.DataFrame:
        path = os.path.join(self.raw_data_path, 'promotions.csv')
        df   = pd.read_csv(path)

        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns={
            'start_date'     : 'start_date',
            'end_date'       : 'end_date',
            'discount_value' : 'discount_value',
            'stackable_flag' : 'stackable_flag',
        })

        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date']   = pd.to_datetime(df['end_date'])

        print(f"Promotions loaded  : {len(df):,} rows")
        return df

    def _load_web_traffic(self) -> pd.DataFrame:
        path = os.path.join(self.raw_data_path, 'web_traffic.csv')
        df   = pd.read_csv(path)

        df.columns = df.columns.str.strip().str.lower()
        df['date'] = pd.to_datetime(df['date'])

        print(f"Web traffic loaded : {len(df):,} rows")
        return df