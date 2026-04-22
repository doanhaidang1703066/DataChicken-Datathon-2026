import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class WalkForwardBacktester:
    """
    Walk-forward validation cho time-series
    """

    def __init__(self, model, df, date_col='date', target_col='revenue'):
        self.model = model
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.results = []

    def _metrics(self, y_true, y_pred):
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }

    def run_backtest(self, splits):
        print(f"\n🚀 Running {len(splits)} folds backtest...\n")

        for i, split in enumerate(splits):
            train_end = pd.to_datetime(split['train_end'])
            val_end = pd.to_datetime(split['val_end'])

            train_df = self.df[self.df[self.date_col] <= train_end].copy()
            val_df = self.df[
                (self.df[self.date_col] > train_end) &
                (self.df[self.date_col] <= val_end)
            ].copy()

            print(f"--- FOLD {i+1} ---")
            print(f"Train size: {len(train_df)}")
            print(f"Val size  : {len(val_df)}")

            # FIT
            self.model.fit(train_df, self.target_col, self.date_col)

            # PREDICT
            preds = self.model.predict(val_df, self.date_col)

            # METRICS
            metrics = self._metrics(val_df[self.target_col], preds)

            print(f"MAE : {metrics['MAE']:.2f}")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(f"R2  : {metrics['R2']:.4f}\n")

            self.results.append({
                'fold': i + 1,
                'metrics': metrics,
                'train_df': train_df,
                'val_df': val_df,
                'preds': preds
            })

        print("✅ Backtest completed")
        return self.results