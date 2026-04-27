import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class WalkForwardBacktester:
    """
    Three-fold walk-forward validation designed for long-horizon forecasting
    with regime drift.

    The three folds serve distinct diagnostic purposes:
        Fold A (primary)  -- validate on 2022, the year closest to test 2023.
                             All ablation decisions should be based on this fold.
        Fold B (stability) -- validate on 2021, checks whether an improvement
                              is stable across years or only works for 2022.
        Fold C (horizon)  -- rolling 12-month window ending 2022-06,
                             simulates the continuous forecast horizon closest
                             to the actual 18-month test submission.

    Using all three folds guards against overfitting to any single validation
    year and gives a clearer picture of generalization across time.
    """

    # Fold definitions as class-level constants for transparency
    FOLDS = {
        'A': {
            'train_end' : '2021-12-31',
            'val_start' : '2022-01-01',
            'val_end'   : '2022-12-31',
            'description': 'Primary -- validate 2022, closest regime to test',
        },
        'B': {
            'train_end' : '2020-12-31',
            'val_start' : '2021-01-01',
            'val_end'   : '2021-12-31',
            'description': 'Stability -- validate 2021, cross-year consistency check',
        },
        'C': {
            'train_end' : '2021-06-30',
            'val_start' : '2021-07-01',
            'val_end'   : '2022-06-30',
            'description': 'Horizon -- rolling 12-month window, closest to submission length',
        },
    }

    def __init__(self, model, feat_df: pd.DataFrame,
                 target_rev='revenue', target_cogs='cogs', date_col='date'):
        """
        Parameters
        ----------
        model    : EnsembleForecaster instance (unfitted)
        feat_df  : full feature DataFrame from FeatureEngineer including targets
        """
        self.model       = model
        self.feat_df     = feat_df.copy().sort_values(date_col).reset_index(drop=True)
        self.target_rev  = target_rev
        self.target_cogs = target_cogs
        self.date_col    = date_col
        self.results     = {}

    # ==========================================================================
    # PUBLIC: RUN
    # ==========================================================================

    def run(self, folds: list = None):
        """
        Run validation on the specified folds.
        folds: list of fold keys to run, e.g. ['A', 'B', 'C'].
               Defaults to all three folds if not specified.
        """
        folds = folds or ['A', 'B', 'C']

        print(f"Running {len(folds)}-fold walk-forward backtest...")
        print("=" * 60)

        for fold_key in folds:
            self._run_fold(fold_key)

        self._print_summary(folds)
        return self.results

    # ==========================================================================
    # PUBLIC: SUMMARY TABLE
    # ==========================================================================

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising metrics across all completed folds.
        """
        records = []
        for fold_key, res in self.results.items():
            row = {'fold': fold_key}
            row.update(res['metrics'])
            records.append(row)
        return pd.DataFrame(records)

    # ==========================================================================
    # INTERNAL: SINGLE FOLD
    # ==========================================================================

    def _run_fold(self, fold_key: str):
        fold_def = self.FOLDS[fold_key]
        print(f"\nFold {fold_key}: {fold_def['description']}")
        print(f"  Train  : start -> {fold_def['train_end']}")
        print(f"  Validate: {fold_def['val_start']} -> {fold_def['val_end']}")

        train_df = self.feat_df[
            self.feat_df[self.date_col] <= pd.Timestamp(fold_def['train_end'])
        ].copy()

        val_df = self.feat_df[
            (self.feat_df[self.date_col] >= pd.Timestamp(fold_def['val_start'])) &
            (self.feat_df[self.date_col] <= pd.Timestamp(fold_def['val_end']))
        ].copy()

        print(f"  Train size   : {len(train_df):,} days")
        print(f"  Val size     : {len(val_df):,} days")

        # Refit model from scratch on this fold's training data
        from src.models import EnsembleForecaster  
        
        fold_model = EnsembleForecaster(
            ridge_alpha    = self.model.ridge_alpha,
            lgb_params     = self.model.lgb_params,
            alpha          = self.model.alpha,
            w_ridge        = self.model.w_ridge,
            w_prophet      = self.model.w_prophet,
            w_lgb          = self.model.w_lgb,
            cal_revenue    = self.model.cal_revenue,
            cal_cogs       = self.model.cal_cogs,
            q_boost        = self.model.q_boost,
            era_weight     = self.model.era_weight,
            off_era_weight = self.model.off_era_weight,
            random_state   = self.model.random_state,
        )
        fold_model.fit(train_df, self.target_rev, self.target_cogs)

        # Predict on validation -- cal_revenue and cal_cogs remain at 1.0
        # during validation since calibration is tuned on the leaderboard only
        preds = fold_model.predict(val_df)

        metrics = self._compute_metrics(
            val_df[self.target_rev].values,
            val_df[self.target_cogs].values,
            preds['revenue'].values,
            preds['cogs'].values,
        )

        self._print_fold_metrics(metrics)

        # Spike decomposition -- top 10% actual revenue days
        spike_metrics = self._spike_decomposition(
            val_df[self.target_rev].values,
            preds['revenue'].values,
        )

        self.results[fold_key] = {
            'metrics'       : metrics,
            'spike_metrics' : spike_metrics,
            'train_df'      : train_df,
            'val_df'        : val_df,
            'preds'         : preds,
            'model'         : fold_model,
        }

    # ==========================================================================
    # INTERNAL: METRICS
    # ==========================================================================

    def _compute_metrics(
        self,
        y_rev  : np.ndarray,
        y_cogs : np.ndarray,
        p_rev  : np.ndarray,
        p_cogs : np.ndarray,
    ) -> dict:
        return {
            'revenue_mae'  : mean_absolute_error(y_rev,  p_rev),
            'revenue_rmse' : np.sqrt(mean_squared_error(y_rev,  p_rev)),
            'revenue_r2'   : r2_score(y_rev,  p_rev),
            'cogs_mae'     : mean_absolute_error(y_cogs, p_cogs),
            'cogs_rmse'    : np.sqrt(mean_squared_error(y_cogs, p_cogs)),
            'cogs_r2'      : r2_score(y_cogs, p_cogs),
        }

    def _spike_decomposition(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray,
        quantile: float = 0.90,
    ) -> dict:
        """
        Decompose MAE into spike days (top 10%) and normal days.
        Spike days dominate total MAE -- tracking them separately reveals
        whether model changes help the baseline or the tails.
        """
        threshold  = np.percentile(y_true, quantile * 100)
        spike_mask = y_true >= threshold
        abs_err    = np.abs(y_true - y_pred)

        mae_spike   = float(np.mean(abs_err[spike_mask]))
        mae_normal  = float(np.mean(abs_err[~spike_mask]))
        n_spike     = int(spike_mask.sum())
        pct_from_spikes = (
            mae_spike * n_spike /
            (mae_spike * n_spike + mae_normal * (~spike_mask).sum()) * 100
        )

        print(f"  Spike decomposition (p{int(quantile*100)}):")
        print(f"    Spike days ({n_spike})  MAE: {mae_spike:,.0f}")
        print(f"    Normal days          MAE: {mae_normal:,.0f}")
        print(f"    % of total MAE from spikes: {pct_from_spikes:.1f}%")

        return {
            'threshold'        : threshold,
            'n_spike'          : n_spike,
            'mae_spike'        : mae_spike,
            'mae_normal'       : mae_normal,
            'pct_from_spikes'  : pct_from_spikes,
        }

    def _print_fold_metrics(self, metrics: dict):
        print(f"  Revenue  -- MAE: {metrics['revenue_mae']:>12,.0f}  "
              f"RMSE: {metrics['revenue_rmse']:>12,.0f}  "
              f"R2: {metrics['revenue_r2']:.4f}")
        print(f"  COGS     -- MAE: {metrics['cogs_mae']:>12,.0f}  "
              f"RMSE: {metrics['cogs_rmse']:>12,.0f}  "
              f"R2: {metrics['cogs_r2']:.4f}")

    def _print_summary(self, folds: list):
        print("\n" + "=" * 60)
        print("Summary across folds")
        print("=" * 60)

        df = self.summary()
        if df.empty:
            return

        # Average across completed folds
        numeric_cols = df.select_dtypes(include=np.number).columns
        avg = df[numeric_cols].mean()

        print(df[['fold'] + list(numeric_cols)].to_string(index=False))
        print("\nAverage:")
        for col in numeric_cols:
            print(f"  {col:<20}: {avg[col]:,.4f}")