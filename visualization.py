import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats


class ModelVisualizer:
    """
    Visualization suite for backtest results and long-horizon forecasts.

    All methods accept the results dict produced by WalkForwardBacktester
    or the forecast DataFrame produced by EnsembleForecaster.predict().
    """

    def __init__(self):
        sns.set_style('whitegrid')
        plt.rcParams.update({
            'font.size'      : 11,
            'axes.titlesize' : 12,
            'axes.labelsize' : 11,
        })

    # ==========================================================================
    # 1. METRICS TABLE
    # ==========================================================================

    def plot_metrics_table(self, results: dict):
        """
        Print and return a DataFrame of metrics across all completed folds.
        """
        records = []
        for fold_key, res in results.items():
            row = {'fold': fold_key}
            row.update(res['metrics'])
            records.append(row)

        df = pd.DataFrame(records)
        print("\nMetrics across folds:")
        print(df.to_string(index=False))
        print("\nAverage:")
        print(df.select_dtypes(include=np.number).mean().to_frame('average').T.to_string())
        return df

    # ==========================================================================
    # 2. PLOT EACH FOLD
    # ==========================================================================

    def plot_each_fold(self, results: dict):
        """
        For each fold: actual vs predicted with MAE band, absolute error panel,
        and spike decomposition printed to console.
        """
        for fold_key, res in results.items():
            val_df = res['val_df'].copy()
            preds  = res['preds']

            actual = val_df['revenue'].values
            pred   = preds['revenue'].values
            dates  = pd.to_datetime(val_df['date'].values)

            spike_threshold = np.percentile(actual, 90)
            spike_mask      = actual >= spike_threshold
            abs_error       = np.abs(actual - pred)
            mean_mae        = abs_error.mean()

            fig = plt.figure(figsize=(15, 9))
            gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

            # --- Top panel: actual vs predicted ---
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(dates, actual, linewidth=1.5, label='Actual',    color='steelblue')
            ax1.plot(dates, pred,   linewidth=1.5, label='Predicted',
                     color='darkorange', linestyle='--', alpha=0.9)
            ax1.scatter(dates[spike_mask], actual[spike_mask],
                        color='red', s=18, zorder=5, label='Spike days (top 10%)')
            ax1.fill_between(dates,
                             pred - mean_mae,
                             pred + mean_mae,
                             alpha=0.10, color='darkorange', label='Mean MAE band')
            ax1.set_title(f"Fold {fold_key} -- Actual vs Predicted Revenue")
            ax1.set_ylabel("Revenue")
            ax1.legend(fontsize=9)
            ax1.tick_params(axis='x', rotation=30)

            # --- Bottom panel: absolute error + 30-day rolling ---
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(dates, abs_error, alpha=0.35, color='steelblue', label='Daily |error|')
            rolling = pd.Series(abs_error, index=dates).rolling('30D').mean()
            ax2.plot(rolling.index, rolling.values, color='navy',
                     linewidth=2, label='30-day rolling MAE')
            ax2.scatter(dates[spike_mask], abs_error[spike_mask],
                        color='red', s=18, zorder=5, label='Spike day error')
            ax2.set_ylabel("Absolute Error")
            ax2.set_xlabel("Date")
            ax2.legend(fontsize=8)
            ax2.tick_params(axis='x', rotation=30)

            plt.suptitle(f"Fold {fold_key} Diagnostic", fontsize=13, y=1.01)
            plt.tight_layout()
            plt.show()

            # Spike decomposition printed below each plot
            print(f"Fold {fold_key} spike decomposition:")
            print(f"  Spike threshold (p90) : {spike_threshold:,.0f}")
            print(f"  Spike days            : {spike_mask.sum()} / {len(actual)}")
            print(f"  MAE spike days        : {abs_error[spike_mask].mean():,.0f}")
            print(f"  MAE normal days       : {abs_error[~spike_mask].mean():,.0f}")
            pct = (abs_error[spike_mask].sum() /
                   abs_error.sum() * 100)
            print(f"  % total MAE from spikes: {pct:.1f}%\n")

    # ==========================================================================
    # 3. RESIDUAL ANALYSIS
    # ==========================================================================

    def plot_residuals(self, results: dict):
        """
        Four-panel residual analysis across all folds combined:
            - Residual histogram
            - QQ plot
            - Residual vs actual (heteroscedasticity)
            - Cumulative error distribution (bias direction)
        """
        all_residuals = []
        all_actuals   = []

        for res in results.values():
            actual = res['val_df']['revenue'].values
            pred   = res['preds']['revenue'].values
            all_residuals.extend(actual - pred)
            all_actuals.extend(actual)

        residuals = np.array(all_residuals)
        actuals   = np.array(all_actuals)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Histogram
        axes[0, 0].hist(residuals, bins=60, edgecolor='k', alpha=0.75)
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero error')
        axes[0, 0].axvline(np.median(residuals), color='orange',
                           linestyle='--', label=f'Median={np.median(residuals):,.0f}')
        axes[0, 0].set_title("Residual Distribution")
        axes[0, 0].set_xlabel("Residual (Actual - Predicted)")
        axes[0, 0].legend(fontsize=9)

        # QQ plot
        stats.probplot(residuals, dist='norm', plot=axes[0, 1])
        axes[0, 1].set_title("QQ Plot\n"
                              "(Deviation at tails = fat-tailed spike errors)")

        # Residual vs actual
        axes[1, 0].scatter(actuals, residuals, alpha=0.15, s=8)
        axes[1, 0].axhline(0, color='red', linestyle='--')
        axes[1, 0].set_xlabel("Actual Revenue")
        axes[1, 0].set_ylabel("Residual")
        axes[1, 0].set_title("Residual vs Actual\n"
                              "(Fan shape = model underfits high revenue days)")

        # CDF
        sorted_res = np.sort(residuals)
        cdf        = np.arange(1, len(sorted_res) + 1) / len(sorted_res)
        axes[1, 1].plot(sorted_res, cdf)
        axes[1, 1].axvline(0, color='red',    linestyle='--', label='Zero error')
        axes[1, 1].axvline(np.median(residuals), color='orange',
                           linestyle='--',
                           label=f'Median={np.median(residuals):,.0f}')
        axes[1, 1].set_title("Cumulative Error Distribution\n"
                              "(Curve left of 0 = systematic underestimation)")
        axes[1, 1].set_xlabel("Residual")
        axes[1, 1].set_ylabel("CDF")
        axes[1, 1].legend(fontsize=9)

        plt.suptitle("Residual Analysis -- All Folds Combined", fontsize=13)
        plt.tight_layout()
        plt.show()

        print("Residual summary (all folds):")
        print(f"  Mean     : {np.mean(residuals):,.0f}")
        print(f"  Median   : {np.median(residuals):,.0f}")
        print(f"  Std      : {np.std(residuals):,.0f}")
        print(f"  Skew     : {stats.skew(residuals):.3f}")
        print(f"  Kurtosis : {stats.kurtosis(residuals):.3f}")

    # ==========================================================================
    # 4. SCATTER PLOT
    # ==========================================================================

    def plot_scatter(self, results: dict):
        """
        Linear and log-scale scatter of predicted vs actual revenue.
        Spike days highlighted in red.
        """
        all_actual = []
        all_pred   = []

        for res in results.values():
            all_actual.extend(res['val_df']['revenue'].values)
            all_pred.extend(res['preds']['revenue'].values)

        actual = np.array(all_actual)
        pred   = np.array(all_pred)

        spike_threshold = np.percentile(actual, 90)
        spike_mask      = actual >= spike_threshold

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, log_scale in zip(axes, [False, True]):
            a = np.maximum(actual, 1) if log_scale else actual
            p = np.maximum(pred,   1) if log_scale else pred

            ax.scatter(a[~spike_mask], p[~spike_mask],
                       alpha=0.2, s=8,  label='Normal days', color='steelblue')
            ax.scatter(a[spike_mask],  p[spike_mask],
                       alpha=0.6, s=15, label='Spike days (top 10%)', color='red')

            lim = [min(a.min(), p.min()), max(a.max(), p.max())]
            ax.plot(lim, lim, 'k--', linewidth=1, label='Perfect fit')

            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_title("Predicted vs Actual (Log Scale)\n"
                             "(Reveals low-revenue fit quality)")
            else:
                ax.set_title("Predicted vs Actual (Linear)\n"
                             "(Red dots below diagonal = spike underestimation)")

            ax.set_xlabel("Actual Revenue")
            ax.set_ylabel("Predicted Revenue")
            ax.legend(fontsize=9)

        plt.tight_layout()
        plt.show()

    # ==========================================================================
    # 5. SHAP SUMMARY
    # ==========================================================================

    def plot_shap_summary(self, lgb_booster, feat_df: pd.DataFrame):
        """
        SHAP beeswarm + mean absolute SHAP bar chart for a fitted LGB booster.
        Pass the LGB booster directly (e.g. fold_model._lgb_base_rev).
        """
        import shap
        import lightgbm as lgb

        exclude = {'date', 'revenue', 'cogs'}
        feature_cols = [c for c in feat_df.columns if c not in exclude]
        X = feat_df[feature_cols].copy().fillna(0)

        explainer  = shap.TreeExplainer(lgb_booster)
        shap_values = explainer.shap_values(X)

        # Beeswarm
        explanation = shap.Explanation(
            values        = shap_values,
            base_values   = explainer.expected_value,
            data          = X.values,
            feature_names = feature_cols,
        )
        shap.summary_plot(explanation, X, show=True)

        # Mean absolute SHAP bar chart
        mean_shap = (
            pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
            .sort_values(ascending=True)
            .tail(25)
        )

        fig, ax = plt.subplots(figsize=(9, 9))
        mean_shap.plot(kind='barh', ax=ax, color='steelblue',
                       edgecolor='k', alpha=0.8)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top 25 Features by Mean Absolute SHAP\n"
                     "(Direction-agnostic feature importance)")
        plt.tight_layout()
        plt.show()

    # ==========================================================================
    # 6. FORECAST PLOT
    # ==========================================================================

    def plot_forecast(
        self,
        train_df       : pd.DataFrame,
        feat_test      : pd.DataFrame,
        forecast_df    : pd.DataFrame,
        date_col       : str   = 'date',
        target_col     : str   = 'revenue',
        train_tail_days: int   = 365,
    ):
        """
        Two-panel forecast visualization:
            Top    -- training tail + forecast line with shaded forecast region
            Bottom -- monthly aggregated forecast bar chart for sanity check
        """
        train_tail = train_df.sort_values(date_col).tail(train_tail_days)

        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.4)

        # --- Top: full timeline ---
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(
            pd.to_datetime(train_tail[date_col]),
            train_tail[target_col],
            linewidth=1.5, color='steelblue',
            label=f'Actual Revenue (last {train_tail_days} days)',
        )
        ax1.plot(
            pd.to_datetime(forecast_df[date_col]),
            forecast_df[target_col],
            linewidth=2.0, color='darkorange',
            label='Forecast (1.5 years)',
        )
        ax1.axvline(
            pd.to_datetime(train_df[date_col].max()),
            color='red', linestyle='--', linewidth=1.2, label='Forecast start',
        )
        ax1.axvspan(
            pd.to_datetime(forecast_df[date_col].min()),
            pd.to_datetime(forecast_df[date_col].max()),
            alpha=0.05, color='darkorange',
        )
        ax1.set_title("Revenue Forecast -- Training Tail and 1.5-Year Horizon")
        ax1.set_ylabel("Revenue")
        ax1.legend(fontsize=10)
        ax1.tick_params(axis='x', rotation=30)

        # --- Bottom: monthly aggregated bar ---
        monthly = (
            forecast_df.copy()
            .assign(month=pd.to_datetime(forecast_df[date_col]).dt.to_period('M'))
            .groupby('month')[target_col]
            .sum()
        )

        ax2 = fig.add_subplot(gs[1])
        ax2.bar(
            range(len(monthly)), monthly.values,
            color='darkorange', alpha=0.8, edgecolor='k', linewidth=0.5,
        )
        ax2.set_xticks(range(len(monthly)))
        ax2.set_xticklabels(
            [str(p) for p in monthly.index],
            rotation=45, ha='right', fontsize=8,
        )
        ax2.set_title("Monthly Aggregated Forecast (Sanity Check)")
        ax2.set_ylabel("Monthly Revenue")

        plt.suptitle("Final Forecast", fontsize=14)
        plt.tight_layout()
        plt.show()

        print("Forecast summary:")
        print(f"  Period     : {forecast_df[date_col].min()} -> "
              f"{forecast_df[date_col].max()}")
        print(f"  Days       : {len(forecast_df)}")
        print(f"  Total rev  : {forecast_df[target_col].sum():,.0f}")
        print(f"  Daily avg  : {forecast_df[target_col].mean():,.0f}")
        print(f"  Daily max  : {forecast_df[target_col].max():,.0f}")
        print(f"  Daily min  : {forecast_df[target_col].min():,.0f}")
        print(f"  Zero days  : {(forecast_df[target_col] == 0).sum()}")