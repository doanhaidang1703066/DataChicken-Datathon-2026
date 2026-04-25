import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


class ModelVisualizer:
    """
    Visualize kết quả backtest + phân tích model
    """

    def __init__(self):
        sns.set(style="whitegrid")

    # =========================
    # 1. METRICS TABLE
    # =========================
    def plot_metrics_table(self, results):
        records = []

        for res in results:
            row = {
                'Fold': res['fold'],
                'MAE': res['metrics']['MAE'],
                'RMSE': res['metrics']['RMSE'],
                'R2': res['metrics']['R2']
            }
            records.append(row)

        df_metrics = pd.DataFrame(records)

        print("\n📊 METRICS TABLE:")
        print(df_metrics)

        print("\n📊 AVERAGE PERFORMANCE:")
        print(df_metrics.mean(numeric_only=True).to_frame(name='Average'))

        return df_metrics

    # =========================
    # 2. PLOT ACTUAL vs PRED
    # =========================
    def plot_backtest_results(self, results):
        plt.figure(figsize=(14, 6))

        for res in results:
            df = res['val_df']
            preds = res['preds']

            plt.plot(df['date'], df['revenue'], label=f"Actual Fold {res['fold']}")
            plt.plot(df['date'], preds, '--', label=f"Pred Fold {res['fold']}")

        plt.title("Actual vs Predicted Revenue (Backtest)")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # =========================
    # 3. PLOT EACH FOLD (CLEAR)
    # =========================
    def plot_each_fold(self, results):
        for res in results:
            df   = res['val_df'].copy()
            preds = res['preds']

            # [ADDED] Spike threshold: top 10% of actual revenue in this fold
            spike_threshold = df['revenue'].quantile(0.90)
            spike_mask      = df['revenue'] >= spike_threshold

            fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [3, 1]})

            # --- TOP: Actual vs Predicted ---
            ax = axes[0]
            ax.plot(df['date'], df['revenue'], label='Actual', linewidth=1.5)
            ax.plot(df['date'], preds, '--', label='Predicted',
                    linewidth=1.5, alpha=0.85)

            # [ADDED] Highlight spike days so it's visually obvious which days are missed
            ax.scatter(
                df.loc[spike_mask, 'date'],
                df.loc[spike_mask, 'revenue'],
                color='red', zorder=5, s=20, label=f'Spike days (top 10%)'
            )

            # [ADDED] Shade the MAE band ± around prediction for visual context
            ax.fill_between(
                df['date'],
                preds - np.abs(df['revenue'].values - preds).mean(),
                preds + np.abs(df['revenue'].values - preds).mean(),
                alpha=0.12, color='orange', label='±MAE band'
            )

            ax.set_title(f"Fold {res['fold']} — Actual vs Predicted", fontsize=13)
            ax.set_ylabel("Revenue")
            ax.legend()
            ax.tick_params(axis='x', rotation=45)

            # --- BOTTOM: Absolute Error over time with 30-day rolling ---
            ax2 = axes[1]
            abs_error = np.abs(df['revenue'].values - preds)
            ax2.plot(df['date'], abs_error, alpha=0.4, color='steelblue', label='Daily |Error|')

            # [ADDED] Rolling 30-day mean error — shows if error worsens further into future
            rolling_error = pd.Series(abs_error).rolling(30, min_periods=1).mean().values
            ax2.plot(df['date'], rolling_error, color='darkblue',
                     linewidth=2, label='30-day rolling MAE')

            # [ADDED] Spike day error highlighted in red
            ax2.scatter(df.loc[spike_mask, 'date'], abs_error[spike_mask.values],
                        color='red', zorder=5, s=20, label='Spike day error')

            ax2.set_ylabel("Absolute Error")
            ax2.set_xlabel("Date")
            ax2.legend(fontsize=8)
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()

            # [ADDED] Print spike vs non-spike MAE decomposition per fold
            mae_spike    = np.mean(abs_error[spike_mask.values])
            mae_nonspike = np.mean(abs_error[~spike_mask.values])
            pct_mae_from_spikes = (
                mae_spike * spike_mask.sum() /
                (mae_spike * spike_mask.sum() + mae_nonspike * (~spike_mask).sum()) * 100
            )
            print(f"\n📊 Fold {res['fold']} Spike Decomposition:")
            print(f"   Spike threshold (p90) : {spike_threshold:,.0f}")
            print(f"   Spike days            : {spike_mask.sum()} / {len(df)}")
            print(f"   MAE spike days        : {mae_spike:,.0f}")
            print(f"   MAE non-spike days    : {mae_nonspike:,.0f}")
            print(f"   % of total MAE from spikes: {pct_mae_from_spikes:.1f}%\n")

    # =========================
    # 4. RESIDUAL ANALYSIS
    # =========================
    def plot_residuals(self, results):
        all_residuals = []
        all_actuals   = []

        for res in results:
            df    = res['val_df']
            preds = res['preds']
            all_residuals.extend(df['revenue'].values - preds)
            all_actuals.extend(df['revenue'].values)

        residuals = np.array(all_residuals)
        actuals   = np.array(all_actuals)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --- [ORIGINAL] Residual histogram ---
        axes[0, 0].hist(residuals, bins=60, edgecolor='k', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--')
        axes[0, 0].set_title("Residual Distribution")
        axes[0, 0].set_xlabel("Residual (Actual - Predicted)")

        # --- [REPLACED] Boxplot → QQ plot: reveals fat tails driving spike misses ---
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("QQ Plot — Are Residuals Normal?\n"
                              "(Deviation at tails = fat-tailed errors = spike days)")

        # --- [ADDED] Residual vs Actual — reveals heteroscedasticity ---
        # If errors grow with actual revenue: model underestimates high-revenue days
        axes[1, 0].scatter(actuals, residuals, alpha=0.2, s=10)
        axes[1, 0].axhline(0, color='red', linestyle='--')
        axes[1, 0].set_xlabel("Actual Revenue")
        axes[1, 0].set_ylabel("Residual")
        axes[1, 0].set_title("Residual vs Actual\n"
                              "(Fan shape = model underfits high revenue days)")

        # --- [ADDED] Signed error CDF — shows directional bias clearly ---
        sorted_res = np.sort(residuals)
        cdf        = np.arange(1, len(sorted_res) + 1) / len(sorted_res)
        axes[1, 1].plot(sorted_res, cdf)
        axes[1, 1].axvline(0, color='red', linestyle='--', label='Zero error')
        axes[1, 1].axvline(np.median(residuals), color='orange',
                           linestyle='--', label=f'Median={np.median(residuals):,.0f}')
        axes[1, 1].set_title("Cumulative Error Distribution\n"
                              "(Curve left of 0 = systematic underestimation)")
        axes[1, 1].set_xlabel("Residual")
        axes[1, 1].set_ylabel("CDF")
        axes[1, 1].legend()

        plt.suptitle("Residual Analysis", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.show()

        # [ADDED] Summary stats
        print(f"\n📊 Residual Summary:")
        print(f"   Mean   : {np.mean(residuals):,.0f}  ← should be near 0 (no bias)")
        print(f"   Median : {np.median(residuals):,.0f}  ← robust bias indicator")
        print(f"   Std    : {np.std(residuals):,.0f}")
        print(f"   Skew   : {stats.skew(residuals):.3f}  ← >0 = under-predict right tail")
        print(f"   Kurtosis: {stats.kurtosis(residuals):.3f}  ← >3 = fat tails (spike misses)")

    # =========================
    # 5. ERROR OVER TIME
    # =========================
    def plot_error_over_time(self, results):
        plt.figure(figsize=(14, 6))

        for res in results:
            df    = res['val_df']
            preds = res['preds']
            error = np.abs(df['revenue'].values - preds)

            plt.plot(df['date'], error, alpha=0.3,
                     label=f"Fold {res['fold']} daily |error|")

            # [ADDED] 30-day rolling mean — key signal: does error grow over time?
            rolling = pd.Series(error, index=df['date']).rolling('30D').mean()
            plt.plot(rolling.index, rolling.values, linewidth=2,
                     label=f"Fold {res['fold']} 30D rolling MAE")

        plt.title("Absolute Error over Time\n"
                  "(Rising rolling MAE = model degrades further into forecast horizon)")
        plt.xlabel("Date")
        plt.ylabel("Error")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # =========================
    # 6. PRED vs ACTUAL SCATTER
    # =========================
    def plot_scatter(self, results):
        actual_all = []
        pred_all   = []

        for res in results:
            actual_all.extend(res['val_df']['revenue'].values)
            pred_all.extend(res['preds'])

        actual_all = np.array(actual_all)
        pred_all   = np.array(pred_all)

        # [ADDED] Spike mask for scatter coloring
        spike_threshold = np.percentile(actual_all, 90)
        spike_mask      = actual_all >= spike_threshold

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- [ORIGINAL] Linear scale scatter ---
        axes[0].scatter(actual_all[~spike_mask], pred_all[~spike_mask],
                        alpha=0.2, s=8, label='Normal days')
        axes[0].scatter(actual_all[spike_mask], pred_all[spike_mask],
                        alpha=0.6, s=15, color='red', label='Spike days (top 10%)')
        max_val = max(actual_all.max(), pred_all.max())
        axes[0].plot([0, max_val], [0, max_val], 'k--', label='Perfect fit')
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title("Predicted vs Actual (Linear)\n"
                          "(Red dots below diagonal = spike underestimation)")
        axes[0].legend()

        # --- [ADDED] Log scale scatter — separates compressed low-revenue days ---
        safe_actual = np.maximum(actual_all, 1)
        safe_pred   = np.maximum(pred_all, 1)
        axes[1].scatter(safe_actual[~spike_mask], safe_pred[~spike_mask],
                        alpha=0.2, s=8, label='Normal days')
        axes[1].scatter(safe_actual[spike_mask], safe_pred[spike_mask],
                        alpha=0.6, s=15, color='red', label='Spike days (top 10%)')
        axes[1].plot([safe_actual.min(), safe_actual.max()],
                     [safe_actual.min(), safe_actual.max()], 'k--', label='Perfect fit')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel("Actual (log scale)")
        axes[1].set_ylabel("Predicted (log scale)")
        axes[1].set_title("Predicted vs Actual (Log Scale)\n"
                          "(Shows low-revenue fit quality without spike compression)")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    # =========================
    # 7. SHAP (OPTIONAL)
    # =========================
    def plot_shap_summary(self, model, df):
        import shap

        X = df.copy()

        # bỏ raw cols
        X = X.drop(columns=['date', 'revenue', 'cogs'], errors='ignore')

        # align feature
        X = X.reindex(columns=model.features_used)

        # 🔥 kiểm tra object thật sự
        obj_cols = X.columns[X.dtypes == 'object']
        print("OBJECT COLS:", obj_cols)

        # 🔥 force convert từng cell (quan trọng)
        for c in X.columns:
            X[c] = pd.to_numeric(
                X[c].astype(str).str.replace('[', '').str.replace(']', ''),
                errors='coerce'
            )

        X = X.fillna(0)

        # 1. Lấy booster từ model của bạn
        booster = model.xgb_model.get_booster()

        # 2. Thủ thuật: Ép XGBoost xuất ra mảng đóng góp trực tiếp
        dmatrix = xgb.DMatrix(X)
        raw_shap = booster.predict(dmatrix, pred_contribs=True)

        # 3. Tách biệt SHAP values và Base value
        shap_values    = raw_shap[:, :-1]
        expected_value = raw_shap[0, -1]

        # 4. Tạo đối tượng Explanation "chuẩn"
        explanation = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=X.values,
            feature_names=X.columns.tolist()
        )

        # 5. Beeswarm summary plot
        shap.summary_plot(explanation, X)

        # [ADDED] Mean |SHAP| bar chart — cleaner feature ranking than beeswarm alone
        mean_shap = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X.columns
        ).sort_values(ascending=True).tail(25)

        plt.figure(figsize=(8, 9))
        mean_shap.plot(kind='barh', color='steelblue', edgecolor='k', alpha=0.8)
        plt.xlabel("Mean |SHAP value|")
        plt.title("Top 25 Features by Mean Absolute SHAP\n"
                  "(= average impact on model output, direction-agnostic)")
        plt.tight_layout()
        plt.show()

    # =========================
    # 8. FORECAST PLOT
    # =========================
    def plot_forecast(self,
                      train_df,
                      test_df,
                      model,
                      date_col='date',
                      target_col='revenue',
                      # [ADDED] Only show last N days of training to avoid squashing forecast
                      train_tail_days=365):

        # =========================
        # 1. CLEAN & SORT
        # =========================
        train_df = train_df.sort_values(date_col)
        test_df  = test_df.sort_values(date_col)

        # =========================
        # 2. FORECAST
        # =========================
        test_pred = model.predict(test_df)

        # [ADDED] Tail of training for visual continuity
        train_tail = train_df.tail(train_tail_days)

        # =========================
        # 3. PLOT
        # =========================
        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={'height_ratios': [3, 1]})

        # --- TOP: Full timeline ---
        ax = axes[0]
        ax.plot(train_tail[date_col], train_tail[target_col],
                label=f'Actual Revenue (last {train_tail_days}d)',
                linewidth=1.5, color='steelblue')
        ax.plot(test_df[date_col], test_pred,
                label='Forecast (1.5Y)', linewidth=2, color='darkorange')

        # SPLIT LINE
        ax.axvline(train_df[date_col].max(), color='red',
                   linestyle='--', label='Forecast Start')

        # [ADDED] Shade forecast region for visual clarity
        ax.axvspan(test_df[date_col].min(), test_df[date_col].max(),
                   alpha=0.05, color='orange')

        ax.set_title("Revenue Forecast: Actual History vs Future Prediction", fontsize=13)
        ax.set_ylabel("Revenue")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        # --- BOTTOM: Monthly aggregated forecast ---
        # [ADDED] Aggregating to monthly removes daily noise and shows macro trend clearly
        test_df_plot = test_df[[date_col]].copy()
        test_df_plot['forecast'] = test_pred
        test_df_plot['month']    = test_df_plot[date_col].dt.to_period('M')
        monthly_forecast = test_df_plot.groupby('month')['forecast'].sum()

        ax2 = axes[1]
        ax2.bar(
            range(len(monthly_forecast)),
            monthly_forecast.values,
            color='darkorange', alpha=0.8, edgecolor='k'
        )
        ax2.set_xticks(range(len(monthly_forecast)))
        ax2.set_xticklabels(
            [str(p) for p in monthly_forecast.index],
            rotation=45, ha='right', fontsize=8
        )
        ax2.set_title("Monthly Aggregated Forecast\n"
                      "(Sanity check: does the monthly total look reasonable?)")
        ax2.set_ylabel("Monthly Revenue")

        plt.tight_layout()
        plt.show()

        # [ADDED] Print forecast summary statistics
        print(f"\n📊 Forecast Summary:")
        print(f"   Period         : {test_df[date_col].min().date()} → {test_df[date_col].max().date()}")
        print(f"   Days forecast  : {len(test_df)}")
        print(f"   Total forecast : {test_pred.sum():,.0f}")
        print(f"   Daily avg      : {test_pred.mean():,.0f}")
        print(f"   Daily max      : {test_pred.max():,.0f}")
        print(f"   Daily min      : {test_pred.min():,.0f}")
        print(f"   Days with pred=0: {(test_pred == 0).sum()}")
