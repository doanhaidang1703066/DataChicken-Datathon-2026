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
            df = res['val_df']
            preds = res['preds']

            plt.figure(figsize=(12, 5))
            plt.plot(df['date'], df['revenue'], label='Actual')
            plt.plot(df['date'], preds, '--', label='Predicted')

            plt.title(f"Fold {res['fold']} - Actual vs Pred")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # =========================
    # 4. RESIDUAL ANALYSIS
    # =========================
    def plot_residuals(self, results):
        all_residuals = []

        for res in results:
            df = res['val_df']
            preds = res['preds']

            residuals = df['revenue'].values - preds
            all_residuals.extend(residuals)

        residuals = np.array(all_residuals)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(residuals, bins=50)
        plt.title("Residual Distribution")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=residuals)
        plt.title("Residual Boxplot")

        plt.tight_layout()
        plt.show()

    # =========================
    # 5. ERROR OVER TIME
    # =========================
    def plot_error_over_time(self, results):
        plt.figure(figsize=(14, 6))

        for res in results:
            df = res['val_df']
            preds = res['preds']

            error = np.abs(df['revenue'] - preds)

            plt.plot(df['date'], error, label=f"Fold {res['fold']}")

        plt.title("Absolute Error over Time")
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
        pred_all = []

        for res in results:
            actual_all.extend(res['val_df']['revenue'].values)
            pred_all.extend(res['preds'])

        plt.figure(figsize=(6, 6))
        plt.scatter(actual_all, pred_all, alpha=0.3)

        max_val = max(max(actual_all), max(pred_all))
        plt.plot([0, max_val], [0, max_val])  # diagonal

        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual Scatter")
        plt.tight_layout()
        plt.show()

    # =========================
    # 7. SHAP (OPTIONAL)
    # =========================
    def plot_shap_summary(self, model, df):

        import shap

        X = df.copy()

        # bỏ raw cols
        X = X.drop(columns=['date','revenue','cogs'], errors='ignore')

        # align feature
        X = X.reindex(columns=model.features_used)

        # 🔥 kiểm tra object thật sự
        obj_cols = X.columns[X.dtypes == 'object']
        print("OBJECT COLS:", obj_cols)

        # 🔥 force convert từng cell (quan trọng)
        for c in X.columns:
            X[c] = pd.to_numeric(X[c].astype(str).str.replace('[','').str.replace(']',''), errors='coerce')

        X = X.fillna(0)

        # 1. Lấy booster từ model của bạn
        booster = model.xgb_model.get_booster()

        # 2. Thủ thuật: Ép XGBoost xuất ra mảng đóng góp trực tiếp
        # Cách này bỏ qua bước SHAP tự đọc file JSON lỗi thời
        dmatrix = xgb.DMatrix(X)
        # pred_contribs=True trả về: [SHAP values..., Expected Value]
        raw_shap = booster.predict(dmatrix, pred_contribs=True)

        # 3. Tách biệt SHAP values và Base value (Expected Value)
        # Cột cuối cùng của raw_shap chính là giá trị base_score gây lỗi lúc nãy
        shap_values = raw_shap[:, :-1]
        expected_value = raw_shap[0, -1] 

        # 4. Tạo đối tượng Explanation "chuẩn" để vẽ Summary Plot
        # Đây là cách code "sạch" nhất, không phụ thuộc vào việc SHAP có đọc được JSON hay không
        explanation = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=X.values,
            feature_names=X.columns.tolist()
        )

        # 5. Vẽ đồ thị
        shap.summary_plot(explanation, X)