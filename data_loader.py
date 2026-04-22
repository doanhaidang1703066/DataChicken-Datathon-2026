import pandas as pd
import os


class DataLoader:
    """
    Load toàn bộ dữ liệu + chuẩn hóa schema.
    """

    def __init__(self, raw_data_path="data/raw"):
        self.raw_data_path = raw_data_path

    def _load_csv(self, filename, date_cols=None):
        path = os.path.join(self.raw_data_path, filename)

        if not os.path.exists(path):
            print(f"⚠️ Missing file: {filename}")
            return None

        try:
            df = pd.read_csv(path, parse_dates=date_cols)

            # Normalize column names
            df.columns = [col.strip().lower() for col in df.columns]

            return df

        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None

    def load_all_data(self):
        print("🚀 Loading raw data...")

        data = {}

        # MASTER
        data['products'] = self._load_csv('products.csv')
        data['customers'] = self._load_csv('customers.csv', ['signup_date'])
        data['promotions'] = self._load_csv('promotions.csv', ['start_date', 'end_date'])
        data['geography'] = self._load_csv('geography.csv')

        # TRANSACTION
        data['orders'] = self._load_csv('orders.csv', ['order_date'])
        data['order_items'] = self._load_csv('order_items.csv')
        data['payments'] = self._load_csv('payments.csv')
        data['shipments'] = self._load_csv('shipments.csv', ['ship_date', 'delivery_date'])
        data['returns'] = self._load_csv('returns.csv', ['return_date'])
        data['reviews'] = self._load_csv('reviews.csv', ['review_date'])

        # ANALYTICAL
        data['sales'] = self._load_csv('sales.csv', ['Date'])

        # OPERATIONAL
        data['inventory'] = self._load_csv('inventory.csv', ['snapshot_date'])
        data['web_traffic'] = self._load_csv('web_traffic.csv', ['date'])

        # SAMPLE SUBMISSION (ONLY FOR SUBMISSION, NOT USED IN BACKTEST)
        data['sample_submission'] = self._load_csv('sample_submission.csv')

        loaded = sum(v is not None for v in data.values())
        print(f"✅ Loaded {loaded} tables")

        return data