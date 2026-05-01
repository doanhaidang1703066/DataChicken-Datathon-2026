# VinTelligence Datathon 2026 — The Gridbreakers (Vòng 1)

**Bài toán**: Dự báo doanh thu và COGS hàng ngày, giai đoạn 01/01/2023 – 01/07/2024
 https://www.kaggle.com/competitions/datathon-2026-round-1
 
**Đội thi**: _**Data Chicken**_

**Thành viên**: Doãn Hải Đăng · Lê Trọng Đạt · Vũ Xuân Duy Anh · Phạm Lê Hải Nam

---

## Nên Xem Gì Trước

| Thứ tự | File | Nội dung |
|--------|------|---------|
| 1 | [`report/report.pdf`](./report/report.pdf) | toàn bộ phân tích, kết quả, methodology trong 4 trang nội dung chính |
| 2 | [`submission/submission.csv`](./submission/submission.csv) | file nộp Kaggle, 548 dòng dự đoán, giữ đúng format `sample_submission.csv` |
| 3 | [`notebooks/mcq-answers.ipynb`](./notebooks/mcq-answers.ipynb) | notebook chứa code và đáp án Part 1 theo đề cập nhật |
| 4 | [`notebooks/eda.ipynb`](./notebooks/eda.ipynb) | notebook chứa code EDA của Part 2, là nguồn của các phân tích trong Báo cáo |
| 5 | [`notebooks/vin-datathon-time-forecasting.ipynb`](./notebooks/vin-datathon-time-forecasting.ipynb) | Pipeline Part 3: feature engineering, model ensemble, walkforward backtesting, SHAP, submission export |
| 6 | [`notebooks/time-forecasting-ablation-study.ipynb`](./notebooks/time-forecasting-ablation-study.ipynb) | Ablation Study: Bổ sung 2 Pretrained Zero-shot Foundation Model: Chronos-2 và TimesFM 2.5 |

---

## Sơ Đồ Quan Hệ Bảng

![ERD giản lược](ERD.png)

---

## Cấu trúc thư mục
```
DATACHICKEN-DATATHON-2026/
├── data/
│   ├── sales.csv                   # train target — 3,833 dòng, 2012-07-04 → 2022-12-31
│   ├── sample_submission.csv       # template submission — 548 dòng, 2023-01-01 → 2024-07-01
│   └── *.csv                       # orders, products, customers, payments, ...
├── src/                            # source code chứa các class có trong notebook
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── validation.py
│   ├── visualization.py
├── notebooks/
│   ├── mcq-answers.ipynb                           #Part 1 Answers
│   ├── eda.ipynb                                   #Part 2 EDA  
│   ├── baseline.ipynb
│   ├── vin-datathon-time-forecasting.ipynb         # Final Submission
│   ├── time-forecasting-ablation-study.ipynb       # Ablation Study
├── Submission/
│   └── submission.csv
├── Report/
│   └── report.pdf
│   └── SHAP_dashboard.png
│   └── *
├── requirements.txt
└── README.md

```

## Nội dung Notebook
| Notebook | Mục đích |
| :--- | :--- |
| mcq-answers.ipynb | File trả lời câu hỏi trắc nghiệm Part 1 |
| eda.ipynb | File phân tích dữ liệu thăm dò (EDA) Part 2 |
| baseline.ipynb | File baseline gốc của chương trình |
| vin-datathon-time-forecasting.ipynb   | File model sinh final submission |
| time-forecasting-ablation-study.ipynb  | File chứa ý tưởng cải thiện model theo hướng Foundation Model Ensemble |

---


## Kiến trúc mô hình
Mô hình chính tuân theo **Ensemble method** bằng cách kết hợp các mô hình học máy và foundation model có kiến trúc khác nhau để tạo ra sự đa dạng gồm:

| Model | Training Interval | Ý nghĩa |
| :--- | :--- | :--- |
| Ridge Regression | Full history | Mô hình tuyến tính tham số học một hàm tuyến tính toàn cục trên feature space. Ổn định, ít bị nhiễu bởi outlier/regime shift, đóng vai trò là 'mỏ neo' giữ cho mô hình không bị overshoot |
|  LightGBM  | Full history (chú trọng 2014-2018) | Mô hình cây gradient boosting phân chia feature space giúp nắm được tương tác giữa các feature. Mô hình sử dụng 1 LGB base + 4 LGB quarterly specialists nhằm nắm bắt đặc trưng và mức độ biến động riêng theo từng quý (đặc biệt là Q3) |
| Prophet  | Post-2019 | Mô hình phân rã có cấu trúc cộng tính: Trend (piecewise linear có changepoint) + Seasonality (Fourier đa tần số) + Noise (Holidays), rất mạnh ở việc extrapolate seasonal component cho horizon dài. Tuy nhiên piecewise trend rất nhạy cảm với jump lớn (ví dụ bước nhảy 2019–2020 do COVID) |
| Chronos | Post-2019 | Encoder-based foundation model chạy zero-shot với known future covariates (promo flags, Tet, is_odd_year, EOM flags) để học seasonality structure độc lập với revenue level, giúp nhìn được bức tranh toàn cục theo 2 hướng |
| TimesFM | Full history | Decoder-based foundation model chỉ handle đơn biến revenue, chạy zero-shot chỉ dựa trên dữ liệu revenue trong quá khứ để xây dựng tất cả trend, level, seasonality,..  |

---

## Feature engineering
- Calendar basics: year, month, day, dayofweek, dayofyear, quarter, is_weekend, is_odd_year, dim (days in month)

- Regime flags: regime_pre2019, regime_2019, regime_post2019, t_days (centered at 2020-01-01), t_years

- Fourier features: Annual (k=1..5 : sin_y1..5, cos_y1..5), Weekly (k=1..2 : sin_w1..2, cos_w1..2), Monthly (k=1..2 : sin_m1..2, cos_m1..2)

- End-of-month / start-of-month: days_to_eom, days_from_som, k=1..3 : is_last1..3, is_first1..3

- Tet features: tet_days_diff, tet_in_7, tet_in_14, tet_before_7, tet_after_7, tet_on sử dụng Lookup table thủ công ( vì ngày Tết mỗi năm không có định vào 1 ngày dương lịch)

- Vietnamese holidays + Sale days: new_year(1/1), womens_day(3/8), national_day(9/2), dd_1111(11/11), black_friday(last Fri of Nov), etc.

- Promo windows: Dựa vào dữ liệu trong **data/promotions.csv** về các chiến dịch Sales trong quá khứ như spring_sale, mid_year, etc. Với mỗi chiến dịch ta có các features: promo_{name}, promo_{name}_since, promo_{name}_until, promo_{name}_disc


---

## Ablation Study
- Ý tưởng: Bổ sung thêm 2 Foundation Model là **Chronos-2** và **TimesFM 2.5** ensemble vào 3 model cơ bản nhằm tăng thêm sự đa dạng trong kiến trúc mô hình, từ đó có khả năng cải thiện được kết quả của mô hình.

- Kết quả: Khi thử thêm 2 FM vào với weight trong khoảng 0.0-0.15 với sự hy sinh weight của LGB blend, kết quả MAE của mô hình ensemble mới có sự tăng giảm không đáng kể so với mô hình cũ (~ 0.5-5k MAE). 

- Kết luận và hành động: LGB đóng vai trò quan trọng trong kiến trúc mô hình và đưa ra prediction tốt, nên giữ weight và tạm thời không sử dụng đến 2 Foundation Model để bảo toàn sự đơn giản và hiệu quả của mô hình hiện tại.

---


## Chạy repo (Reproduce)
### Chạy trên Kaggle
1. Import notebook [`vin-datathon-time-forecasting.ipynb `](http://github.com/doanhaidang1703066/DataChicken-Datathon-2026/blob/main/notebooks/vin-datathon-time-forecasting.ipynb) trong cuộc thi trên Kaggle,
2. Sau đó chạy "Run all cells" 
3. Khi chạy xong tất cả các cells, ta có được file submission.csv để nộp:

   ```
   /kaggle/working/submission.csv
   ```

---

## Submission
- File [`submission.csv`](https://github.com/doanhaidang1703066/DataChicken-Datathon-2026/blob/main/submission/submission.csv) nằm trong thư mục *submission*

Source code and results available at: https://github.com/doanhaidang1703066/DataChicken-Datathon-2026
