import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

# Load dữ liệu thô
def clean_data(file_path, output_path):
    df = pd.read_csv(file_path)
    initial_rows = df.shape[0]
    print(f"Số dòng ban đầu: {initial_rows}")

    # Chuẩn hóa tên cột
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.rename(columns={"méta_score": "meta_score"})

    # Nhóm các cột để xử lý
    money_cols = ["budget", "opening_weekend_gross", "grossworldwwide", "gross_us_canada"]

    categorical_cols = [
        "mpa", "countries_origin", "production_company",
        "genres", "languages", "stars"
    ]

    text_cols = ["description", "awards_content", "filming_locations"]

    # XỬ LÝ VOTES (chuyển K, M thành số)
    def convert_votes(v):
        if pd.isna(v):
            return np.nan
        v = str(v).upper()
        if "K" in v:
            return float(v.replace("K", "")) * 1_000
        if "M" in v:
            return float(v.replace("M", "")) * 1_000_000
        return float(v)

    df["votes"] = df["votes"].apply(convert_votes)

    # XỬ LÝ CỘT TIỀN TỆ (loại bỏ ký tự không phải số)
    for col in money_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ép kiểu numeric cho các cột cần thiết
    for col in ["year", "rating", "meta_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Loại bỏ các dòng thiếu year hoặc rating (cốt lõi)
    df = df.dropna(subset=["year", "rating"])

    # Điền missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    for col in text_cols:
        df[col] = df[col].fillna("No information")

    # LOG TRANSFORM để giảm skewness (giữ cột gốc)
    log_cols = [
        "votes", "budget",
        "opening_weekend_gross",
        "grossworldwwide", "gross_us_canada"
    ]

    for col in log_cols:
        df[col] = df[col].clip(lower=0)  # tránh log(0)
        df[f"{col}_log"] = np.log1p(df[col])

    # XỬ LÝ OUTLIERS bằng Winsorization (capping tại 1% và 99%)
    cols_to_winsorize = [
        'votes_log',
        'budget_log',
        'opening_weekend_gross_log',
        'grossworldwwide_log',
        'gross_us_canada_log',
        'rating',
        'meta_score'
    ]

    for col in cols_to_winsorize:
        if col in df.columns:
            df[col] = winsorize(df[col], limits=[0.01, 0.01])


    # KIỂM TRA SAU XỬ LÝ
    final_rows = df.shape[0]
    print(f"\nSố dòng sau tiền xử lý: {final_rows}")
    print(f"Số dòng bị loại bỏ: {initial_rows - final_rows}")

    print("\nGiá trị khuyết sau xử lý (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    # LƯU FILE CLEAN
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu dữ liệu sạch tại: {output_path}")