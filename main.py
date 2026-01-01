from src.preproccess import concat_data, data_cleaning
import os
import pandas as pd

def main():
    # Gom dữ liệu phim từ các file CSV
    base_path = f"{os.getcwd()}/data/raw/Data/"
    output_path = f"{os.getcwd()}/data/raw/all_movies_data.csv"
    concat_data.concat_movie_data(base_path=base_path, output_path=output_path)
    
    df = pd.read_csv(output_path) 
    # (Khuyến nghị) ép kiểu Year để tránh lỗi
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Lọc dữ liệu theo năm (1975–2025)
    df = df[(df["Year"] >= 1975) & (df["Year"] <= 2025)]

    # Lưu lại dữ liệu sau khi lọc
    filtered_output_path = f"{os.getcwd()}/data/raw/all_movies_data_1975_2025.csv"
    df.to_csv(filtered_output_path, index=False)

    
if __name__ == "__main__":
    main()