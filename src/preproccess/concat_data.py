import pandas as pd
import glob
import os

def concat_movie_data(base_path, output_path):
    all_files = glob.glob(f"{base_path}/*/merged_movies_data_*.csv")

    print(os.getcwd())
    print(f"Tìm thấy {len(all_files)} files:")
    for f in all_files:
        print(f"  - {f}")

    if all_files:
        df_list = []
        for file in all_files:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"Đã đọc {file}: {len(df)} dòng")
        
        final_df = pd.concat(df_list, ignore_index=True)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        print(f"Tổng số phim: {len(final_df):,}")
        print(f"File đã lưu: {output_path}")
    else:
        print("Không tìm thấy file nào!")