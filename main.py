from src.preproccess import concat_data, data_cleaning
import os

def main():
    # Gom dữ liệu phim từ các file CSV
    base_path = f"{os.getcwd()}/data/raw/Data/"
    output_path = f"{os.getcwd()}/data/raw/all_movies_data.csv"
    concat_data.concat_movie_data(base_path=base_path, output_path=output_path)
    
    # Làm sạch dữ liệu
if __name__ == "__main__":
    main()