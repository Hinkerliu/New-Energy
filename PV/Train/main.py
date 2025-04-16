from feature_engineering import create_feature_dataset
from model import train_lightgbm_model, predict_with_lightgbm
from data_utils import get_meteo_for_day
from evaluation import evaluate_model
from submission import generate_submission_files
from config import OUTPUT_PATH
import os
import pandas as pd

def main():
    solar_stations = [6, 7, 8, 9, 10]
    train_start_date = '2024-01-01'
    train_end_date = '2024-11-29'
    test_start_date = '2024-11-30'
    test_end_date = '2024-12-30'
    results = []
    models_dict = {}
    for station_id in solar_stations:
        print(f"\n开始处理光伏场站 {station_id}...")
        features_df = create_feature_dataset(station_id, train_start_date, train_end_date)
        if features_df is not None:
            models = train_lightgbm_model(features_df, station_id)
            if models:
                models_dict[station_id] = models
                eval_result = evaluate_model(
                    station_id, models, test_start_date, test_end_date,
                    predict_func=predict_with_lightgbm,
                    get_meteo_for_day_func=get_meteo_for_day
                )
                if eval_result:
                    results.append(eval_result)
    if results:
        results_df = pd.DataFrame(results)
        print("\n所有光伏场站评估结果:")
        print(results_df)
        avg_accuracy = results_df['avg_daily_accuracy'].mean()
        print(f"\n所有光伏场站平均准确率: {avg_accuracy:.4f}")
        results_df.to_csv(os.path.join(OUTPUT_PATH, "solar_stations_results.csv"), index=False)
    submission_dir = os.path.join(OUTPUT_PATH, "submission")
    if models_dict:
        generate_submission_files(
            solar_stations, '20241230', models_dict, submission_dir,
            predict_func=predict_with_lightgbm,
            get_meteo_for_day_func=get_meteo_for_day
        )

if __name__ == "__main__":
    main()