from submission import generate_submission_files
import pandas as pd

def main():
    # 光伏场站ID
    solar_stations = [6, 7, 8, 9, 10]
    # 预测日期范围
    start_date = pd.to_datetime('20250101', format='%Y%m%d')
    end_date = pd.to_datetime('20250228', format='%Y%m%d')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    prediction_dates = [date.strftime('%Y%m%d') for date in date_range]

    print(f"开始为{len(prediction_dates)}天生成预测结果")
    print(f"预测日期范围: {prediction_dates[0]} 至 {prediction_dates[-1]}")

    # 生成所有日期的预测结果，使用相对路径
    output_dir = r"..\..\output"  # 修改为正确的相对路径
    generate_submission_files(solar_stations, prediction_dates, output_dir)

    print("所有预测任务已完成。")

if __name__ == "__main__":
    main()