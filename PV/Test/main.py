"""
主程序入口 - 光伏发电预测系统
"""
import pandas as pd
from submission import generate_submission_files
from config import SOLAR_STATIONS, START_DATE, END_DATE, OUTPUT_PATH

def main():
    """主函数"""
    # 创建预测日期列表
    start_date = pd.to_datetime(START_DATE, format='%Y%m%d')
    end_date = pd.to_datetime(END_DATE, format='%Y%m%d')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    prediction_dates = [date.strftime('%Y%m%d') for date in date_range]
    
    print("=" * 50)
    print("光伏发电预测系统")
    print("=" * 50)
    print(f"预测站点: {SOLAR_STATIONS}")
    print(f"预测日期范围: {prediction_dates[0]} 至 {prediction_dates[-1]} (共 {len(prediction_dates)} 天)")
    print("=" * 50)
    
    # 生成所有日期的预测结果
    output_zip = generate_submission_files(SOLAR_STATIONS, prediction_dates, OUTPUT_PATH)
    
    print("=" * 50)
    print(f"预测完成! 提交文件: {output_zip}")
    print("=" * 50)

if __name__ == "__main__":
    main()