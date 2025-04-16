"""
工具模块 - 提供辅助功能
"""
import warnings
import datetime as dt

# 忽略警告
warnings.filterwarnings('ignore')

def format_log(message, level="INFO"):
    """
    格式化日志消息
    
    参数:
        message (str): 日志消息
        level (str): 日志级别
        
    返回:
        str: 格式化后的日志消息
    """
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[{timestamp}] [{level}] {message}"

def print_info(message):
    """
    打印信息级别日志
    
    参数:
        message (str): 日志消息
    """
    print(format_log(message, "INFO"))

def print_warning(message):
    """
    打印警告级别日志
    
    参数:
        message (str): 日志消息
    """
    print(format_log(message, "WARNING"))

def print_error(message):
    """
    打印错误级别日志
    
    参数:
        message (str): 日志消息
    """
    print(format_log(message, "ERROR"))