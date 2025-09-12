
import pandas as pd
import sys
import os
pd.set_option('display.max_rows', None)  # None 表示显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
# pd.set_option('display.width', 1000)  # 设置合适的宽度值
pd.set_option('display.max_colwidth', None)

def read_parquet_file(file_path):
    """
    读取.parquet文件并显示基本信息
    
    Args:
        file_path (str): .parquet文件路径
    """
    try:
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        
        print(f"文件路径: {file_path}")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("\n前5行数据:")
        import pdb;pdb.set_trace()
        print(df.head(1))
        
        # print(f"\n数据类型:")
        # print(df.dtypes)
        
        # print(f"\n基本统计信息:")
        # print(df.describe())
        
        return df
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def main():
    # 如果没有指定文件路径，使用默认文件
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 使用第一个找到的parquet文件作为示例
        # file_path = "/data0/jzzhang/verl/datasets/math500/test.parquet"
        # file_path="/data0/jzzhang/verl/datasets/NuminaMath-CoT/data/train-00000-of-00005.parquet" # 171k
        # file_path="/data0/jzzhang/verl/datasets/NuminaMath-CoT/data/train-00001-of-00005.parquet" # 171k
        # file_path="/data0/jzzhang/verl/datasets/NuminaMath-CoT/data/train-00002-of-00005.parquet" # 171k
        # file_path="/data0/jzzhang/verl/datasets/NuminaMath-CoT/data/train-00003-of-00005.parquet" # 171k
        # file_path="/data0/jzzhang/verl/datasets/NuminaMath-CoT/data/train-00004-of-00005.parquet" # 171k
        # file_path="/data0/jzzhang/verl/datasets/NuminaMath-CoT/data/test-00000-of-00001.parquet" # 100
        # file_path="/data0/jzzhang/verl/datasets/NuminaMath-CoT/NuminaMath171k_test_math_format.parquet"
        # file_path="/data0/jzzhang/verl/datasets/NuminaMath-CoT/NuminaMath171k_SFT_math_format.parquet"
        # file_path="/data0/jzzhang/verl/datasets/aime25/aime2025_math_format_repeat3.parquet"
        # file_path="/data0/jzzhang/verl/datasets/aime24/aime2024_math_format_repeat3.parquet"
        # file_path="/data0/jzzhang/verl/datasets/amc/amc_math_format_repeat3.parquet"
        # file_path="/data0/jzzhang/verl/datasets/amc/data/train-00000-of-00001.parquet"
        file_path="/data0/jzzhang/verl/datasets/NuminaMath-CoT/NuminaMath342k_math_format.parquet"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        print("请指定一个有效的.parquet文件路径")
        return
    
    df = read_parquet_file(file_path)
    
    if df is not None:
        print(f"\n成功读取文件，共{len(df)}行数据")

if __name__ == "__main__":
    main() 