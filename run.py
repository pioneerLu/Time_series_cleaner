import argparse
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(__file__))

from src.data_cleaner import DataProcessor, Config

def get_npy_data_length(npy_path: str) -> int:
    """
    从npy文件中获取数据长度
    :param npy_path: npy文件路径
    :return: 数据长度
    """
    try:
        data = np.load(npy_path)
        if len(data.shape) >= 2:
            return data.shape[1]  # 返回第二个维度的大小
        return data.shape[0]  # 如果是一维数组，返回第一个维度的大小
    except Exception as e:
        print(f"读取npy文件时发生错误：{str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Time Series Data Cleaner')
    
    # 基本参数
    parser.add_argument('--input', type=str, required=True, help='输入npy文件路径')
    parser.add_argument('--data_length', type=int, help='数据长度（可选，默认从输入文件中自动获取）')
    parser.add_argument('--dataset_name', type=str, default='dataset', help='数据集名称（默认：dataset）')
    parser.add_argument('--base_dir', type=str, help='基础目录路径（可选）')
    parser.add_argument('--keep_temp_files', action='store_true', help='是否保留临时文件')
    
    # 数据筛选参数
    parser.add_argument('--seasonality_threshold', type=float, default=0.64, help='周期性阈值（默认：0.64）')
    parser.add_argument('--trend_threshold', type=float, default=0.75, help='趋势性阈值（默认：0.75）')
    parser.add_argument('--shifting_threshold', type=float, default=0.24, help='漂移性阈值（默认：0.24）')
    parser.add_argument('--transition_threshold', type=float, default=0.09, help='转移性阈值（默认：0.09）')
    parser.add_argument('--long_term_jsd_threshold', type=float, default=0.3, help='长期JSD阈值（默认：0.3）')
    
    # 零值检测参数
    parser.add_argument('--enable_zero_check', action='store_true', default=False, help='是否启用零值检测（默认：False）')
    parser.add_argument('--zero_check_len', type=int, default=100, help='零值检测的检查长度（默认：100）')
    parser.add_argument('--zero_ratio_threshold', type=float, default=0.9, help='零值比例阈值（默认：0.9）')
    parser.add_argument('--zero_streak_threshold', type=int, default=50, help='零值连续阈值（默认：50）')
    parser.add_argument('--near_zero_threshold', type=float, default=0.005, help='接近零的阈值（默认：0.005）')
    parser.add_argument('--zero_std_threshold', type=float, default=1.0, help='零值检测的标准差阈值（默认：1.0）')
    
    # 处理模式
    parser.add_argument('--mode', type=str, choices=['full', 'features', 'filter'], default='full',
                      help='处理模式：full-完整处理, features-仅计算特征, filter-仅筛选数据（默认：full）')
    
    args = parser.parse_args()
    
    # 自动获取数据长度
    if args.data_length is None:
        data_length = get_npy_data_length(args.input)
        if data_length is None:
            print("错误：无法从输入文件中获取数据长度，请手动指定 --data_length 参数")
            return
        args.data_length = data_length
        print(f"获取到数据长度：{data_length}")
    
    # 创建配置对象
    config = Config(
        data_length=args.data_length,
        dataset_name=args.dataset_name,
        base_dir=args.base_dir,
        keep_temp_files=args.keep_temp_files,
        
        # 数据筛选参数
        seasonality_threshold=args.seasonality_threshold,
        trend_threshold=args.trend_threshold,
        shifting_threshold=args.shifting_threshold,
        transition_threshold=args.transition_threshold,
        long_term_jsd_threshold=args.long_term_jsd_threshold,
        
        # 零值检测参数
        enable_zero_check=args.enable_zero_check,
        zero_check_len=args.zero_check_len,
        zero_ratio_threshold=args.zero_ratio_threshold,
        zero_streak_threshold=args.zero_streak_threshold,
        near_zero_threshold=args.near_zero_threshold,
        zero_std_threshold=args.zero_std_threshold
    )
    
    processor = DataProcessor(config)
    
    # 根据模式处理数据
    if args.mode == 'full':
        output_path = processor.process_npy(args.input)
    elif args.mode == 'features':
        processor.calculate_features(args.input)
        output_path = None
    elif args.mode == 'filter':
        output_path = processor.filter_data(args.input)
    
    print("处理完成！")

if __name__ == '__main__':
    main() 