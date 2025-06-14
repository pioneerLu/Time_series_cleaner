import os
from pathlib import Path
import datetime

class Config:
    def __init__(self, 
                 data_length: int = 5112,
                 base_dir: str = None,
                 keep_temp_files: bool = False,
                 dataset_name: str = "dataset",
                 # 数据筛选参数
                 seasonality_threshold: float = 0.64,  # 周期性阈值
                 trend_threshold: float = 0.75,        # 趋势性阈值
                 shifting_threshold: float = 0.24,     # 漂移性阈值
                 transition_threshold: float = 0.09,   # 转移性阈值
                 long_term_jsd_threshold: float = 0.3,  # 长期JSD阈值
                 # 零值检测相关参数
                 enable_zero_check: bool = True,
                 zero_check_len: int = 100,
                 zero_ratio_threshold: float = 0.9,
                 zero_streak_threshold: int = 50,
                 near_zero_threshold: float = 0.005,
                 zero_std_threshold: float = 1.0):
        """
        初始化配置
        :param data_length: 数据长度
        :param base_dir: 基础目录，如果为None则使用默认目录
        :param keep_temp_files: 是否保留临时文件
        :param dataset_name: 数据集名称，用于生成输出文件名
        :param seasonality_threshold: 周期性阈值
        :param trend_threshold: 趋势性阈值
        :param shifting_threshold: 漂移性阈值
        :param transition_threshold: 转移性阈值
        :param long_term_jsd_threshold: 长期JSD阈值
        :param enable_zero_check: 是否启用零值检测
        :param zero_check_len: 零值检测的检查长度
        :param zero_ratio_threshold: 零值比例阈值
        :param zero_streak_threshold: 零值连续阈值
        :param near_zero_threshold: 接近零的阈值
        :param zero_std_threshold: 零值检测的标准差阈值
        """
        self.data_length = data_length
        self.keep_temp_files = keep_temp_files
        self.dataset_name = dataset_name
        
        # 数据筛选参数
        self.seasonality_threshold = seasonality_threshold
        self.trend_threshold = trend_threshold
        self.shifting_threshold = shifting_threshold
        self.transition_threshold = transition_threshold
        self.long_term_jsd_threshold = long_term_jsd_threshold
        
        # 零值检测相关参数
        self.enable_zero_check = enable_zero_check
        self.zero_check_len = zero_check_len
        self.zero_ratio_threshold = zero_ratio_threshold
        self.zero_streak_threshold = zero_streak_threshold
        self.near_zero_threshold = near_zero_threshold
        self.zero_std_threshold = zero_std_threshold
        
        # 设置基础目录
        self.base_dir = os.path.abspath(base_dir if base_dir else os.path.join(os.path.expanduser("~"), "data_cleaner"))
            
        # 创建必要的目录
        self.temp_dir = os.path.join(self.base_dir, "temp")
        self.output_dir = os.path.join(self.base_dir, "output")
        
        # 使用数据集名称创建目录
        self.characteristics_dir = os.path.join(self.temp_dir, "characteristics", dataset_name)
        self.visualization_dir = os.path.join(self.temp_dir, "visualization", dataset_name)
        self.csv_dir = os.path.join(self.temp_dir, "csv", dataset_name)
        self.npy_output_path = os.path.join(self.output_dir, f"{dataset_name}.npy")
        
        # 创建所有必要的目录
        directories = [
            self.base_dir,
            self.temp_dir,
            self.output_dir,
            self.characteristics_dir,
            self.visualization_dir,
            self.csv_dir
        ]
        
        # 确保所有目录都被创建
        for dir_path in directories:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"创建目录：{dir_path}")
            except Exception as e:
                print(f"创建目录 {dir_path} 时发生错误：{str(e)}")
                raise
            
    def cleanup(self):
        """
        清理临时文件
        """
        if not self.keep_temp_files:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir) 