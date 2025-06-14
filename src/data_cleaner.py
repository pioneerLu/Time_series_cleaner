import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Optional
from scipy.stats import skew, kurtosis
from scipy.signal import argrelextrema
from rpy2.robjects import FloatVector
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from src.model import TimeSeriesFeatureExtractor, StatisticalCalculator, TimeSeriesProcessor
from src.config import Config

class DataProcessor:
    def __init__(self, config: Optional[Config] = None):
        """
        初始化数据处理器
        :param config: 配置对象，如果为None则使用默认配置
        """
        self.config = config if config is not None else Config()
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.stat_calculator = StatisticalCalculator()
        
    def process_npy(self, input_npy_path: str, skip_feature_calculation: bool = False) -> str:
        """
        处理npy文件并生成筛选后的数据
        :param input_npy_path: 输入npy文件路径
        :param skip_feature_calculation: 是否跳过特征计算步骤
        :return: 输出npy文件路径
        """
        # 1. 将npy转换为csv
        self._npy2csv(input_npy_path)
        
        # 2. 生成特征报告（可选）
        if not skip_feature_calculation:
            processor = TimeSeriesProcessor(output_dir=self.config.characteristics_dir)
            processor.process_path(self.config.csv_dir)
        
        # 3. 筛选数据
        good_data = self._filter_data()
        
        # 4. 保存筛选后的数据
        self._save_filtered_data(good_data)
        
        # 5. 清理临时文件
        if not self.config.keep_temp_files:
            self.config.cleanup()
            
        return self.config.npy_output_path
        
    def calculate_features(self, input_npy_path: str) -> None:
        """
        仅计算特征，不进行数据筛选
        :param input_npy_path: 输入npy文件路径
        """
        # 1. 将npy转换为csv
        self._npy2csv(input_npy_path)
        
        # 2. 生成特征报告
        processor = TimeSeriesProcessor(output_dir=self.config.characteristics_dir)
        processor.process_path(self.config.csv_dir)
        
    def filter_data(self, input_npy_path: str) -> str:
        """
        仅进行数据筛选，假设特征已经计算完成
        :param input_npy_path: 输入npy文件路径
        :return: 输出npy文件路径
        """
        # 1. 将npy转换为csv（如果csv文件不存在）
        if not os.path.exists(self.config.csv_dir):
            self._npy2csv(input_npy_path)
        
        # 2. 筛选数据
        good_data = self._filter_data()
        
        # 3. 保存筛选后的数据
        self._save_filtered_data(good_data)
        
        # 4. 清理临时文件
        if not self.config.keep_temp_files:
            self.config.cleanup()
            
        return self.config.npy_output_path
        
    def _npy2csv(self, npy_path: str):
        """
        将npy文件转换为csv文件
        :param npy_path: npy文件路径
        """
        os.makedirs(self.config.csv_dir, exist_ok=True)
        data = np.load(npy_path)
        for i in range(data.shape[0]):
            df = pd.DataFrame({
                'date': range(data.shape[1]), 
                'data': data[i].flatten(),
                'cols': 0
            })
            df.to_csv(os.path.join(self.config.csv_dir, f'{data.shape[1]}_{i}.csv'), index=False)
            
    def _filter_data(self) -> List[str]:
        """
        根据特征筛选数据
        :return: 筛选后的数据ID列表
        """
        good_data = []
        dir = self.config.characteristics_dir

        names = []
        for file in os.listdir(dir):
            if file.startswith('DATA_characteristics'):
                names.append(file)
                
        for name in names:
            temp = name.split('/')[-1].split('.')[0].split('_')[-1]
            df = pd.read_csv(os.path.join(dir, name))
            
            path2 = os.path.join(self.config.csv_dir, f'{self.config.data_length}_{temp}.csv')
            df2 = pd.read_csv(path2)
            
            if self._check_conditions(df) and not self._is_zero_or_constant(df2['data'].values):
                good_data.append(temp)

                data_clean = self._fix_anomalies(df2['data'].values)
                plt.figure(figsize=(10, 4))
                plt.title(f'Data ID: {temp}')
                plt.plot(data_clean)
                plt.savefig(os.path.join(self.config.visualization_dir, f'plot_{temp}.png'))
                plt.close()
                
        return good_data
    
    def _check_conditions(self, df: pd.DataFrame) -> bool:
        """
        检查数据是否满足筛选条件
        :param df: 特征数据框
        :return: 是否满足条件
        """
        if df['Trend'].values < self.config.trend_threshold:
            return False
        if df['Transition'].values > self.config.transition_threshold:
            return False
        if df['Seasonality'].values < self.config.seasonality_threshold:
            return False
        if abs(df['Shifting'].values) > self.config.shifting_threshold:
            return False
        if df['Long_term_jsd'].values > self.config.long_term_jsd_threshold:
            return False
        return True
    
    def _is_zero_or_constant(self, ts: np.ndarray) -> bool:
        """
        检查时间序列是否为零或常数
        :param ts: 时间序列数据
        :return: 是否为零或常数
        """
        if not self.config.enable_zero_check:
            return False
            
        initial_segment = ts[:self.config.zero_check_len]
        
        zero_ratio = np.mean(initial_segment == 0)
        streak = 0
        for val in ts:
            if val == 0:
                streak += 1
            else:
                break
        too_many_zeros = zero_ratio > self.config.zero_ratio_threshold or streak > self.config.zero_streak_threshold
        
        near_zero_mean = np.mean(initial_segment) < self.config.near_zero_threshold
        low_std = np.std(initial_segment) < self.config.zero_std_threshold
        nearly_constant = near_zero_mean and low_std
        
        return too_many_zeros or nearly_constant
    
    def _fix_anomalies(self, series_data: np.ndarray, threshold: float = 7, 
                      window_size: int = 3) -> np.ndarray:
        """
        修复时间序列中的异常值
        :param series_data: 时间序列数据
        :param threshold: 异常值阈值
        :param window_size: 滑动窗口大小
        :return: 修复后的时间序列
        """
        if not isinstance(series_data, pd.Series):
            series_data = pd.Series(series_data)
        fixed_series = series_data.copy()
        
        # 计算滑动均值和标准差
        rolling_mean = series_data.rolling(window=window_size, center=True).mean()
        rolling_std = series_data.rolling(window=window_size, center=True).std()
        
        # 填充滑动窗口边缘的NaN值
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.replace(0, series_data.std())
        
        threshold_anomalies = np.abs(series_data - rolling_mean) > threshold * rolling_std
        
        # 检测极端低值
        median_value = series_data.median()
        extreme_low_anomalies = series_data < median_value * 0.2
        
        # 合并两种异常检测
        anomalies = threshold_anomalies | extreme_low_anomalies
        anomaly_indices = np.where(anomalies)[0]
        
        if len(anomaly_indices) == 0:
            return series_data.values
        
        for idx in anomaly_indices:
            # 寻找前后最近的非异常点
            left_idx = idx - 1
            while left_idx >= 0 and anomalies[left_idx]:
                left_idx -= 1
            
            right_idx = idx + 1
            while right_idx < len(series_data) and anomalies[right_idx]:
                right_idx += 1
            
            # 线性插值
            if left_idx >= 0 and right_idx < len(series_data):
                fixed_series[idx] = np.interp(idx, [left_idx, right_idx], 
                                           [series_data[left_idx], series_data[right_idx]])
            elif left_idx >= 0:
                fixed_series[idx] = series_data[left_idx]
            elif right_idx < len(series_data):
                fixed_series[idx] = series_data[right_idx]
        
        return fixed_series.values
    
    def _save_filtered_data(self, good_data: List[str]):
        """
        保存筛选后的数据
        :param good_data: 筛选后的数据ID列表
        """
        output_npy = None
        
        for data_idx in good_data:
            data_path = os.path.join(self.config.csv_dir, f'{self.config.data_length}_{data_idx}.csv')
            df = pd.read_csv(data_path)
            data_clean = self._fix_anomalies(df['data'].values)
            data = data_clean.reshape(1, -1, 1)
            
            if output_npy is None:
                output_npy = data
            else:
                output_npy = np.concatenate((output_npy, data), axis=1)
                
        if output_npy is None:
            print(f"没有筛选到数据")
            return
            
        # 保存为npy文件
        os.makedirs(os.path.dirname(self.config.npy_output_path), exist_ok=True)
        np.save(self.config.npy_output_path, output_npy)
        print(f"保存筛选后的数据到: {self.config.npy_output_path}")
        print(f"数据形状: {output_npy.shape}")

if __name__ == "__main__":
    # 示例用法
    config = Config(
        data_length=900,
        base_dir="/path/to/your/directory", 
        keep_temp_files=False  # 是否保留临时文件
    )
    
    processor = DataProcessor(config)
    input_npy_path = "/path/to/your/input.npy"
    output_npy_path = processor.process_npy(input_npy_path) 
    


