# Time Series Data Cleaner

这是一个用于清理和筛选时间序列数据的工具。该工具可以帮助您：
1. 将npy格式的时间序列数据转换为csv格式
2. 生成时间序列特征报告
3. 根据特征筛选高质量的数据
4. 修复数据中的异常值
5. 生成数据可视化结果
6. 检测并过滤零值或常数值序列

## 数据格式要求

### 输入数据格式
- 文件格式：`.npy`文件
- 数据形状：`(n_samples, sequence_length, 1)`
  - `n_samples`: 样本数量
  - `sequence_length`: 每个时间序列的长度
  - `1`: 特征维度（单变量时间序列）
- 数据类型：`float32`或`float64`

示例：
```python
import numpy as np

# 示例数据
n_samples = 100
sequence_length = 725
data = np.random.randn(n_samples, sequence_length, 1)
np.save('data.npy', data)
```

### 输出数据格式
- 文件格式：`.npy`文件
- 数据形状：`(n_filtered_samples, sequence_length, 1)`
  - `n_filtered_samples`: 筛选后的样本数量
  - `sequence_length`: 时间序列长度（与输入相同）
  - `1`: 特征维度

## 安装

```
conda create -n env python=3.10 r-base=4.3.1
conda activate env
```

```
export R_HOME=$CONDA_PREFIX/lib/R
export PATH=$PATH:$R_HOME/bin
```

```
conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge r-tidyverse r-Rcatch22 r-forecast r-tsfeatures -y
```

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行使用

工具提供了命令行接口，支持以下功能：

1. 完整处理：
```bash
python run.py --base_dir '/home/rwkv/RWKV-TS/data_cleaner' --input data/data.npy  --mode full --dataset_name test_dataset
```

2. 仅计算特征：
```bash
python run.py --base_dir '/home/rwkv/RWKV-TS/data_cleaner' --input data/data.npy --mode features --keep_temp_files --dataset_name test_dataset
```

3. 仅进行数据筛选（假设特征已计算完成）：
```bash
python run.py --base_dir '/home/rwkv/RWKV-TS/data_cleaner' --input data/data.npy  --mode filter --keep_temp_files --dataset_name test_dataset
```

#### 命令行参数说明

基本参数：
- `--input`: 输入npy文件路径（必需）
- `--output`: 输出npy文件路径（可选）
- `--data_length`: 数据长度（默认：725）
- `--dataset_name`: 数据集名称（默认：dataset）
- `--base_dir`: 基础目录路径（可选）
- `--keep_temp_files`: 是否保留临时文件

处理模式：
- `--mode`: 处理模式（默认：full）
  - `full`: 完整处理
  - `features`: 仅计算特征
  - `filter`: 仅筛选数据

数据筛选参数：
- `--seasonality_threshold`: 周期性阈值（默认：0.64）
- `--trend_threshold`: 趋势性阈值（默认：0.75）
- `--shifting_threshold`: 漂移性阈值（默认：0.24）
- `--transition_threshold`: 转移性阈值（默认：0.09）
- `--long_term_jsd_threshold`: 长期JSD阈值（默认：0.3）

零值检测参数：
- `--enable_zero_check`: 是否启用零值检测（默认：True）
- `--zero_check_len`: 零值检测的检查长度（默认：100）
- `--zero_ratio_threshold`: 零值比例阈值（默认：0.9）
- `--zero_streak_threshold`: 零值连续阈值（默认：50）
- `--near_zero_threshold`: 接近零的阈值（默认：0.005）
- `--zero_std_threshold`: 零值检测的标准差阈值（默认：1.0）



### 基本用法

```python
from data_cleaner import DataProcessor, Config

# 创建配置对象
config = Config(
    data_length=900,  # 数据长度
    dataset_name="my_dataset",  # 数据集名称
    base_dir="/path/to/your/directory", 
    keep_temp_files=False,  # 是否保留临时文件
    
    # 数据筛选参数
    seasonality_threshold=0.64,  # 周期性阈值
    trend_threshold=0.75,        # 趋势性阈值
    shifting_threshold=0.24,     # 漂移性阈值
    transition_threshold=0.09,   # 转移性阈值
    long_term_jsd_threshold=0.3  # 长期JSD阈值
)

# 创建数据处理器
processor = DataProcessor(config)

# 处理数据
input_npy_path = "/path/to/your/input.npy"
output_npy_path = processor.process_npy(input_npy_path)
```

### 分步处理

如果您想分步处理数据，或者重复使用已计算的特征，可以使用以下方法：

1. 仅计算特征：
```python
# 计算特征并保存到临时目录
processor.calculate_features(input_npy_path)
```

2. 仅进行数据筛选（假设特征已计算完成）：
```python
# 使用已计算的特征进行筛选
output_npy_path = processor.filter_data(input_npy_path)
```

3. 完整处理但跳过特征计算：
```python
# 使用已计算的特征进行完整处理
output_npy_path = processor.process_npy(input_npy_path, skip_feature_calculation=True)
```

注意：使用分步处理时，请确保：
- 设置 `keep_temp_files=True` 以保留临时文件
- 使用相同的配置参数（特别是 `data_length` 和 `dataset_name`）
- 确保临时文件目录结构完整

### 配置选项

`Config` 类提供以下配置选项：

- `data_length`: 数据长度（默认：725）
- `dataset_name`: 数据集名称（默认："dataset"）
- `base_dir`: 基础目录路径（默认：用户主目录下的time-series-data）
- `keep_temp_files`: 是否保留临时文件（默认：False）
- `seasonality_threshold`: 周期性阈值（默认：0.64）
- `trend_threshold`: 趋势性阈值（默认：0.75）
- `shifting_threshold`: 漂移性阈值（默认：0.24）
- `transition_threshold`: 转移性阈值（默认：0.09）
- `long_term_jsd_threshold`: 长期JSD阈值（默认：0.3）

零值检测相关参数：
- `enable_zero_check`: 是否启用零值检测（默认：True）
- `zero_check_len`: 零值检测的检查长度（默认：100）
- `zero_ratio_threshold`: 零值比例阈值（默认：0.9）
- `zero_streak_threshold`: 零值连续阈值（默认：50）
- `near_zero_threshold`: 接近零的阈值（默认：0.005）
- `zero_std_threshold`: 零值检测的标准差阈值（默认：1.0）

### 输出目录结构

```
time-series-data/
├── temp/                    # 临时文件目录
│   ├── csv/                # CSV文件目录
│   ├── characteristics/    # 特征报告目录
│   └── visualization/      # 可视化结果目录
└── output/                 # 最终输出目录
    └── {dataset_name}_{data_length}.npy  # 筛选后的数据文件
```

## 数据筛选标准

该工具使用以下标准来筛选数据：

1. 趋势强度 > trend_threshold（默认：0.75）
2. 转换率 < transition_threshold（默认：0.09）
3. 季节性强度 > seasonality_threshold（默认：0.64）
4. 漂移率 < shifting_threshold（默认：0.24）
5. 长期JSD < long_term_jsd_threshold（默认：0.3）

## 零值检测

工具提供了可配置的零值检测功能，用于过滤掉无效的时间序列数据。零值检测包括以下检查：

1. 零值比例检查：检查序列中零值的比例是否超过阈值
2. 零值连续检查：检查序列开头是否有过多的连续零值
3. 常数值检查：检查序列是否接近常数值（均值和标准差都很小）

可以通过配置参数来控制零值检测的行为：

```python
config = Config(
    # 启用/禁用零值检测
    enable_zero_check=True,
    
    # 零值检测参数
    zero_check_len=100,        # 检查序列的前100个点
    zero_ratio_threshold=0.9,  # 零值比例阈值
    zero_streak_threshold=50,  # 连续零值阈值
    near_zero_threshold=0.005, # 接近零的阈值
    zero_std_threshold=1.0     # 标准差阈值
)
```

如果不需要零值检测，可以将其禁用：

```python
config = Config(
    enable_zero_check=False  # 禁用零值检测
)
```

## 异常值处理

工具会自动检测并修复以下类型的异常值：

1. 基于滑动窗口的统计异常值
2. 极端低值（低于中位数的20%）

## 致谢

本项目基于以下论文的工作：

Qiu, X., Hu, J., Zhou, L., Wu, X., Du, J., Zhang, B., Guo, C., Zhou, A., Jensen, C. S., Sheng, Z., & Yang, B. (2024). TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods. Proceedings of the VLDB Endowment, 17(9), 2363-2377.

## 依赖项

- numpy
- pandas
- matplotlib
- scipy
- rpy2
- statsmodels
- scikit-learn

## license

MIT License 
