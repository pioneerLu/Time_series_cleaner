# Time Series Data Cleaner

一个用于清理和筛选时间序列数据的工具，基于 R 的 `tsfeatures` 库进行特征提取，支持智能数据筛选和异常值修复。

## 功能特性

- 时间序列特征提取（使用 R 的 tsfeatures 库）
- 基于多维度指标的数据筛选（趋势、季节性、转换、漂移、JSD）
- 异常值自动检测和修复
- 零值和常数值序列过滤
- 灵活的处理模式（完整处理/仅特征/仅筛选）

## 安装

### 前置要求

- Python 3.8+
- R 4.3.1+
- Conda（推荐）

### 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n data_cleaner python=3.10 r-base=4.3.1
conda activate data_cleaner

# 2. 设置 R 环境变量
export R_HOME=$CONDA_PREFIX/lib/R
export PATH=$PATH:$R_HOME/bin

# 3. 安装 R 包
conda install -c conda-forge r-tidyverse r-Rcatch22 r-forecast r-tsfeatures -y

# 4. 安装 Python 依赖
pip install -r requirements.txt
```

## 快速开始

### 命令行使用

```bash
# 完整处理
python run.py --input data.npy --mode full --dataset_name my_dataset

# 仅计算特征
python run.py --input data.npy --mode features --keep_temp_files --dataset_name my_dataset

# 仅筛选数据（特征已计算）
python run.py --input data.npy --mode filter --keep_temp_files --dataset_name my_dataset
```

使用 `--help` 查看所有可用参数：
```bash
python run.py --help
```

### Python API

```python
from src.data_cleaner import DataProcessor, Config

# 创建配置
config = Config(
    data_length=725,
    dataset_name="my_dataset",
    base_dir="/path/to/output",
    keep_temp_files=False,
    seasonality_threshold=0.64,
    trend_threshold=0.75,
    shifting_threshold=0.24,
    transition_threshold=0.09,
    long_term_jsd_threshold=0.3
)

# 处理数据
processor = DataProcessor(config)
output_path = processor.process_npy("input.npy")
```

## 数据格式

### 输入格式
- **文件格式**: `.npy` 文件
- **数据形状**: `(n_samples, sequence_length, 1)`
  - `n_samples`: 样本数量
  - `sequence_length`: 每个时间序列的长度
  - `1`: 特征维度（单变量时间序列）
- **数据类型**: `float32` 或 `float64`

示例：
```python
import numpy as np

# 生成示例数据
n_samples = 100
sequence_length = 725
data = np.random.randn(n_samples, sequence_length, 1)
np.save('data.npy', data)
```

### 输出格式
- **文件格式**: `.npy` 文件
- **数据形状**: `(n_filtered_samples, sequence_length, 1)`
  - `n_filtered_samples`: 筛选后的样本数量
  - `sequence_length`: 时间序列长度（与输入相同）
  - `1`: 特征维度

## 配置参数

### 数据筛选参数（默认值）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `seasonality_threshold` | 0.64 | 周期性强度阈值 |
| `trend_threshold` | 0.75 | 趋势强度阈值 |
| `shifting_threshold` | 0.24 | 漂移率阈值 |
| `transition_threshold` | 0.09 | 转移率阈值 |
| `long_term_jsd_threshold` | 0.3 | 长期JSD阈值 |

### 零值检测参数（默认值）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_zero_check` | True | 是否启用零值检测 |
| `zero_check_len` | 100 | 检查长度 |
| `zero_ratio_threshold` | 0.9 | 零值比例阈值 |
| `zero_streak_threshold` | 50 | 连续零值阈值 |
| `near_zero_threshold` | 0.005 | 接近零的阈值 |
| `zero_std_threshold` | 1.0 | 标准差阈值 |

## 数据筛选标准

数据必须满足以下所有条件才会被保留：

1. 趋势强度 > `trend_threshold`（默认：0.75）
2. 转换率 < `transition_threshold`（默认：0.09）
3. 季节性强度 > `seasonality_threshold`（默认：0.64）
4. 漂移率 < `shifting_threshold`（默认：0.24）
5. 长期JSD < `long_term_jsd_threshold`（默认：0.3）
6. 通过零值检测（如果启用）

## 输出目录结构

```
{base_dir}/
├── temp/                    # 临时文件目录
│   ├── csv/                # CSV文件目录
│   ├── characteristics/    # 特征报告目录
│   └── visualization/      # 可视化结果目录
└── output/                 # 最终输出目录
    └── {dataset_name}.npy  # 筛选后的数据文件
```

## 命令行参数

### 基本参数
- `--input`: 输入npy文件路径（必需）
- `--data_length`: 数据长度（可选，默认从输入文件自动获取）
- `--dataset_name`: 数据集名称（默认：dataset）
- `--base_dir`: 基础目录路径（可选）
- `--keep_temp_files`: 是否保留临时文件

### 处理模式
- `--mode`: 处理模式（默认：full）
  - `full`: 完整处理
  - `features`: 仅计算特征
  - `filter`: 仅筛选数据

### 数据筛选参数
所有筛选参数都有合理的默认值，通常无需修改。如需调整，请参考上方的配置参数表格。

## 分步处理

如果需要分步处理数据或重复使用已计算的特征：

```python
# 1. 仅计算特征
processor.calculate_features(input_npy_path)

# 2. 仅进行数据筛选（假设特征已计算完成）
output_npy_path = processor.filter_data(input_npy_path)

# 3. 完整处理但跳过特征计算
output_npy_path = processor.process_npy(input_npy_path, skip_feature_calculation=True)
```

注意：使用分步处理时，请确保：
- 设置 `keep_temp_files=True` 以保留临时文件
- 使用相同的配置参数（特别是 `data_length` 和 `dataset_name`）

## 异常值处理

工具会自动检测并修复以下类型的异常值：
1. **统计异常值**: 基于滑动窗口，超过阈值倍标准差的值
2. **极端低值**: 低于中位数20%的值

修复方法：使用线性插值或前向/后向填充。

## 致谢

本项目基于以下论文的工作：

> Qiu, X., Hu, J., Zhou, L., Wu, X., Du, J., Zhang, B., Guo, C., Zhou, A., Jensen, C. S., Sheng, Z., & Yang, B. (2024). TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods. Proceedings of the VLDB Endowment, 17(9), 2363-2377.

## 贡献

欢迎贡献。
