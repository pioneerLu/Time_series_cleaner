# Time Series Data Cleaner

给预训练时序模型准备单变量序列：读入 `.npy`，丢掉不可用样本，按长度写出干净数据。

默认模式只依赖 NumPy / pandas，不跑 R。可选的统计特征严筛需要 R（`tsfeatures` 等）。

## 安装

预训练清洗：

```bash
pip install -r requirements.txt
```

统计特征严筛额外需要 Conda + R：

```bash
conda create -n data_cleaner python=3.10 r-base=4.3.1
conda activate data_cleaner
export R_HOME=$CONDA_PREFIX/lib/R
export PATH=$PATH:$R_HOME/bin
conda install -c conda-forge r-tidyverse r-Rcatch22 r-forecast r-tsfeatures -y
pip install -r requirements-r.txt
```

## 输入 / 输出

- 输入：单个 `.npy`，或含多个 `.npy` 的目录（默认递归子目录）
- 形状规则：
  - `(N, T, 1)` / `(N, T, C)`：按样本拆条；`C>1` 时 `--channel_mode flatten`（默认）拆通道，`first` 只留第 0 通道
  - `(T,)`、`(T, 1)`、`(1, T)`：单条序列
  - 其它二维：必须指定 `--layout NT`（每行一条）或 `--layout TC`（每列一个通道），否则报错退出
  - `object` 数组：每个元素当作一条序列
- 输出（默认 `output/`）：
  - `data_{T}.npy`，形状 `(N, T, 1)`，`float32`
  - `manifest.json`：保留 / 丢弃统计
  - `index.jsonl`：溯源（来源文件、下标、输出位置）
  - `progress.json`：增量进度

## 预训练清洗（默认）

```bash
python run.py --input path/to/npy_or_dir
python run.py --input path/to/npy_or_dir --output /tmp/clean --min_length 64 --overwrite
python run.py --input wind.npy --layout TC
python run.py --input data/ --nan_policy split --no-dedup
python run.py --input data/ --no-recursive --disable_zero_check
```

默认会丢弃：

- 长度小于 `--min_length`（默认 32）
- 含 NaN / Inf（`--nan_policy interp` 插值，`split` 切成有限子段）
- 全零、近常数（峰峰值过小）、开头近零过多（`--disable_zero_check` 关闭）
- 与已保留样本取值完全相同的重复序列（`--no-dedup` 关闭）

默认不使用 R、不画图、不修补异常值。需要修补时加 `--fix_anomalies`。

中断后可对同一输出目录再次运行：已完成且未改动的文件会跳过。源文件变更会重建输出。`--overwrite` 强制重跑。

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--layout` | `auto` | 二维含义：`NT` 每行一条，`TC` 每列一个通道 |
| `--channel_mode` | `flatten` | 三维 `(N,T,C)`：拆通道或只留第 0 通道 |
| `--nan_policy` | `drop` | `drop` / `interp` / `split` |
| `--min_length` | `32` | 最短序列长度 |
| `--recursive` | 开 | `--no-recursive` 只扫当前目录 |
| `--dedup` | 开 | `--no-dedup` 不去重 |
| `--overwrite` | 关 | 忽略进度，重新清洗 |
| `--fix_anomalies` | 关 | 用滑动窗口修补尖峰 / 极端低值 |
| `--keep_temp_files` | 关 | 保留分片临时文件 |

零值检测相关：`--disable_zero_check`、`--zero_check_len`、`--zero_ratio_threshold`、`--zero_streak_threshold`、`--near_zero_threshold`、`--zero_std_threshold`（默认 `0`，不用标准差判常数，避免误杀零均值序列）。

## 统计特征严筛（可选）

需要 R，且一次处理一个 npy。会先把序列写成 CSV，用 `tsfeatures` / Catch22 提特征，再按趋势、季节性、转换、漂移、长期 JSD 筛选。

```bash
python run.py --input data.npy --mode full --dataset_name my_dataset --keep_temp_files
python run.py --input data.npy --mode features --keep_temp_files --dataset_name my_dataset
python run.py --input data.npy --mode filter --keep_temp_files --dataset_name my_dataset
```

`--mode full` 是完整流程（提特征 + 筛选）。`--mode features` 只提特征，`--mode filter` 假定特征已算好。加 `--visualize` 才会保存通过样本的图。

筛选阈值（通常不用改）：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--trend_threshold` | 0.75 | 趋势强度下限 |
| `--seasonality_threshold` | 0.64 | 季节性强度下限 |
| `--transition_threshold` | 0.09 | 转换率上限 |
| `--shifting_threshold` | 0.24 | 漂移率上限 |
| `--long_term_jsd_threshold` | 0.3 | 长期分布偏移上限 |

输出为 `output/{dataset_name}.npy`，形状 `(N, T, 1)`。

## 脚本

```bash
bash scripts/process_pretrain.sh
bash scripts/process_full.sh
```

可通过环境变量覆盖路径，例如 `INPUT=data OUTPUT=/tmp/clean MIN_LENGTH=64 bash scripts/process_pretrain.sh`。使用前把默认 `--input` 改成你的数据路径。

## 测试

```bash
python -m unittest tests.test_pretrain -v
```
