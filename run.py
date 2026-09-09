import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))

from src.config import Config


def get_npy_data_length(npy_path: str):
    try:
        data = np.load(npy_path, mmap_mode="r", allow_pickle=True)
        if getattr(data, "ndim", 0) >= 2:
            return int(data.shape[1])
        return int(data.shape[0])
    except Exception as e:
        print(f"读取npy文件时发生错误：{e}")
        return None


def main():
    try:
        _run()
    except (ValueError, FileNotFoundError) as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)


def _run():
    parser = argparse.ArgumentParser(
        description="Time Series Data Cleaner - 预训练时序数据清洗",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py --input data/ --mode pretrain
  python run.py --input data/a.npy --mode pretrain --min_length 64 --overwrite
  python run.py --input data/a.npy --mode full --dataset_name my_dataset --keep_temp_files
        """,
    )
    parser.add_argument("--input", type=str, required=True, help="输入 npy 文件或目录")
    parser.add_argument("--output", type=str, help="输出目录（默认：<base_dir>/output）")
    parser.add_argument("--base_dir", type=str, help="工作目录（默认：仓库根目录）")
    parser.add_argument("--data_length", type=int, help="特征筛选模式的序列长度（默认从文件读取）")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="特征筛选输出文件名")
    parser.add_argument("--keep_temp_files", action="store_true", help="保留临时文件")
    parser.add_argument("--overwrite", action="store_true", help="忽略已有进度，重新清洗")
    parser.add_argument(
        "--layout",
        type=str,
        choices=["auto", "NT", "TC"],
        default="auto",
        help="二维数组的含义：NT 每行一条序列，TC 每列一个通道；auto 仅在无歧义时可用",
    )
    parser.add_argument("--min_length", type=int, default=32, help="最短序列长度（pretrain）")
    parser.add_argument(
        "--channel_mode",
        type=str,
        choices=["flatten", "first"],
        default="flatten",
        help="三维 (N,T,C)：flatten 拆通道，first 只保留第 0 通道",
    )
    parser.add_argument(
        "--nan_policy",
        type=str,
        choices=["drop", "interp", "split"],
        default="drop",
        help="非有限值：drop 整条丢，interp 插值，split 切成有限子段",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="递归扫描输入目录中的 npy（--no-recursive 关闭）",
    )
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="按值去重（--no-dedup 关闭）",
    )
    parser.add_argument("--fix_anomalies", action="store_true", help="保守修补异常值")
    parser.add_argument("--visualize", action="store_true", help="特征筛选时保存通过样本的图")
    parser.add_argument("--seasonality_threshold", type=float, default=0.64)
    parser.add_argument("--trend_threshold", type=float, default=0.75)
    parser.add_argument("--shifting_threshold", type=float, default=0.24)
    parser.add_argument("--transition_threshold", type=float, default=0.09)
    parser.add_argument("--long_term_jsd_threshold", type=float, default=0.3)
    parser.add_argument("--enable_zero_check", action="store_true", help="特征筛选模式显式开启零值检测")
    parser.add_argument("--disable_zero_check", action="store_true", help="关闭零值/常数检测")
    parser.add_argument("--zero_check_len", type=int, default=100)
    parser.add_argument("--zero_ratio_threshold", type=float, default=0.9)
    parser.add_argument("--zero_streak_threshold", type=int, default=50)
    parser.add_argument("--near_zero_threshold", type=float, default=0.005)
    parser.add_argument("--zero_std_threshold", type=float, default=0.0,
                       help="额外的标准差上限，0 表示不用该项（避免误杀零均值序列）")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "full", "features", "filter"],
        default="pretrain",
        help="pretrain: 预训练清洗（默认，不跑 R）；full/features/filter: 基于统计特征的严筛",
    )
    args = parser.parse_args()
    mode = "tfb" if args.mode == "full" else args.mode

    if mode == "pretrain":
        enable_zero_check = not args.disable_zero_check
        config = Config(
            dataset_name=args.dataset_name,
            base_dir=args.base_dir,
            output_dir=args.output,
            keep_temp_files=args.keep_temp_files,
            mode="pretrain",
            layout=args.layout,
            channel_mode=args.channel_mode,
            nan_policy=args.nan_policy,
            recursive=args.recursive,
            dedup=args.dedup,
            min_length=args.min_length,
            visualize=args.visualize,
            fix_anomalies=args.fix_anomalies,
            overwrite=args.overwrite,
            enable_zero_check=enable_zero_check,
            zero_check_len=args.zero_check_len,
            zero_ratio_threshold=args.zero_ratio_threshold,
            zero_streak_threshold=args.zero_streak_threshold,
            near_zero_threshold=args.near_zero_threshold,
            zero_std_threshold=args.zero_std_threshold,
        )
        from src.pretrain import PretrainCleaner

        output_dir = PretrainCleaner(config).run(args.input)
        print(f"处理完成，输出目录：{output_dir}")
        return

    if args.data_length is None:
        if os.path.isdir(args.input):
            print("错误：特征筛选模式请传入单个 npy 文件，或指定 --data_length")
            return
        data_length = get_npy_data_length(args.input)
        if data_length is None:
            print("错误：无法从输入文件中获取数据长度，请手动指定 --data_length")
            return
        args.data_length = data_length
        print(f"获取到数据长度：{data_length}")

    enable_zero_check = args.enable_zero_check and not args.disable_zero_check
    config = Config(
        data_length=args.data_length,
        dataset_name=args.dataset_name,
        base_dir=args.base_dir,
        output_dir=args.output,
        keep_temp_files=args.keep_temp_files,
        mode="tfb",
        visualize=args.visualize,
        fix_anomalies=True,
        seasonality_threshold=args.seasonality_threshold,
        trend_threshold=args.trend_threshold,
        shifting_threshold=args.shifting_threshold,
        transition_threshold=args.transition_threshold,
        long_term_jsd_threshold=args.long_term_jsd_threshold,
        enable_zero_check=enable_zero_check,
        zero_check_len=args.zero_check_len,
        zero_ratio_threshold=args.zero_ratio_threshold,
        zero_streak_threshold=args.zero_streak_threshold,
        near_zero_threshold=args.near_zero_threshold,
        zero_std_threshold=args.zero_std_threshold,
    )
    from src.data_cleaner import DataProcessor

    processor = DataProcessor(config)
    if mode == "features":
        processor.calculate_features(args.input)
        print("特征计算完成")
        return
    if mode == "filter":
        output_path = processor.filter_data(args.input)
    else:
        output_path = processor.process_npy(args.input)
    if output_path:
        print(f"处理完成，输出文件：{output_path}")


if __name__ == "__main__":
    main()
