import os
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Config:
    def __init__(
        self,
        data_length: int = 5112,
        base_dir: str = None,
        output_dir: str = None,
        keep_temp_files: bool = False,
        dataset_name: str = "dataset",
        mode: str = "pretrain",
        layout: str = "auto",
        channel_mode: str = "flatten",
        nan_policy: str = "drop",
        recursive: bool = True,
        dedup: bool = True,
        min_length: int = 32,
        visualize: bool = False,
        fix_anomalies: bool = False,
        overwrite: bool = False,
        seasonality_threshold: float = 0.64,
        trend_threshold: float = 0.75,
        shifting_threshold: float = 0.24,
        transition_threshold: float = 0.09,
        long_term_jsd_threshold: float = 0.3,
        enable_zero_check: bool = True,
        zero_check_len: int = 100,
        zero_ratio_threshold: float = 0.9,
        zero_streak_threshold: int = 50,
        near_zero_threshold: float = 0.005,
        zero_std_threshold: float = 0.0,
    ):
        self.data_length = data_length
        self.keep_temp_files = keep_temp_files
        self.dataset_name = dataset_name
        self.mode = mode
        self.layout = layout
        self.channel_mode = channel_mode
        self.nan_policy = nan_policy
        self.recursive = recursive
        self.dedup = dedup
        self.min_length = min_length
        self.visualize = visualize
        self.fix_anomalies = fix_anomalies
        self.overwrite = overwrite

        self.seasonality_threshold = seasonality_threshold
        self.trend_threshold = trend_threshold
        self.shifting_threshold = shifting_threshold
        self.transition_threshold = transition_threshold
        self.long_term_jsd_threshold = long_term_jsd_threshold

        self.enable_zero_check = enable_zero_check
        self.zero_check_len = zero_check_len
        self.zero_ratio_threshold = zero_ratio_threshold
        self.zero_streak_threshold = zero_streak_threshold
        self.near_zero_threshold = near_zero_threshold
        self.zero_std_threshold = zero_std_threshold

        self.base_dir = os.path.abspath(base_dir if base_dir else REPO_ROOT)
        self.output_dir = os.path.abspath(output_dir or os.path.join(self.base_dir, "output"))
        self.temp_dir = os.path.join(self.base_dir, "temp")
        self.shard_dir = os.path.join(self.output_dir, ".shards")
        self.progress_path = os.path.join(self.output_dir, "progress.json")
        self.manifest_path = os.path.join(self.output_dir, "manifest.json")

        self.characteristics_dir = os.path.join(self.temp_dir, "characteristics", dataset_name)
        self.visualization_dir = os.path.join(self.temp_dir, "visualization", dataset_name)
        self.csv_dir = os.path.join(self.temp_dir, "csv", dataset_name)
        self.npy_output_path = os.path.join(self.output_dir, f"{dataset_name}.npy")

    def ensure_pretrain_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.shard_dir, exist_ok=True)

    def ensure_tfb_dirs(self):
        for dir_path in (
            self.base_dir,
            self.temp_dir,
            self.output_dir,
            self.characteristics_dir,
            self.visualization_dir,
            self.csv_dir,
        ):
            os.makedirs(dir_path, exist_ok=True)

    def cleanup(self):
        if not self.keep_temp_files and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
