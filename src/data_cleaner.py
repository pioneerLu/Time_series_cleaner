import numpy as np
import os
from typing import List, Optional
import matplotlib.pyplot as plt

from src.config import Config
from src.filters import fix_anomalies, is_zero_or_constant


class DataProcessor:
    def __init__(self, config: Optional[Config] = None):
        self.config = config if config is not None else Config(mode="tfb")
        self.config.ensure_tfb_dirs()

    def process_npy(self, input_npy_path: str, skip_feature_calculation: bool = False) -> str:
        from src.model import TimeSeriesProcessor

        self._npy2csv(input_npy_path)
        if not skip_feature_calculation:
            processor = TimeSeriesProcessor(output_dir=self.config.characteristics_dir)
            processor.process_path(self.config.csv_dir)
        good_data = self._filter_data()
        self._save_filtered_data(good_data)
        if not self.config.keep_temp_files:
            self.config.cleanup()
        return self.config.npy_output_path

    def calculate_features(self, input_npy_path: str) -> None:
        from src.model import TimeSeriesProcessor

        self._npy2csv(input_npy_path)
        processor = TimeSeriesProcessor(output_dir=self.config.characteristics_dir)
        processor.process_path(self.config.csv_dir)

    def filter_data(self, input_npy_path: str) -> str:
        if not os.path.exists(self.config.csv_dir):
            self._npy2csv(input_npy_path)
        good_data = self._filter_data()
        self._save_filtered_data(good_data)
        if not self.config.keep_temp_files:
            self.config.cleanup()
        return self.config.npy_output_path

    def _npy2csv(self, npy_path: str):
        os.makedirs(self.config.csv_dir, exist_ok=True)
        data = np.load(npy_path)
        for i in range(data.shape[0]):
            import pandas as pd

            df = pd.DataFrame({
                "date": range(data.shape[1]),
                "data": data[i].flatten(),
                "cols": 0,
            })
            df.to_csv(os.path.join(self.config.csv_dir, f"{data.shape[1]}_{i}.csv"), index=False)

    def _filter_data(self) -> List[str]:
        import pandas as pd

        good_data = []
        characteristics_dir = self.config.characteristics_dir
        feature_files = [
            name for name in os.listdir(characteristics_dir)
            if name.startswith("DATA_characteristics")
        ]
        for feature_file in feature_files:
            data_id = feature_file.split("/")[-1].split(".")[0].split("_")[-1]
            feature_df = pd.read_csv(os.path.join(characteristics_dir, feature_file))
            csv_path = os.path.join(self.config.csv_dir, f"{self.config.data_length}_{data_id}.csv")
            if not os.path.exists(csv_path):
                continue
            csv_df = pd.read_csv(csv_path)
            values = csv_df["data"].values
            if self._check_conditions(feature_df) and not self._is_zero_or_constant(values):
                good_data.append(data_id)
                if self.config.visualize:
                    cleaned = fix_anomalies(values)
                    os.makedirs(self.config.visualization_dir, exist_ok=True)
                    plt.figure(figsize=(10, 4))
                    plt.title(f"Data ID: {data_id}")
                    plt.plot(cleaned)
                    plt.savefig(os.path.join(self.config.visualization_dir, f"plot_{data_id}.png"))
                    plt.close()
        return good_data

    def _check_conditions(self, df) -> bool:
        if df["Trend"].values < self.config.trend_threshold:
            return False
        if df["Transition"].values > self.config.transition_threshold:
            return False
        if df["Seasonality"].values < self.config.seasonality_threshold:
            return False
        if abs(df["Shifting"].values) > self.config.shifting_threshold:
            return False
        if df["Long_term_jsd"].values > self.config.long_term_jsd_threshold:
            return False
        return True

    def _is_zero_or_constant(self, ts: np.ndarray) -> bool:
        if not self.config.enable_zero_check:
            return False
        return is_zero_or_constant(
            ts,
            zero_check_len=self.config.zero_check_len,
            zero_ratio_threshold=self.config.zero_ratio_threshold,
            zero_streak_threshold=self.config.zero_streak_threshold,
            near_zero_threshold=self.config.near_zero_threshold,
            zero_std_threshold=self.config.zero_std_threshold,
        )

    def _save_filtered_data(self, good_data: List[str]):
        import pandas as pd

        if not good_data:
            print("警告：没有筛选到符合条件的数据")
            return
        filtered_samples = []
        for data_id in good_data:
            csv_path = os.path.join(self.config.csv_dir, f"{self.config.data_length}_{data_id}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            cleaned = fix_anomalies(df["data"].values)
            filtered_samples.append(cleaned.reshape(1, -1, 1))
        if not filtered_samples:
            print("警告：没有有效的数据可以保存")
            return
        output_npy = np.concatenate(filtered_samples, axis=0)
        os.makedirs(os.path.dirname(self.config.npy_output_path), exist_ok=True)
        np.save(self.config.npy_output_path, output_npy)
        print(f"保存筛选后的数据到: {self.config.npy_output_path}")
        print(f"数据形状: {output_npy.shape}")
