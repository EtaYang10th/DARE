# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class SampleTrainingTracker:
    """Track per-sample training counts and plot epoch-wise histograms."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.state_path = os.path.join(self.output_dir, "sample_training_counts.json")
        self.samples: dict[str, dict] = {}
        self.initialized = False

    @staticmethod
    def _normalize_index(raw_index):
        if isinstance(raw_index, np.generic):
            raw_index = raw_index.item()
        if isinstance(raw_index, (int, np.integer)):
            return int(raw_index)
        if isinstance(raw_index, str):
            return raw_index
        raise TypeError(f"Unsupported sample index type: {type(raw_index)}")

    @staticmethod
    def _extract_prompt_from_row(row, prompt_key: str) -> str:
        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict):
            question = extra_info.get("question")
            if isinstance(question, str):
                return question

        prompt_obj = row.get(prompt_key)
        if isinstance(prompt_obj, str):
            return prompt_obj
        if isinstance(prompt_obj, (list, dict)):
            return json.dumps(prompt_obj)
        if prompt_obj is None:
            return ""
        return str(prompt_obj)

    def initialize_from_dataframe(self, dataframe, prompt_key: str):
        if self.initialized:
            return

        for _, row in dataframe.iterrows():
            row_dict = row.to_dict()
            extra_info = row_dict.get("extra_info")
            if not isinstance(extra_info, dict) or "index" not in extra_info:
                raise KeyError("Dataset row missing extra_info.index; cannot build sample tracker.")
            index_value = self._normalize_index(extra_info["index"])
            index_key = str(index_value)
            if index_key in self.samples:
                continue
            self.samples[index_key] = {
                "index": index_value,
                "prompt": self._extract_prompt_from_row(row_dict, prompt_key),
                "train_count": 0,
            }

        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, dict) or "samples" not in existing:
                raise ValueError(f"Invalid tracker file schema: {self.state_path}")
            existing_samples = existing["samples"]
            if not isinstance(existing_samples, list):
                raise ValueError(f"Invalid samples field in tracker file: {self.state_path}")

            for item in existing_samples:
                if not isinstance(item, dict):
                    raise ValueError("Each tracked sample must be a dict.")
                if "index" not in item or "train_count" not in item:
                    raise ValueError("Tracked sample must contain index and train_count.")
                index_value = self._normalize_index(item["index"])
                index_key = str(index_value)
                if index_key in self.samples:
                    loaded_count = item["train_count"]
                    if not isinstance(loaded_count, int) or loaded_count < 0:
                        raise ValueError("train_count must be a non-negative integer.")
                    self.samples[index_key]["train_count"] = loaded_count

        self.initialized = True
        self.save_state()

    def increment_from_batch_indices(self, batch_indices):
        unique_indices = np.unique(batch_indices)
        for raw_index in unique_indices:
            index_value = self._normalize_index(raw_index)
            index_key = str(index_value)
            if index_key not in self.samples:
                raise KeyError(f"Batch contains unknown sample index: {index_value}")
            self.samples[index_key]["train_count"] += 1

    def save_state(self):
        sample_list = sorted(self.samples.values(), key=lambda x: str(x["index"]))
        payload = {
            "samples": sample_list,
            "num_samples": len(sample_list),
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def plot_epoch_histogram(self, epoch: int):
        train_counts = [item["train_count"] for item in self.samples.values()]
        if len(train_counts) == 0:
            raise ValueError("No samples available for histogram plotting.")

        unique_counts, sample_frequencies = np.unique(np.array(train_counts, dtype=int), return_counts=True)

        plt.figure(figsize=(8, 5))
        plt.bar(unique_counts, sample_frequencies, width=0.8, color="tab:blue")
        plt.xlabel("Training Count")
        plt.ylabel("Number of Samples")
        plt.title(f"Sample Training Count Distribution (Epoch {epoch})")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        figure_path = os.path.join(self.plots_dir, "sample_training_count_hist.png")
        plt.savefig(figure_path, dpi=150)
        plt.close()
        return figure_path
