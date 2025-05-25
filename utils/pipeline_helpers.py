import logging
import sys
import os
import polars as pl
import json
from typing import Any, Dict


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


class StepMetadataLogger:
    def __init__(self, output_dir: str, filename: str = "step_metadata.json"):
        self.output_dir = output_dir
        self.metadata: Dict[str, Any] = {}
        self.filepath = os.path.join(output_dir, filename)

    def add_stat(self, key: str, value: Any):
        self.metadata[key] = value

    def add_stats(self, stats: Dict[str, Any]):
        self.metadata.update(stats)

    def log_stats(self, level=logging.INFO):
        for key, value in self.metadata.items():
            logging.log(level, f"[METADATA] {key}: {value}")

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logging.info(f"Step metadata saved to {self.filepath}")

    def get(self, key: str, default=None):
        return self.metadata.get(key, default)