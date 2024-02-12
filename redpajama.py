import datasets
from datasets import load_dataset
from pathlib import Path
target_dir = Path('datasets/redpajama/')
datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_dir)

ds = load_dataset("togethercomputer/RedPajama-Data-1T", 'default', data_dir=target_dir)
