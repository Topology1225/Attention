import datasets
from datasets import load_dataset
from pathlib import Path
target_dir = Path('./wmt14')

ds = load_dataset("togethercomputer/RedPajama-Data-1T", 'default', cache_dir=target_dir)
