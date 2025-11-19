import os
import json
import pathlib
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.index_builder import build_index_from_json
from src.preprocessing.chunking import SectionRecursiveConfig, SectionRecursiveStrategy, DocumentChunker

project_root = pathlib.Path(__file__).parent.parent
json_file_path = project_root / 'data' / 'threads_clean.json'
artifacts_dir = project_root / 'index' / 'sections'
embedding_model_path = 'models/Qwen3-Embedding-4B-Q5_K_M.gguf'
index_prefix = 'threads_index'

# ----- starting chunking ------ #
chunk_config = SectionRecursiveConfig(
    recursive_chunk_size=256,
    recursive_overlap=64
)
strategy = SectionRecursiveStrategy(config=chunk_config)
chunker = DocumentChunker(
    keep_tables=True,
    strategy=strategy
)
# ----- done chunking ------ #

with open(json_file_path, 'r') as f:
    threads_data = json.load(f)

build_index_from_json(
    threads_data=threads_data,
    chunker=chunker,
    chunk_config=chunk_config,
    embedding_model_path=embedding_model_path,
    artifacts_dir=artifacts_dir,
    index_prefix=index_prefix,
    do_visualize=False
)