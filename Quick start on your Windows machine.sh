pip install requests tiktoken tqdm colorama

# Run with no teacher first to test (just filename-based instructions)
python metrostack_dataset_builder.py --project C:\path\to\MetroStack --out-dir ./dataset

# Then with local Ollama as teacher (best quality, free)
python metrostack_dataset_builder.py --project C:\path\to\MetroStack \
  --teacher ollama --ollama-model qwen2.5-coder:7b --out-dir ./dataset