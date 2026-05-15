# Step 1 — Generate 500+ pairs from your existing code
python metrostack_synthetic_gen_v2.py --project C:/MetroStack \
    --model qwen2.5-coder:7b --max-files 120 --out ./dataset

# Step 2 — Train (WSL2, NVIDIA GPU)
python metrobot_train.py \
    --train ./dataset/metrostack_train.jsonl \
    --val   ./dataset/metrostack_val.jsonl \
    --rank 16 --epochs 3 --out ./metrobot_output

# Step 3 — Deploy
cd ./metrobot_output/gguf
ollama create metrobot -f Modelfile
ollama run metrobot